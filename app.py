#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time
import pandas as pd
from io import BytesIO
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, stable model for deterministic SQL text output
MODEL_NAME = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; }
.main-header { font-size: 2.2rem; color: #4a4a4a; font-weight: 700; margin-bottom: .25rem; }
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.0rem; border-radius: 12px; color: white; text-align: center; margin: .5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,.1);
}
.executive-summary {
    background-color: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 6px rgba(0,0,0,.05);
    margin-bottom: 1.2rem; border-left: 4px solid #764ba2;
}
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #667eea; }
.small-muted { color:#6c757d; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# KEYS & CONNECTIONS
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it to your app Secrets or .env")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=LLM_TEMPERATURE, request_timeout=60, max_retries=3)

llm = get_llm()

@st.cache_resource
def get_engine_and_schema():
    """Create engine and return schema string for the single allowed table."""
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
    except KeyError:
        st.error("❌ Missing DB_* secrets. Please add DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.")
        st.stop()

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
    )
    # smoke test
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    # Build schema doc from information_schema for the allowed table
    schema_rows = []
    q = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q)).fetchall()
        for c, t in rows:
            schema_rows.append(f'- "{c}" ({t})')
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(schema_rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

def make_sql_prompt(question: str, schema_text: str) -> str:
    # ---- SINGLE (EXECUTIVE) PROMPT — replaces any duplicates ----
    return f"""
You are Voylla DesignGPT Executive Edition, an expert SQL/analytics assistant for Voylla jewelry data designed for executive use.

Return a single VALID PostgreSQL SELECT query for the user's request. Follow these rules strictly:

GENERAL
- Read-only SELECT statements only.
- Only use table voylla."voylla_design_ai".
- Always exclude cancelled orders: WHERE "Sale Order Item Status" != 'CANCELLED'.
- Use double-quotes for all identifiers.
- If time period is vague (e.g., "this quarter", "last 6 months"), infer sensible filters using "Date".
- No explanations or markdown; output ONLY the SQL.

EXECUTIVE METRICS
- Revenue: SUM("Amount")
- Units: SUM("Qty")
- AOV: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100
- Growth Rate: use window functions (e.g., LAG) for period-over-period comparisons when needed.

SCHEMA
{schema_text}

USER QUESTION
{question}
"""

def generate_sql(question: str) -> str:
    prompt = make_sql_prompt(question, schema_doc)
    sql = llm.invoke(prompt).content.strip()
    # strip possible codefences if present
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
        sql = sql.strip()
    # safety
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains a non read-only keyword.")
    if 'voylla_design_ai' not in sql:
        raise ValueError('SQL must reference voylla."voylla_design_ai".')
    if 'SELECT' not in sql.upper():
        raise ValueError("No SELECT detected.")
    return sql

def run_sql_to_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}")

def summarize_for_executives(df: pd.DataFrame, user_q: str) -> str:
    # token-light preview
    preview_csv = df.head(50).to_csv(index=False)
    prompt = f"""
You are Voylla DesignGPT Executive Edition. In 6–9 crisp bullet points:
- Key findings (prioritize trends/opportunities/risks)
- 2–4 actionable recommendations
- Call out best/worst performers when relevant
Avoid tables and emojis. Keep it executive and concise.

USER QUESTION:
{user_q}

RESULTS PREVIEW (CSV):
{preview_csv}
"""
    return llm.invoke(prompt).content.strip()

def auto_chart(df: pd.DataFrame):
    if df.empty:
        return None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if not num_cols:
        return None
    y = num_cols[0]
    x = cat_cols[0] if cat_cols else None
    try:
        if x:
            work = df.copy()
            if len(work) > 15:
                work = work.nlargest(15, y) if y in work.columns else work.head(15)
            fig = px.bar(work, x=x, y=y, title=f"{y} by {x}")
        else:
            fig = px.line(df, y=y, title=f"Trend of {y}")
        fig.update_layout(height=420, showlegend=False)
        return fig
    except Exception:
        return None

def should_show_table(user_q: str, df: pd.DataFrame) -> bool:
    """Heuristic: only show table when it aids inspection."""
    q = user_q.lower()
    wants_detail = any(k in q for k in ["table", "list", "rows", "raw", "detail", "download", "export"])
    compact_enough = (len(df) <= 100 and len(df.columns) <= 8)
    ranked_or_top = any(k in q for k in ["top ", "best ", "rank", "highest", "lowest"])
    return wants_detail or ranked_or_top or compact_enough

def should_show_chart(user_q: str, df: pd.DataFrame) -> bool:
    """Heuristic: show chart for trends/comparisons, not for wide or very long tables."""
    q = user_q.lower()
    trendy = any(k in q for k in ["trend", "growth", "yoy", "qoq", "mom", "over time", "by month", "by week"])
    compare = " by " in q or any(k in q for k in ["compare", "vs", "split", "breakdown"])
    sized_ok = 2 <= len(df) <= 500
    return (trendy or compare) and sized_ok

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)

    with engine.connect() as conn:
        try:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM voylla."voylla_design_ai" 
                WHERE "Sale Order Item Status" != 'CANCELLED'
            """)).scalar()
            st.success(f"✅ Connected: {count:,} active records")
        except Exception as e:
            st.error(f"❌ Connection issue: {e}")

    st.markdown("---")
    st.header("💡 Executive Questions")
    presets = [
        "Show me top 10 products by revenue this quarter",
        "What are our best performing channels by growth rate?",
        "Compare this year's revenue to last year by month",
        "Which design styles have the highest average order value?",
        "Show me channel-wise revenue and units this month"
    ]
    for q in presets:
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state["auto_q"] = q

    st.markdown("---")
    st.subheader("Display Preferences")
    smart_display = st.checkbox("Smart display (summary first; table/chart only when helpful)", value=True)
    force_table = st.checkbox("Always show table", value=False)
    force_chart = st.checkbox("Always show chart", value=False)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state["last_df"] = None
            st.rerun()
    with col2:
        st.caption("Agent-free • stable • no verbose logs")

# =========================
# SESSION
# =========================
if "chat" not in st.session_state: st.session_state.chat = []
if "auto_q" not in st.session_state: st.session_state.auto_q = None
if "last_df" not in st.session_state: st.session_state.last_df = None
if "last_sql" not in st.session_state: st.session_state.last_sql = ""

# =========================
# HEADER
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics — agent-free & reliable")

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"] if m["role"]=="assistant" else m["content"])

# Always-visible input (keeps mobile keyboard)
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

if inp:
    st.session_state.chat.append({"role":"user","content":inp})
    with st.chat_message("user"):
        st.markdown(inp)

    with st.spinner("Polishing your insights… 💎"):
        try:
            sql = generate_sql(inp)
            df = run_sql_to_df(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = sql
            summary = summarize_for_executives(df, inp)
        except Exception as e:
            summary = f"⚠️ Could not complete request: {e}"
            df = pd.DataFrame()

    # Assistant message
    with st.chat_message("assistant"):
        if summary:
            st.markdown(f"<div class='assistant-message'>{summary}</div>", unsafe_allow_html=True)

    # Show SQL (collapsible)
    with st.expander("View generated SQL"):
        st.code(st.session_state.last_sql, language="sql")

    # Smart, non-intrusive display of results
    if not df.empty:
        # Decide whether to show table/chart
        show_tbl = force_table or (not smart_display) or should_show_table(inp, df)
        show_cht = force_chart or (not smart_display) or should_show_chart(inp, df)

        # Put table and chart behind expanders so they don't clutter the view
        if show_tbl:
            with st.expander("View results table"):
                st.dataframe(df, use_container_width=True)
                st.caption(
                    "<div class='small-muted'>Tip: Use the download button below for the full dataset.</div>",
                    unsafe_allow_html=True
                )

        if show_cht:
            fig = auto_chart(df)
            if fig:
                with st.expander("View chart"):
                    st.plotly_chart(fig, use_container_width=True)

# Export
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    export_df = st.session_state.last_df.copy()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
        meta = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'SQL'],
            'Value': [len(export_df), len(export_df.columns),
                      datetime.now().strftime("%Y-%m-%d %H:%M"),
                      st.session_state.last_sql[:32000]]
        })
        meta.to_excel(writer, index=False, sheet_name='Summary')

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "💾 Download Executive Report",
        data=output.getvalue(),
        file_name=f"voylla_executive_report_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="download_exec"
    )

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "MoM", "trending", "best performing" •
Say "show me channel-wise performance this quarter" to start.
</div>
""", unsafe_allow_html=True)
