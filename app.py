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
MODEL_NAME = "gpt-4.1-mini"   # <- stable with LangChain and great at following format
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
    return f"""
You are a senior data analyst. Return a single **valid PostgreSQL** SELECT query for the question.
STRICT RULES:
- Read-only SELECT statements only.
- Only use table voylla."voylla_design_ai".
- Always filter out cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'.
- If time period is vague (e.g., "this quarter", "last 6 months"), infer sensible filters using "Date".
- Use double-quotes for all identifiers.
- Do not add explanations, markdown, or fencing; output ONLY the SQL.

SCHEMA:
{schema_text}

# EXECUTIVE REPORTING GUIDELINES
- Focus on business insights, not just data
- Highlight trends, opportunities, and risks
- Compare performance metrics (YoY, MoM, QoQ)
- Use clear, concise language appropriate for executives
- Provide actionable recommendations when possible

# DATABASE SCHEMA: voylla."voylla_design_ai"

## KEY COLUMNS FOR EXECUTIVE ANALYSIS
### Business Metrics
- "Date" (timestamp) — Transaction date
- "Channel" (text) — Sales platform (Cloudtail, FLIPKART, MYNTRA, NYKAA, etc.)
- "Sale Order Item Status" (text) — Filter with: WHERE "Sale Order Item Status" != 'CANCELLED'
- "Qty" (integer) — Units sold
- "Amount" (numeric) — Revenue (Qty × price)
- "MRP" (numeric) — Maximum Retail Price
- "Cost Price" (numeric) — Unit cost

### Design Intelligence
- "Design Style" (text) — Aesthetic (Tribal, Contemporary, Traditional/Ethnic, Minimalist)
- "Form" (text) — Shape (Triangle, Stud, Hoop, Jhumka, Ear Cuff)
- "Metal Color" (text) — Finish (Antique Silver, Yellow Gold, Rose Gold, Silver, Antique Gold, Oxidized Black)
- "Look" (text) — Occasion/vibe (Oxidized, Everyday, Festive, Party, Wedding)
- "Central Stone" (text) — Primary gemstone

# MANDATORY FILTERS
- Always exclude cancelled orders: WHERE "Sale Order Item Status" != 'CANCELLED'
- For time-based questions, use appropriate date ranges
- When comparing channels, ensure fair comparison by including only common time periods

# EXECUTIVE METRICS
- Revenue: SUM("Amount")
- Units: SUM("Qty")
- Average Order Value: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100
- Growth Rate: Use LAG() function for period-over-period comparisons

# RESPONSE FORMATTING FOR EXECUTIVES
1. Start with a concise executive summary of key findings
2. Present data in clean, well-formatted markdown tables
3. Highlight the most important insights in bold
4. Include visualizations when appropriate (charts will be auto-generated)
5. End with actionable recommendations or suggested next analyses


QUESTION:
{question}
"""

def generate_sql(question: str) -> str:
    prompt = make_sql_prompt(question, schema_doc)
    sql = llm.invoke(prompt).content.strip()
    # strip possible codefences if the model adds them
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
        sql = sql.strip()
    # safety
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains a non read-only keyword.")
    if "voylla_design_ai" not in sql:
        raise ValueError("SQL must reference voylla.\"voylla_design_ai\".")
    return sql

def run_sql_to_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}")

def summarize_for_executives(df: pd.DataFrame, user_q: str) -> str:
    # Keep token-light by downsampling preview
    preview_csv = df.head(50).to_csv(index=False)
    prompt = f"""
You are an executive analyst. Using the user's question and the CSV preview of results,
write a concise executive summary with:
1) Key findings (bullets),
2) 2-4 actionable recommendations,
3) If applicable, call out best/worst performers.

Be crisp. No markdown tables here.

USER QUESTION:
{user_q}

RESULTS PREVIEW (CSV):
{preview_csv}
"""
    return llm.invoke(prompt).content.strip()

def auto_chart(df: pd.DataFrame):
    # pick a sensible default: first object column as x, largest numeric as y
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
            # top-15 for readability
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state["last_df"] = None
            st.session_state["last_sql"] = ""
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
        st.markdown(m["content"])

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

    # NEW: persist assistant message so it survives reruns (e.g. after download)
    if summary:
        st.session_state.chat.append({"role": "assistant", "content": summary})  # NEW

    # Show SQL (collapsible)
    with st.expander("View generated SQL"):
        st.code(st.session_state.last_sql, language="sql")

    # Data table & chart
    if not df.empty:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        fig = auto_chart(df)
        if fig:
            st.subheader("📊 Data Visualization")
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

    # NEW: reset buffer before passing to download_button so data is served correctly after rerun
    output.seek(0)  # NEW

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
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" •
Say "show me channel-wise performance this quarter" to start.
</div>
""", unsafe_allow_html=True)
