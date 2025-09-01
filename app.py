#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time, json
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configuration (fast + stable)
MODEL_NAME = "gpt-4.1-mini"   # more deterministic and quick for SQL/JSON tasks
LLM_TEMPERATURE = 0.05

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
.assistant-message { 
    background-color: #f8f9fa; 
    border-radius: 12px; 
    padding: 1rem; 
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.insight-card {
    background-color: #fff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #ffd700;
}
.recommendation-card {
    background-color: #f0f8ff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #4682b4;
}
.stButton button { width: 100%; }
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
    return ChatOpenAI(model=MODEL_NAME, temperature=LLM_TEMPERATURE, request_timeout=60, max_retries=2)

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
    rows = []
    q = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        for c, t, n in conn.execute(text(q)).fetchall():
            rows.append(f'- "{c}" ({t}, nullable: {n})')
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# MEMORY
# =========================
@st.cache_resource
def get_memory():
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")

memory = get_memory()

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\\b", re.I)

def make_sql_prompt(question: str, schema_text: str, history: list | None = None) -> str:
    history_text = ""
    if history:
        # only last few utterances to keep prompt tight
        compact = []
        for m in history[-4:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            compact.append(f"{role}: {content}")
        history_text = "\\n# CONVERSATION HISTORY (last 4):\\n" + "\\n".join(compact)

    return f"""
You are a senior data analyst generating a single **valid PostgreSQL** SELECT query.

STRICT RULES:
- Read-only SELECT statements only.
- Use only table voylla."voylla_design_ai" (alias allowed).
- Always exclude cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'.
- Use ONLY columns that exist in the schema. Do not invent columns.
- If the request needs revenue, use SUM("Amount"); units use SUM("Qty").
- Profit/margin proxy is allowed using "Cost Price": (SUM("Amount") - SUM("Cost Price" * "Qty")) AS profit.
- If time period is vague (e.g., "this quarter", "last 6 months"), derive filter using column "Date".
- Use double-quotes for all identifiers.
- Output ONLY the SQL (no markdown, no fencing, no commentary).

SCHEMA:
{schema_text}
{history_text}

QUESTION:
{question}
""".strip()

def refine_sql_prompt(question: str, schema_text: str, error_msg: str, last_sql: str) -> str:
    return f"""
Your previous SQL caused an error.

QUESTION:
{question}

SCHEMA:
{schema_text}

PREVIOUS SQL (incorrect):
{last_sql}

DB ERROR:
{error_msg}

TASK:
Return a corrected **single** PostgreSQL SELECT query that obeys ALL rules:
- Read-only; use only voylla."voylla_design_ai".
- Must include WHERE "Sale Order Item Status" != 'CANCELLED'.
- Use only columns from the schema above.
- Output only the SQL (no Markdown, no backticks, no explanations).
""".strip()

def enforce_cancelled_filter(sql: str) -> str:
    """Ensure the CANCELLED filter exists; if missing, inject it safely."""
    if re.search(r'"Sale Order Item Status"\\s*!=\\s*\'CANCELLED\'', sql, re.I):
        return sql
    # find presence of WHERE
    if re.search(r'\\bWHERE\\b', sql, re.I):
        # add AND condition before GROUP BY / ORDER BY / LIMIT / ;
        return re.sub(
            r'\\bWHERE\\b',
            'WHERE "Sale Order Item Status" != \'CANCELLED\' AND ',
            sql,
            flags=re.I,
            count=1
        )
    else:
        # insert WHERE before GROUP BY/ORDER BY/LIMIT/;
        split_pat = r'\\b(GROUP\\s+BY|ORDER\\s+BY|LIMIT)\\b'
        m = re.search(split_pat, sql, re.I)
        if m:
            pos = m.start()
            prefix, suffix = sql[:pos], sql[pos:]
            return f'{prefix} WHERE "Sale Order Item Status" != \'CANCELLED\' {suffix}'
        else:
            # no trailing clause; just append WHERE
            sql_wo_semicolon = sql.strip().rstrip(';')
            return f'{sql_wo_semicolon} WHERE "Sale Order Item Status" != \'CANCELLED\';'

def enforce_schema_qual(sql: str) -> str:
    """Ensure table is schema-qualified."""
    # If query already has voylla."voylla_design_ai", return.
    if 'voylla."voylla_design_ai"' in sql:
        return sql
    # Replace unqualified references "voylla_design_ai" with voylla."voylla_design_ai"
    sql = re.sub(r'\\b"voylla_design_ai"\\b', 'voylla."voylla_design_ai"', sql)
    sql = re.sub(r'\\bvoylla_design_ai\\b', 'voylla."voylla_design_ai"', sql)
    return sql

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*", "", s).strip()
        s = s[:-3] if s.endswith("```") else s
    return s.strip()

def validate_read_only(sql: str):
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains a non read-only keyword.")

def _llm_sql(question: str, history: list | None = None) -> str:
    prompt = make_sql_prompt(question, schema_doc, history)
    raw = llm.invoke(prompt).content
    sql = strip_code_fences(raw)
    sql = enforce_schema_qual(sql)
    sql = enforce_cancelled_filter(sql)
    validate_read_only(sql)
    if 'voylla."voylla_design_ai"' not in sql:
        raise ValueError("SQL must reference voylla.\"voylla_design_ai\".")
    return sql

@st.cache_data(show_spinner=False, ttl=600)
def run_sql_to_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)

def generate_and_run_sql(question: str, history: list | None = None) -> tuple[pd.DataFrame, str]:
    """Generate SQL; execute; if fails, auto-refine once with DB error message."""
    sql1 = _llm_sql(question, history)
    try:
        df = run_sql_to_df(sql1)
        return df, sql1
    except Exception as e:
        # Ask the model to fix using the actual DB error
        fix_prompt = refine_sql_prompt(question, schema_doc, str(e), sql1)
        sql2 = strip_code_fences(llm.invoke(fix_prompt).content)
        sql2 = enforce_schema_qual(sql2)
        sql2 = enforce_cancelled_filter(sql2)
        validate_read_only(sql2)
        if 'voylla."voylla_design_ai"' not in sql2:
            raise RuntimeError(f"Second SQL still invalid. Error was: {e}")
        # Second try
        df = run_sql_to_df(sql2)
        return df, sql2

def analyze_data_fast(df: pd.DataFrame, user_q: str) -> dict:
    """Deterministic, no-LLM analysis for speed and fewer mistakes."""
    if df is None or df.empty:
        return {
            "executive_summary": "No data returned for the query.",
            "key_metrics": {},
            "insights": [],
            "recommendations": [],
            "followup_questions": []
        }

    # Basic metrics (guarded)
    rev = float(df["Amount"].sum()) if "Amount" in df.columns else None
    units = float(df["Qty"].sum()) if "Qty" in df.columns else None
    aov = (rev / units) if (rev is not None and units and units != 0) else None
    profit = None
    margin_pct = None
    if all(col in df.columns for col in ["Cost Price", "Qty", "Amount"]):
        profit = float((df["Amount"] - df["Cost Price"] * df["Qty"]).sum())
        margin_pct = (profit / rev * 100.0) if (rev and rev != 0) else None

    # Top splits, if present
    splits = []
    for col in ["Channel", "Design Style", "Metal Color", "Form", "Look"]:
        if col in df.columns:
            grp = (df.groupby(col)["Amount"].sum().sort_values(ascending=False).head(5)
                   if "Amount" in df.columns else
                   df.groupby(col).size().sort_values(ascending=False).head(5))
            splits.append((col, grp))

    insights = []
    if splits:
        col, grp = splits[0]
        top_name = str(grp.index[0])
        top_val = float(grp.iloc[0])
        share = (top_val / rev * 100.0) if (rev and rev != 0 and "Amount" in df.columns) else None
        insights.append({
            "title": f"Top {col}",
            "description": f"{top_name} leads the {col.lower()} mix.",
            "impact": "Prioritize inventory & marketing on the leading segment.",
            "data_support": f"{top_name}: {top_val:,.2f}" + (f" ({share:.1f}% of revenue)" if share else "")
        })

    if margin_pct is not None:
        insights.append({
            "title": "Healthy profit signal" if margin_pct >= 20 else "Margin risk",
            "description": "Overall margin outlook from returned dataset.",
            "impact": "Adjust pricing or costs based on realized margins.",
            "data_support": f"Profit: {profit:,.2f}, Margin: {margin_pct:.1f}%"
        })

    recs = [
        {
            "title": "Double down on top segment",
            "description": "Allocate budget and placements to the best performing channel/segment.",
            "expected_impact": "Higher ROAS and faster sell-through"
        },
        {
            "title": "Focus price-band mix",
            "description": "Optimize assortment around price bands contributing most to AOV.",
            "expected_impact": "Improved revenue per unit"
        }
    ]
    if margin_pct is not None and margin_pct < 20:
        recs.append({
            "title": "Cost review",
            "description": "Review BOM and vendor terms for high-volume SKUs with weak margins.",
            "expected_impact": "2–5% margin lift"
        })

    key_metrics = {}
    if rev is not None: key_metrics["Revenue"] = f"₹{rev:,.2f}"
    if units is not None: key_metrics["Units"] = f"{units:,.0f}"
    if aov is not None: key_metrics["AOV"] = f"₹{aov:,.2f}"
    if margin_pct is not None: key_metrics["Margin %"] = f"{margin_pct:.1f}%"

    return {
        "executive_summary": "Snapshot built from the returned dataset with priority splits and margin proxy.",
        "key_metrics": key_metrics,
        "insights": insights,
        "recommendations": recs,
        "followup_questions": [
            "Drill down by channel and month",
            "Compare top 10 SKUs YoY",
            "Check price-band contribution and returns rate"
        ]
    }

def analyze_data_llm(df: pd.DataFrame, user_q: str, history: list | None = None) -> dict:
    """LLM analysis (kept for depth); guarded with JSON parse fallback."""
    analysis_prompt = f"""
You are an executive data analyst at Voylla.
Analyze the provided data to answer the user's question and provide actionable insights.

USER QUESTION: {user_q}

DATA PREVIEW (first 20 rows):
{df.head(20).to_csv(index=False)}

DATA STRUCTURE:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}

ANALYSIS REQUIREMENTS:
1. Executive summary
2. 3–5 insights with supporting data
3. Trends/patterns/anomalies
4. Compare metrics (YoY/MoM/QoQ) if relevant
5. 3–5 actionable recommendations
6. 2–3 follow-up questions

Return strict JSON with:
{{
  "executive_summary": "...",
  "key_metrics": {{
    "metric1": "value1"
  }},
  "insights": [
    {{"title": "...", "description": "...", "impact": "...", "data_support": "..."}}
  ],
  "recommendations": [
    {{"title": "...", "description": "...", "expected_impact": "..."}}
  ],
  "followup_questions": ["...", "..."]
}}
""".strip()

    try:
        raw = llm.invoke(analysis_prompt).content.strip()
        return json.loads(raw)
    except Exception:
        return analyze_data_fast(df, user_q)  # safe fallback

def create_advanced_visualizations(df: pd.DataFrame):
    """Create multiple visualizations based on data."""
    visualizations = []
    if df is None or df.empty:
        return visualizations

    # Time series if Date-like & numeric present
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if date_cols and numeric_cols:
        date_col = date_cols[0]
        try:
            dft = df.copy()
            dft[date_col] = pd.to_datetime(dft[date_col])
            ts = dft.groupby(pd.Grouper(key=date_col, freq='M'))[numeric_cols[0]].sum().reset_index()
            fig = px.line(ts, x=date_col, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]} Over Time")
            fig.update_layout(height=380)
            visualizations.append(fig)
        except Exception:
            pass

    # Top categories
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        top_df = df.groupby(cat_col)[numeric_cols[0]].sum().nlargest(10).reset_index()
        fig = px.bar(top_df, x=cat_col, y=numeric_cols[0], title=f"Top 10 {cat_col} by {numeric_cols[0]}")
        fig.update_layout(height=380, xaxis_tickangle=-45)
        visualizations.append(fig)

    # Distribution
    if numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        fig.update_layout(height=380)
        visualizations.append(fig)

    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Metrics")
        fig.update_layout(height=380)
        visualizations.append(fig)

    return visualizations

def display_analysis(analysis: dict):
    """Render analysis cleanly."""
    st.markdown("### 📋 Executive Summary")
    st.info(analysis.get("executive_summary", "No summary available."))

    # Key Metrics
    km = analysis.get("key_metrics", {})
    if km:
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(min(len(km), 4))
        for i, (metric, value) in enumerate(km.items()):
            cols[i % len(cols)].metric(metric, value)

    # Insights
    ins = analysis.get("insights", [])
    if ins:
        st.markdown("### 💡 Key Insights")
        for insight in ins:
            with st.expander(f"🔍 {insight.get('title', 'Insight')}"):
                st.markdown(f"**Description**: {insight.get('description', '')}")
                st.markdown(f"**Impact**: {insight.get('impact', '')}")
                if insight.get('data_support'):
                    st.markdown(f"**Data Support**: {insight.get('data_support')}")

    # Recommendations
    recs = analysis.get("recommendations", [])
    if recs:
        st.markdown("### 🎯 Recommendations")
        for rec in recs:
            st.success(f"**{rec.get('title', 'Recommendation')}**: {rec.get('description', '')}")
            if rec.get('expected_impact'):
                st.caption(f"Expected impact: {rec.get('expected_impact')}")

    # Follow-ups
    fqs = analysis.get("followup_questions", [])
    if fqs:
        st.markdown("### 🔎 Suggested Follow-ups")
        for i, question in enumerate(fqs):
            if st.button(question, key=f"followup_{i}"):
                st.session_state.auto_q = question
                st.rerun()

def make_excel_download(df: pd.DataFrame, analysis: dict, sql: str) -> BytesIO:
    """Create a multi-sheet Excel with data + analysis + metadata."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Executive_Report')

        # Analysis summary
        if analysis:
            analysis_data = []
            for section in ["executive_summary", "key_metrics", "insights", "recommendations"]:
                if section in analysis:
                    if section == "key_metrics":
                        for k, v in analysis[section].items():
                            analysis_data.append({"Section": section, "Content": f"{k}: {v}"})
                    elif section in ("insights", "recommendations"):
                        for item in analysis[section]:
                            analysis_data.append({"Section": section, "Content": str(item)})
                    else:
                        analysis_data.append({"Section": section, "Content": analysis[section]})
            pd.DataFrame(analysis_data).to_excel(writer, index=False, sheet_name='Analysis_Summary')

        meta = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'SQL'],
            'Value': [len(df), len(df.columns),
                      datetime.now().strftime("%Y-%m-%d %H:%M"),
                      sql[:32000]]
        })
        meta.to_excel(writer, index=False, sheet_name='Metadata')
    output.seek(0)
    return output

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)

    # DB health + rolling revenue
    with engine.connect() as conn:
        try:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM voylla."voylla_design_ai"
                WHERE "Sale Order Item Status" != 'CANCELLED'
            """)).scalar()
            revenue = conn.execute(text("""
                SELECT SUM("Amount") FROM voylla."voylla_design_ai"
                WHERE "Sale Order Item Status" != 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '30 days'
            """)).scalar()
            st.success(f"✅ Connected: {count:,} active records")
            st.metric("30-Day Revenue", f"₹{revenue:,.2f}" if revenue else "N/A")
        except Exception as e:
            st.error(f"❌ Connection issue: {e}")

    st.markdown("---")
    st.header("⚙️ Options")
    st.checkbox("⚡ Quick Analysis (no LLM)", key="quick_mode", value=True,
                help="Faster & deterministic. Turn off to use the LLM for deeper narrative analysis.")

    st.markdown("---")
    st.header("💡 Executive Questions")
    presets = [
        "Show me top 10 products by revenue this quarter",
        "What are our best performing channels by growth rate?",
        "Compare this year's revenue to last year by month",
        "Which design styles have the highest average order value?",
        "Show me channel-wise revenue and units this month",
        "Analyze sales trends by metal color and design style",
        "What is our profit margin by product category?",
        "Identify seasonal patterns in our sales data"
    ]
    for q in presets:
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state["auto_q"] = q
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat = []
            st.session_state.results = []
            st.session_state.auto_q = None
            memory.clear()
            st.rerun()
    with col2:
        st.caption("Advanced AI • Context-aware • Executive Insights")

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state: st.session_state.chat = []
if "auto_q" not in st.session_state: st.session_state.auto_q = None
# Persistent RESULTS HISTORY: list of dicts {question, sql, df, analysis, ts}
if "results" not in st.session_state: st.session_state.results = []

# =========================
# HEADER
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics — Results persist across questions")

# Render prior chat (lightweight)
for m in st.session_state.chat[-12:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Always-visible input (keeps mobile keyboard)
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

# =========================
# RUN PIPELINE
# =========================
if inp:
    st.session_state.chat.append({"role": "user", "content": inp})
    with st.chat_message("user"):
        st.markdown(inp)

    with st.spinner("Conducting analysis… 💎"):
        try:
            # 1) Generate + run SQL (with auto-refine on error)
            df, sql = generate_and_run_sql(inp, st.session_state.chat)

            # 2) Analysis (fast or LLM)
            if st.session_state.get("quick_mode", True):
                analysis = analyze_data_fast(df, inp)
            else:
                analysis = analyze_data_llm(df, inp, st.session_state.chat)

            # 3) Save to memory + results history (PERSIST)
            memory.save_context({"input": inp}, {"output": f"Analysis ready: {analysis.get('executive_summary', '')}"})
            st.session_state.results.append({
                "question": inp,
                "sql": sql,
                "df": df,
                "analysis": analysis,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        except Exception as e:
            err = f"⚠️ Could not complete request: {e}"
            st.session_state.chat.append({"role": "assistant", "content": err})
            st.error(err)

# =========================
# RENDER ALL RUNS (PERSISTENT)
# =========================
# Newest first
for i, res in enumerate(reversed(st.session_state.results)):
    idx = len(st.session_state.results) - 1 - i  # original index
    with st.container(border=True):
        st.subheader(f"Run #{idx + 1} • {res['ts']}")
        st.markdown(f"**Question:** {res['question']}")

        # Analysis section
        with st.chat_message("assistant"):
            display_analysis(res["analysis"])

        # Show SQL (collapsible)
        with st.expander("View generated SQL"):
            st.code(res["sql"], language="sql")

        # Visualizations
        if res["df"] is not None and not res["df"].empty:
            st.markdown("#### 📈 Visualizations")
            figs = create_advanced_visualizations(res["df"])
            if figs:
                cols = st.columns(2)
                for j, fig in enumerate(figs):
                    cols[j % 2].plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📋 Data Preview")
            st.dataframe(res["df"], use_container_width=True, height=360)

            # Per-run Excel download
            excel_buf = make_excel_download(res["df"], res["analysis"], res["sql"])
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "💾 Download Executive Report (this run)",
                data=excel_buf.getvalue(),
                file_name=f"voylla_executive_report_run{idx+1}_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"download_exec_{idx}_{ts}"
            )

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "trending", "best performing" •
Try "analyze profitability by design style" or "show me seasonal trends" for advanced insights.
</div>
""", unsafe_allow_html=True)
