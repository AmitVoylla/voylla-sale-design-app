#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, random
import pandas as pd
from io import BytesIO
import plotly.express as px
from datetime import datetime

# ---------- Spinner messages ----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as file:
        lines = [line.strip().split('. ', 1)[1] for line in file if '. ' in line]
else:
    lines = [
        "Crunching the numbers with a sparkle ✨",
        "Polishing your insights… 💎",
        "Crafting your jewelry analytics… 💍",
    ]

# ---------- Config ----------
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- LLM ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, request_timeout=60, max_retries=3)

llm = get_llm()

# ---------- DB Connection ----------
@st.cache_resource
def get_engine_and_schema():
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
    except KeyError:
        st.error("❌ Missing DB_* secrets.")
        st.stop()

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

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

# ---------- Helpers ----------
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

def make_sql_prompt(question: str, schema_text: str) -> str:
    return f"""
You are an expert SQL analyst for Voylla jewelry data.
Generate a valid PostgreSQL SELECT query for the question.
RULES:
- Use only voylla."voylla_design_ai".
- Always include WHERE "Sale Order Item Status" != 'CANCELLED'.
- For time-based questions, infer sensible date filters using "Date".
- Use double-quotes for all identifiers.
- Output ONLY the SQL query.
SCHEMA:
{schema_text}
QUESTION:
{question}
"""

def generate_sql(question: str) -> str:
    prompt = make_sql_prompt(question, schema_doc)
    sql = llm.invoke(prompt).content.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains dangerous keywords.")
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
    preview_csv = df.head(50).to_csv(index=False)
    prompt = f"""
You are an executive analyst for Voylla jewelry. Using the user's question and CSV preview of results, provide a concise executive summary with:
1) Key findings (2-4 bullets, focus on trends, opportunities, risks)
2) 2-4 actionable recommendations
3) Highlight best/worst performers if relevant
4) Suggest if a visualization (e.g., bar chart, line chart) would enhance understanding
Use clear, professional language suitable for executives.
USER QUESTION:
{user_q}
RESULTS PREVIEW (CSV):
{preview_csv}
"""
    return llm.invoke(prompt).content.strip()

def should_display_chart(df: pd.DataFrame, summary: str) -> bool:
    if df.empty:
        return False
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    return "visualization" in summary.lower() and len(num_cols) >= 1 and len(cat_cols) >= 1 and len(df) <= 100

def auto_chart(df: pd.DataFrame):
    if df.empty:
        return None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if not num_cols or not cat_cols:
        return None
    y = num_cols[0]
    x = cat_cols[0]
    try:
        work = df.copy()
        if len(work) > 15:
            work = work.nlargest(15, y) if y in work.columns else work.head(15)
        fig = px.bar(work, x=x, y=y, title=f"{y} by {x}")
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception:
        return None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    try:
        with engine.connect() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM voylla."voylla_design_ai" 
                WHERE "Sale Order Item Status" != 'CANCELLED'
            """)).scalar()
            st.success(f"✅ Connected: {count:,} active records")
    except Exception as e:
        st.error(f"❌ Connection issue: {e}")

    st.markdown("---")
    st.header("💡 Executive Questions")
    questions = [
        "Show me top 10 products by revenue this quarter",
        "What are our best performing channels by growth rate?",
        "Compare this year's revenue to last year by month",
        "Which design styles have the highest average order value?",
        "Which metal colors are trending this season?"
    ]
    for q in questions:
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state.auto_question = q

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_df = None
            st.rerun()
    with col2:
        st.caption("Voylla DesignGPT v2.0")

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auto_question" not in st.session_state:
    st.session_state.auto_question = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

# ---------- UI ----------
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='assistant-message'>{message['content']}</div>" if message["role"] == "assistant" else message["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask an executive question about sales or design trends…")
if st.session_state.auto_question:
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner(random.choice(lines)):
        try:
            sql = generate_sql(user_input)
            df = run_sql_to_df(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = sql
            summary = summarize_for_executives(df, user_input)
        except Exception as e:
            summary = f"⚠️ Could not complete request: {e}"
            df = pd.DataFrame()

    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant-message'>{summary}</div>", unsafe_allow_html=True)

    with st.expander("View generated SQL"):
        st.code(st.session_state.last_sql, language="sql")

    if not df.empty:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        if should_display_chart(df, summary):
            fig = auto_chart(df)
            if fig:
                st.subheader("📊 Data Visualization")
                st.plotly_chart(fig, use_container_width=True)

# ---------- Export ----------
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "💾 Download Executive Report",
        data=output.getvalue(),
        file_name=f"voylla_executive_report_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="executive_download"
    )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities • 
Use terms like "YoY", "QoQ", "market share", "trending", "best performing"
</div>
""", unsafe_allow_html=True)
