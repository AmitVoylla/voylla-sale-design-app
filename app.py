#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd
from io import BytesIO, StringIO
import re
import random
import plotly.express as px
from datetime import datetime
import time

# ===================== CONFIG =====================

st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

SPINNER_LINES = [
    "Analyzing jewelry trends with precision…",
    "Generating strategic insights…",
    "Processing your data query…",
    "Crafting actionable analytics…",
    "Polishing your report…",
]

# ===================== STYLES =====================

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; }
.main-header { font-size: 2.5rem; color: #4a4a4a; font-weight: 700; margin-bottom: 0.5rem; }
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] { color: #212529 !important; font-size: 1rem; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.executive-summary { background-color: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem; border-left: 4px solid #764ba2; }
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# ===================== SECRETS / LLM =====================

TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        lines = [ln.strip().split('. ', 1)[1] for ln in f if '. ' in ln]
else:
    lines = SPINNER_LINES

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets or environment.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def get_llm():
    # small, fast, reliable
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, request_timeout=60, max_retries=3)

llm = get_llm()

# ===================== DATABASE =====================

@st.cache_resource
def get_database_connection():
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            db_host = st.secrets["DB_HOST"]
            db_port = st.secrets["DB_PORT"]
            db_name = st.secrets["DB_NAME"]
            db_user = st.secrets["DB_USER"]
            db_password = st.secrets["DB_PASSWORD"]

            engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = SQLDatabase(engine, include_tables=["voylla_design_ai"], schema="voylla")
            return db
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Database connection failed after {max_retries} attempts: {e}")
            st.stop()

db = get_database_connection()

# Expose the engine for direct SQL fallbacks
ENGINE = db._engine  # (private attr, but stable in practice)

# ===================== HELPERS =====================

def markdown_to_dataframe(markdown_text: str):
    """Parse a markdown table into a DataFrame."""
    if not markdown_text or '|' not in markdown_text:
        return None
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    table_start = None
    for i, line in enumerate(lines):
        if '|' in line and i+1 < len(lines) and re.match(r'^[\s\|:-]+$', lines[i+1]):
            table_start = i
            break
    if table_start is None:
        return None
    table_lines = []
    for i in range(table_start, len(lines)):
        line = lines[i]
        if '|' in line and not re.match(r'^[\s\|:-]+$', line):
            table_lines.append(line)
        elif len(table_lines) > 0:
            break
    if len(table_lines) < 2:
        return None
    cleaned = []
    for ln in table_lines:
        if not ln.startswith('|'):
            ln = '|' + ln
        if not ln.endswith('|'):
            ln = ln + '|'
        cleaned.append(ln)
    try:
        df = pd.read_csv(StringIO("\n".join(cleaned)), sep="|", skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        return df
    except Exception:
        return None

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Executive_Report"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        meta = {
            'Metric': ['Total Rows', 'Total Columns', 'Export Date'],
            'Value': [len(df), len(df.columns), datetime.now().strftime("%Y-%m-%d %H:%M")]
        }
        pd.DataFrame(meta).to_excel(writer, index=False, sheet_name='Summary')
    return output.getvalue()

def top_channels_sales(limit: int = 10) -> pd.DataFrame:
    """Direct SQL fallback for Top Channels KPI."""
    sql = """
    SELECT
      "Channel",
      SUM("Qty")   AS total_units_sold,
      SUM("Amount") AS total_revenue
    FROM voylla."voylla_design_ai"
    WHERE "Sale Order Item Status" != 'CANCELLED'
    GROUP BY "Channel"
    ORDER BY total_revenue DESC
    LIMIT :lim
    """
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params={"lim": limit})
    return df

# ===================== SESSION =====================

for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, k=10
    )

# ===================== AGENT (SILENT) =====================

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,  # <-- hide ReAct Thought/Action traces
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=20,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ===================== SIDEBAR =====================

with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Checking database connection..."):
            result = db.run('SELECT COUNT(*) as total_records FROM voylla."voylla_design_ai" WHERE "Sale Order Item Status" != \'CANCELLED\'')
            m = re.search(r'\d+', result)
            if m:
                st.success(f"✅ Connected: {m.group(0)} active records")
    except Exception as e:
        st.error(f"❌ Connection issue: {e}")

    st.markdown("---")
    st.header("💡 Executive Questions")
    with st.expander("📈 Performance Overview", expanded=True):
        for q in [
            "Show me top 10 products by revenue this quarter",
            "What are our best performing channels by growth rate?",
            "Compare this year's revenue to last year by month",
            "What is our profit margin trend over the last 6 months?",
            "Which design styles have the highest average order value?"
        ]:
            if st.button(f"• {q}", key=f"exec_{hash(q)}"):
                st.session_state.auto_question = q

    with st.expander("🎨 Design Intelligence"):
        for q in [
            "Which metal colors are trending this season?",
            "What are the top 3 success combinations for wedding look?",
            "How do Traditional/Ethnic vs contemporary designs perform?",
            "Show me the performance of different stone settings"
        ]:
            if st.button(f"• {q}", key=f"design_{hash(q)}"):
                st.session_state.auto_question = q

    with st.expander("📊 Channel Analysis"):
        for q in [
            "Compare AOV across all channels",
            "Which designs perform best on each platform?",
            "Show me channel growth rates over time"
        ]:
            if st.button(f"• {q}", key=f"channel_{hash(q)}"):
                st.session_state.auto_question = q

    st.markdown("---")
    col1, _ = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    st.markdown("---")
    st.caption("Voylla DesignGPT v2.0 • Executive Edition")

# ===================== HEADER / HERO =====================

st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics for Executive Decision Making")

if not st.session_state.chat_history:
    st.markdown("""
    <div class='executive-summary'>
      <h3>📋 Executive Summary</h3>
      <p>Welcome to Voylla's Executive Analytics Dashboard. This AI-powered tool provides:</p>
      <ul>
        <li>Real-time sales performance analytics</li>
        <li>Design intelligence and trend analysis</li>
        <li>Channel performance comparisons</li>
        <li>Success combination identification</li>
      </ul>
      <p>Ask questions in natural language or use the sample questions in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)

# Quick stat cards (static)
try:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card">📊 Real-time Analytics</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card">💎 Sales Insights</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card">🚀 Growth Opportunities</div>', unsafe_allow_html=True)
except:
    pass

# ===================== CHAT HISTORY RENDER =====================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ===================== CHAT INPUT =====================

chat_prompt = "Ask an executive question about sales or design trends…"
user_input = st.chat_input(chat_prompt, key="chat_box")

# Sidebar quick question queued?
if st.session_state.get("auto_question"):
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

# ===================== QUICK INTENT: TOP CHANNELS =====================

def maybe_handle_top_channels_intent(text: str) -> bool:
    """If user's text is asking for channels with most sales, handle directly."""
    if not text:
        return False
    pattern = r"(channels?).*(most|top).*(sales|revenue|qty|units)"
    if re.search(pattern, text, re.I):
        with st.spinner(random.choice(lines).strip()):
            df_res = top_channels_sales(10)
        st.session_state.last_df = df_res

        st.subheader("Top 10 Channels by Revenue")
        st.dataframe(df_res, use_container_width=True)

        st.subheader("📊 Revenue by Channel")
        chart = px.bar(df_res.head(10), x="Channel", y="total_revenue",
                       title="Revenue by Channel (Top 10)")
        chart.update_layout(xaxis_tickangle=-30, height=420, showlegend=False)
        st.plotly_chart(chart, use_container_width=True)

        excel_bytes = df_to_excel_bytes(df_res, sheet_name="Top_Channels")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "💾 Download as Excel",
            data=excel_bytes,
            file_name=f"top_channels_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_top_channels_{ts}"
        )

        # Compact executive summary to chat
        if not df_res.empty:
            leader = df_res.iloc[0]
            summary = (
                f"**Executive Summary:** {leader['Channel']} leads with revenue ₹{leader['total_revenue']:,.2f} "
                f"and units {int(leader['total_units_sold'])}. "
                f"Top 3 channels: {', '.join(df_res['Channel'].head(3))}."
            )
            st.session_state.chat_history.append({"role": "assistant", "content": summary})
            with st.chat_message("assistant"):
                st.markdown(f"<div class='assistant-message'>{summary}</div>", unsafe_allow_html=True)
        return True
    return False

# ===================== HANDLE TURN =====================

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Short-circuit common KPI
    if maybe_handle_top_channels_intent(user_input):
        st.stop()

    # Otherwise, use the silent agent
    with st.spinner(random.choice(lines).strip()):
        try:
            result = st.session_state.agent_executor.invoke({"input": user_input})
            response = result.get("output", "")
            st.session_state.last_query_result = response
        except Exception as e:
            err = str(e)
            if "rate limit" in err.lower():
                response = "⏱️ System is experiencing high demand. Please try again."
            elif "connection" in err.lower() or "timeout" in err.lower():
                response = "🔌 Database connection issue. Please try again."
            else:
                response = f"I hit an error. Please retry. Error: {err[:200]}"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)

    # Parse any markdown table from the agent answer and show chart + download
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res
        st.subheader("📊 Data Visualization")
        # Simple heuristic: pick first numeric with a categorical
        num_cols = df_res.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df_res.select_dtypes(include=['object', 'string']).columns.tolist()
        if num_cols and cat_cols:
            fig = px.bar(df_res.head(10), x=cat_cols[0], y=num_cols[0],
                         title=f"{num_cols[0]} by {cat_cols[0]}")
            fig.update_layout(xaxis_tickangle=-30, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Data Table & Download"):
            st.dataframe(df_res, use_container_width=True)
            excel_bytes = df_to_excel_bytes(df_res)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "💾 Download this table as Excel",
                data=excel_bytes,
                file_name=f"voylla_table_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"inline_dl_{timestamp}"
            )

# ===================== GLOBAL EXPORT =====================

if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        rows_choice = st.selectbox("Rows to export:", [100, 500, 1000, "All"], index=1)
        export_df = st.session_state.last_df.copy() if rows_choice == "All" else st.session_state.last_df.iloc[:int(rows_choice)].copy()
    with col2:
        st.caption(f"📋 {len(export_df)} rows × {len(export_df.columns)} columns")
    with col3:
        excel_bytes = df_to_excel_bytes(export_df)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "💾 Download Executive Report",
            data=excel_bytes,
            file_name=f"voylla_executive_report_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="executive_download"
        )

# ===================== FOOTER =====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" •
Request visualizations with "show me a chart of..."
</div>
""", unsafe_allow_html=True)
