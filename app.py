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
import re
import random
import plotly.express as px
from datetime import datetime
import time
from io import BytesIO

# ---------- Helper: table + Excel (CSV fallback) ----------
def show_table_with_excel_download(
    df: pd.DataFrame,
    title: str = "Table + Download",
    filename_prefix: str = "voylla_table",
    expanded: bool = True,
    key_suffix: str = "table"
):
    if df is None or df.empty:
        return
    with st.expander(title, expanded=expanded):
        st.dataframe(df, use_container_width=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        file_base = f"{filename_prefix}_{timestamp}"

        # Try Excel first
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            output.seek(0)
            st.download_button(
                "💾 Download as Excel",
                data=output.getvalue(),
                file_name=f"{file_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"dl_xlsx_{key_suffix}"
            )
        except Exception as e:
            st.warning(f"Excel export unavailable ({e}). Offering CSV instead.")
            st.download_button(
                "💾 Download as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{file_base}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_csv_{key_suffix}"
            )

def coerce_numeric_commas(df: pd.DataFrame) -> pd.DataFrame:
    """Turn '1,234' strings into numbers where possible."""
    if df is None or df.empty: return df
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(s, errors="ignore")
    return df

# ---------- Spinner copy ----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip().split('. ', 1)[1] for line in f if '. ' in line]
else:
    lines = [
        "Crunching the numbers with a sparkle ✨",
        "Polishing your insights… 💎",
        "Setting the stones in your report…",
        "Crafting your jewelry analytics… 💍",
        "Mining data gems for you… ⛏️",
        "Designing your perfect answer… ✨",
    ]

# ---------- Secrets ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM ----------
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, request_timeout=60, max_retries=3)
llm = get_llm()

# ---------- DB ----------
@st.cache_resource
def get_database_connection():
    max_retries, retry_delay = 3, 2
    for attempt in range(max_retries):
        try:
            db_host = st.secrets["DB_HOST"]
            db_port = st.secrets["DB_PORT"]
            db_name = st.secrets["DB_NAME"]
            db_user = st.secrets["DB_USER"]
            db_password = st.secrets["DB_PASSWORD"]

            engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
                pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = SQLDatabase(engine, include_tables=["voylla_design_ai"], schema="voylla")
            return db
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Database connection failed: {e}")
            st.stop()
db = get_database_connection()

# ---------- Markdown table -> DataFrame ----------
def markdown_to_dataframe(markdown_text: str):
    if not markdown_text or '|' not in markdown_text: return None
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    table_start = None
    for i, line in enumerate(lines):
        if '|' in line and i+1 < len(lines) and re.match(r'^[\s\|:-]+$', lines[i+1]):
            table_start = i; break
    if table_start is None: return None

    table_lines = []
    for i in range(table_start, len(lines)):
        line = lines[i]
        if '|' in line and not re.match(r'^[\s\|:-]+$', line):
            table_lines.append(line)
        elif len(table_lines) > 0:
            break
    if len(table_lines) < 2: return None

    cleaned = []
    for line in table_lines:
        if not line.startswith('|'): line = '|' + line
        if not line.endswith('|'): line = line + '|'
        cleaned.append(line)

    try:
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(cleaned)), sep="|", skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Table parsing error: {e}")
        return None

# ---------- Conversation context ----------
def get_conversation_context():
    if not st.session_state.chat_history: return ""
    recent = st.session_state.chat_history[-6:]
    parts = ["### Conversation History:"]
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content'][:500]}")
    return "\n".join(parts)

# ---------- Chart builder ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    if df is None or df.empty: return None
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if len(numeric_cols) == 0: return None

    if chart_type == "auto":
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            chart_type = "bar" if len(df) <= 15 else "line"
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
        else:
            chart_type = "line"

    try:
        if chart_type == "bar" and len(categorical_cols) >= 1:
            top_df = df.nlargest(min(10, len(df)), numeric_cols[0])
            fig = px.bar(top_df, x=categorical_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} by {categorical_cols[0]}")
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            return fig
        elif chart_type == "line":
            fig = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}")
            fig.update_layout(height=400, showlegend=False)
            return fig
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                             title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            fig.update_layout(height=400, showlegend=False)
            return fig
    except Exception as e:
        st.error(f"Chart creation error: {e}")
    return None

# ---------- Page config & styles ----------
st.set_page_config(page_title="Voylla DesignGPT - Executive Dashboard", page_icon="💎", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; }
.main-header { font-size: 2.5rem; color: #4a4a4a; font-weight: 700; margin-bottom: 0.5rem; }
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] { color: #212529 !important; font-size: 1rem; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,.1); }
.executive-summary { background-color: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,.05); margin-bottom: 1.5rem; border-left: 4px solid #764ba2; }
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- Memory & Agent ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True,
        memory=st.session_state.memory, max_iterations=15,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Checking database connection..."):
            result = db.run('SELECT COUNT(*) as total_records FROM voylla."voylla_design_ai" WHERE "Sale Order Item Status" != \'CANCELLED\'')
            if result:
                count = re.search(r'\d+', result)
                if count: st.success(f"✅ Connected: {count.group(0)} active records")
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
            "How do traditional vs contemporary designs perform?",
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
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()
    st.caption("Voylla DesignGPT v2.0 • Executive Edition")

# ---------- Header ----------
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics for Executive Decision Making")

# Executive Summary (first load)
if not st.session_state.chat_history:
    st.markdown("""
    <div class='executive-summary'>
      <h3>📋 Executive Summary</h3>
      <p>Welcome to Voylla's Executive Analytics Dashboard. This AI-powered tool provides real-time performance analytics, design intelligence, channel comparisons, and success-combination insights. Use the sample questions in the sidebar or ask your own.</p>
    </div>
    """, unsafe_allow_html=True)

# Quick tiles
try:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="metric-card">📊 Real-time Analytics</div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card">💎 Sales Insights</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="metric-card">🚀 Growth Opportunities</div>', unsafe_allow_html=True)
except: pass

# Chat history render
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ---------- Chat input (always rendered) ----------
chat_prompt = "Ask an executive question about sales or design trends…"
user_input = st.chat_input(chat_prompt, key="chat_box")
if st.session_state.get("auto_question"):
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Executive-focused prompt (always on)
    conversation_context = get_conversation_context()
    prompt = f"""
You are Voylla DesignGPT Executive Edition, an expert SQL/analytics assistant for Voylla jewelry data.

# EXECUTIVE REPORTING GUIDELINES
- Lead with 2–4 key findings.
- Focus on trends, YoY/MoM, risks & opportunities.
- Use clear language and concise bullets.
- End with 3 actionable recommendations.

# DATABASE SCHEMA: voylla."voylla_design_ai"
(Use "Sale Order Item Status" != 'CANCELLED' in every aggregation.)

# CONVERSATION CONTEXT
{conversation_context}

# CURRENT REQUEST
{user_input}
"""

    random_template = random.choice(lines)
    with st.spinner(random_template.strip()):
        try:
            time.sleep(0.4)
            response = st.session_state.agent_executor.run(prompt)
            st.session_state.last_query_result = response
        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower():
                response = "⏱️ System is experiencing high demand. Please try again shortly."
            elif "connection" in msg.lower() or "timeout" in msg.lower():
                response = "🔌 Database connection issue. Please try again in a moment."
            elif "parsing" in msg.lower():
                response = "I had trouble understanding that request. Could you rephrase it more specifically?"
            else:
                response = f"Sorry, I hit an error: {msg[:160]}..."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)

    # Parse any markdown table, build chart, and show download
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        df_res = coerce_numeric_commas(df_res)
        st.session_state.last_df = df_res

        st.subheader("📊 Data Visualization")
        chart = create_chart_from_dataframe(df_res)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        # Always provide table + download here
        show_table_with_excel_download(
            st.session_state.last_df,
            title="Table + Download",
            filename_prefix="voylla_executive_table",
            expanded=True,
            key_suffix="lastdf"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "AOV", "top 10", "growth rate" •
Say "show me a chart of..." for visuals
</div>
""", unsafe_allow_html=True)
