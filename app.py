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
from io import BytesIO
import re
import random
import plotly.express as px
from datetime import datetime
import time

# ---------- Spinner messages -----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as file:
        lines = [line.strip().split('. ', 1)[1] for line in file if '. ' in line]
else:
    lines = [
        "Crunching the numbers with a sparkle ✨",
        "Polishing your insights… 💎",
        "Setting the stones in your report…",
        "Crafting your jewelry analytics… 💍",
        "Mining data gems for you… ⛏️",
        "Designing your perfect answer… ✨",
    ]

# ---------- Secrets handling ----------
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

# ---------- DB Connection ----------
@st.cache_resource
def get_database_connection():
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

    db = SQLDatabase(
        engine,
        include_tables=["voylla_design_ai"],
        schema="voylla"
    )
    return db

db = get_database_connection()

# ---------- Markdown parser ----------
def markdown_to_dataframe(markdown_text: str):
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
    cleaned_lines = []
    for line in table_lines:
        if not line.startswith('|'):
            line = '|' + line
        if not line.endswith('|'):
            line = line + '|'
        cleaned_lines.append(line)
    try:
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(cleaned_lines)), sep="|", skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        return df
    except:
        return None

# ---------- Conversation context ----------
def get_conversation_context():
    if not st.session_state.chat_history:
        return ""
    recent_history = st.session_state.chat_history[-6:]
    context_parts = ["### Conversation History:"]
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts)

# ---------- Chart helper ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    if df is None or df.empty:
        return None
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','string']).columns.tolist()
    if len(numeric_cols) == 0:
        return None
    try:
        if chart_type == "auto":
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                chart_type = "bar" if len(df) <= 15 else "line"
            elif len(numeric_cols) >= 2:
                chart_type = "scatter"
            else:
                chart_type = "line"
        if chart_type == "bar":
            fig = px.bar(df.head(10), x=categorical_cols[0], y=numeric_cols[0])
        elif chart_type == "line":
            fig = px.line(df, y=numeric_cols[0])
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        else:
            return None
        fig.update_layout(height=400, showlegend=False)
        return fig
    except:
        return None

# ---------- Safe query ----------
def execute_safe_query(query):
    dangerous_keywords = ['drop','delete','update','insert','alter','truncate','create','grant','revoke']
    if any(k in query.lower() for k in dangerous_keywords):
        return "Error: Dangerous SQL blocked."
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing query: {str(e)}"

# ---------- UI ----------
st.set_page_config(page_title="Voylla DesignGPT - Executive Dashboard", page_icon="💎", layout="wide")

# Session state init
for key in ["chat_history","last_df","last_query_result","auto_question","executive_mode"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key=="chat_history" else (True if key=="executive_mode" else None)

# Memory & Agent
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=5,   # reduced
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ---------- Main Chat ----------
chat_prompt = "Ask an executive question about sales or design trends…"
user_input = st.chat_input(chat_prompt, key="chat_box")

if st.session_state.get("auto_question"):
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation_context = get_conversation_context()
    prompt = f"""
You are Voylla DesignGPT, SQL/analytics assistant for executive reporting.

# CONTEXT
{conversation_context}

# EXECUTIVE REQUEST
{user_input}
"""
    random_template = random.choice(lines)

    with st.spinner(random_template.strip()):
        try:
            response = st.session_state.agent_executor.invoke({"input": prompt})
            response_text = response.get("output","")
        except Exception as e:
            st.warning("⚠️ Agent struggled. Switching to fallback mode…")
            try:
                # Ask LLM to only return SQL
                sql_only_prompt = f"Write only a safe PostgreSQL SELECT query (no explanations) for this request:\n{user_input}\nRemember: Always filter WHERE \"Sale Order Item Status\" != 'CANCELLED'"
                sql_response = llm.invoke(sql_only_prompt)
                sql_query = sql_response.content.strip()
                result = execute_safe_query(sql_query)
                if isinstance(result,str):
                    response_text = result
                else:
                    response_text = "✅ Fallback SQL executed."
                    # Convert result to DataFrame if possible
                    df_res = markdown_to_dataframe(str(result))
                    if df_res is not None:
                        st.session_state.last_df = df_res
            except Exception as e2:
                response_text = f"❌ Both agent and fallback failed: {e2}"

    st.session_state.chat_history.append({"role":"assistant","content":response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Try parse table & chart
    df_res = markdown_to_dataframe(response_text)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res
        st.subheader("📊 Data Visualization")
        chart = create_chart_from_dataframe(df_res)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        with st.expander("View Data Table"):
            st.dataframe(df_res, use_container_width=True)

# ---------- Excel Export ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    col1, col2 = st.columns([1,3])
    with col1:
        max_rows = st.selectbox("Rows to export:", [100,500,1000,"All"], index=1)
    with col2:
        st.caption(f"📋 {len(st.session_state.last_df)} rows × {len(st.session_state.last_df.columns)} columns")
    export_df = st.session_state.last_df if max_rows=="All" else st.session_state.last_df.head(int(max_rows))
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Executive_Report")
        summary_data = {"Metric":["Total Rows","Total Columns","Export Date"],
                        "Value":[len(export_df),len(export_df.columns),datetime.now().strftime("%Y-%m-%d %H:%M")]}
        pd.DataFrame(summary_data).to_excel(writer,index=False,sheet_name="Summary")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("💾 Download Executive Report",
                       data=output.getvalue(),
                       file_name=f"voylla_executive_report_{timestamp}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True,
                       key="executive_download")
