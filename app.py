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
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ---------- Enhanced spinner messages -----------
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

# ---------- Enhanced secrets handling ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM with error recovery ----------
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, request_timeout=60, max_retries=3)
llm = get_llm()

# ---------- Enhanced DB CONNECTION with caching and retries ----------
@st.cache_resource
def get_database_connection():
    max_retries = 3
    retry_delay = 2  # seconds
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
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = SQLDatabase(
                engine,
                include_tables=["voylla_design_ai"],
                schema="voylla"
            )
            return db
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Database connection failed after {max_retries} attempts: {str(e)}")
            st.stop()

db = get_database_connection()

# ---------- Helpers ----------
def markdown_to_dataframe(markdown_text: str):
    """Parse a markdown table into a DataFrame with better error handling."""
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
    except Exception as e:
        st.error(f"Table parsing error: {str(e)}")
        return None

def create_chart_from_dataframe(df, chart_type="auto"):
    """Create appropriate charts based on DataFrame structure."""
    if df is None or df.empty:
        return None
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if len(numeric_cols) == 0:
        return None
    if chart_type == "auto":
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            chart_type = "bar" if len(df) <= 15 else "line"
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
        else:
            chart_type = "line"
    try:
        if chart_type == "bar" and len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            top_df = df.nlargest(10, numeric_cols[0]) if len(df) > 10 else df
            fig = px.bar(top_df, x=categorical_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                         color=numeric_cols[0], color_continuous_scale="viridis")
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "line" and len(numeric_cols) >= 1:
            date_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'month', 'year', 'day']):
                    date_col = col
                    break
            if date_col and date_col in df.columns:
                try:
                    df_sorted = df.sort_values(by=date_col)
                    fig = px.line(df_sorted, x=date_col, y=numeric_cols[0],
                                  title=f"Trend of {numeric_cols[0]} over time")
                except:
                    fig = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}")
            else:
                fig = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}")
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                             title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        else:
            return None
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return None

def execute_safe_query(query):
    """Execute SQL query with safety checks and error handling."""
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create', 'grant', 'revoke']
    if any(keyword in query.lower() for keyword in dangerous_keywords):
        return "Error: Query contains potentially dangerous operations."
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Executive_Report"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        summary_data = {
            'Metric': ['Total Rows', 'Total Columns', 'Export Date'],
            'Value': [len(df), len(df.columns), datetime.now().strftime("%Y-%m-%d %H:%M")]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
    return output.getvalue()

# ---------- Page ----------
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Always-on styling (Executive look, no toggle)
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; }
.main-header { font-size: 2.5rem; color: #4a4a4a; font-weight: 700; margin-bottom: 0.5rem; }
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] { color: #212529 !important; font-size: 1rem; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.executive-summary { background-color: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem; border-left: 4px solid #764ba2; }
.success-indicator { color: #28a745; font-weight: bold; }
.warning-indicator { color: #ffc107; font-weight: bold; }
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# ---------- Initialize session state ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- Memory / Agent ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=15,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Checking database connection..."):
            result = db.run("SELECT COUNT(*) as total_records FROM voylla.\"voylla_design_ai\" WHERE \"Sale Order Item Status\" != 'CANCELLED'")
            if result:
                record_count = re.search(r'\d+', result)
                if record_count:
                    st.success(f"✅ Connected: {record_count.group(0)} active records")
    except Exception as e:
        st.error(f"❌ Connection issue: {str(e)}")

    st.markdown("---")
    st.header("💡 Executive Questions")
    with st.expander("📈 Performance Overview", expanded=True):
        executive_questions = [
            "Show me top 10 products by revenue this quarter",
            "What are our best performing channels by growth rate?",
            "Compare this year's revenue to last year by month",
            "What is our profit margin trend over the last 6 months?",
            "Which design styles have the highest average order value?"
        ]
        for q in executive_questions:
            if st.button(f"• {q}", key=f"exec_{hash(q)}"):
                st.session_state.auto_question = q
    with st.expander("🎨 Design Intelligence"):
        design_questions = [
            "Which metal colors are trending this season?",
            "What are the top 3 success combinations for wedding look?",
            "How do traditional vs contemporary designs perform?",
            "Show me the performance of different stone settings"
        ]
        for q in design_questions:
            if st.button(f"• {q}", key=f"design_{hash(q)}"):
                st.session_state.auto_question = q
    with st.expander("📊 Channel Analysis"):
        channel_questions = [
            "Compare AOV across all channels",
            "Which designs perform best on each platform?",
            "Show me channel growth rates over time"
        ]
        for q in channel_questions:
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

# ---------- Header ----------
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics for Executive Decision Making")

# Always show executive summary on first load
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

# Quick stats cards
try:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="metric-card">📊 Real-time Analytics</div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card">💎 Sales Insights</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="metric-card">🚀 Growth Opportunities</div>', unsafe_allow_html=True)
except:
    pass

# ---------- Render chat history ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ---------- Chat input (always visible) ----------
chat_prompt = "Ask an executive question about sales or design trends…"
user_input = st.chat_input(chat_prompt, key="chat_box")

# If a sidebar question is queued, process it this run
if st.session_state.get("auto_question"):
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

# ---------- Conversation context builder ----------
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

# ---------- Handle a new prompt ----------
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation_context = get_conversation_context()
    prompt = f"""
You are Voylla DesignGPT Executive Edition, an expert SQL/analytics assistant for Voylla jewelry data analysis designed for executive use.

# CONVERSATION CONTEXT
{conversation_context}

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
- "Design Style" (text)
- "Form" (text)
- "Metal Color" (text)
- "Look" (text)
- "Central Stone" (text)

# MANDATORY FILTERS
- Always exclude cancelled orders: WHERE "Sale Order Item Status" != 'CANCELLED'
- For time-based questions, use appropriate date ranges
- When comparing channels, include only common time periods

# EXECUTIVE METRICS
- Revenue: SUM("Amount")
- Units: SUM("Qty")
- Average Order Value: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100
- Growth Rate: Use LAG() for period-over-period comparisons

# RESPONSE FORMATTING
1) Start with a concise executive summary
2) Present data in clean markdown tables
3) Bold the most important insights
4) Include charts when appropriate
5) End with actionable recommendations

# CURRENT EXECUTIVE REQUEST
{user_input}
"""

    random_template = random.choice(lines)
    with st.spinner(random_template.strip()):
        try:
            time.sleep(0.5)
            response = st.session_state.agent_executor.run(prompt)
            st.session_state.last_query_result = response
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                response = "⏱️ System is experiencing high demand. Please wait a moment and try again."
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                response = "🔌 Database connection issue. Please try again in a moment."
            elif "parsing" in error_msg.lower():
                response = "I had trouble understanding that request. Could you please rephrase your question more specifically?"
            else:
                response = f"I encountered an error. Please try again or rephrase your question. Error: {error_msg[:100]}..."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)

    # ---------- Parse table & show chart + INLINE DOWNLOAD ----------
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res

        st.subheader("📊 Data Visualization")
        chart = create_chart_from_dataframe(df_res)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

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

# ---------- Global export block (optional; kept for convenience) ----------
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "💾 Download Executive Report",
            data=excel_bytes,
            file_name=f"voylla_executive_report_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="executive_download"
        )

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" •
Request visualizations with "show me a chart of..."
</div>
""", unsafe_allow_html=True)
