#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from io import BytesIO
import re
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import sys

# ---------- Configure Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------- Spinner Messages ----------
SPINNER_MESSAGES = [
    "Polishing insights with a sparkle ✨",
    "Crafting your jewelry analytics… 💍",
    "Mining data gems for you… ⛏️",
    "Designing your perfect report… 📊",
    "Analyzing trends with precision… 🔍",
]

# ---------- Secrets Handling with Robust Fallback ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API key not found. Please configure it in the environment or Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM Initialization with Caching and Error Handling ----------
@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, request_timeout=30)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        st.error("Failed to connect to language model. Please try again later.")
        st.stop()

llm = get_llm()

# ---------- Database Connection with Enhanced Robustness ----------
@st.cache_resource(show_spinner=False)
def get_database_connection():
    try:
        db_host = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
        db_port = st.secrets.get("DB_PORT", os.getenv("DB_PORT"))
        db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
        db_user = st.secrets.get("DB_USER", os.getenv("DB_USER"))
        db_password = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))

        if not all([db_host, db_port, db_name, db_user, db_password]):
            raise ValueError("Missing database credentials")

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=10
        )
        
        db = SQLDatabase(
            engine,
            include_tables=["voylla_design_ai"],
            schema="voylla",
            sample_rows_in_table_info=2
        )
        return db, engine
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        st.error(f"Unable to connect to database: {e}")
        st.stop()

db, engine = get_database_connection()

# ---------- Markdown to DataFrame Parser with Robust Error Handling ----------
def markdown_to_dataframe(markdown_text: str):
    try:
        if not markdown_text or '|' not in markdown_text:
            logger.warning("No valid markdown table found")
            return None

        lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        header_idx = None
        for i in range(len(lines) - 1):
            if '|' in lines[i] and re.match(r'^[\s\|\-:]+$', lines[i + 1]):
                header_idx = i
                break

        if header_idx is None:
            logger.warning("No valid table header found")
            return None

        table_lines = [row for row in lines[header_idx:] if '|' in row and not re.match(r'^[\s\|\-:]+$', row)]
        if len(table_lines) < 2:
            logger.warning("Insufficient table rows")
            return None

        normalized = [f"|{row.strip()}|" if not row.startswith('|') else row for row in table_lines]
        header_cols = len(normalized[0].split('|')) - 2
        cleaned_rows = [r for r in normalized if len(r.split('|')) - 2 == header_cols]

        if not cleaned_rows:
            logger.warning("No consistent table rows")
            return None

        df = pd.read_csv(
            StringIO("\n".join(cleaned_rows)),
            sep=r'\s*\|\s*',
            engine='python',
            skipinitialspace=True
        )
        df = df.dropna(how='all', axis=1)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()

        logger.info(f"Parsed markdown table with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Table parsing error: {e}")
        st.warning(f"Could not parse table: {e}")
        return None

# ---------- Chart Generation with Enhanced Visuals ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    try:
        if df is None or df.empty:
            logger.warning("Empty or invalid DataFrame for chart")
            return None

        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

        if not numeric_cols:
            logger.warning("No numeric columns for chart")
            return None

        if chart_type == "auto":
            if len(df) <= 50 and categorical_cols:
                chart_type = "bar"
            elif len(numeric_cols) >= 2:
                chart_type = "scatter"
            else:
                chart_type = "line"

        fig = None
        if chart_type == "bar" and categorical_cols and numeric_cols:
            fig = px.bar(
                df, x=categorical_cols[0], y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                color=categorical_cols[0], color_discrete_sequence=px.colors.qualitative.Bold
            )
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(
                df, x=numeric_cols[0], y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                color=numeric_cols[1], size=numeric_cols[1],
                color_continuous_scale=px.colors.sequential.Viridis
            )
        elif chart_type == "line" and numeric_cols:
            fig = px.line(
                df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

        if fig:
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            logger.info(f"Generated {chart_type} chart")
            return fig
        return None
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None

# ---------- UI Configuration ----------
st.set_page_config(
    page_title="Voylla DesignGPT",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #fff1eb, #ace0f9);
    color: #1a1a1a;
    font-family: 'Arial', sans-serif;
}
.stChatMessage {
    border-radius: 10px;
    margin: 0.5rem;
    padding: 1rem;
}
.stChatMessage[data-testid="stChatMessage-user"] {
    background-color: #e6f3ff;
}
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background-color: #f0f0f0;
}
.metric-card {
    background: linear-gradient(135deg, #6b48ff, #00ddeb);
    padding: 1rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.success-indicator {
    color: #28a745;
    font-weight: bold;
}
.warning-indicator {
    color: #ff9800;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar with Enhanced Functionality ----------
with st.sidebar:
    st.header("📊 System Status")
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) as total_records FROM voylla.\"voylla_design_ai\"").fetchone()
            record_count = result[0] if result else "Unknown"
            st.markdown(f'<div class="success-indicator">✅ Connected to voylla_design_ai</div>', unsafe_allow_html=True)
            st.info(f"📈 Total records: {record_count:,}")
    except Exception as e:
        logger.error(f"Connection health check failed: {e}")
        st.markdown('<div class="warning-indicator">⚠️ Database connection issue</div>', unsafe_allow_html=True)

    st.header("💡 Sample Questions")
    with st.expander("📈 Sales Performance", expanded=True):
        sample_questions = [
            "Top 10 SKUs by revenue this year",
            "Best performing designs last 90 days",
            "Revenue trends by month for last 6 months",
            "Channel-wise AOV comparison"
        ]
        for q in sample_questions:
            if st.button(f"• {q}", key=f"perf_{q}"):
                st.session_state.auto_question = q

    with st.expander("🎨 Design Insights"):
        design_questions = [
            "Popular Form × Metal Color combinations",
            "Success combinations for Wedding look",
            "Central Stone performance by category",
            "Traditional vs Contemporary sales"
        ]
        for q in design_questions:
            if st.button(f"• {q}", key=f"design_{q}"):
                st.session_state.auto_question = q

    st.header("⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    with col2:
        show_charts = st.checkbox("📊 Auto-generate Charts", value=True)

    if "chat_history" in st.session_state:
        st.caption(f"💭 History: {len(st.session_state.chat_history)} messages")

# ---------- Main Interface ----------
st.title("💎 Voylla DesignGPT")
st.caption("Your AI-powered jewelry analytics assistant")

# Quick Metrics
try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">📊 Real-time Insights</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">💰 Sales Performance</div>', unsafe_allow_html=True)
except Exception as e:
    logger.warning(f"Metric cards rendering failed: {e}")

# ---------- Session State Initialization ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- Agent and Memory Setup ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )

if "agent_executor" not in st.session_state:
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            memory=st.session_state.memory,
            max_iterations=12,
            early_stopping_method="generate"
        )
        logger.info("Agent executor initialized successfully")
    except Exception as e:
        logger.error(f"Agent initialization error: {e}")
        st.error("Failed to initialize analytics agent. Please try again.")
        st.stop()

# ---------- Conversation Context Builder ----------
def get_conversation_context():
    if not st.session_state.chat_history:
        return ""
    recent_history = st.session_state.chat_history[-6:]
    context_parts = []
    for i, msg in enumerate(recent_history):
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts)

# ---------- Render Chat History ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Handle User Input ----------
if "auto_question" in st.session_state and st.session_state.auto_question:
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None
else:
    user_input = st.chat_input("Ask about sales, designs, or trends…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- Enhanced System Prompt ----------
    conversation_context = get_conversation_context()
    is_follow_up = any(word in user_input.lower() for word in [
        "show me", "can you", "what about", "how about", "also", "and",
        "but", "however", "additionally", "compare", "versus", "breakdown",
        "details", "more", "expand"
    ]) or len(st.session_state.chat_history) > 1

    prompt = f"""
You are Voylla DesignGPT, a professional analytics assistant for Voylla jewelry data.

# CONVERSATION CONTEXT
{"🔄 Follow-up question detected. Build on prior analysis." if is_follow_up else ""}
Recent conversation:
{conversation_context}

# SAFETY PROTOCOLS
- **Read-only**: Never execute DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE
- **Complete results**: Return all rows unless LIMIT is specified
- **PostgreSQL compliance**: Use double-quoted column names, PostgreSQL functions
- **Error handling**: Suggest simpler queries on failure

# DATABASE SCHEMA: voylla."voylla_design_ai"
## Key Columns
- Transaction: "Date" (timestamp), "Channel" (text), "Sale Order Item Status" (text), "Qty" (integer), "Amount" (numeric), "MRP" (numeric), "Cost Price" (numeric), "Discount" (numeric)
- Product: "EAN" (text), "Product Code" (text), "Collection" (text), "Category" (text), "Sub-Category" (text)
- Design: "Design Style" (text), "Form" (text), "Metal Color" (text), "Look" (text), "Craft Style" (text), "Central Stone" (text), "Surrounding Layout" (text), "Stone Setting" (text), "Style Motif" (text)

# MANDATORY FILTERS
- Exclude cancelled orders: WHERE "Sale Order Item Status" <> 'CANCELLED'
- Default to all data unless date specified
- Group by "Channel" for platform comparisons

# METRICS
- Total Quantity: SUM("Qty")
- Total Revenue: SUM("Amount")
- AOV: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Success Score: Rank by SUM("Qty") and SUM("Amount")
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100

# OUTPUT FORMATTING
- Use clean markdown tables
- Provide context: "Based on X records from Y date range"
- Highlight insights: **Bold** key findings
- Suggest follow-ups: "Would you like to explore [specific aspect]?"

# CURRENT REQUEST
{user_input}
"""

    with st.spinner(random.choice(SPINNER_MESSAGES)):
        try:
            response = st.session_state.agent_executor.run(prompt)
            st.session_state.last_query_result = response
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            response = f"⚠️ Error processing request: {e}. Please try rephrasing or simplifying your query."
            st.session_state.last_query_result = response

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # ---------- Table and Chart Rendering ----------
    df = markdown_to_dataframe(response)
    if df is not None and not df.empty:
        st.session_state.last_df = df
        if show_charts:
            chart = create_chart_from_dataframe(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

# ---------- Download Functionality ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_rows = st.selectbox("Export rows:", [100, 500, 1000, "All"], index=1)
        export_df = st.session_state.last_df if max_rows == "All" else st.session_state.last_df.iloc[:int(max_rows)]
    
    with col2:
        st.caption(f"📋 {len(export_df)} rows × {len(export_df.columns)} columns")
    
    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Voylla_Analytics')
        st.download_button(
            "📥 Download Excel",
            data=output.getvalue(),
            file_name=f"voylla_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #333; font-size: 0.9em;'>
💡 <b>Tips for Best Results:</b> Ask specific questions like "Top designs by revenue" or "Wedding look performance" • 
Use follow-ups like "break down by channel" • Enable auto-charts for instant visuals
</div>
""", unsafe_allow_html=True)

