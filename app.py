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

# =========================
# CONFIG & SETUP
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced spinner messages
SPINNER_MESSAGES = [
    "Crunching the numbers with a sparkle ✨",
    "Polishing your insights… 💎",
    "Setting the stones in your report…",
    "Crafting your jewelry analytics… 💍",
    "Mining data gems for you… ⛏️",
    "Designing your perfect answer… ✨",
]

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
    color: #212529;
}
.main-header {
    font-size: 2.5rem;
    color: #4a4a4a;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.executive-summary {
    background-color: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
    border-left: 4px solid #764ba2;
}
.assistant-message {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# KEYS & CONNECTIONS
# =========================
load_dotenv()

# Initialize session state early
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None
if "auto_question" not in st.session_state:
    st.session_state.auto_question = None
if "executive_mode" not in st.session_state:
    st.session_state.executive_mode = True

# API Key handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("🔑 No OpenAI key found – please add it in your app's Secrets or .env")
        st.stop()

os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

llm = get_llm()

@st.cache_resource
def get_database_connection():
    """Enhanced database connection with better error handling."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Try secrets first, then environment
            try:
                db_host = st.secrets["DB_HOST"]
                db_port = st.secrets["DB_PORT"]
                db_name = st.secrets["DB_NAME"]
                db_user = st.secrets["DB_USER"]
                db_password = st.secrets["DB_PASSWORD"]
            except:
                db_host = os.getenv("DB_HOST")
                db_port = os.getenv("DB_PORT")
                db_name = os.getenv("DB_NAME")
                db_user = os.getenv("DB_USER")
                db_password = os.getenv("DB_PASSWORD")
            
            if not all([db_host, db_port, db_name, db_user, db_password]):
                raise ValueError("Missing database configuration")
                
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

# =========================
# HELPER FUNCTIONS
# =========================
def get_conversation_context():
    """Build conversation context for better responses."""
    if not st.session_state.chat_history:
        return ""
    
    recent_history = st.session_state.chat_history[-6:]
    context_parts = ["### Conversation History:"]
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

def markdown_to_dataframe(markdown_text: str):
    """Parse markdown table into DataFrame with better error handling."""
    if not markdown_text or '|' not in markdown_text:
        return None
        
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    
    # Find table
    table_start = None
    for i, line in enumerate(lines):
        if '|' in line and i+1 < len(lines) and re.match(r'^[\s\|:-]+$', lines[i+1]):
            table_start = i
            break
    
    if table_start is None:
        return None
    
    # Extract table rows
    table_lines = []
    for i in range(table_start, len(lines)):
        line = lines[i]
        if '|' in line and not re.match(r'^[\s\|:-]+$', line):
            table_lines.append(line)
        elif len(table_lines) > 0:
            break
    
    if len(table_lines) < 2:
        return None
    
    try:
        # Clean and format
        cleaned_lines = []
        for line in table_lines:
            if not line.startswith('|'):
                line = '|' + line
            if not line.endswith('|'):
                line = line + '|'
            cleaned_lines.append(line)
        
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(cleaned_lines)), sep="|", skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Clean data
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        return None

def should_show_chart(question: str, df: pd.DataFrame) -> bool:
    """Determine if chart visualization is appropriate."""
    if df.empty or len(df) < 2:
        return False
    
    chart_keywords = [
        'trend', 'compare', 'comparison', 'growth', 'over time', 'by month', 'by year',
        'top', 'ranking', 'performance', 'distribution', 'analysis', 'breakdown',
        'versus', 'vs', 'chart', 'graph', 'visualize', 'show me'
    ]
    
    no_chart_keywords = [
        'count', 'total', 'sum', 'average', 'mean', 'specific', 'exact',
        'list all', 'show all', 'details'
    ]
    
    question_lower = question.lower()
    
    # Check for explicit no-chart requests
    for keyword in no_chart_keywords:
        if keyword in question_lower and not any(ck in question_lower for ck in chart_keywords):
            return False
    
    # Check for chart-friendly requests
    for keyword in chart_keywords:
        if keyword in question_lower:
            return True
    
    # Data characteristics check
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    return bool(numeric_cols) and len(df) <= 50

def create_smart_chart(df, question: str):
    """Create appropriate chart based on data structure and question."""
    if df is None or df.empty or not should_show_chart(question, df):
        return None
    
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'month', 'year'])]
    
    if not numeric_cols:
        return None
    
    try:
        # Time series for date data
        if date_cols and len(df) > 1:
            df_sorted = df.sort_values(by=date_cols[0])
            fig = px.line(df_sorted, x=date_cols[0], y=numeric_cols[0], 
                         title=f"{numeric_cols[0]} Over Time")
                         
        # Bar chart for categorical comparison
        elif categorical_cols and len(df) <= 20:
            work_df = df.nlargest(15, numeric_cols[0]) if len(df) > 15 else df
            fig = px.bar(work_df, x=categorical_cols[0], y=numeric_cols[0], 
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                        color=numeric_cols[0], color_continuous_scale="viridis")
            fig.update_layout(xaxis_tickangle=-45)
            
        # Scatter plot for correlation
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        else:
            return None
            
        fig.update_layout(height=400, showlegend=False)
        return fig
        
    except Exception:
        return None

# =========================
# AGENT SETUP
# =========================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=8  # Reduced for better performance
    )

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,  # Reduced verbosity
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=10,  # Reduced iterations
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    
    # Connection status
    try:
        with st.spinner("Checking database connection..."):
            result = db.run("SELECT COUNT(*) FROM voylla.\"voylla_design_ai\" WHERE \"Sale Order Item Status\" != 'CANCELLED'")
            if result:
                record_count = re.search(r'\d+', result)
                if record_count:
                    st.success(f"✅ Connected: {record_count.group(0)} active records")
    except Exception as e:
        st.error(f"❌ Connection issue: {str(e)}")
    
    st.markdown("---")
    st.header("💡 Executive Questions")
    
    # Sample questions with unique keys
    with st.expander("📈 Performance Overview", expanded=True):
        exec_questions = [
            "Show me top 10 products by revenue this quarter",
            "What are our best performing channels by growth rate?",
            "Compare this year's revenue to last year by month",
            "Which design styles have the highest average order value?",
            "Show me channel-wise revenue and units this month"
        ]
        for i, q in enumerate(exec_questions):
            if st.button(f"• {q}", key=f"exec_perf_{i}"):
                st.session_state.auto_question = q

    with st.expander("🎨 Design Intelligence"):
        design_questions = [
            "Which metal colors are trending this season?",
            "What are the top 3 success combinations for wedding look?",
            "How do traditional vs contemporary designs perform?",
            "Show me the performance of different stone settings"
        ]
        for i, q in enumerate(design_questions):
            if st.button(f"• {q}", key=f"design_intel_{i}"):
                st.session_state.auto_question = q

    with st.expander("📊 Channel Analysis"):
        channel_questions = [
            "Compare AOV across all channels",
            "Which designs perform best on each platform?",
            "Show me channel growth rates over time"
        ]
        for i, q in enumerate(channel_questions):
            if st.button(f"• {q}", key=f"channel_anal_{i}"):
                st.session_state.auto_question = q
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            # Clear all session state
            for key in ['chat_history', 'last_df', 'last_query_result', 'auto_question']:
                if key in st.session_state:
                    if key == 'chat_history':
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        st.session_state.executive_mode = st.checkbox("Executive Mode", value=True, key="exec_toggle")
    
    st.caption("Voylla DesignGPT v2.0 • Executive Edition")

# =========================
# MAIN INTERFACE
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics for Executive Decision Making")

# Executive summary for first visit
if st.session_state.executive_mode and not st.session_state.chat_history:
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
    with col1:
        st.markdown('<div class="metric-card">📊 Real-time Analytics</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">💎 Sales Insights</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">🚀 Growth Opportunities</div>', unsafe_allow_html=True)
except:
    pass

# =========================
# CHAT INTERFACE
# =========================
# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and st.session_state.executive_mode:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input (always visible)
user_input = st.chat_input("Ask an executive question about sales or design trends…", key="main_chat_input")

# Handle auto-question from sidebar
if st.session_state.get("auto_question"):
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None

# Process user input
if user_input:
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Build enhanced prompt
    conversation_context = get_conversation_context()
    
    enhanced_prompt = f"""
You are Voylla DesignGPT Executive Edition, an expert SQL/analytics assistant for executive decision making.

{conversation_context}

# EXECUTIVE GUIDELINES
- Focus on business insights and actionable recommendations
- Highlight trends, opportunities, and risks
- Use executive-appropriate language
- Compare performance metrics when relevant
- Present data in clean, formatted markdown tables

# DATABASE: voylla."voylla_design_ai"
## KEY EXECUTIVE METRICS
- Revenue: SUM("Amount")
- Units: SUM("Qty")
- AOV: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100

## MANDATORY FILTERS
- Always exclude cancelled orders: WHERE "Sale Order Item Status" != 'CANCELLED'
- Use appropriate date ranges for time-based questions

## RESPONSE FORMAT
1. Executive summary of key findings
2. Clean markdown tables
3. Bold important insights
4. Actionable recommendations

CURRENT REQUEST: {user_input}

Remember: Focus on business impact and executive decision-making needs.
"""
    
    # Execute with error handling
    random_message = random.choice(SPINNER_MESSAGES)
    
    with st.spinner(random_message):
        try:
            time.sleep(0.3)  # Brief delay for UX
            
            response = st.session_state.agent_executor.invoke({"input": enhanced_prompt})
            
            # Handle different response formats
            if isinstance(response, dict):
                response_text = response.get('output', str(response))
            else:
                response_text = str(response)
                
            st.session_state.last_query_result = response_text
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                response_text = "⏱️ System is experiencing high demand. Please wait a moment and try again."
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                response_text = "🔌 Database connection issue. Please try again in a moment."
            elif "parsing" in error_msg.lower():
                response_text = "I had trouble understanding that request. Could you please rephrase your question more specifically?"
            else:
                response_text = f"I encountered an error processing your request. Please try again or rephrase your question. Error: {error_msg[:100]}..."
    
    # Add assistant response to chat
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    
    with st.chat_message("assistant"):
        if st.session_state.executive_mode:
            st.markdown(f"<div class='assistant-message'>{response_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(response_text)
    
    # Parse and visualize results
    df_result = markdown_to_dataframe(response_text)
    if df_result is not None and not df_result.empty:
        st.session_state.last_df = df_result
        
        # Smart chart generation
        chart = create_smart_chart(df_result, user_input)
        if chart:
            st.subheader("📊 Data Visualization")
            st.plotly_chart(chart, use_container_width=True)
        
        # Conditional table display
        if len(df_result) <= 25 or 'detail' in user_input.lower() or 'list' in user_input.lower():
            with st.expander("📋 View Data Table", expanded=len(df_result) <= 10):
                st.dataframe(df_result, use_container_width=True)

# =========================
# EXPORT FUNCTIONALITY
# =========================
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        max_rows = st.selectbox("Export rows:", [100, 500, 1000, "All"], index=1, key="export_rows")
        if max_rows == "All":
            export_df = st.session_state.last_df.copy()
        else:
            export_df = st.session_state.last_df.iloc[:int(max_rows)].copy()
    
    with col2:
        st.caption(f"📋 {len(export_df)} rows × {len(export_df.columns)} columns")
    
    with col3:
        # Generate Excel file
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'Export Time'],
                    'Value': [
                        len(export_df), 
                        len(export_df.columns),
                        datetime.now().strftime("%Y-%m-%d"),
                        datetime.now().strftime("%H:%M:%S")
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
            
            output.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            st.download_button(
                "💾 Download Executive Report",
                data=output.getvalue(),
                file_name=f"voylla_executive_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="final_download"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities • 
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" • 
Request specific analyses like "show me top performers" or "compare channels"
</div>
""", unsafe_allow_html=True)
