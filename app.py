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
from io import BytesIO

def show_table_with_excel_download(df: pd.DataFrame, title: str = "View Data Table", filename_prefix: str = "voylla_table"):
    """Render a dataframe and a download-as-Excel button together."""
    if df is None or df.empty:
        return

    with st.expander(title, expanded=False):
        st.dataframe(df, use_container_width=True)

        # Build Excel in-memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="💾 Download as Excel",
            data=output.getvalue(),
            file_name=f"{filename_prefix}_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_{filename_prefix}_{timestamp}"
        )



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
            
            # Test connection with proper SQLAlchemy text() function
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

# ---------- Enhanced markdown parser ----------
def markdown_to_dataframe(markdown_text: str):
    """Parse a markdown table into a DataFrame with better error handling."""
    if not markdown_text or '|' not in markdown_text:
        return None
        
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    
    # Find the table header and separator
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
    
    # Clean and normalize the table
    cleaned_lines = []
    for line in table_lines:
        # Ensure proper pipe formatting
        if not line.startswith('|'):
            line = '|' + line
        if not line.endswith('|'):
            line = line + '|'
        cleaned_lines.append(line)
    
    # Convert to DataFrame
    try:
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(cleaned_lines)), sep="|", skipinitialspace=True)
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
        
        # Clean column names and data
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Table parsing error: {str(e)}")
        return None

# ---------- Enhanced conversation context ----------
def get_conversation_context():
    """Build enhanced conversation context with better formatting."""
    if not st.session_state.chat_history:
        return ""
    
    # Get last 6 exchanges for better context
    recent_history = st.session_state.chat_history[-6:]
    
    context_parts = ["### Conversation History:"]
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]  # Limit length
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

# ---------- Chart generation helper ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    """Create appropriate charts based on DataFrame structure."""
    if df is None or df.empty:
        return None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    # Auto-detect best chart type
    if chart_type == "auto":
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            if len(df) <= 15:
                chart_type = "bar"
            else:
                chart_type = "line"
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
        else:
            chart_type = "line"
    
    try:
        if chart_type == "bar" and len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # For better visualization, limit to top 10 categories
            if len(df) > 10:
                top_df = df.nlargest(10, numeric_cols[0])
            else:
                top_df = df
            
            fig = px.bar(top_df, x=categorical_cols[0], y=numeric_cols[0], 
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                        color=numeric_cols[0], color_continuous_scale="viridis")
            fig.update_layout(xaxis_tickangle=-45)
            
        elif chart_type == "line" and len(numeric_cols) >= 1:
            # Try to find a date or sequential column
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

# ---------- Enhanced query validation and execution ----------
def execute_safe_query(query):
    """Execute SQL query with safety checks and error handling."""
    # Safety checks - prevent any destructive operations
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate', 'create', 'grant', 'revoke']
    if any(keyword in query.lower() for keyword in dangerous_keywords):
        return "Error: Query contains potentially dangerous operations."
    
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

# ---------- Enhanced UI ----------
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling for CEO presentation
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
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] {
    color: #212529 !important;
    font-size: 1rem;
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
.success-indicator {
    color: #28a745;
    font-weight: bold;
}
.warning-indicator {
    color: #ffc107;
    font-weight: bold;
}
.assistant-message {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    border-left: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# ---------- Initialize session state ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question", "executive_mode"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "executive_mode":
            st.session_state[key] = True
        else:
            st.session_state[key] = None

# ---------- Enhanced Memory / Agent ----------
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
        # early_stopping_method="generate",
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ---------- Enhanced Sidebar ----------
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    
    # Add connection health check
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
    
    # Categorized sample questions for executive use
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
    st.header("🔧 Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        st.session_state.executive_mode = st.checkbox("Executive Mode", value=True, key="exec_mode_check")
    
    st.markdown("---")
    st.caption("Voylla DesignGPT v2.0 • Executive Edition")

# ---------- Main Title and Header ----------
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics for Executive Decision Making")

# Add executive summary if in executive mode
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

# Add quick stats if available
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

# ---------- Render chat history ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and st.session_state.executive_mode:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ---------- Handle auto-generated questions ----------
# if "auto_question" in st.session_state and st.session_state.auto_question:
#     user_input = st.session_state.auto_question
#     st.session_state.auto_question = None
# else:
#     user_input = st.chat_input("Ask an executive question about sales or design trends…")

# ---------- Handle auto-generated questions (fixed to keep chat box visible) ----------

# 1) Always render the chat input so the keyboard never disappears on mobile
chat_prompt = "Ask an executive question about sales or design trends…"
user_input = st.chat_input(chat_prompt, key="chat_box")

# 2) If a sidebar button queued a question, process it this run
if st.session_state.get("auto_question"):
    # Use the queued question for this turn, but keep the chat box rendered
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None  # clear the queue


if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- ENHANCED SYSTEM PROMPT FOR EXECUTIVE USE ----------
    conversation_context = get_conversation_context()
    
    # Enhanced prompt with better context awareness for executive needs
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

# CURRENT EXECUTIVE REQUEST
{user_input}

Remember: You are speaking to company executives. Be insightful, professional, and focused on business impact.
"""

    # ---------- Enhanced execution with better error handling ----------
    random_template = random.choice(lines)
    
    with st.spinner(random_template.strip()):
        try:
            # Add a small delay to make the spinner more visible
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
                response = f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question. Error: {error_msg[:100]}..."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        if st.session_state.executive_mode:
            st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown(response)

    # ---------- Enhanced table parsing and visualization ----------
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res
        
        # Auto-generate charts for executive mode
        if st.session_state.executive_mode and len(df_res) > 1:
            st.subheader("📊 Data Visualization")
            chart = create_chart_from_dataframe(df_res)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # # Show data preview
            # with st.expander("View Data Table"):
            #     st.dataframe(df_res, use_container_width=True)

            # Table + Excel download in one place
            show_table_with_excel_download(
                df_res,
                title="View Data Table + Excel",
                filename_prefix="voylla_executive_table"
            )


# ---------- Enhanced download functionality ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        MAX_EXPORT_ROWS = st.selectbox("Rows to export:", [100, 500, 1000, "All"], index=1)
        if MAX_EXPORT_ROWS == "All":
            export_df = st.session_state.last_df.copy()
        else:
            export_df = st.session_state.last_df.iloc[:int(MAX_EXPORT_ROWS)].copy()
    
    with col2:
        # Show data summary
        st.caption(f"📋 {len(export_df)} rows × {len(export_df.columns)} columns")
    
    with col3:
        # Enhanced download options
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
            # Add a summary sheet
            summary_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Export Date'],
                'Value': [len(export_df), len(export_df.columns), datetime.now().strftime("%Y-%m-%d %H:%M")]
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        st.download_button(
            "💾 Download Executive Report",
            data=output.getvalue(),
            file_name=f"voylla_executive_report_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="executive_download"
        )

# ---------- Footer with executive tips ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities • 
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" • 
Request visualizations with "show me a chart of..."
</div>
""", unsafe_allow_html=True)
