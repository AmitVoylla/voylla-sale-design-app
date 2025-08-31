#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.agent_types import AgentType
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ENHANCED CONFIGURATION ----------
st.set_page_config(
    page_title="Voylla DesignGPT | CEO Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PROFESSIONAL STYLING ----------
st.markdown("""
<style>
/* Executive Dashboard Theme */
.stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: #ffffff;
}

.main-header {
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.metric-container {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    text-align: center;
    color: white;
}

.status-success {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.status-warning {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.chat-container {
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    color: #333;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.insight-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #ffd700;
}

.stDataFrame {
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Enhanced buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Chart styling */
.plotly-chart {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- ENHANCED SPINNER MESSAGES ----------
EXECUTIVE_MESSAGES = [
    "🔍 Analyzing market performance data...",
    "💎 Processing jewelry analytics...",
    "📊 Generating executive insights...",
    "⚡ Optimizing query performance...",
    "🎯 Extracting actionable intelligence...",
    "💰 Calculating revenue metrics...",
    "🏆 Identifying top performers...",
    "📈 Building trend analysis...",
    "🔮 Predicting market opportunities...",
    "💡 Crafting strategic recommendations..."
]

# ---------- ROBUST DATABASE CONNECTION ----------
@st.cache_resource
def get_database_connection():
    """Enhanced database connection with better error handling"""
    try:
        # Validate all required secrets
        required_secrets = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [s for s in required_secrets if s not in st.secrets]
        
        if missing:
            st.error(f"❌ Missing database secrets: {', '.join(missing)}")
            st.stop()
        
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"] 
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
        
        # Enhanced connection string with better parameters
        connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
            connect_args={
                "connect_timeout": 30,
                "application_name": "VoyllaDesignGPT"
            }
        )
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            result.fetchone()
        
        db = SQLDatabase(
            engine,
            include_tables=["voylla_design_ai"],
            schema="voylla",
            sample_rows_in_table_info=3
        )
        
        return db, engine
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("💡 Please verify your database credentials in Streamlit secrets")
        st.stop()

# ---------- ENHANCED LLM CONFIGURATION ----------
@st.cache_resource
def get_llm():
    """Initialize LLM with better parameters for SQL generation"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    
    if not api_key:
        st.error("🔑 OpenAI API key required. Add to secrets or .env file")
        st.stop()
    
    return ChatOpenAI(
        model="gpt-4-turbo-preview",  # More powerful model
        temperature=0,  # Deterministic for SQL
        max_tokens=4000,
        request_timeout=60
    )

# ---------- INTELLIGENT DATA PARSER ----------
def parse_sql_result(result_text: str):
    """Enhanced parsing for SQL results with multiple format support"""
    if not result_text:
        return None
        
    # Try to extract markdown table
    table_pattern = r'\|.*?\|'
    table_lines = [line for line in result_text.split('\n') if re.match(table_pattern, line.strip())]
    
    if len(table_lines) >= 3:  # Header + separator + at least one data row
        return parse_markdown_table(table_lines)
    
    # Try to extract SQL result set (common format)
    lines = result_text.split('\n')
    data_lines = []
    headers = None
    
    for i, line in enumerate(lines):
        if '|' in line and not re.match(r'^[\s\|\-:]+$', line):
            if headers is None:
                headers = [col.strip() for col in line.split('|') if col.strip()]
            else:
                row_data = [col.strip() for col in line.split('|') if col.strip()]
                if len(row_data) == len(headers):
                    data_lines.append(row_data)
    
    if headers and data_lines:
        try:
            df = pd.DataFrame(data_lines, columns=headers)
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            return df
        except Exception:
            pass
    
    return None

def parse_markdown_table(table_lines):
    """Enhanced markdown table parser"""
    try:
        # Clean and normalize lines
        cleaned_lines = []
        for line in table_lines:
            line = line.strip()
            if not line.startswith('|'):
                line = '|' + line
            if not line.endswith('|'):
                line += '|'
            cleaned_lines.append(line)
        
        # Parse header
        header_line = cleaned_lines[0]
        headers = [col.strip() for col in header_line.split('|')[1:-1] if col.strip()]
        
        # Parse data (skip separator line)
        data_rows = []
        for line in cleaned_lines[2:]:  # Skip header and separator
            row_data = [col.strip() for col in line.split('|')[1:-1]]
            if len(row_data) == len(headers):
                data_rows.append(row_data)
        
        if not data_rows:
            return None
            
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Enhanced type conversion
        for col in df.columns:
            # Try numeric conversion
            if df[col].str.replace(r'[,$%]', '', regex=True).str.replace('.', '', 1).str.isdigit().all():
                df[col] = pd.to_numeric(df[col].str.replace(r'[,$%]', '', regex=True), errors='ignore')
            # Try date conversion for common date patterns
            elif df[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        return df
        
    except Exception as e:
        logger.error(f"Table parsing error: {e}")
        return None

# ---------- INTELLIGENT CHART GENERATOR ----------
def create_executive_chart(df, chart_type="auto"):
    """Generate executive-level charts with intelligent type detection"""
    if df is None or df.empty or len(df) < 2:
        return None
    
    try:
        # Detect column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Executive color scheme
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8']
        
        # Intelligent chart selection
        if chart_type == "auto":
            if date_cols and numeric_cols:
                chart_type = "timeline"
            elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and len(df) <= 25:
                chart_type = "bar"
            elif len(numeric_cols) >= 2:
                chart_type = "scatter"
            else:
                chart_type = "bar"
        
        # Create appropriate chart
        if chart_type == "timeline" and date_cols:
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], 
                         color=categorical_cols[0] if categorical_cols else None,
                         title="Performance Timeline")
            
        elif chart_type == "bar" and categorical_cols:
            # Top N for better readability
            display_df = df.head(15) if len(df) > 15 else df
            fig = px.bar(display_df, x=categorical_cols[0], y=numeric_cols[0],
                        color=categorical_cols[1] if len(categorical_cols) > 1 else None,
                        title=f"Top Performance: {numeric_cols[0]} by {categorical_cols[0]}")
            fig.update_xaxis(tickangle=45)
            
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           color=categorical_cols[0] if categorical_cols else None,
                           size=numeric_cols[2] if len(numeric_cols) > 2 else None,
                           title="Performance Correlation Analysis")
            
        elif chart_type == "pie" and categorical_cols and numeric_cols:
            # Top 10 for pie chart readability
            pie_data = df.nlargest(10, numeric_cols[0])
            fig = px.pie(pie_data, values=numeric_cols[0], names=categorical_cols[0],
                        title="Market Share Distribution")
            
        else:
            # Fallback: simple bar chart
            fig = px.bar(df.head(20), x=df.columns[0], y=numeric_cols[0] if numeric_cols else df.columns[1],
                        title="Data Overview")
        
        # Executive styling
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=16,
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Professional color scheme
        if hasattr(fig, 'data') and fig.data:
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'marker'):
                    trace.marker.color = colors[i % len(colors)]
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None

# ---------- ENHANCED QUERY TEMPLATES ----------
EXECUTIVE_QUERY_TEMPLATES = {
    "revenue_performance": """
    SELECT 
        "Channel",
        COUNT(DISTINCT "Product Code") as unique_products,
        SUM("Qty") as total_units,
        ROUND(SUM("Amount")::numeric, 2) as total_revenue,
        ROUND(AVG("Amount")::numeric, 2) as avg_order_value,
        ROUND((SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100, 2) as profit_margin_pct
    FROM voylla."voylla_design_ai"
    WHERE "Sale Order Item Status" <> 'CANCELLED'
        AND "Date" >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY "Channel"
    ORDER BY total_revenue DESC;
    """,
    
    "top_performers": """
    SELECT 
        "Design Style",
        "Form", 
        "Metal Color",
        "Look",
        SUM("Qty") as total_qty,
        ROUND(SUM("Amount")::numeric, 2) as total_revenue,
        COUNT(DISTINCT "Channel") as channels_sold,
        ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 2) as avg_price_per_unit
    FROM voylla."voylla_design_ai"
    WHERE "Sale Order Item Status" <> 'CANCELLED'
    GROUP BY "Design Style", "Form", "Metal Color", "Look"
    HAVING SUM("Qty") >= 10
    ORDER BY total_revenue DESC, total_qty DESC
    LIMIT 20;
    """,
    
    "trend_analysis": """
    SELECT 
        EXTRACT(YEAR FROM "Date"::date) as year,
        EXTRACT(MONTH FROM "Date"::date) as month,
        "Design Style",
        SUM("Qty") as units_sold,
        ROUND(SUM("Amount")::numeric, 2) as revenue,
        COUNT(DISTINCT "Product Code") as unique_designs
    FROM voylla."voylla_design_ai"
    WHERE "Sale Order Item Status" <> 'CANCELLED'
        AND "Date" >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY year, month, "Design Style"
    ORDER BY year DESC, month DESC, revenue DESC;
    """
}

# ---------- ENHANCED SQL AGENT ----------
class EnhancedSQLAgent:
    def __init__(self, db, engine, llm):
        self.db = db
        self.engine = engine
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        self.agent = create_sql_agent(
            llm=llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=15,
            early_stopping_method="generate"
        )
    
    def execute_query(self, user_query: str, context: str = "") -> tuple:
        """Execute query with enhanced error handling and fallbacks"""
        try:
            # Enhanced prompt for better SQL generation
            enhanced_prompt = f"""
            You are an expert SQL analyst for Voylla jewelry business intelligence.
            
            CONTEXT: {context}
            
            CRITICAL REQUIREMENTS:
            1. ALWAYS exclude cancelled orders: WHERE "Sale Order Item Status" <> 'CANCELLED'
            2. Use double quotes for all column names: "Column Name"
            3. Return complete results (no arbitrary LIMIT unless requested)
            4. Use PostgreSQL syntax and functions
            5. Handle NULL values properly with NULLIF and COALESCE
            6. Always include relevant business metrics
            
            SCHEMA REMINDER - voylla."voylla_design_ai" key columns:
            - "Date", "Channel", "Qty", "Amount", "Product Code"
            - "Design Style", "Form", "Metal Color", "Look", "Category"
            - "Central Stone", "Style Motif", "Cost Price", "MRP"
            
            USER QUERY: {user_query}
            
            Generate a complete, accurate SQL query and provide insights.
            """
            
            result = self.agent.run(enhanced_prompt)
            return result, None
            
        except Exception as e:
            error_msg = str(e)
            
            # Try fallback with simpler query
            if "timeout" in error_msg.lower():
                return "⏱️ Query timeout. Please try a more specific question or smaller date range.", error_msg
            elif "syntax" in error_msg.lower():
                return "🔧 SQL syntax issue detected. Trying simpler approach...", error_msg
            else:
                return f"❌ Query execution failed: {error_msg}", error_msg

# ---------- EXECUTIVE DASHBOARD FUNCTIONS ----------
def get_executive_summary():
    """Generate real-time executive summary"""
    try:
        with st.session_state.engine.connect() as conn:
            # Quick performance metrics
            summary_query = text("""
            SELECT 
                COUNT(DISTINCT "Product Code") as total_products,
                SUM("Qty") as total_units_sold,
                ROUND(SUM("Amount")::numeric, 0) as total_revenue,
                COUNT(DISTINCT "Channel") as active_channels,
                ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 2) as avg_unit_price
            FROM voylla."voylla_design_ai"
            WHERE "Sale Order Item Status" <> 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '30 days'
            """)
            
            result = conn.execute(summary_query).fetchone()
            if result:
                return {
                    'products': int(result[0]) if result[0] else 0,
                    'units': int(result[1]) if result[1] else 0,
                    'revenue': float(result[2]) if result[2] else 0,
                    'channels': int(result[3]) if result[3] else 0,
                    'avg_price': float(result[4]) if result[4] else 0
                }
    except Exception as e:
        logger.error(f"Executive summary error: {e}")
    
    return None

def format_currency(amount):
    """Format currency for executive presentation"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.1f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.1f}L"
    else:
        return f"₹{amount:,.0f}"

# ---------- INITIALIZE ENHANCED SESSION STATE ----------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "chat_history": [],
        "last_df": None,
        "last_query_result": None,
        "auto_question": None,
        "executive_mode": True,
        "chart_preference": "auto"
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize everything
initialize_session_state()

# Get database connection and LLM
db, engine = get_database_connection()
st.session_state.engine = engine
llm = get_llm()

# Initialize enhanced SQL agent
if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = EnhancedSQLAgent(db, engine, llm)

# ---------- EXECUTIVE HEADER ----------
st.markdown("""
<div class="main-header">
    <h1>💎 Voylla DesignGPT</h1>
    <h3>Executive Business Intelligence Dashboard</h3>
    <p>AI-Powered Jewelry Analytics & Strategic Insights</p>
</div>
""", unsafe_allow_html=True)

# ---------- REAL-TIME EXECUTIVE METRICS ----------
exec_summary = get_executive_summary()
if exec_summary:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{exec_summary['products']:,}</h3>
            <p>Active Products</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{exec_summary['units']:,}</h3>
            <p>Units Sold (30d)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{format_currency(exec_summary['revenue'])}</h3>
            <p>Revenue (30d)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{exec_summary['channels']}</h3>
            <p>Sales Channels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <h3>₹{exec_summary['avg_price']:,.0f}</h3>
            <p>Avg Unit Price</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- ENHANCED SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🔍 System Status")
    
    # Connection status
    try:
        test_result = db.run("SELECT COUNT(*) FROM voylla.\"voylla_design_ai\" LIMIT 1")
        record_count = re.findall(r'\d+', test_result)[0] if test_result else "Unknown"
        st.markdown(f'<div class="status-success">✅ Connected | {record_count} records</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="status-warning">⚠️ Connection issue: {str(e)[:50]}...</div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Executive Queries")
    
    # Strategic query categories
    with st.expander("💰 Revenue & Performance", expanded=True):
        strategic_queries = [
            "Top 10 revenue generating product combinations this quarter",
            "Channel performance comparison with profit margins", 
            "Best performing design styles by revenue last 90 days",
            "Premium vs budget product performance analysis"
        ]
        for q in strategic_queries:
            if st.button(q, key=f"strat_{hash(q)}"):
                st.session_state.auto_question = q
    
    with st.expander("🎨 Design Intelligence"):
        design_queries = [
            "Most successful metal color and form combinations",
            "Trending design styles by channel performance",
            "Central stone preferences driving highest AOV",
            "Contemporary vs Traditional design revenue split"
        ]
        for q in design_queries:
            if st.button(q, key=f"design_{hash(q)}"):
                st.session_state.auto_question = q
    
    with st.expander("📊 Market Analytics"):
        market_queries = [
            "Seasonal trends in jewelry categories",
            "Channel-wise customer preferences analysis",
            "Growth opportunities by underperforming segments",
            "Price optimization insights by category"
        ]
        for q in market_queries:
            if st.button(q, key=f"market_{hash(q)}"):
                st.session_state.auto_question = q
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_resource.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Settings
    st.markdown("### ⚙️ Display Settings")
    auto_charts = st.checkbox("📈 Auto-generate charts", value=True)
    show_sql = st.checkbox("🔍 Show SQL queries", value=False)
    executive_mode = st.checkbox("👔 Executive summaries", value=True)

# ---------- ENHANCED CHAT INTERFACE ----------
st.markdown("### 💬 Ask Your Business Questions")

# Render chat history with enhanced styling
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="chat-container">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ---------- ENHANCED QUERY PROCESSING ----------
# Handle auto-generated questions
if st.session_state.auto_question:
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None
else:
    user_input = st.chat_input("Ask me about sales trends, design performance, or strategic insights...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Build conversation context
    context = ""
    if len(st.session_state.chat_history) > 2:
        recent_messages = st.session_state.chat_history[-4:]
        context = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in recent_messages])
    
    # Show professional spinner
    spinner_message = random.choice(EXECUTIVE_MESSAGES)
    
    with st.spinner(spinner_message):
        start_time = time.time()
        
        # Execute query with enhanced error handling
        response, error = st.session_state.sql_agent.execute_query(user_input, context)
        
        execution_time = time.time() - start_time
        
        # Enhance response for executive presentation
        if not error and response:
            if executive_mode:
                response += f"\n\n*📊 Analysis completed in {execution_time:.1f}s*"
        
        st.session_state.last_query_result = response
    
    # Display response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-container">{response}</div>', unsafe_allow_html=True)
    
    # ---------- ENHANCED DATA VISUALIZATION ----------
    parsed_df = parse_sql_result(response)
    
    if parsed_df is not None and not parsed_df.empty:
        st.session_state.last_df = parsed_df
        
        # Show data with enhanced formatting
        st.markdown("### 📋 Data Results")
        st.dataframe(
            parsed_df, 
            use_container_width=True,
            hide_index=True
        )
        
        # Auto-generate executive charts
        if auto_charts and len(parsed_df) > 1:
            st.markdown("### 📊 Visual Analytics")
            
            chart = create_executive_chart(parsed_df)
            if chart:
                st.plotly_chart(chart, use_container_width=True, key=f"chart_{len(st.session_state.chat_history)}")
            
            # Offer additional chart types
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            with chart_col1:
                if st.button("📊 Bar Chart"):
                    alt_chart = create_executive_chart(parsed_df, "bar")
                    if alt_chart:
                        st.plotly_chart(alt_chart, use_container_width=True)
            
            with chart_col2:
                if st.button("🥧 Pie Chart"):
                    alt_chart = create_executive_chart(parsed_df, "pie")
                    if alt_chart:
                        st.plotly_chart(alt_chart, use_container_width=True)
            
            with chart_col3:
                if st.button("📈 Scatter Plot"):
                    alt_chart = create_executive_chart(parsed_df, "scatter")
                    if alt_chart:
                        st.plotly_chart(alt_chart, use_container_width=True)
        
        # Executive insights generation
        if executive_mode and len(parsed_df) > 0:
            insights = generate_executive_insights(parsed_df, user_input)
            if insights:
                st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

# ---------- EXECUTIVE INSIGHTS GENERATOR ----------
def generate_executive_insights(df, query):
    """Generate strategic insights for executive presentation"""
    insights = []
    
    try:
        # Revenue insights
        if 'revenue' in df.columns.str.lower().str.join(' '):
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'amount' in col.lower()]
            if revenue_cols:
                total_revenue = df[revenue_cols[0]].sum()
                top_performer = df.loc[df[revenue_cols[0]].idxmax()]
                insights.append(f"💰 **Total Revenue**: {format_currency(total_revenue)}")
                insights.append(f"🏆 **Top Performer**: {top_performer.iloc[0]} generating {format_currency(top_performer[revenue_cols[0]])}")
        
        # Quantity insights
        qty_cols = [col for col in df.columns if 'qty' in col.lower() or 'units' in col.lower()]
        if qty_cols:
            total_qty = df[qty_cols[0]].sum()
            insights.append(f"📦 **Total Units**: {total_qty:,}")
        
        # Performance distribution
        if len(df) > 1:
            insights.append(f"📊 **Performance Spread**: Top 20% accounts for {(df.head(int(len(df)*0.2)).sum().sum() / df.sum().sum() * 100):.0f}% of total value")
        
        # Channel insights
        if 'channel' in df.columns.str.lower().str.join(' '):
            channel_cols = [col for col in df.columns if 'channel' in col.lower()]
            if channel_cols:
                unique_channels = df[channel_cols[0]].nunique()
                insights.append(f"🌐 **Channel Diversity**: {unique_channels} active sales channels")
        
        return "**📈 EXECUTIVE INSIGHTS**\n\n" + "\n\n".join(insights)
        
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        return None

# ---------- ENHANCED DOWNLOAD FUNCTIONALITY ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        export_rows = st.selectbox(
            "Rows to export:",
            [100, 500, 1000, "All"],
            index=2,
            key="export_rows_selector"
        )
    
    with col2:
        export_format = st.selectbox(
            "Format:",
            ["Excel", "CSV"],
            key="export_format_selector"
        )
    
    with col3:
        include_charts = st.checkbox("Include charts", value=True)
    
    with col4:
        # Prepare export data
        if export_rows == "All":
            export_df = st.session_state.last_df.copy()
        else:
            export_df = st.session_state.last_df.head(export_rows).copy()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='VoyllaAnalytics', index=False)
                
                # Add metadata sheet
                metadata = pd.DataFrame({
                    'Export Info': [
                        'Generated by Voylla DesignGPT',
                        f'Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                        f'Total Rows: {len(export_df)}',
                        f'Total Columns: {len(export_df.columns)}',
                        'Data Source: voylla_design_ai table'
                    ]
                })
                metadata.to_excel(writer, sheet_name='Export_Info', index=False)
            
            st.download_button(
                "📥 Download Executive Report",
                data=output.getvalue(),
                file_name=f"voylla_executive_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV Data",
                data=csv_data,
                file_name=f"voylla_data_{timestamp}.csv",
                mime="text/csv",
                type="primary"
            )

# ---------- QUICK ANALYSIS SHORTCUTS ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("### ⚡ Quick Analysis")
    
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    with analysis_col1:
        if st.button("🔍 Deep Dive Analysis"):
            columns = ', '.join(st.session_state.last_df.columns[:3])
            st.session_state.auto_question = f"Show me detailed breakdown and trends for {columns} with insights"
    
    with analysis_col2:
        if st.button("📊 Performance Ranking"):
            st.session_state.auto_question = "Rank all items by performance and show top 15 with success factors"
    
    with analysis_col3:
        if st.button("🎯 Strategic Recommendations"):
            st.session_state.auto_question = "Based on this data, what are the top 5 strategic recommendations for business growth?"

# ---------- EXECUTIVE TEMPLATES ----------
def get_executive_templates():
    """Predefined executive-level analysis templates"""
    return {
        "quarterly_review": EXECUTIVE_QUERY_TEMPLATES["revenue_performance"],
        "top_performers": EXECUTIVE_QUERY_TEMPLATES["top_performers"], 
        "trend_analysis": EXECUTIVE_QUERY_TEMPLATES["trend_analysis"]
    }

# ---------- ADVANCED ERROR RECOVERY ----------
def handle_query_errors(error_msg: str, original_query: str):
    """Intelligent error recovery with suggestions"""
    error_lower = error_msg.lower()
    
    if "timeout" in error_lower:
        return "⏱️ **Query Timeout Recovery**\n\nThe query was too complex. Try:\n- Adding date filters (last 30/60/90 days)\n- Limiting to specific channels\n- Focusing on fewer product attributes"
    
    elif "syntax" in error_lower or "column" in error_lower:
        return "🔧 **SQL Syntax Recovery**\n\nLet me suggest a simpler approach:\n- Use basic column names\n- Try 'show me sales by channel'\n- Or 'top 10 products by revenue'"
    
    elif "connection" in error_lower:
        return "🔌 **Connection Recovery**\n\nDatabase connection issue. Please:\n- Wait 30 seconds and try again\n- Check your internet connection\n- Contact technical support if issues persist"
    
    else:
        return f"❌ **Error Recovery Mode**\n\nEncountered: {error_msg[:100]}...\n\nTry rephrasing your question or use one of the suggested queries from the sidebar."

# ---------- PERFORMANCE MONITORING ----------
def log_performance_metrics():
    """Log performance for monitoring"""
    if len(st.session_state.chat_history) > 0:
        session_duration = len(st.session_state.chat_history) * 30  # Rough estimate
        success_rate = sum(1 for msg in st.session_state.chat_history if msg["role"] == "assistant" and "error" not in msg["content"].lower()) / len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant"])
        
        st.sidebar.markdown(f"""
        ### 📈 Session Analytics
        - **Queries**: {len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])}
        - **Success Rate**: {success_rate:.1%}
        - **Avg Response**: ~{session_duration/max(len(st.session_state.chat_history), 1):.1f}s
        """)

# Call performance monitoring
log_performance_metrics()

# ---------- FOOTER WITH CEO-READY FEATURES ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 12px; color: white;'>
    <h4>🚀 Voylla DesignGPT - Executive Edition</h4>
    <p><strong>Powered by Advanced AI Analytics</strong> | Real-time Business Intelligence | Strategic Decision Support</p>
    <p>💡 <em>Ask complex questions like:</em> "Compare Q4 performance across channels with profitability analysis" | 
    "Show trending design combinations for premium segment" | "Strategic recommendations for inventory optimization"</p>
</div>
""", unsafe_allow_html=True)

# ---------- HEALTH CHECK & MONITORING ----------
if st.sidebar.button("🔍 System Health Check"):
    with st.sidebar:
        with st.spinner("Running diagnostics..."):
            health_status = {}
            
            # Database health
            try:
                test_query = db.run("SELECT COUNT(*), MAX(\"Date\"), MIN(\"Date\") FROM voylla.\"voylla_design_ai\" WHERE \"Sale Order Item Status\" <> 'CANCELLED'")
                health_status["database"] = "✅ Healthy"
            except Exception as e:
                health_status["database"] = f"❌ Issue: {str(e)[:50]}"
            
            # LLM health  
            try:
                test_response = llm.invoke("Test query")
                health_status["ai_model"] = "✅ Responsive"
            except Exception as e:
                health_status["ai_model"] = f"❌ Issue: {str(e)[:50]}"
            
            # Memory health
            memory_size = len(st.session_state.chat_history)
            if memory_size < 50:
                health_status["memory"] = f"✅ Optimal ({memory_size} messages)"
            else:
                health_status["memory"] = f"⚠️ High usage ({memory_size} messages)"
            
            # Display health status
            st.markdown("**🏥 System Health**")
            for component, status in health_status.items():
                st.markdown(f"- **{component.title()}**: {status}")

# ---------- PERFORMANCE OPTIMIZATION ----------
# Clean up old session data periodically
if len(st.session_state.chat_history) > 100:
    st.session_state.chat_history = st.session_state.chat_history[-50:]  # Keep last 50 messages
    st.info("🧹 Cleaned up chat history for optimal performance")

# ---------- DEBUG MODE (Hidden unless needed) ----------
if st.secrets.get("DEBUG_MODE", False):
    with st.expander("🔧 Debug Information"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        if st.session_state.last_query_result:
            st.text_area("Last Query Result:", st.session_state.last_query_result, height=200)
        if st.session_state.last_df is not None:
            st.write("DataFrame Info:")
            st.write(f"Shape: {st.session_state.last_df.shape}")
            st.write("Columns:", st.session_state.last_df.columns.tolist())
            st.write("Data Types:", st.session_state.last_df.dtypes.to_dict())
            
