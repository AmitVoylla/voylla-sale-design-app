#!/usr/bin/env python
# coding: utf-8
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configuration
MODEL_NAME = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.1

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
.conversation-item {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #e0e0e0;
}
.stButton button {
    use_container_width: 100%;
}
.css-1d391kg {padding: 1.5rem;}
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
    return ChatOpenAI(model=MODEL_NAME, temperature=LLM_TEMPERATURE, request_timeout=120, max_retries=3)

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
    schema_rows = []
    q = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q)).fetchall()
        for c, t, n in rows:
            schema_rows.append(f'- "{c}" ({t}, nullable: {n})')
    
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(schema_rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# MEMORY & CONVERSATION MANAGEMENT
# =========================
@st.cache_resource
def get_memory():
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Don't cache memory - we want it to persist within session but not across sessions
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

memory = st.session_state.conversation_memory

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

def make_sql_prompt(question: str, schema_text: str, history: list = None) -> str:
    history_text = ""
    if history:
        # Only use last few relevant exchanges for context
        recent_history = history[-6:] if len(history) > 6 else history
        history_text = "\n# CONVERSATION HISTORY:\n" + "\n".join([
            f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: Generated SQL for analysis"
            for m in recent_history
        ])
    
    return f"""
You are an expert PostgreSQL data analyst specializing in jewelry business intelligence.
Generate ONLY a valid PostgreSQL SELECT query to answer the user's question.

CRITICAL REQUIREMENTS:
1. Use ONLY voylla."voylla_design_ai" table
2. ALWAYS exclude cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'
3. Use double quotes for ALL column names: "Column Name"
4. Return ONLY the SQL query - NO explanations, markdown, or code blocks
5. For time periods, use appropriate date filters on "Date" column
6. For aggregations, use proper GROUP BY clauses
7. For rankings/top items, use ORDER BY with LIMIT

SAFETY / GUARANTEES
- **Never** run mutating queries. Disallow: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
- Read-only analytical SELECTs only.
- Return complete tables (no truncation) unless the user explicitly asks for a LIMIT.
- Use **PostgreSQL** syntax and always **double-quote** column names.

Design Intelligence Attributes
- **"Design Style"** (text) — Aesthetic (Tribal, Contemporary, Traditional/Ethnic, Minimalist)
- **"Form"** (text) — Shape (Triangle, Stud, Hoop, Jhumka, Ear Cuff)
- **"Metal Color"** (text) — Finish (Antique Silver, Yellow Gold, Rose Gold, Silver, Antique Gold, Oxidized Black)
- **"Look"** (text) — Occasion/vibe (Oxidized, Everyday, Festive, Party, Wedding)
- **"Craft Style"** (text) — Technique (Handcrafted, etc.)
- **"Central Stone"** (text) — Primary gemstone
- **"Surrounding Layout"** (text) — Stone arrangement
- **"Stone Setting"** (text) — Mounting style
- **"Style Motif"** (text) — Design theme (Geometric, Floral, Abstract)

QUERY PATTERNS
- For **trending designs**: GROUP BY date bucket (e.g., month) and Design Style
- For **success combinations**: GROUP BY 3–5 traits (avoid overly wide groups); order by SUM("Qty") DESC then SUM("Amount") DESC
- For **channel breakdown**: include "Channel" in SELECT/GROUP BY
- For **top SKUs**: group or filter by "Product Code"
- Always **quote** column names and fully qualify the table as voylla."voylla_design_ai"


COMMON PATTERNS:
- Revenue analysis: SUM("Amount")
- Unit sales: SUM("Qty") 
- Average order value: SUM("Amount")/NULLIF(SUM("Qty"),0)
- Profit margin: (SUM("Amount") - SUM("Cost Price" * "Qty"))/NULLIF(SUM("Amount"),0) * 100
- Time periods: DATE_TRUNC('month', "Date") for monthly data
- Recent data: "Date" >= CURRENT_DATE - INTERVAL '30 days'
- Inventory : SUM("Inventory")
- Product Code is also called "SKU"
- Type column consists two types 'Online' And 'Offline'


OUTPUT RULES
- Return results as a markdown table with **all** rows (unless user asks a LIMIT)
- If a question is ambiguous, make a **reasonable assumption** and state it briefly above the table
- For follow-up questions, acknowledge the previous context: "Building on the previous analysis..." or "Expanding on those results..."

For Non-Database Questions:
- Respond naturally and helpfully
- Handle greetings, casual chat, and general knowledge  
- Be conversational and friendly


SCHEMA:
{schema_text}
{history_text}

USER QUESTION: {question}

SQL QUERY:"""

def generate_sql(question: str, history: list = None) -> str:
    """Generate SQL with improved accuracy and context awareness"""
    prompt = make_sql_prompt(question, schema_doc, history)
    
    try:
        response = llm.invoke(prompt)
        sql = response.content.strip()
        
        # Clean up the response
        if sql.startswith("```"):
            sql = re.sub(r"^```[a-zA-Z0-9]*\n?", "", sql)
            sql = sql.replace("```", "")
        
        sql = sql.strip()
        
        # Ensure basic requirements
        if not sql.upper().startswith("SELECT"):
            raise ValueError("Query must be a SELECT statement")
        
        if DANGEROUS.search(sql):
            raise ValueError("Generated SQL contains non-read-only operations")
        
        if "voylla_design_ai" not in sql:
            raise ValueError("SQL must reference the voylla_design_ai table")
        
        # Add cancelled filter if not present
        if "CANCELLED" not in sql.upper():
            if "WHERE" in sql.upper():
                sql = sql.replace("WHERE", 'WHERE "Sale Order Item Status" != \'CANCELLED\' AND')
            else:
                sql += ' WHERE "Sale Order Item Status" != \'CANCELLED\''
        
        return sql
        
    except Exception as e:
        st.error(f"SQL generation error: {e}")
        raise

def run_sql_to_df(sql: str) -> pd.DataFrame:
    """Execute SQL with better error handling"""
    with engine.connect() as conn:
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        except Exception as e:
            st.error(f"Database error: {e}")
            raise RuntimeError(f"SQL execution error: {e}")

def analyze_data_improved(df: pd.DataFrame, user_q: str, history: list = None) -> dict:
    """Improved analysis with more structured and accurate insights"""
    
    if df.empty:
        return {
            "executive_summary": "No data found for the specified criteria.",
            "key_metrics": {},
            "insights": [],
            "recommendations": [],
            "followup_questions": []
        }
    
    # Create more focused analysis prompt
    analysis_prompt = f"""
You are a senior business analyst at Voylla jewelry company. Analyze the data and provide actionable insights.

USER QUESTION: {user_q}

DATA SUMMARY:
- Total Records: {len(df)}
- Columns: {list(df.columns)}
- Date Range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}

KEY DATA POINTS:
{df.head(10).to_string()}

STATISTICAL SUMMARY:
{df.describe().to_string() if not df.empty else 'No numeric data'}

Provide your analysis in the following JSON format:
{{
  "executive_summary": "Clear 2-3 sentence summary of key findings",
  "key_metrics": {{
    "total_revenue": "value if applicable",
    "total_units": "value if applicable", 
    "avg_order_value": "value if applicable",
    "top_performer": "value if applicable"
  }},
  "insights": [
    {{
      "title": "Specific insight title",
      "description": "Detailed explanation with numbers",
      "impact": "Business impact level (High/Medium/Low)",
      "data_support": "Specific numbers from the data"
    }}
  ],
  "recommendations": [
    {{
      "title": "Actionable recommendation",
      "description": "What to do and why",
      "expected_impact": "Expected business outcome"
    }}
  ],
  "followup_questions": [
    "Specific analytical question based on findings",
    "Another relevant question"
  ]
}}
"""
    
    try:
        response = llm.invoke(analysis_prompt).content.strip()
        # Clean JSON response
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()

        import re, json
        
        def safe_json_loads(response: str):
            # Extract JSON block only
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                return {}
            
            json_str = match.group(0)
        
            # Fix common issues:
            # 1. Replace unquoted keys with quoted keys
            json_str = re.sub(r'(\s*)([A-Za-z0-9_]+)(\s*):', r'\1"\2"\3:', json_str)
            # 2. Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
        
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Final JSON parse failed:", e)
                return {}


        analysis = safe_json_loads(response)


        
        # analysis = json.loads(response)
        
        # Calculate actual metrics from data
        actual_metrics = {}
        if 'Amount' in df.columns:
            actual_metrics['Total Revenue'] = f"₹{df['Amount'].sum():,.2f}"
        if 'Qty' in df.columns:
            actual_metrics['Total Units'] = f"{df['Qty'].sum():,}"
        if 'Amount' in df.columns and 'Qty' in df.columns and df['Qty'].sum() > 0:
            actual_metrics['Avg Order Value'] = f"₹{df['Amount'].sum() / df['Qty'].sum():.2f}"
        
        # Merge with calculated metrics
        analysis['key_metrics'].update(actual_metrics)
        
        return analysis
        
    except Exception as e:
        st.warning(f"Analysis parsing error: {e}")
        # Fallback analysis
        return {
            "executive_summary": f"Analysis completed for {len(df)} records. Key patterns identified in the data.",
            "key_metrics": {
                "Total Records": str(len(df)),
                "Columns Analyzed": str(len(df.columns))
            },
            "insights": [
                {
                    "title": "Data Overview",
                    "description": f"Analyzed {len(df)} records across {len(df.columns)} dimensions",
                    "impact": "Medium",
                    "data_support": f"Dataset contains {len(df)} rows"
                }
            ],
            "recommendations": [
                {
                    "title": "Further Analysis Needed",
                    "description": "Consider drilling down into specific metrics for deeper insights",
                    "expected_impact": "Better understanding of business trends"
                }
            ],
            "followup_questions": [
                "What are the top performing categories?",
                "How do sales vary by time period?"
            ]
        }

def create_smart_visualizations(df: pd.DataFrame, analysis: dict, question: str):
    """Create relevant visualizations based on data and question context"""
    visualizations = []
    
    if df.empty:
        return visualizations
    
    try:
        # 1. Time series if date data exists
        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            
            # Revenue over time
            if 'Amount' in df.columns:
                time_df = df_copy.groupby(df_copy['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
                time_df['Date'] = time_df['Date'].dt.to_timestamp()
                
                fig = px.line(time_df, x='Date', y='Amount', 
                             title="Revenue Trend Over Time",
                             labels={'Amount': 'Revenue (₹)', 'Date': 'Month'})
                fig.update_layout(height=400)
                visualizations.append(fig)
        
        # 2. Category analysis
        categorical_cols = ['Channel', 'Design Style', 'Form', 'Metal Color', 'Look', 'Central Stone']
        available_cats = [col for col in categorical_cols if col in df.columns]
        
        if available_cats and 'Amount' in df.columns:
            cat_col = available_cats[0]
            top_df = df.groupby(cat_col)['Amount'].sum().nlargest(10).reset_index()
            
            fig = px.bar(top_df, x=cat_col, y='Amount',
                        title=f"Top 10 {cat_col} by Revenue",
                        labels={'Amount': 'Revenue (₹)'})
            fig.update_layout(height=400, xaxis_tickangle=-45)
            visualizations.append(fig)
        
        # 3. Distribution charts
        if 'Amount' in df.columns:
            fig = px.histogram(df, x='Amount', bins=20,
                              title="Revenue Distribution",
                              labels={'Amount': 'Revenue (₹)', 'count': 'Frequency'})
            fig.update_layout(height=400)
            visualizations.append(fig)
        
        # 4. Performance comparison
        if len(available_cats) >= 2 and 'Amount' in df.columns:
            pivot_df = df.groupby([available_cats[0], available_cats[1]])['Amount'].sum().reset_index()
            top_categories = df.groupby(available_cats[0])['Amount'].sum().nlargest(5).index
            pivot_df = pivot_df[pivot_df[available_cats[0]].isin(top_categories)]
            
            fig = px.bar(pivot_df, x=available_cats[0], y='Amount', color=available_cats[1],
                        title=f"Revenue by {available_cats[0]} and {available_cats[1]}",
                        labels={'Amount': 'Revenue (₹)'})
            fig.update_layout(height=400, xaxis_tickangle=-45)
            visualizations.append(fig)
            
    except Exception as e:
        st.warning(f"Visualization error: {e}")
    
    return visualizations

def display_analysis(analysis: dict):
    """Display analysis in structured format"""
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    st.markdown(f"<div class='executive-summary'>{analysis.get('executive_summary', 'No summary available.')}</div>", 
                unsafe_allow_html=True)
    
    # Key Metrics
    if analysis.get("key_metrics"):
        st.markdown("### 📊 Key Metrics")
        metrics = analysis["key_metrics"]
        cols = st.columns(min(len(metrics), 4))
        for i, (metric, value) in enumerate(metrics.items()):
            cols[i % 4].metric(metric, value)
    
    # Insights
    if analysis.get("insights"):
        st.markdown("### 💡 Key Insights")
        for insight in analysis["insights"]:
            with st.expander(f"🔍 {insight.get('title', 'Insight')}"):
                st.markdown(f"**Description:** {insight.get('description', '')}")
                st.markdown(f"**Business Impact:** {insight.get('impact', '')}")
                if insight.get('data_support'):
                    st.markdown(f"**Supporting Data:** {insight.get('data_support')}")
    
    # Recommendations
    if analysis.get("recommendations"):
        st.markdown("### 🎯 Strategic Recommendations")
        for i, rec in enumerate(analysis["recommendations"]):
            st.markdown(f"<div class='recommendation-card'><strong>{rec.get('title', 'Recommendation')}</strong><br>{rec.get('description', '')}<br><em>Expected Impact: {rec.get('expected_impact', '')}</em></div>", 
                       unsafe_allow_html=True)
    
    # Follow-up Questions
    if analysis.get("followup_questions"):
        st.markdown("### 🔍 Suggested Follow-up Questions")
        cols = st.columns(2)
        for i, question in enumerate(analysis["followup_questions"]):
            if cols[i % 2].button(f"📊 {question}", key=f"followup_{hash(question)}_{i}"):
                st.session_state.auto_q = question
                st.rerun()

# =========================
# SESSION STATE INITIALIZATION
# =========================
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
if "analysis_history" not in st.session_state: 
    st.session_state.analysis_history = []
if "auto_q" not in st.session_state: 
    st.session_state.auto_q = None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    
    # Connection status
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
    
    # Preset questions
    st.header("💡 Quick Analysis")
    presets = [
        "Show top 10 channels by revenue this month",
        "Compare design styles performance year over year", 
        "Analyze profit margins by metal color",
        "What are seasonal sales patterns?",
        "Which products have highest average order value?",
        "Show revenue trends for last 6 months",
        "Compare channel performance quarter by quarter",
        "Identify best performing product combinations"
    ]
    
    for q in presets:
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state["auto_q"] = q
            st.rerun()
    
    st.markdown("---")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All", width='stretch'):
            st.session_state.chat_history = []
            st.session_state.analysis_history = []
            st.session_state.conversation_memory.clear()
            st.rerun()
    
    with col2:
        show_history = st.checkbox("Show History", value=True)

# =========================
# MAIN INTERFACE
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence & Sales Analytics — Persistent Analysis History")

# Display conversation history
if show_history and st.session_state.analysis_history:
    st.markdown("### 📚 Analysis History")
    
    # Show recent analyses in expandable sections
    for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
        with st.expander(f"Q{len(st.session_state.analysis_history)-i}: {item['question'][:80]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_analysis(item['analysis'])
            
            with col2:
                if item['visualizations']:
                    st.markdown("#### 📈 Charts")
                    for viz in item['visualizations']:
                        st.plotly_chart(viz, width='stretch')
                
                if not item['data'].empty:
                    st.markdown("#### 📋 Data")
                    st.dataframe(item['data'].head(), width='stretch')
                    
                    # Quick export
                    csv = item['data'].to_csv(index=False)
                    st.download_button(
                        "💾 Export CSV",
                        data=csv,
                        file_name=f"analysis_{i}.csv",
                        mime="text/csv",
                        width='stretch'
                    )

# Chat input
inp = st.chat_input("Ask any question about sales, design trends, or business performance...", key="main_input")

# Handle auto questions
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

if inp:
    st.markdown("---")
    st.markdown(f"### 🤔 Question: {inp}")
    
    with st.spinner("🔍 Analyzing data and generating insights..."):
        try:
            # Generate SQL
            sql = generate_sql(inp, st.session_state.chat_history)
            
            # Execute query
            df = run_sql_to_df(sql)
            
            # Perform analysis
            analysis = analyze_data_improved(df, inp, st.session_state.chat_history)
            
            # Create visualizations
            visualizations = create_smart_visualizations(df, analysis, inp)
            
            # Store in history
            analysis_item = {
                'question': inp,
                'sql': sql,
                'data': df,
                'analysis': analysis,
                'visualizations': visualizations,
                'timestamp': datetime.now()
            }
            st.session_state.analysis_history.append(analysis_item)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": inp})
            st.session_state.chat_history.append({"role": "assistant", "content": analysis.get("executive_summary", "Analysis completed")})
            
            # Update memory
            memory.save_context(
                {"input": inp}, 
                {"output": f"Generated SQL and analyzed {len(df)} records. Key insight: {analysis.get('executive_summary', '')[:100]}"}
            )
            
        except Exception as e:
            st.error(f"❌ Analysis Error: {str(e)}")
            st.stop()
    
    # Display current analysis
    st.markdown("### 📊 Current Analysis")
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_analysis(analysis)
    
    with col2:
        # Show SQL
        with st.expander("🔍 Generated SQL"):
            st.code(sql, language="sql")
        
        # Quick stats
        if not df.empty:
            st.markdown("#### 📈 Quick Stats")
            st.info(f"**Records:** {len(df):,}\n\n**Columns:** {len(df.columns)}")
    
    # Visualizations
    if visualizations:
        st.markdown("### 📈 Data Visualizations")
        
        # Display in grid
        if len(visualizations) == 1:
            st.plotly_chart(visualizations[0], width='stretch')
        elif len(visualizations) == 2:
            col1, col2 = st.columns(2)
            col1.plotly_chart(visualizations[0], width='stretch')
            col2.plotly_chart(visualizations[1], width='stretch')
        else:
            for i in range(0, len(visualizations), 2):
                cols = st.columns(2)
                for j, viz in enumerate(visualizations[i:i+2]):
                    cols[j].plotly_chart(viz, width='stretch')
    
    # Data table
    if not df.empty:
        st.markdown("### 📋 Data Table")
        st.dataframe(df, width='stretch', height=400)
        
        # Export options
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv,
                file_name=f"voylla_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            # Excel export with analysis
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add analysis sheet
                analysis_df = pd.DataFrame([
                    ['Question', inp],
                    ['Executive Summary', analysis.get('executive_summary', '')],
                    ['Total Records', len(df)],
                    ['Generated SQL', sql]
                ], columns=['Field', 'Value'])
                analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
            
            output.seek(0)
            st.download_button(
                "📊 Download Excel",
                data=output.getvalue(),
                file_name=f"voylla_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )
        
        with col3:
            st.info(f"**{len(df):,}** records analyzed")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
💎 <strong>Voylla DesignGPT</strong> | Advanced Business Intelligence<br>
💡 <em>Tips:</em> Ask about trends, comparisons, profitability, seasonal patterns, or growth opportunities<br>
🔄 All previous analyses are preserved above for reference and comparison
</div>
""", unsafe_allow_html=True)
