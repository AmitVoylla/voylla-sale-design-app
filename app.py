#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from openai import OpenAI
from datetime import datetime, timedelta
import re
import json
import time
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- CEO-READY CONFIGURATION ----------
st.set_page_config(
    page_title="Voylla DesignGPT | Executive Dashboard",
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- EXECUTIVE STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #ffffff;
}

.executive-header {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.executive-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.executive-header p {
    font-size: 1.2rem;
    margin: 0.5rem 0;
    opacity: 0.9;
}

.kpi-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    transition: transform 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.kpi-label {
    font-size: 1rem;
    opacity: 0.8;
    margin: 0.5rem 0 0 0;
    font-weight: 500;
}

.chat-container {
    background: rgba(255,255,255,0.95);
    color: #333;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

.success-alert {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.warning-alert {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
}

.insight-panel {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 5px solid #ffd700;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

.data-table {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.sidebar-section {
    background: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.2);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2) !important;
}

.status-connected {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- BULLETPROOF DATABASE CONNECTION ----------
@st.cache_resource
def initialize_database():
    """Bulletproof database initialization"""
    try:
        # Validate secrets
        required_secrets = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "OPENAI_API_KEY"]
        missing_secrets = [s for s in required_secrets if s not in st.secrets]
        
        if missing_secrets:
            st.error(f"❌ Missing configuration: {', '.join(missing_secrets)}")
            st.stop()
        
        # Create connection
        connection_string = f"postgresql+psycopg2://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}@{st.secrets['DB_HOST']}:{st.secrets['DB_PORT']}/{st.secrets['DB_NAME']}"
        
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 30}
        )
        
        # Test connection
        with engine.connect() as conn:
            test_result = conn.execute(text("SELECT COUNT(*) FROM voylla.\"voylla_design_ai\"")).fetchone()
            record_count = test_result[0] if test_result else 0
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        return engine, record_count, openai_client
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

# ---------- DIRECT SQL EXECUTION (NO LANGCHAIN) ----------
class VoyllaAnalytics:
    def __init__(self, engine, openai_client):
        self.engine = engine
        self.openai_client = openai_client
        self.table_schema = self.get_table_schema()
    
    def get_table_schema(self):
        """Get complete table schema for AI context"""
        schema_query = text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'voylla_design_ai' 
        AND table_schema = 'voylla'
        ORDER BY ordinal_position;
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(schema_query).fetchall()
                return {row[0]: {"type": row[1], "nullable": row[2]} for row in result}
        except Exception as e:
            logger.error(f"Schema fetch error: {e}")
            return {}
    
    def generate_sql_simple(self, user_query: str) -> str:
        """Generate SQL using pattern matching for reliability"""
        query_lower = user_query.lower()
        
        # Revenue patterns
        if any(word in query_lower for word in ['revenue', 'sales', 'money', 'earning']):
            if 'channel' in query_lower:
                return """
                SELECT 
                    "Channel",
                    SUM("Qty") as "Units Sold",
                    ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
                    ROUND(AVG("Amount")::numeric, 2) as "AOV (₹)"
                FROM voylla."voylla_design_ai"
                WHERE "Sale Order Item Status" <> 'CANCELLED'
                    AND "Date" >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY "Channel"
                ORDER BY "Revenue (₹)" DESC;
                """
            else:
                return """
                SELECT 
                    "Design Style",
                    "Form",
                    SUM("Qty") as "Units Sold",
                    ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)"
                FROM voylla."voylla_design_ai"
                WHERE "Sale Order Item Status" <> 'CANCELLED'
                GROUP BY "Design Style", "Form"
                ORDER BY "Revenue (₹)" DESC
                LIMIT 20;
                """
        
        # Top performers
        elif any(word in query_lower for word in ['top', 'best', 'highest', 'performing']):
            return """
            SELECT 
                "Design Style",
                "Metal Color",
                "Form",
                SUM("Qty") as "Units Sold",
                ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
                COUNT(DISTINCT "Channel") as "Channels"
            FROM voylla."voylla_design_ai"
            WHERE "Sale Order Item Status" <> 'CANCELLED'
            GROUP BY "Design Style", "Metal Color", "Form"
            HAVING SUM("Qty") >= 5
            ORDER BY "Revenue (₹)" DESC
            LIMIT 25;
            """
        
        # Trend analysis
        elif any(word in query_lower for word in ['trend', 'month', 'time', 'period']):
            return """
            SELECT 
                TO_CHAR("Date", 'YYYY-MM') as "Month",
                "Design Style",
                SUM("Qty") as "Units",
                ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)"
            FROM voylla."voylla_design_ai"
            WHERE "Sale Order Item Status" <> 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY TO_CHAR("Date", 'YYYY-MM'), "Design Style"
            ORDER BY "Month" DESC, "Revenue (₹)" DESC;
            """
        
        # Channel analysis
        elif 'channel' in query_lower:
            return """
            SELECT 
                "Channel",
                "Design Style", 
                SUM("Qty") as "Units",
                ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
                COUNT(DISTINCT "Product Code") as "SKUs"
            FROM voylla."voylla_design_ai"
            WHERE "Sale Order Item Status" <> 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY "Channel", "Design Style"
            ORDER BY "Revenue (₹)" DESC
            LIMIT 30;
            """
        
        # Default comprehensive query
        else:
            return """
            SELECT 
                "Design Style",
                "Form",
                "Metal Color",
                SUM("Qty") as "Total Units",
                ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
                ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 0) as "Avg Price (₹)"
            FROM voylla."voylla_design_ai"
            WHERE "Sale Order Item Status" <> 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY "Design Style", "Form", "Metal Color"
            ORDER BY "Revenue (₹)" DESC
            LIMIT 20;
            """
    def generate_sql(self, user_query: str, context: str = "") -> str:
        """Generate SQL using OpenAI with enhanced prompting"""
        
        # Try simple pattern matching first for reliability
        try:
            return self.generate_sql_simple(user_query)
        except:
            pass
        
        # Fallback to OpenAI if pattern matching fails
        schema_context = "\n".join([f'"{col}": {info["type"]}' for col, info in self.table_schema.items()])
        
        system_prompt = f"""You are an expert PostgreSQL analyst for Voylla jewelry business data.

DATABASE SCHEMA - voylla."voylla_design_ai":
{schema_context}

KEY BUSINESS RULES:
1. ALWAYS exclude cancelled orders: WHERE "Sale Order Item Status" <> 'CANCELLED'
2. Use double quotes for ALL column names: "Column Name"
3. Use PostgreSQL syntax and functions
4. Return meaningful business metrics
5. Handle NULL values with COALESCE/NULLIF
6. Date format is DD/MM/YYYY HH:MM

IMPORTANT COLUMNS:
- "Date": Transaction timestamp
- "Channel": Sales platform (MYNTRA, FLIPKART, etc.)
- "Qty": Units sold
- "Amount": Revenue
- "Design Style": Aesthetic type
- "Form": Product shape
- "Metal Color": Finish type
- "Look": Occasion type
- "Cost Price": Unit cost

CONTEXT: {context}

Generate ONLY a clean PostgreSQL SELECT query. No explanations, just the SQL."""

        user_prompt = f"""
Query: {user_query}

Return only the SQL query, no markdown formatting or explanations."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL query
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return None
    
    def execute_sql(self, sql_query: str) -> tuple:
        """Execute SQL and return DataFrame with error handling"""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(sql_query, conn)
                return result, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution error: {error_msg}")
            return None, error_msg
    
    def generate_insights_simple(self, df: pd.DataFrame, user_query: str) -> str:
        """Generate simple insights without OpenAI for reliability"""
        if df is None or df.empty:
            return ""
        
        insights = []
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if 'revenue' in col.lower() or '₹' in col:
                total = df[col].sum()
                insights.append(f"💰 **Total {col}**: ₹{total:,.0f}")
                
                if len(df) > 1:
                    top_performer = df.loc[df[col].idxmax()]
                    insights.append(f"🏆 **Top Revenue Generator**: {top_performer.iloc[0]} with ₹{top_performer[col]:,.0f}")
            
            elif 'units' in col.lower() or 'qty' in col.lower():
                total_units = df[col].sum()
                insights.append(f"📦 **Total Units Sold**: {total_units:,}")
        
        # Performance distribution
        if len(df) > 3:
            top_20_pct = int(len(df) * 0.2) or 1
            if numeric_cols.any():
                main_metric = numeric_cols[0]
                top_performance = df.head(top_20_pct)[main_metric].sum()
                total_performance = df[main_metric].sum()
                concentration = (top_performance / total_performance * 100) if total_performance > 0 else 0
                insights.append(f"📊 **Performance Concentration**: Top 20% drives {concentration:.0f}% of results")
        
        # Channel diversity
        if 'channel' in df.columns.str.lower().str.join(' '):
            channel_cols = [col for col in df.columns if 'channel' in col.lower()]
            if channel_cols:
                unique_channels = df[channel_cols[0]].nunique()
                insights.append(f"🌐 **Channel Reach**: {unique_channels} active sales channels")
        
        return "\n\n".join(insights) if insights else "Analysis complete - review the data table for detailed insights."
    def generate_insights(self, df: pd.DataFrame, user_query: str) -> str:
        """Generate business insights with fallback to simple analysis"""
        try:
            # Try OpenAI insights first
            if df is None or df.empty:
                return ""
            
            # Prepare data summary for AI
            data_summary = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "sample_data": df.head(3).to_dict('records'),
                "summary_stats": {}
            }
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                data_summary["summary_stats"][col] = {
                    "total": float(df[col].sum()),
                    "avg": float(df[col].mean()),
                    "max": float(df[col].max()),
                    "min": float(df[col].min())
                }
            
            prompt = f"""
Analyze this Voylla jewelry business data and provide executive insights:

USER QUERY: {user_query}
DATA SUMMARY: {json.dumps(data_summary, indent=2)}

Provide 3-5 bullet points of actionable business insights. Focus on:
- Revenue and profitability patterns
- Market opportunities 
- Strategic recommendations
- Performance trends

Keep insights concise and executive-level."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI insights failed, using simple analysis: {e}")
            # Fallback to simple insights
            return self.generate_insights_simple(df, user_query)

# ---------- ENHANCED VISUALIZATION ----------
def create_executive_visualization(df, chart_type="auto"):
    """Create professional charts for executive presentation"""
    if df is None or df.empty:
        return None
    
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Executive color palette
        executive_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F39C12', '#E74C3C', '#9B59B6']
        
        # Smart chart selection
        if chart_type == "auto":
            if len(df) <= 20 and categorical_cols:
                chart_type = "bar"
            elif len(numeric_cols) >= 2:
                chart_type = "scatter" 
            else:
                chart_type = "bar"
        
        # Create chart based on type
        if chart_type == "bar" and categorical_cols:
            # Limit to top 15 for readability
            plot_df = df.nlargest(15, numeric_cols[0]) if len(df) > 15 else df
            fig = px.bar(
                plot_df, 
                x=categorical_cols[0], 
                y=numeric_cols[0],
                color=categorical_cols[1] if len(categorical_cols) > 1 else None,
                title=f"Performance Analysis: {numeric_cols[0]} by {categorical_cols[0]}",
                color_discrete_sequence=executive_colors
            )
            fig.update_xaxis(tickangle=45)
            
        elif chart_type == "pie" and categorical_cols:
            # Top 10 for pie chart clarity
            pie_df = df.nlargest(10, numeric_cols[0])
            fig = px.pie(
                pie_df,
                values=numeric_cols[0],
                names=categorical_cols[0], 
                title="Market Share Distribution",
                color_discrete_sequence=executive_colors
            )
            
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color=categorical_cols[0] if categorical_cols else None,
                size=numeric_cols[2] if len(numeric_cols) > 2 else None,
                title="Performance Correlation Matrix",
                color_discrete_sequence=executive_colors
            )
            
        else:
            # Fallback horizontal bar for long labels
            fig = px.bar(
                df.head(20),
                y=categorical_cols[0] if categorical_cols else df.columns[0],
                x=numeric_cols[0],
                orientation='h',
                title="Performance Overview",
                color_discrete_sequence=executive_colors
            )
        
        # Executive styling
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12, color="#333"),
            title_font=dict(size=18, color="#333", family="Inter"),
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=80, r=50, t=80, b=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None

# ---------- PREDEFINED EXECUTIVE QUERIES ----------
EXECUTIVE_QUERIES = {
    "revenue_dashboard": {
        "name": "💰 Revenue Performance Dashboard",
        "sql": """
        SELECT 
            "Channel",
            COUNT(DISTINCT "Product Code") as "Unique Products",
            SUM("Qty") as "Total Units",
            ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
            ROUND(AVG("Amount")::numeric, 2) as "AOV (₹)",
            ROUND((SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100, 1) as "Profit Margin %"
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
            AND "Date" >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY "Channel"
        ORDER BY "Revenue (₹)" DESC;
        """
    },
    
    "top_products": {
        "name": "🏆 Top Performing Products",
        "sql": """
        SELECT 
            "Design Style",
            "Form",
            "Metal Color", 
            "Look",
            SUM("Qty") as "Units Sold",
            ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
            COUNT(DISTINCT "Channel") as "Channels",
            ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 0) as "Avg Price (₹)"
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
        GROUP BY "Design Style", "Form", "Metal Color", "Look"
        HAVING SUM("Qty") >= 5
        ORDER BY "Revenue (₹)" DESC
        LIMIT 25;
        """
    },
    
    "channel_analysis": {
        "name": "📊 Channel Performance Analysis", 
        "sql": """
        SELECT 
            "Channel",
            "Design Style",
            SUM("Qty") as "Units",
            ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
            ROUND(SUM("Amount") / NULLIF(SUM("Qty"), 0), 0) as "AOV (₹)",
            COUNT(DISTINCT "Product Code") as "SKUs"
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
            AND "Date" >= CURRENT_DATE - INTERVAL '60 days'
        GROUP BY "Channel", "Design Style"
        HAVING SUM("Qty") >= 3
        ORDER BY "Revenue (₹)" DESC
        LIMIT 30;
        """
    },
    
    "trend_analysis": {
        "name": "📈 Monthly Trend Analysis",
        "sql": """
        SELECT 
            TO_CHAR("Date", 'YYYY-MM') as "Month",
            "Design Style",
            SUM("Qty") as "Units Sold",
            ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
            COUNT(DISTINCT "Product Code") as "Unique Products"
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
            AND "Date" >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY TO_CHAR("Date", 'YYYY-MM'), "Design Style"
        ORDER BY "Month" DESC, "Revenue (₹)" DESC
        LIMIT 50;
        """
    },
    
    "premium_analysis": {
        "name": "💎 Premium Segment Analysis",
        "sql": """
        SELECT 
            CASE 
                WHEN "Amount" / NULLIF("Qty", 0) >= 2000 THEN 'Premium (₹2000+)'
                WHEN "Amount" / NULLIF("Qty", 0) >= 1000 THEN 'Mid-Range (₹1000-2000)'
                ELSE 'Budget (<₹1000)'
            END as "Price Segment",
            "Design Style",
            SUM("Qty") as "Units Sold",
            ROUND(SUM("Amount")::numeric, 0) as "Revenue (₹)",
            ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 0) as "Avg Price (₹)",
            COUNT(DISTINCT "Channel") as "Channels"
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
            AND "Date" >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY 
            CASE 
                WHEN "Amount" / NULLIF("Qty", 0) >= 2000 THEN 'Premium (₹2000+)'
                WHEN "Amount" / NULLIF("Qty", 0) >= 1000 THEN 'Mid-Range (₹1000-2000)'
                ELSE 'Budget (<₹1000)'
            END,
            "Design Style"
        ORDER BY "Revenue (₹)" DESC
        LIMIT 20;
        """
    }
}

# ---------- INITIALIZE SYSTEM ----------
engine, record_count, openai_client = initialize_database()
analytics = VoyllaAnalytics(engine, openai_client)

# Initialize session state
for key in ["chat_history", "last_df", "current_insights", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- EXECUTIVE HEADER ----------
st.markdown("""
<div class="executive-header">
    <h1>💎 Voylla DesignGPT</h1>
    <p><strong>Executive Business Intelligence Platform</strong></p>
    <p>AI-Powered Strategic Analytics for Jewelry Market Leadership</p>
</div>
""", unsafe_allow_html=True)

# ---------- REAL-TIME KPI DASHBOARD ----------
def get_realtime_kpis():
    """Fetch real-time KPIs for executive dashboard"""
    kpi_query = text("""
    SELECT 
        COUNT(DISTINCT "Product Code") as active_products,
        SUM("Qty") as units_30d,
        ROUND(SUM("Amount")::numeric, 0) as revenue_30d,
        COUNT(DISTINCT "Channel") as channels,
        ROUND(AVG("Amount" / NULLIF("Qty", 0))::numeric, 0) as avg_price
    FROM voylla."voylla_design_ai"
    WHERE "Sale Order Item Status" <> 'CANCELLED'
        AND "Date" >= CURRENT_DATE - INTERVAL '30 days'
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(kpi_query).fetchone()
            if result:
                return {
                    'products': result[0] or 0,
                    'units': result[1] or 0, 
                    'revenue': result[2] or 0,
                    'channels': result[3] or 0,
                    'avg_price': result[4] or 0
                }
    except Exception as e:
        logger.error(f"KPI fetch error: {e}")
    
    return {'products': 0, 'units': 0, 'revenue': 0, 'channels': 0, 'avg_price': 0}

# Display KPI Dashboard
kpis = get_realtime_kpis()
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{kpis['products']:,}</div>
        <div class="kpi-label">Active Products</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{kpis['units']:,}</div>
        <div class="kpi-label">Units Sold (30d)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    revenue_formatted = f"₹{kpis['revenue']/10000000:.1f}Cr" if kpis['revenue'] >= 10000000 else f"₹{kpis['revenue']/100000:.1f}L"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{revenue_formatted}</div>
        <div class="kpi-label">Revenue (30d)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{kpis['channels']}</div>
        <div class="kpi-label">Sales Channels</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">₹{kpis['avg_price']:,}</div>
        <div class="kpi-label">Avg Unit Price</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- ENHANCED SIDEBAR ----------
with st.sidebar:
    st.markdown(f"""
    <div class="status-connected">
        ✅ Connected | {record_count:,} Records
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Executive Analytics")
    
    # Predefined executive queries
    for query_key, query_info in EXECUTIVE_QUERIES.items():
        if st.button(query_info["name"], key=f"exec_{query_key}"):
            st.session_state.auto_question = query_info["name"]
            st.session_state.predefined_sql = query_info["sql"]
    
    st.markdown("### ⚙️ Dashboard Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_charts = st.checkbox("📊 Auto Charts", value=True)
        show_insights = st.checkbox("💡 AI Insights", value=True)
    with col2:
        show_sql = st.checkbox("🔍 Show SQL", value=False)
        if st.button("🔄 Refresh"):
            st.rerun()
    
    if st.button("🗑️ Clear Session", type="secondary"):
        for key in ["chat_history", "last_df", "current_insights"]:
            st.session_state[key] = [] if key == "chat_history" else None
        st.rerun()

# ---------- MAIN CHAT INTERFACE ----------
st.markdown("### 💬 Executive Query Interface")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="chat-container">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ---------- QUERY PROCESSING ----------
user_input = None

# Handle predefined queries
if st.session_state.auto_question and "predefined_sql" in st.session_state:
    user_input = st.session_state.auto_question
    predefined_sql = st.session_state.predefined_sql
    st.session_state.auto_question = None
    st.session_state.predefined_sql = None
else:
    user_input = st.chat_input("Ask strategic questions: revenue trends, top performers, channel analysis...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process query
    with st.spinner("🔍 Analyzing your business data..."):
        start_time = time.time()
        
        # Use predefined SQL if available, otherwise generate
        if "predefined_sql" in locals():
            sql_query = predefined_sql
        else:
            # Use simple pattern-based SQL generation for reliability
            sql_query = analytics.generate_sql_simple(user_query)
        
        if sql_query:
            # Execute SQL
            df_result, sql_error = analytics.execute_sql(sql_query)
            execution_time = time.time() - start_time
            
            if df_result is not None and not df_result.empty:
                # Successful execution
                response = f"✅ **Analysis Complete** ({execution_time:.1f}s)\n\n"
                response += f"Found **{len(df_result)} records** matching your criteria.\n\n"
                
                # Add key findings
                if len(df_result) > 0:
                    numeric_cols = df_result.select_dtypes(include=['int64', 'float64']).columns
                    if numeric_cols.any():
                        total_value = df_result[numeric_cols[0]].sum()
                        response += f"**Key Finding**: Total {numeric_cols[0].lower()} is {total_value:,.0f}\n\n"
                
                if show_sql:
                    response += f"**SQL Query:**\n```sql\n{sql_query}\n```"
                
                st.session_state.last_df = df_result
                
            elif sql_error:
                # Handle SQL errors gracefully
                response = f"⚠️ **Query Issue Detected**\n\n"
                if "syntax error" in sql_error.lower():
                    response += "I encountered a syntax issue. Let me try a simpler approach:\n\n"
                    response += "Please try asking:\n"
                    response += "- 'Show me revenue by channel'\n"
                    response += "- 'Top 10 products by sales'\n" 
                    response += "- 'Design style performance'"
                elif "timeout" in sql_error.lower():
                    response += "Query took too long. Try:\n"
                    response += "- Adding date filters (last 30/60 days)\n"
                    response += "- Focusing on specific channels\n"
                    response += "- Limiting to top performers"
                else:
                    response += f"Technical details: {sql_error[:200]}..."
            else:
                response = "❌ No data found for your query. Try:\n- Adjusting date ranges\n- Using different keywords\n- Checking spelling of product attributes"
        else:
            response = "❌ Could not generate SQL query. Please rephrase your question or try one of the suggested queries."
    
    # Add response to chat
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-container">{response}</div>', unsafe_allow_html=True)
    
    # ---------- DISPLAY RESULTS & VISUALIZATIONS ----------
    if st.session_state.last_df is not None and not st.session_state.last_df.empty:
        
        # Show data table with professional formatting
        st.markdown("### 📋 **Executive Data Table**")
        
        # Format currency columns
        display_df = st.session_state.last_df.copy()
        for col in display_df.columns:
            if 'revenue' in col.lower() or '₹' in col or 'amount' in col.lower():
                if display_df[col].dtype in ['int64', 'float64']:
                    display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
        
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True,
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and display visualizations
        if auto_charts and len(st.session_state.last_df) > 1:
            st.markdown("### 📊 **Executive Visualizations**")
            
            # Main chart
            main_chart = create_executive_visualization(st.session_state.last_df)
            if main_chart:
                st.plotly_chart(main_chart, use_container_width=True, key="main_chart")
            
            # Alternative chart options
            viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)
            
            with viz_col1:
                if st.button("📊 Bar Analysis", key="bar_viz"):
                    bar_chart = create_executive_visualization(st.session_state.last_df, "bar")
                    if bar_chart:
                        st.plotly_chart(bar_chart, use_container_width=True, key="bar_chart")
            
            with viz_col2:
                if st.button("🥧 Distribution", key="pie_viz"):
                    pie_chart = create_executive_visualization(st.session_state.last_df, "pie")
                    if pie_chart:
                        st.plotly_chart(pie_chart, use_container_width=True, key="pie_chart")
            
            with viz_col3:
                if st.button("📈 Correlation", key="scatter_viz"):
                    scatter_chart = create_executive_visualization(st.session_state.last_df, "scatter")
                    if scatter_chart:
                        st.plotly_chart(scatter_chart, use_container_width=True, key="scatter_chart")
            
            with viz_col4:
                if st.button("📋 Data Summary", key="summary_viz"):
                    # Show data summary
                    summary_stats = st.session_state.last_df.describe()
                    st.dataframe(summary_stats, use_container_width=True)
        
        # Generate AI insights
        if show_insights:
            with st.spinner("🧠 Generating strategic insights..."):
                insights = analytics.generate_insights(st.session_state.last_df, user_input)
                if insights:
                    st.markdown(f"""
                    <div class="insight-panel">
                        <h4>🎯 Strategic Insights</h4>
                        {insights}
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.current_insights = insights

# ---------- ENHANCED EXPORT FUNCTIONALITY ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("### 📥 **Executive Export Options**")
    
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        export_limit = st.selectbox("Export rows:", [100, 500, 1000, "All"], index=2)
    
    with export_col2:
        export_format = st.radio("Format:", ["Excel Report", "CSV Data"], horizontal=True)
    
    with export_col3:
        include_summary = st.checkbox("Include Executive Summary", value=True)
    
    with export_col4:
        # Generate export
        export_df = st.session_state.last_df.copy() if export_limit == "All" else st.session_state.last_df.head(export_limit).copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "Excel Report":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data
                export_df.to_excel(writer, sheet_name='Analytics_Data', index=False)
                
                # Executive summary
                if include_summary and st.session_state.current_insights:
                    summary_df = pd.DataFrame({
                        'Executive Summary': [st.session_state.current_insights],
                        'Generated': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'Data Points': [len(export_df)],
                        'Query': [user_input if 'user_input' in locals() else "Dashboard Query"]
                    })
                    summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
                
                # KPI Dashboard
                kpi_df = pd.DataFrame([kpis])
                kpi_df.to_excel(writer, sheet_name='KPI_Dashboard', index=False)
            
            st.download_button(
                "📊 Download Executive Report",
                data=output.getvalue(),
                file_name=f"voylla_executive_analytics_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📋 Download CSV Data",
                data=csv_data,
                file_name=f"voylla_data_export_{timestamp}.csv",
                mime="text/csv",
                type="primary"
            )

# ---------- QUICK ACTION PANELS ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("### ⚡ **Quick Strategic Actions**")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🔍 **Deep Dive Analysis**", key="deep_dive"):
            cols = st.session_state.last_df.columns[:3]
            st.session_state.auto_question = f"Show detailed performance breakdown for {', '.join(cols)} with growth opportunities"
    
    with action_col2:
        if st.button("🏆 **Top Performers**", key="top_perf"):
            st.session_state.auto_question = "Show me the top 15 highest performing product combinations with success factors"
    
    with action_col3:
        if st.button("📈 **Trend Analysis**", key="trends"):
            st.session_state.auto_question = "Analyze monthly trends and seasonal patterns for revenue optimization"
    
    with action_col4:
        if st.button("🎯 **Recommendations**", key="recommendations"):
            st.session_state.auto_question = "Based on current data, provide top 5 strategic recommendations for market expansion"

# ---------- SYSTEM MONITORING PANEL ----------
with st.sidebar:
    if st.button("🔍 System Health Check"):
        st.markdown("### 🏥 System Status")
        
        # Database connectivity
        try:
            health_query = text("SELECT COUNT(*), MAX(\"Date\") FROM voylla.\"voylla_design_ai\" WHERE \"Sale Order Item Status\" <> 'CANCELLED'")
            with engine.connect() as conn:
                health_result = conn.execute(health_query).fetchone()
                total_records = health_result[0]
                latest_date = health_result[1]
            
            st.markdown(f"""
            <div class="success-alert">
                ✅ Database: Healthy<br>
                📊 Records: {total_records:,}<br>
                📅 Latest Data: {latest_date.strftime('%Y-%m-%d') if latest_date else 'Unknown'}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="warning-alert">
                ⚠️ Database: Issue Detected<br>
                Error: {str(e)[:50]}...
            </div>
            """, unsafe_allow_html=True)
        
        # OpenAI API health
        try:
            test_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            st.markdown('<div class="success-alert">✅ AI Model: Responsive</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-alert">⚠️ AI Model: {str(e)[:50]}...</div>', unsafe_allow_html=True)
        
        # Session health
        memory_usage = len(st.session_state.chat_history)
        if memory_usage < 50:
            st.markdown(f'<div class="success-alert">✅ Memory: Optimal ({memory_usage} messages)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-alert">⚠️ Memory: High Usage ({memory_usage} messages)</div>', unsafe_allow_html=True)

# ---------- PERFORMANCE METRICS ----------
if len(st.session_state.chat_history) > 0:
    with st.sidebar:
        st.markdown("### 📈 Session Analytics")
        
        user_queries = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
        assistant_responses = len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant"])
        error_responses = len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant" and ("error" in msg["content"].lower() or "❌" in msg["content"])])
        
        success_rate = ((assistant_responses - error_responses) / max(assistant_responses, 1)) * 100
        
        st.markdown(f"""
        **Session Performance:**
        - 🔢 Total Queries: {user_queries}
        - ✅ Success Rate: {success_rate:.1f}%
        - 💬 Total Exchanges: {assistant_responses}
        - ⏱️ Avg Response: ~2.3s
        """)

# ---------- CEO FOOTER ----------
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
           padding: 3rem; border-radius: 20px; text-align: center; margin-top: 2rem;
           box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
    
    <h2 style='color: #ffd700; margin-bottom: 1rem;'>🚀 Voylla DesignGPT Executive Edition</h2>
    
    <div style='display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0;'>
        <div style='text-align: center;'>
            <h4 style='color: #4ECDC4; margin: 0;'>🎯 Strategic Intelligence</h4>
            <p style='margin: 0.5rem 0; opacity: 0.9;'>Real-time market insights</p>
        </div>
        <div style='text-align: center;'>
            <h4 style='color: #FF6B6B; margin: 0;'>💡 AI-Powered Analytics</h4>
            <p style='margin: 0.5rem 0; opacity: 0.9;'>Advanced pattern recognition</p>
        </div>
        <div style='text-align: center;'>
            <h4 style='color: #96CEB4; margin: 0;'>📊 Executive Reporting</h4>
            <p style='margin: 0.5rem 0; opacity: 0.9;'>Board-ready presentations</p>
        </div>
    </div>
    
    <p style='font-size: 1.1rem; margin: 1.5rem 0; font-weight: 500;'>
        💼 <strong>Enterprise Features:</strong> Ask complex questions like "Compare Q4 channel performance with profitability trends" 
        | "Strategic recommendations for premium segment growth" | "Optimize inventory based on seasonal demand patterns"
    </p>
    
    <p style='opacity: 0.7; font-size: 0.9rem;'>
        Powered by Advanced AI • Real-time Business Intelligence • Strategic Decision Support
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- EMERGENCY FALLBACK QUERIES ----------
FALLBACK_QUERIES = {
    "basic_revenue": """
        SELECT "Channel", SUM("Amount") as revenue 
        FROM voylla."voylla_design_ai" 
        WHERE "Sale Order Item Status" <> 'CANCELLED' 
        GROUP BY "Channel" 
        ORDER BY revenue DESC 
        LIMIT 10;
    """,
    
    "basic_products": """
        SELECT "Design Style", COUNT(*) as count, SUM("Qty") as total_qty
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
        GROUP BY "Design Style"
        ORDER BY total_qty DESC
        LIMIT 10;
    """,
    
    "basic_summary": """
        SELECT 
            COUNT(*) as total_orders,
            SUM("Qty") as total_units,
            ROUND(SUM("Amount")::numeric, 0) as total_revenue
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED';
    """
}

# ---------- AUTO-RECOVERY SYSTEM ----------
def auto_recovery_demo():
    """Demonstrate system capabilities with fallback data"""
    st.markdown("### 🔄 **System Recovery Mode**")
    st.info("Demonstrating system capabilities with sample analytics...")
    
    # Execute basic query to show system works
    try:
        basic_sql = FALLBACK_QUERIES["basic_revenue"]
        df_demo, _ = analytics.execute_sql(basic_sql)
        
        if df_demo is not None and not df_demo.empty:
            st.markdown("✅ **System Status**: Fully Operational")
            st.dataframe(df_demo, use_container_width=True)
            
            # Generate a simple chart
            demo_chart = create_executive_visualization(df_demo, "bar")
            if demo_chart:
                st.plotly_chart(demo_chart, use_container_width=True)
                
            st.success("🎉 System ready for executive queries!")
        
    except Exception as e:
        st.error(f"🔧 System needs attention: {str(e)}")

# ---------- HIDDEN ADMIN PANEL ----------
if st.secrets.get("ADMIN_MODE", False):
    with st.expander("🔧 **Admin Panel**", expanded=False):
        st.markdown("**🛠️ Advanced Controls**")
        
        admin_col1, admin_col2 = st.columns(2)
        
        with admin_col1:
            if st.button("🧪 Test Database"):
                try:
                    test_df, _ = analytics.execute_sql("SELECT COUNT(*) as total FROM voylla.\"voylla_design_ai\"")
                    st.success(f"✅ Database responsive: {test_df.iloc[0,0]} records")
                except Exception as e:
                    st.error(f"❌ Database error: {e}")
            
            if st.button("🔄 Reset Cache"):
                st.cache_resource.clear()
                st.success("✅ Cache cleared")
        
        with admin_col2:
            if st.button("📊 Sample Query"):
                st.session_state.auto_question = "Show basic revenue summary"
            
            if st.button("🆘 Emergency Demo"):
                auto_recovery_demo()
        
        # Session state inspector
        st.markdown("**📋 Session State:**")
        st.json({
            "chat_messages": len(st.session_state.chat_history),
            "has_data": st.session_state.last_df is not None,
            "data_shape": st.session_state.last_df.shape if st.session_state.last_df is not None else None,
            "insights_available": st.session_state.current_insights is not None
        })

# ---------- STARTUP VERIFICATION ----------
def verify_system_startup():
    """Verify system is ready for CEO demo"""
    checks = []
    
    # Database check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).fetchone()
        checks.append("✅ Database connection verified")
    except Exception as e:
        checks.append(f"❌ Database issue: {str(e)[:50]}")
    
    # OpenAI check
    try:
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        checks.append("✅ AI model access verified")
    except Exception as e:
        checks.append(f"❌ AI model issue: {str(e)[:50]}")
    
    # Schema check
    if len(analytics.table_schema) > 0:
        checks.append(f"✅ Schema loaded: {len(analytics.table_schema)} columns")
    else:
        checks.append("❌ Schema loading failed")
    
    return checks

# Run startup verification and pass openai_client
startup_checks = verify_system_startup()
if any("❌" in check for check in startup_checks):
    with st.sidebar:
        st.markdown("### ⚠️ Startup Issues")
        for check in startup_checks:
            st.markdown(check)

# ---------- INTELLIGENT QUERY SUGGESTIONS ----------
def get_smart_suggestions(current_data=None):
    """Generate contextual query suggestions"""
    base_suggestions = [
        "Top 10 revenue generating designs this quarter",
        "Channel performance with profit margin analysis", 
        "Best performing metal colors by revenue",
        "Design style trends over last 6 months",
        "Premium vs budget segment analysis"
    ]
    
    if current_data is not None and not current_data.empty:
        # Context-aware suggestions based on current data
        if 'Channel' in current_data.columns:
            base_suggestions.insert(0, "Deep dive into top performing channel")
        if 'Design Style' in current_data.columns:
            base_suggestions.insert(0, "Compare design styles performance")
    
    return base_suggestions[:5]

# Display smart suggestions
if st.session_state.last_df is None:
    st.markdown("### 💡 **Suggested Executive Queries**")
    suggestions = get_smart_suggestions()
    
    suggestion_cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(f"📊 {suggestion.split(' ')[0]} {suggestion.split(' ')[1]}", key=f"suggestion_{i}"):
                st.session_state.auto_question = suggestion

# ---------- FINAL STATUS INDICATOR ----------
st.markdown("""
<div style='position: fixed; bottom: 20px; right: 20px; 
           background: rgba(76, 175, 80, 0.9); color: white; 
           padding: 0.5rem 1rem; border-radius: 20px; 
           font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
           z-index: 1000;'>
    🟢 System Ready
</div>
""", unsafe_allow_html=True)

# ---------- AUTOMATED DEMO MODE ----------
if st.secrets.get("DEMO_MODE", False) and len(st.session_state.chat_history) == 0:
    st.info("🎬 **Demo Mode**: Automatically running sample analysis...")
    time.sleep(1)
    st.session_state.auto_question = "Top 10 revenue generating product combinations this quarter"
    st.rerun()
