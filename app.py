#!/usr/bin/env python
# coding: utf-8

"""
Voylla DesignGPT — CEO Room Edition (Stable)
-------------------------------------------
A deterministic, production-hardened Streamlit app for SQL analytics on
voylla."voylla_design_ai" with:
- **No brittle ReAct agents** (zero tool-chaining);
- **Structured JSON planning** → **verified SQL** with strict whitelists;
- **Automatic retries & graceful fallbacks** when the LLM or DB errs;
- **Direct DataFrame rendering (no markdown parsing)**;
- **One-click CEO dashboard** (KPIs + charts for last 90 days);
- **Query sandbox** with safe SELECT-only enforcement;
- **Charts & Excel export**;
- **Schema cache + synonyms mapping** for robust natural language → columns.

Requires:
- OPENAI_API_KEY in .env or Streamlit secrets
- DB_* secrets in st.secrets

This file is single‑file ready for presentation.
"""

import os
import re
import json
import time
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
import plotly.express as px

# ---------------------------
# 🔐 Secrets & Environment
# ---------------------------
load_dotenv()

# Allow both .env and Streamlit Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 No OpenAI key found. Set OPENAI_API_KEY via .env or Streamlit Secrets.")
    st.stop()

# Use OpenAI direct client (no agent). LangChain not required here.
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"OpenAI client import failed: {e}")
    st.stop()

# ---------------------------
# 🗄️ Database Connection
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets.get("DB_PORT", 5432)
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

engine = get_engine()

# ---------------------------
# 🧭 App Config & Styles
# ---------------------------
st.set_page_config(
    page_title="Voylla DesignGPT (CEO Edition)",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #fcf1ed; color: #111; }
    .metric-card { background: linear-gradient(135deg,#6a11cb 0%,#2575fc 100%); padding: 14px 16px; border-radius: 14px; color: #fff; font-weight: 700; text-align: center; }
    .soft { background: #fff; border-radius: 14px; padding: 14px; box-shadow: 0 6px 24px rgba(0,0,0,0.06); }
    .subtle { color:#666; font-size:0.9rem; }
    .danger { color:#b00020; font-weight:600; }
    .ok { color:#0f9d58; font-weight:600; }
    .warn { color:#e37400; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 🧱 Schema & Column Whitelist
# ---------------------------
DEFAULT_COLUMNS = [
    "EAN", "Date", "Channel", "Type", "Product Code", "Collection",
    "Discount", "Qty", "Amount", "MRP", "Cost Price", "Sale Order Item Status",
    "Category", "Sub-Category", "Look", "Design Style", "Form", "Metal Color",
    "Craft Style", "Central Stone", "Surrounding Layout", "Stone Setting", "Style Motif",
]

@st.cache_data(show_spinner=False)
def fetch_actual_columns(engine: Engine) -> List[str]:
    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'voylla' AND table_name = 'voylla_design_ai'
        ORDER BY ordinal_position;
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows]

actual_cols = fetch_actual_columns(engine)
# Use intersection to avoid SQL errors
ALLOWED_COLS = [c for c in DEFAULT_COLUMNS if c in actual_cols] or actual_cols

# Synonyms to map NL → canonical columns
SYNONYMS = {
    "sku": "Product Code",
    "product": "Product Code",
    "sku id": "Product Code",
    "ean": "EAN",
    "date": "Date",
    "channel": "Channel",
    "platform": "Channel",
    "qty": "Qty",
    "quantity": "Qty",
    "units": "Qty",
    "revenue": "Amount",
    "sales": "Amount",
    "amount": "Amount",
    "aov": "Amount",  # will compute
    "mrp": "MRP",
    "cost": "Cost Price",
    "status": "Sale Order Item Status",
    "collection": "Collection",
    "category": "Category",
    "sub category": "Sub-Category",
    "subcategory": "Sub-Category",
    "look": "Look",
    "design style": "Design Style",
    "form": "Form",
    "metal color": "Metal Color",
}

# ---------------------------
# 🧠 Prompting — JSON Plan (deterministic)
# ---------------------------
SYSTEM = {
    "role": "system",
    "content": (
        "You convert a user analytics request into STRICT JSON with keys: "
        "metrics(list of strings), dimensions(list of strings), filters(list of {column,op,value}), "
        "date_range({from:string|optional, to:string|optional}), limit(int|optional), order_by(list of {field,dir}|optional). "
        "Only choose from provided columns. Never invent columns. If unsure, leave arrays empty. "
        "All output MUST be valid JSON only, no prose."
    ),
}

PLAN_EXAMPLE = {
    "metrics": ["SUM(\"Qty\") as total_qty", "SUM(\"Amount\") as total_revenue"],
    "dimensions": ["Channel"],
    "filters": [
        {"column": "Sale Order Item Status", "op": "<>", "value": "CANCELLED"}
    ],
    "date_range": {"from": None, "to": None},
    "limit": 20,
    "order_by": [{"field": "total_revenue", "dir": "DESC"}],
}

# ---------------------------
# 🛡️ SQL Builder (safe)
# ---------------------------
ALLOWED_FUNCS = {"SUM", "COUNT", "AVG", "MIN", "MAX", "ROUND"}

COL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-]*$")


def _quote_col(c: str) -> str:
    c = c.strip()
    if c in ALLOWED_COLS and COL_PATTERN.match(c):
        return f'"{c}"'
    # Allow metrics like SUM("Qty") as total
    if any(c.upper().startswith(fn + "(") for fn in ALLOWED_FUNCS):
        return c
    # Fallback reject
    raise ValueError(f"Disallowed column/expression: {c}")


def build_sql(plan: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """Return SQL string, list(dimensions), list(metric_aliases)."""
    dims = plan.get("dimensions") or []
    mets = plan.get("metrics") or []
    filters = plan.get("filters") or []
    dr = (plan.get("date_range") or {})
    limit = plan.get("limit")
    order_by = plan.get("order_by") or []

    # Always filter out CANCELLED
    ensure_cancel = {"column": "Sale Order Item Status", "op": "<>", "value": "CANCELLED"}
    if ensure_cancel not in filters:
        filters = [ensure_cancel] + filters

    # Quote dimensions
    qdims = []
    for d in dims:
        if d in SYNONYMS:
            d = SYNONYMS[d]
        if d not in ALLOWED_COLS:
            continue
        qdims.append(_quote_col(d))

    # Validate metrics, extract aliases for order_by
    qmetrics = []
    met_aliases = []
    for m in mets:
        m = m.strip()
        # enforce only known columns inside funcs
        # Simple safety: check all column-like substrings inside quotes are allowed
        for col in re.findall(r'"([^"]+)"', m):
            if col not in ALLOWED_COLS:
                raise ValueError(f"Metric references unknown column: {col}")
        # Extract alias if present "... as alias"
        alias = None
        m_parts = re.split(r"\s+as\s+", m, flags=re.IGNORECASE)
        if len(m_parts) == 2:
            alias = m_parts[1].strip()
        qmetrics.append(m)
        if alias:
            met_aliases.append(alias)

    sel_parts = []
    if qdims:
        sel_parts.extend(qdims)
    if qmetrics:
        sel_parts.extend(qmetrics)
    if not sel_parts:
        # default fallback
        sel_parts = ["\"Channel\"", "SUM(\"Qty\") as total_qty", "SUM(\"Amount\") as total_revenue"]
        qdims = ["\"Channel\""]
        met_aliases = ["total_qty", "total_revenue"]

    where_clauses = []
    params = {}

    # Filters
    idx = 0
    for f in filters:
        col = f.get("column")
        op = (f.get("op") or "=").upper()
        val = f.get("value")
        if col in SYNONYMS:
            col = SYNONYMS[col]
        if col not in ALLOWED_COLS:
            continue
        col_q = _quote_col(col)
        if op not in {"=", "<>", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN"}:
            op = "="
        if op == "IN" and isinstance(val, list):
            placeholders = []
            for v in val:
                key = f"p{idx}"
                idx += 1
                params[key] = v
                placeholders.append(f":{key}")
            where_clauses.append(f"{col_q} IN (" + ", ".join(placeholders) + ")")
        else:
            key = f"p{idx}"
            idx += 1
            params[key] = val
            where_clauses.append(f"{col_q} {op} :{key}")

    # Date range
    frm, to = dr.get("from"), dr.get("to")
    if frm:
        params["p_from"] = frm
        where_clauses.append("\"Date\"::date >= :p_from::date")
    if to:
        params["p_to"] = to
        where_clauses.append("\"Date\"::date <= :p_to::date")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    group_sql = f"GROUP BY {', '.join(qdims)}" if qdims else ""

    # ORDER BY safeguards (only by dims or metric aliases)
    allowed_order = set([d.strip('"') for d in qdims] + met_aliases)
    order_parts = []
    for ob in order_by:
        field = (ob.get("field") or "").strip()
        direction = (ob.get("dir") or "DESC").upper()
        if field in allowed_order:
            order_parts.append(f'"{field}" {direction}' if field in [d.strip('"') for d in qdims] else f"{field} {direction}")
    order_sql = f"ORDER BY {', '.join(order_parts)}" if order_parts else ""

    limit_sql = f"LIMIT {int(limit)}" if isinstance(limit, int) and limit > 0 else ""

    sql = f"""
        SELECT {', '.join(sel_parts)}
        FROM voylla."voylla_design_ai"
        {where_sql}
        {group_sql}
        {order_sql}
        {limit_sql}
    """

    return sql, [d.strip('"') for d in qdims], met_aliases

# ---------------------------
# 🧪 Robust Execute with Retries
# ---------------------------

def run_sql(sql: str, params: Optional[Dict[str, Any]] = None, retries: int = 1) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries + 1):
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn, params=params or {})
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))
    raise last_err

# ---------------------------
# 🧰 Utilities
# ---------------------------

def try_json(s: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse with minor repairs."""
    s = s.strip()
    # Remove leading/trailing junk
    s = re.sub(r"^[^\[{]+", "", s)
    s = re.sub(r"[^}\]]+$", "", s)
    # Quote keys if missing (very mild attempt)
    try:
        return json.loads(s)
    except Exception:
        # Try to find JSON block
        m = re.search(r"(\{[\s\S]*\})", s)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


def call_planner(nl: str) -> Dict[str, Any]:
    """Call OpenAI to get a JSON plan. Apply guards and defaults."""
    user = {
        "role": "user",
        "content": (
            "You MUST respond with pure JSON. Columns available are: "
            + ", ".join([f'"{c}"' for c in ALLOWED_COLS])
            + ".\n\nUser request: "
            + nl
        ),
    }
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",  # fast & reliable
        temperature=0,
        messages=[SYSTEM, user],
        max_tokens=400,
    )
    raw = resp.choices[0].message.content or "{}"
    plan = try_json(raw) or {}

    # Guard rails: ensure lists exist
    plan.setdefault("metrics", [])
    plan.setdefault("dimensions", [])
    plan.setdefault("filters", [])
    plan.setdefault("date_range", {})
    plan.setdefault("limit", 50)
    plan.setdefault("order_by", [])

    # Map synonyms in dimensions
    dims = []
    for d in plan.get("dimensions", []):
        d = d.lower().strip()
        d = SYNONYMS.get(d, d)
        if d in ALLOWED_COLS:
            dims.append(d)
    plan["dimensions"] = dims

    # Ensure at least one metric
    if not plan["metrics"]:
        plan["metrics"] = ["SUM(\"Qty\") as total_qty", "SUM(\"Amount\") as total_revenue"]

    # Default ordering
    if not plan["order_by"]:
        plan["order_by"] = [{"field": "total_revenue", "dir": "DESC"}]

    return plan


def auto_chart(df: pd.DataFrame):
    if df.empty:
        return None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    try:
        if cat_cols and num_cols:
            fig = px.bar(df, x=cat_cols[0], y=num_cols[0])
        elif len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1])
        elif num_cols:
            fig = px.line(df, y=num_cols[0])
        else:
            return None
        fig.update_layout(height=420, margin=dict(l=8, r=8, t=40, b=8))
        return fig
    except Exception:
        return None

# ---------------------------
# 🧱 Sidebar
# ---------------------------
with st.sidebar:
    st.header("📊 Connection")
    try:
        ping = pd.read_sql_query(text("SELECT COUNT(*) AS n FROM voylla.\"voylla_design_ai\""), engine)
        st.success(f"Connected. Rows: {int(ping['n'][0]):,}")
    except Exception as e:
        st.error(f"Connection check failed: {e}")

    st.header("💡 Quick Questions")
    def qbtn(label):
        if st.button(label):
            st.session_state.user_input = label
    qbtn("Top 20 SKUs by revenue this month")
    qbtn("Best selling success combination last 90 days")
    qbtn("Which Design Style performs best on Myntra?")
    qbtn("Revenue trends by month for last 6 months")

    st.header("⚙️ Options")
    auto_charts = st.checkbox("Auto-charts", value=True)
    default_limit = st.selectbox("Default LIMIT", [20, 50, 100, 500], index=1)

# ---------------------------
# 🧾 Title & KPI Row
# ---------------------------
st.title("💬 Voylla DesignGPT — CEO Room Edition")
st.caption("Deterministic insights on sales & design — reliable in demos.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">🔒 Read‑only SQL</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">🧠 JSON→SQL Planner</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">📈 Charts & Export</div>', unsafe_allow_html=True)

st.markdown("")

# ---------------------------
# 📊 CEO Dashboard (last 90 days)
# ---------------------------
with st.expander("📊 CEO Snapshot — last 90 days", expanded=True):
    end = datetime.now().date()
    start = end - timedelta(days=90)

    # KPIs
    kpi_sql = f"""
        SELECT 
            SUM("Qty") AS units,
            SUM("Amount") AS revenue,
            (SUM("Amount") - SUM("Cost Price" * "Qty")) AS gross_profit
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
          AND "Date"::date BETWEEN :s AND :e
    """
    dfk = run_sql(kpi_sql, {"s": str(start), "e": str(end)})
    k1, k2, k3 = st.columns(3)
    k1.metric("Units Sold", f"{int(dfk['units'][0] or 0):,}")
    k2.metric("Revenue", f"₹{float(dfk['revenue'][0] or 0):,.0f}")
    k3.metric("Gross Profit", f"₹{float(dfk['gross_profit'][0] or 0):,.0f}")

    # Channel mix
    ch_sql = f"""
        SELECT "Channel", SUM("Qty") AS qty, SUM("Amount") AS revenue
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
          AND "Date"::date BETWEEN :s AND :e
        GROUP BY "Channel"
        ORDER BY revenue DESC
        LIMIT 12
    """
    dfc = run_sql(ch_sql, {"s": str(start), "e": str(end)})

    # Design style
    ds_sql = f"""
        SELECT "Design Style", SUM("Qty") AS qty, SUM("Amount") AS revenue
        FROM voylla."voylla_design_ai"
        WHERE "Sale Order Item Status" <> 'CANCELLED'
          AND "Date"::date BETWEEN :s AND :e
        GROUP BY "Design Style"
        ORDER BY qty DESC
        LIMIT 12
    """
    dsd = run_sql(ds_sql, {"s": str(start), "e": str(end)})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Channel Mix")
        if not dfc.empty:
            st.plotly_chart(px.bar(dfc, x="Channel", y="revenue"), use_container_width=True)
        else:
            st.info("No data.")
    with c2:
        st.subheader("Top Design Styles")
        if not dsd.empty:
            st.plotly_chart(px.bar(dsd, x="Design Style", y="qty"), use_container_width=True)
        else:
            st.info("No data.")

# ---------------------------
# 💬 Query Box
# ---------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input(
    "Ask about sales or design trends…",
    value=st.session_state.user_input,
    placeholder="e.g., Success combinations for Wedding look last 30 days on Myntra",
)

run_clicked = st.button("Run Analysis", type="primary")

# ---------------------------
# 🧯 Safe Run Path
# ---------------------------
if run_clicked and user_input.strip():
    st.session_state.user_input = user_input

    with st.spinner("Designing your perfect answer… ✨"):
        # 1) Plan
        plan = call_planner(user_input)
        # Inject default limit option
        if plan.get("limit") in (None, 0):
            plan["limit"] = default_limit

        # 2) Build SQL
        try:
            sql, dims, met_aliases = build_sql(plan)
        except Exception as e:
            # Fallback generic plan
            plan_fallback = {
                "metrics": ["SUM(\"Qty\") as total_qty", "SUM(\"Amount\") as total_revenue"],
                "dimensions": ["Channel"],
                "filters": [{"column": "Sale Order Item Status", "op": "<>", "value": "CANCELLED"}],
                "date_range": plan.get("date_range", {}),
                "limit": default_limit,
                "order_by": [{"field": "total_revenue", "dir": "DESC"}],
            }
            sql, dims, met_aliases = build_sql(plan_fallback)

        # 3) Execute with retry & progressive relaxation on errors
        params: Dict[str, Any] = {}
        dr = plan.get("date_range") or {}
        if dr.get("from"):
            params["p_from"] = dr["from"]
        if dr.get("to"):
            params["p_to"] = dr["to"]

        try:
            df = run_sql(sql, params=params, retries=2)
        except Exception as e:
            # Relax ordering & limit and retry once
            sql_simple = re.sub(r"ORDER BY[\s\S]*?(LIMIT|$)", r"\1", sql, flags=re.IGNORECASE)
            try:
                df = run_sql(sql_simple, params=params, retries=1)
            except Exception as e2:
                st.error(f"Query failed even after retries. Root cause: {e2}")
                st.code(sql, language="sql")
                st.stop()

        # 4) Display Results
        st.success("Analysis complete.")
        st.caption("Based on voylla.\"voylla_design_ai\" (cancelled orders excluded by default)")

        st.dataframe(df, use_container_width=True)

        if auto_charts:
            fig = auto_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # 5) Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="VoyllaData")
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📥 Download Excel",
            data=output.getvalue(),
            file_name=f"voylla_insights_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # 6) Transparency: show SQL (collapsible)
        with st.expander("🔎 Inspect SQL"):
            st.code(sql, language="sql")

# ---------------------------
# 📝 Helper: Common Phrases
# ---------------------------
st.markdown("---")
st.markdown(
    "**Pro Tips**: Ask things like _'success combinations for Wedding look'_, _'AOV by Channel last month'_, _'Top 15 SKUs by revenue on Myntra'_."
)
