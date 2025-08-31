#!/usr/bin/env python
# coding: utf-8

"""
Voylla DesignGPT — CEO Room Edition (Stable)
-------------------------------------------
... (docstring unchanged) ...
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 No OpenAI key found. Set OPENAI_API_KEY via .env or Streamlit Secrets.")
    st.stop()

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
# 🧱 Schema & Column Whitelist
# ---------------------------
DEFAULT_COLUMNS = [
    "EAN", "Date", "Channel", "Type", "Product Code", "Collection",
    "Discount", "Qty", "Amount", "MRP", "Cost Price", "Sale Order Item Status",
    "Category", "Sub-Category", "Look", "Design Style", "Form", "Metal Color",
    "Craft Style", "Central Stone", "Surrounding Layout", "Stone Setting", "Style Motif",
]

@st.cache_data(show_spinner=False)
def fetch_actual_columns(_engine: Engine) -> List[str]:
    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'voylla' AND table_name = 'voylla_design_ai'
        ORDER BY ordinal_position;
        """
    )
    with _engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows]

actual_cols = fetch_actual_columns(engine)
# Use intersection to avoid SQL errors
ALLOWED_COLS = [c for c in DEFAULT_COLUMNS if c in actual_cols] or actual_cols

# ... rest of code unchanged ...
