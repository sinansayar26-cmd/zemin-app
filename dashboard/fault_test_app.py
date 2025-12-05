import streamlit as st
st.set_page_config(page_title="ZEMİN - Deprem Erken Uyarı", layout="wide")

import os
import glob
import math
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ------------------- Plotly kontrolü (hata vermesin diye) -------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    st.error("Plotly yüklenemedi – requirements.txt kontrol ediliyor...")
    PLOTLY_OK = False

if not PLOTLY_OK:
    st.stop()

# ------------------- Sayfa Başlığı -------------------
st.title("ZEMİN - Deprem Erken Uyarı Sistemi (v0.3.2)")

# ------------------- Sidebar Kontroller -------------------
st.sidebar.header("Parametreler")
st
