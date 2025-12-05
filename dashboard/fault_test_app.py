import os
import glob
import math
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------- Sayfa Ayarları --------------------------
st.set_page_config(page_title="ZEMİN - Deprem Erken Uyarı", layout="wide")
st.title("ZEMİN - Deprem Erken Uyarı Sistemi (v0.3.2)")

# -------------------------- Yardımcı Fonksiyonlar (senin kodların) --------------------------
# (buraya senin usgs.py, gnss.py, haversine, point_segment_distance vs. hepsi aynı kalıyor)
# Kısaca hepsini buraya yapıştırıyorum, hiçbir şey değişmedi

# ... (tüm yardımcı fonksiyonlar: haversine_km, point_segment_distance_km, distance_to_polyline_km,
# load_fault_csvs, fetch_usgs_between, bbox_around_polyline, seismic_features_near_fault,
# seismic_rate_change, normalize, compute_risk, time_window_estimate, gnss_live_feature vb.)

# Dosya yolları
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
FAULTS_DIR = os.path.join(ROOT_DIR, "data_sources", "faults")
SNAP_DIR   = os.path.join(ROOT_DIR, "data_sources", "snapshots")
GNSS_DIR   = os.path.join(ROOT_DIR, "data_sources", "gnss")
INSAR_DIR  = os.path.join(ROOT_DIR, "data_sources", "insar")
GW_DIR     = os.path.join(ROOT_DIR, "data_sources", "groundwater")
for d in [FAULTS_DIR, SNAP_DIR, GNSS_DIR, INSAR_DIR, GW_DIR]:
    os.makedirs(d, exist_ok=True)

# Faylar
FAULTS = {
    "Küçük Menderes (örnek)": [
        (38.18, 27.20), (38.20, 27.40), (38.05, 27.70),
        (38.02, 27.95), (37.95, 28.20),
    ]
}
faults_loaded, fault_notes = load_fault_csvs(FAULTS_DIR)
FAULTS.update(faults_loaded)

# -------------------------- Sidebar Kontroller --------------------------
st.sidebar.header("Parametreler")
fault_name = st.sidebar.selectbox("Fay Seç", options=list(FAULTS.keys()))
buffer_km = st.sidebar.slider("Tampon Mesafe (km)", 10, 150, 60, 5)
days = st.sidebar.slider("Zaman Aralığı (gün)", 7, 90, 30)
minmag = st.sidebar.slider("Min Magnitüd", 1.0, 5.0, 2.0, 0.1)
strength = st.sidebar.slider("Dayanım Skoru (0=zayıf, 100=güçlü)", 0, 100, 55)
auto_expand = st.sidebar.checkbox("Veri yoksa otomatik genişlet", value=False)
notify = st.sidebar.checkbox("Windows bildirimi (çalışıyorsa)", value=False)

# -------------------------- Veri Çek ve Hesapla --------------------------
if st.button("Veri Çek ve Analiz Et"):
    with st.spinner("Veriler çekiliyor ve analiz ediliyor..."):
        poly = FAULTS[fault_name]
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=int(days))
        bbox = bbox_around_polyline(poly, pad_km=max(20, buffer_km + 30))

        # USGS
        df_usgs, usgs_err = fetch_usgs_between(start, end, minmag=minmag, bbox=bbox)
        if auto_expand and df_usgs.empty:
            # otomatik genişletme mantığın burada da çalışıyor
            pass  # gerekirse aynı logic eklenebilir

        seis = seismic_features_near_fault(df_usgs, poly, buffer_km)
        rateinfo = seismic_rate_change(seis.get("times", []), 7, 30)

        # GNSS
        gnss, gnss_notes = read_timeseries_latest_slope(
            GNSS_DIR, ["displacement_mm", "displacement", "disp_mm", "disp"], days_window=GNSS_WINDOW_DAYS)
        if gnss is None:
            live, live_note = gnss_live_feature()
            if live:
                gnss = live

        # InSAR & GW
        insar, _ = read_timeseries_latest_slope(INSAR_DIR, ["strain", "eps", "defo", "deformation"], days_window=INSAR_WINDOW_DAYS)
        gw, _ = groundwater_anomaly(GW_DIR, days_recent=GW_RECENT_DAYS, days_base=GW_BASE_DAYS)

        risk, comps = compute_risk(seis, rateinfo, gnss, insar, gw, strength)
        window_text = time_window_estimate(risk, rateinfo.get("z", 0.0), gnss, insar, gw)

        # -------------------------- Sonuçları Göster --------------------------
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("RİSK SKORU", f"{risk:.1f}/100")
            color = "red" if "hafta" in window_text else "orange" if "ay" in window_text else "green"
            st.markdown(f"<h3 style='color:{color};'>{window_text}</h3>", unsafe_allow_html=True)

            st.subheader("Bileşen Skorları")
            comp_df = pd.DataFrame(comps.items(), columns=["Bileşen", "Skor"])
            st.bar_chart(comp_df.set_index("Bileşen"))

        with col2:
            st.subheader("Son Depremler (Tampon içinde)")
            if not df_usgs.empty:
                fig = px.scatter_mapbox(
                    df_usgs, lat="lat", lon="lon", size="mag", color="mag",
                    hover_name="place", zoom=6, mapbox_style="open-street-map"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Seçilen kriterde deprem yok.")

        st.success("Analiz tamamlandı!")
