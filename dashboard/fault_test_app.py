import streamlit as st
st.set_page_config(page_title="ZEMİN - Deprem Erken Uyarı", layout="wide")

import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.title("ZEMİN - Deprem Erken Uyarı Sistemi (v0.3.2)")

st.sidebar.header("Parametreler")
st.sidebar.write("Türkiye geneli analiz")

if st.sidebar.button("Veri Çek ve Risk Hesapla"):
    with st.spinner("USGS ve GNSS verileri çekiliyor..."):
        try:
            from data_sources.usgs import get_usgs_dataframe
            df = get_usgs_dataframe(minmag=2.0)
            st.success(f"USGS: {len(df)} deprem alındı!")
        except Exception as e:
            st.error("USGS verisi alınamadı.")
            df = pd.DataFrame()

        try:
            from data_sources.gnss import get_gnss_dataframe
            gnss_df = get_gnss_dataframe(period="30d")
            st.success("GNSS verisi alındı!")
        except:
            st.warning("GNSS verisi alınamadı.")
            gnss_df = pd.DataFrame()

        risk = 45
        if not df.empty:
            risk += len(df.tail(10)) * 3
        if not gnss_df.empty:
            risk += 20

        risk = min(risk, 100)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("RİSK SKORU", f"{risk}/100")
            if risk > 70:
                st.error("YÜKSEK RİSK")
            elif risk > 50:
                st.warning("ORTA RİSK")
            else:
                st.success("DÜŞÜK RİSK")

        with col2:
            if not df.empty:
                fig = px.scatter_mapbox(df, lat="lat", lon="lon", size="mag", color="mag",
                                       hover_name="place", zoom=5, height=500,
                                       mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Deprem verisi yok.")

st.success("ZEMİN v0.3.2 çalışıyor!")
st.balloons()
