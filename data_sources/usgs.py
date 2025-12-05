# -*- coding: utf-8 -*-
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple, Sequence

USGS_DAY_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _download_usgs_geojson() -> dict:
    r = requests.get(USGS_DAY_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _parse_features(js: dict) -> pd.DataFrame:
    feats = js.get("features", []) or []
    rows = []
    for f in feats:
        prop = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", [np.nan, np.nan, np.nan])
        lon, lat = (coords[0], coords[1]) if len(coords) >= 2 else (np.nan, np.nan)
        mag = prop.get("mag", None)
        t_ms = prop.get("time", None)
        rows.append({"time_ms": t_ms, "mag": mag, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Zamanı UTC'ye çevir
    df["time_utc"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True, errors="coerce")
    # Magnitüd sayısal
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    # Sırala ve temizle
    df = df.dropna(subset=["time_utc", "mag"]).sort_values("time_utc").reset_index(drop=True)
    return df

def _apply_common_filters(df: pd.DataFrame, min_mag: float) -> pd.DataFrame:
    df = df[df["mag"] >= float(min_mag)]
    # Her sürümde güvenli: doğrudan tz-aware saat al
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["time_utc"] >= (now - pd.Timedelta(hours=24))]
    return df

def fetch_usgs_last_24h(min_mag: float = 0.0) -> Tuple[List[pd.Timestamp], List[float]]:
    """
    USGS 'all day' feed'den son 24 saati (zaten kapsıyor) çeker; min_mag filtresi uygular.
    Dönüş: (times_utc, mags)
    """
    try:
        js = _download_usgs_geojson()
        df = _parse_features(js)
        if df.empty:
            return [], []
        df = _apply_common_filters(df, min_mag)
        if df.empty:
            return [], []
        return df["time_utc"].to_list(), df["mag"].astype(float).to_list()
    except Exception as e:
        print("[USGS] hata:", e)
        return [], []

def fetch_usgs_last_24h_bbox(
    bbox: Sequence[float],  # [minLon, minLat, maxLon, maxLat]
    min_mag: float = 0.0
) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float]]:
    """
    USGS 'all day' feed'den son 24 saati çeker, min_mag ve bbox uygular.
    Dönüş: (times_utc, mags, lats, lons)
    """
    try:
        js = _download_usgs_geojson()
        df = _parse_features(js)
        if df.empty:
            return [], [], [], []
        df = _apply_common_filters(df, min_mag)
        if df.empty:
            return [], [], [], []

        min_lon, min_lat, max_lon, max_lat = map(float, bbox)
        df = df[
            (df["lat"].between(min_lat, max_lat, inclusive="both")) &
            (df["lon"].between(min_lon, max_lon, inclusive="both"))
        ]
        if df.empty:
            return [], [], [], []

        return (df["time_utc"].to_list(),
                df["mag"].astype(float).to_list(),
                df["lat"].astype(float).to_list(),
                df["lon"].astype(float).to_list())
    except Exception as e:
        print("[USGS] bbox hata:", e)
        return [], [], [], []
