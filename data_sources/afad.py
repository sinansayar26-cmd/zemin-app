# -*- coding: utf-8 -*-
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Sequence
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError, ConnectionError as ReqConnError, ReadTimeout as ReqTimeout

# --- Resmi AFAD HTML (sende TLS kopuyor olabilir) ---
AFAD_LAST_URL = "https://deprem.afad.gov.tr/last-earthquakes.html"

# --- Resmi olmayan ayna (şu an 404 verebilir; yine de fallback olarak tutuyoruz) ---
AFAD_COMMUNITY_URL = "https://api.orhanaydogdu.com.tr/deprem/live.php?limit=200"

TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# --- USGS fallback için import ---
try:
    from data_sources.usgs import fetch_usgs_last_24h, fetch_usgs_last_24h_bbox
except Exception:
    from .usgs import fetch_usgs_last_24h, fetch_usgs_last_24h_bbox  # type: ignore

# Kaynak etiketi (dashboard başlığında göstermek için)
LAST_SOURCE = None  # "AFAD" | "AFAD-mirror" | "USGS-TR-fallback"
def get_afad_last_source():
    return LAST_SOURCE

class AFADFetchError(Exception):
    pass

def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=2, connect=2, read=2, backoff_factor=0.4,
        status_forcelist=(502, 503, 504), raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ---------------- Resmi AFAD HTML yolu ----------------
def _download_html() -> str:
    """
    AFAD sayfasını indirir.
    1) Normal TLS (verify=True)
    2) SSLError/EOF olursa verify=False ile bir deneme daha (uyarı basar)
    """
    s = _requests_session()
    try:
        r = s.get(AFAD_LAST_URL, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except SSLError as e:
        print("[AFAD] TLS uyarısı:", e, " -> verify=False ile tekrar deniyorum.")
        try:
            r = s.get(AFAD_LAST_URL, headers=HEADERS, timeout=TIMEOUT, verify=False)
            r.raise_for_status()
            return r.text
        except Exception as e2:
            raise AFADFetchError(f"AFAD TLS başarısız (verify=False denemesi dahil): {e2}") from e
    except (ReqConnError, ReqTimeout) as e:
        raise AFADFetchError(f"AFAD bağlantı/okuma hatası: {e}") from e
    except requests.RequestException as e:
        raise AFADFetchError(f"AFAD istek hatası: {e}") from e

def _read_afad_table_html() -> pd.DataFrame:
    """
    AFAD son depremler sayfasındaki tabloyu okur (HTML).
    Birden fazla parser dener: lxml -> bs4
    """
    html = _download_html()
    errors = []

    # 1) lxml
    try:
        tables = pd.read_html(html, flavor="lxml")
        if tables:
            df = tables[0].copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception as e:
        errors.append(f"lxml failed: {e}")

    # 2) bs4
    try:
        tables = pd.read_html(html, flavor="bs4")
        if tables:
            df = tables[0].copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception as e:
        errors.append(f"bs4 failed: {e}")

    raise AFADFetchError("AFAD tablosu okunamadı. " + " | ".join(errors))

# ---- HTML normalize yardımcıları ----
def _pick_mag_col(cols: List[str]) -> Optional[str]:
    lower = [str(c).strip().lower() for c in cols]
    order = ["ml", "mw", "md", "m", "mag", "büyüklük"]
    for key in order:
        for c, lc in zip(cols, lower):
            if lc == key or lc.startswith(key + " "):
                return c
    for c, lc in zip(cols, lower):
        if "mag" in lc or "büyükl" in lc:
            return c
    return None

def _pick_lat_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        lc = str(c).lower()
        if "enlem" in lc or "lat" in lc:
            return c
    return None

def _pick_lon_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        lc = str(c).lower()
        if "boylam" in lc or "lon" in lc:
            return c
    return None

def _pick_time_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["tarih", "time", "date", "zaman"]):
            return c
    return None

def _normalize_df_from_html(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    col_time = _pick_time_col(cols)
    col_mag  = _pick_mag_col(cols)
    col_lat  = _pick_lat_col(cols)
    col_lon  = _pick_lon_col(cols)

    if col_time is None or col_mag is None:
        raise AFADFetchError(f"Gerekli sütunlar bulunamadı. Sütunlar: {cols}")

    out = pd.DataFrame()
    t = pd.to_datetime(df[col_time], errors="coerce", dayfirst=True)
    try:
        if t.dt.tz is None:
            t = t.dt.tz_localize("Europe/Istanbul", nonexistent="shift_forward", ambiguous="NaT")
        t = t.dt.tz_convert("UTC")
    except Exception:
        t = pd.to_datetime(df[col_time], errors="coerce", utc=True)
    out["time_utc"] = t

    m = pd.to_numeric(df[col_mag].astype(str).str.replace(",", "."), errors="coerce")
    out["mag"] = m

    if col_lat is not None:
        out["lat"] = pd.to_numeric(df[col_lat].astype(str).str.replace(",", "."), errors="coerce")
    else:
        out["lat"] = np.nan
    if col_lon is not None:
        out["lon"] = pd.to_numeric(df[col_lon].astype(str).str.replace(",", "."), errors="coerce")
    else:
        out["lon"] = np.nan

    out = out.dropna(subset=["time_utc", "mag"])
    out = out.sort_values("time_utc")
    return out.reset_index(drop=True)

# ---------------- Resmi olmayan JSON ayna ----------------
def _read_afad_table_mirror() -> pd.DataFrame:
    """
    Topluluk aynasından JSON çekip DataFrame'e çevirir.
    """
    s = _requests_session()
    r = s.get(AFAD_COMMUNITY_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()  # şu an 404 fırlatabilir
    js = r.json()
    if isinstance(js, dict):
        for k in ["result", "data", "items", "earthquakes"]:
            if k in js and isinstance(js[k], list):
                js = js[k]
                break
    if not isinstance(js, list):
        raise AFADFetchError("Ayna JSON beklenen formatta değil.")

    rows = []
    for it in js:
        ts = it.get("date") or it.get("time") or it.get("timestamp") or it.get("datetime")
        mag = it.get("mag") or it.get("magnitude") or it.get("ml") or it.get("Mw") or it.get("MD")
        lat = it.get("lat") or it.get("latitude") or it.get("enlem")
        lon = it.get("lng") or it.get("lon") or it.get("longitude") or it.get("boylam")
        rows.append({"time_any": ts, "mag_any": mag, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    t = pd.to_datetime(df["time_any"], errors="coerce")
    try:
        if t.dt.tz is None:
            t = t.dt.tz_localize("Europe/Istanbul", nonexistent="shift_forward", ambiguous="NaT")
        t = t.dt.tz_convert("UTC")
    except Exception:
        t = pd.to_datetime(df["time_any"], errors="coerce", utc=True)
    df["time_utc"] = t
    df["mag"] = pd.to_numeric(pd.Series(df["mag_any"]).astype(str).str.replace(",", "."), errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    out = df.dropna(subset=["time_utc", "mag"]).copy().sort_values("time_utc")
    return out.reset_index(drop=True)

# ---------------- Ortak arayüz: 3 kademeli fallback ----------------
def fetch_afad_last_24h(min_mag: float = 0.0) -> Tuple[List[pd.Timestamp], List[float]]:
    """
    Sıra: 1) AFAD HTML -> 2) Ayna JSON -> 3) USGS Türkiye BBOX
    Dönüş: (times_utc, mags)
    """
    global LAST_SOURCE
    # 1) Resmi AFAD
    try:
        raw = _read_afad_table_html()
        df = _normalize_df_from_html(raw)
        LAST_SOURCE = "AFAD"
    except Exception as e:
        print("[AFAD] resmi yol başarısız:", e, " -> ayna JSON deneniyor.")
        # 2) Ayna
        try:
            df = _read_afad_table_mirror()
            LAST_SOURCE = "AFAD-mirror"
        except Exception as e2:
            print("[AFAD] ayna yol da başarısız:", e2, " -> USGS-TR fallback.")
            # 3) USGS Türkiye BBOX fallback (26E-45E, 36N-42.5N)
            usgs_t, usgs_m, *_ = fetch_usgs_last_24h_bbox(bbox=[26.0, 36.0, 45.0, 42.5], min_mag=min_mag)
            LAST_SOURCE = "USGS-TR-fallback"
            print(f"[AFAD] USGS-TR fallback sayısı (24h >= {min_mag}):", len(usgs_m))
            return usgs_t, usgs_m

    now = pd.Timestamp.now(tz="UTC")
    df = df[df["time_utc"] >= (now - pd.Timedelta(hours=24))]
    df = df[df["mag"] >= float(min_mag)]
    print(f"[AFAD] kaynak={LAST_SOURCE}  |  24h event count (>= {min_mag}): {len(df)}")
    if df.empty:
        return [], []
    return df["time_utc"].to_list(), df["mag"].astype(float).to_list()

def fetch_afad_last_24h_bbox(
    bbox: Sequence[float],  # [minLon, minLat, maxLon, maxLat]
    min_mag: float = 0.0
) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float]]:
    """
    Sıra: 1) AFAD HTML -> 2) Ayna JSON -> 3) USGS BBOX fallback
    Dönüş: (times_utc, mags, lats, lons)
    """
    global LAST_SOURCE
    # 1) Resmi AFAD
    try:
        raw = _read_afad_table_html()
        df = _normalize_df_from_html(raw)
        LAST_SOURCE = "AFAD"
    except Exception as e:
        print("[AFAD] resmi yol başarısız:", e, " -> ayna JSON deneniyor.")
        # 2) Ayna
        try:
            df = _read_afad_table_mirror()
            LAST_SOURCE = "AFAD-mirror"
        except Exception as e2:
            print("[AFAD] ayna yol da başarısız:", e2, " -> USGS fallback.")
            usgs_t, usgs_m, usgs_lat, usgs_lon = fetch_usgs_last_24h_bbox(bbox=bbox, min_mag=min_mag)
            LAST_SOURCE = "USGS-TR-fallback"
            return usgs_t, usgs_m, usgs_lat, usgs_lon

    now = pd.Timestamp.now(tz="UTC")
    df = df[df["time_utc"] >= (now - pd.Timedelta(hours=24))]
    df = df[df["mag"] >= float(min_mag)]

    # bbox uygula (lat/lon varsa)
    if {"lat", "lon"}.issubset(df.columns):
        min_lon, min_lat, max_lon, max_lat = map(float, bbox)
        df = df[
            (df["lat"].between(min_lat, max_lat, inclusive="both")) &
            (df["lon"].between(min_lon, max_lon, inclusive="both"))
        ]

    print(f"[AFAD] kaynak={LAST_SOURCE}  |  24h bbox event count (>= {min_mag}): {len(df)}")
    if df.empty:
        return [], [], [], []
    return (df["time_utc"].to_list(),
            df["mag"].astype(float).to_list(),
            df["lat"].astype(float).to_list() if "lat" in df else [np.nan]*len(df),
            df["lon"].astype(float).to_list() if "lon" in df else [np.nan]*len(df))
