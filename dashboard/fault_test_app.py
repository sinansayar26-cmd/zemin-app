# C:\Users\sinan\OneDrive\Masaüstü\deprem_risk_izleme\dashboard\fault_test_app.py
import os, glob, math, time, json
import requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

import streamlit as st
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

# ---- Opsiyonel Windows bildirimi ----
try:
    from win10toast import ToastNotifier
    TOASTER = ToastNotifier()
except Exception:
    TOASTER = None

# ---- GNSS canlı kaynak (data_sources/gnss.py) ----
try:
    # Kullanıcı projedeki dosya: deprem_risk_izleme/data_sources/gnss.py
    from data_sources.gnss import get_gnss_dataframe
except Exception:
    get_gnss_dataframe = None

# =========================
# Konfig (kolay değişen)
# =========================
GNSS_WINDOW_DAYS   = 60     # GNSS eğim penceresi
INSAR_WINDOW_DAYS  = 60     # InSAR eğim penceresi
GW_RECENT_DAYS     = 21     # Yeraltı suyu son pencere
GW_BASE_DAYS       = 90     # Yeraltı suyu referans pencere

AUTO_EXPAND_LIMITS = {      # otomatik genişletme üst sınırları
    "minmag": 1.0,
    "days": 90,
    "buffer_km": 120
}

# =========================
# Dizinler
# =========================
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
FAULTS_DIR = os.path.join(ROOT_DIR, "data_sources", "faults")
SNAP_DIR   = os.path.join(ROOT_DIR, "data_sources", "snapshots")
GNSS_DIR   = os.path.join(ROOT_DIR, "data_sources", "gnss")
INSAR_DIR  = os.path.join(ROOT_DIR, "data_sources", "insar")
GW_DIR     = os.path.join(ROOT_DIR, "data_sources", "groundwater")
for d in [FAULTS_DIR, SNAP_DIR, GNSS_DIR, INSAR_DIR, GW_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# Coğrafi yardımcılar
# =========================
EARTH_R = 6371.0088  # km

def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2 * EARTH_R * math.asin(min(1, math.sqrt(a)))

def point_segment_distance_km(px, py, x1, y1, x2, y2):
    mx = math.cos(math.radians(py))
    x1p, x2p, pxp = x1*mx, x2*mx, px*mx
    y1p, y2p, pyp = y1, y2, py
    vx, vy = (x2p - x1p), (y2p - y1p)
    wx, wy = (pxp - x1p), (pyp - y1p)
    denom = vx*vx + vy*vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
    projx, projy = x1 + (x2 - x1)*t, y1 + (y2 - y1)*t
    return haversine_km(py, px, projy, projx)

def distance_to_polyline_km(lat, lon, polyline):
    mind = float("inf")
    for i in range(len(polyline)-1):
        a = polyline[i]; b = polyline[i+1]
        d = point_segment_distance_km(lon, lat, a[1], a[0], b[1], b[0])
        if d < mind: mind = d
    return mind

# =========================
# Fay tanımı + CSV'den yükleme
# =========================
FAULTS = {
    "Küçük Menderes (örnek)": [
        (38.18, 27.20),(38.20, 27.40),(38.05, 27.70),
        (38.02, 27.95),(37.95, 28.20),
    ]
}

def load_fault_csvs(folder_path):
    faults, notes = {}, []
    for f in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            if not {"lat","lon"}.issubset(df.columns):
                notes.append(f"[FAULT] {os.path.basename(f)}: 'lat,lon' yok sayıldı")
                continue
            poly = [(float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()]
            if len(poly) >= 2:
                name = os.path.splitext(os.path.basename(f))[0]
                faults[f"CSV: {name}"] = poly
                notes.append(f"[FAULT] {os.path.basename(f)}: {len(poly)} nokta yüklendi")
            else:
                notes.append(f"[FAULT] {os.path.basename(f)}: 2'den az nokta")
        except Exception as e:
            notes.append(f"[FAULT] {os.path.basename(f)} hata: {e}")
    return faults, notes

faults_loaded, fault_notes = load_fault_csvs(FAULTS_DIR)
FAULTS.update(faults_loaded)

# =========================
# USGS deprem verisi
# =========================
def fetch_usgs_between(start_utc: datetime, end_utc: datetime, minmag=2.0, bbox=None):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": str(minmag),
        "orderby": "time-asc",
        "limit": "20000",
    }
    if bbox:
        params.update({
            "minlatitude": bbox[0], "minlongitude": bbox[1],
            "maxlatitude": bbox[2], "maxlongitude": bbox[3],
        })
    err = None
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            feats = r.json().get("features", [])
            rows = []
            for f in feats:
                prop = f.get("properties", {})
                geom = f.get("geometry", {})
                lon, lat, depth = (geom.get("coordinates") or [None,None,None])[:3]
                rows.append({
                    "time": datetime.fromtimestamp((prop.get("time") or 0)/1000, tz=timezone.utc),
                    "lat": lat, "lon": lon, "depth_km": depth,
                    "mag": prop.get("mag"), "place": prop.get("place",""),
                    "id": prop.get("ids","") or prop.get("code",""),
                })
            return pd.DataFrame(rows), None
        except Exception as e:
            time.sleep(1.2)
            err = str(e)
    return pd.DataFrame(columns=["time","lat","lon","depth_km","mag","place","id"]), f"USGS hata: {err or 'bilinmiyor'}"

def bbox_around_polyline(polyline, pad_km=50):
    lats = [p[0] for p in polyline]; lons = [p[1] for p in polyline]
    minlat, maxlat = min(lats), max(lats); minlon, maxlon = min(lons), max(lons)
    lat_pad = pad_km/111.32
    midlat = (minlat+maxlat)/2
    lon_pad = pad_km/(111.32*max(0.3, math.cos(math.radians(midlat))))
    return (minlat-lat_pad, minlon-lon_pad, maxlat+lat_pad, maxlon+lon_pad)

# =========================
# Yardımcı: dosya özeti
# =========================
def files_summary(folder, patterns=("*.csv",)):
    found = []
    for pat in patterns:
        found += glob.glob(os.path.join(folder, pat))
    found = [os.path.basename(p) for p in sorted(found)]
    return found

# =========================
# GNSS / InSAR / Yeraltı suyu okuyucular (esnek kolon eşleşmesi)
# =========================
def pick_first_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols: return cols[cand.lower()]
    return None

def read_timeseries_latest_slope(folder, value_candidates, days_window=30, time_candidates=("timestamp","time","date")):
    notes = []
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        return None, ["[TS] klasörde CSV yok"]
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if d.empty:
                notes.append(f"[TS] {os.path.basename(f)} boş")
                continue
            tcol = pick_first_column(d, time_candidates)
            vcol = pick_first_column(d, value_candidates + ["value","val"])
            if not tcol or not vcol:
                notes.append(f"[TS] {os.path.basename(f)} kolon bulunamadı (t:{tcol}, v:{vcol})")
                continue
            d = d[[tcol, vcol]].dropna()
            d.rename(columns={tcol:"timestamp", vcol:"value"}, inplace=True)
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
            d = d.dropna(subset=["timestamp","value"])
            if d.empty:
                notes.append(f"[TS] {os.path.basename(f)} timestamp/value parse edilemedi")
                continue
            d["src"] = os.path.basename(f)
            frames.append(d)
        except Exception as e:
            notes.append(f"[TS] {os.path.basename(f)} hata: {e}")
    if not frames:
        return None, (notes if notes else ["[TS] uygun içerik yok"])
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    end = df["timestamp"].max()
    start = end - pd.Timedelta(days=days_window)
    dfw = df[df["timestamp"].between(start, end)]
    if dfw["timestamp"].nunique() < 2:
        return None, notes + [f"[TS] {days_window} günde 2 farklı zaman noktası yok"]
    t = (dfw["timestamp"] - dfw["timestamp"].min()).dt.total_seconds()/86400.0
    slope = np.polyfit(t, dfw["value"].values, 1)[0]
    vmax = np.nanmax(np.abs(dfw["value"].values))
    return {"slope_per_day": float(slope), "max_abs": float(vmax), "end": str(end)}, notes

def groundwater_anomaly(folder, days_recent=14, days_base=60, time_candidates=("timestamp","time","date")):
    notes = []
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        return None, ["[GW] klasörde CSV yok"]
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if d.empty:
                notes.append(f"[GW] {os.path.basename(f)} boş"); continue
            tcol = pick_first_column(d, time_candidates)
            vcol = pick_first_column(d, ["water_level","wl","waterlevel","level"])
            if not tcol or not vcol:
                notes.append(f"[GW] {os.path.basename(f)} kolon bulunamadı (t:{tcol}, v:{vcol})")
                continue
            d = d[[tcol, vcol]].dropna()
            d.rename(columns={tcol:"timestamp", vcol:"wl"}, inplace=True)
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
            d = d.dropna(subset=["timestamp","wl"])
            d["src"] = os.path.basename(f)
            frames.append(d)
        except Exception as e:
            notes.append(f"[GW] {os.path.basename(f)} hata: {e}")
    if not frames:
        return None, notes if notes else ["[GW] uygun içerik yok"]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    end = df["timestamp"].max()
    recent_start = end - pd.Timedelta(days=days_recent)
    base_start   = end - pd.Timedelta(days=days_recent+days_base)
    base = df[df["timestamp"].between(base_start, recent_start)]
    recent = df[df["timestamp"].between(recent_start, end)]
    if len(base) < 5 or len(recent) < 3:
        return None, notes + [f"[GW] veri yetersiz (base≥5, recent≥3)"]
    mu = base["wl"].mean(); sigma = base["wl"].std(ddof=1) or 1.0
    z = (recent["wl"].mean() - mu)/sigma
    return {"z_anomaly": float(z), "end": str(end)}, notes

# =========================
# GNSS canlı fallback
# =========================
def gnss_live_feature(station: str = "PARI", period: str = "30d"):
    """
    data_sources/gnss.py -> GeoNet Tilde API (ücretsiz) üzerinden GNSS verisini çeker,
    günlük eğimi (mm/gün) döndürür. Türkiye için mekânsal uygunluk değil, pipeline testi içindir.
    """
    if get_gnss_dataframe is None:
        return None, "[GNSS-LIVE] get_gnss_dataframe import edilemedi"

    try:
        df = get_gnss_dataframe(station=station, period=period)
        if df.empty or df["timestamp"].nunique() < 2:
            return None, "[GNSS-LIVE] Veri yetersiz"

        df = df.sort_values("timestamp")
        # displacement metre -> mm
        df["disp_mm"] = df["displacement"] * 1000.0

        end = df["timestamp"].max()
        start = end - pd.Timedelta(days=GNSS_WINDOW_DAYS)
        w = df[df["timestamp"].between(start, end)]
        if w["timestamp"].nunique() < 2:
            return None, "[GNSS-LIVE] Pencerede 2 zaman noktası yok"

        t_days = (w["timestamp"] - w["timestamp"].min()).dt.total_seconds() / 86400.0
        slope_mm_per_day = float(np.polyfit(t_days, w["disp_mm"].values, 1)[0])
        vmax = float(np.nanmax(np.abs(w["disp_mm"].values)))
        return {"slope_per_day": slope_mm_per_day, "max_abs": vmax, "end": str(end)}, None
    except Exception as e:
        return None, f"[GNSS-LIVE] hata: {e}"

# =========================
# Sismik özellik + hız artışı
# =========================
def seismic_features_near_fault(eq_df, fault_poly, buffer_km):
    if eq_df.empty:
        return {"count":0, "max_mag":0.0, "b_value":0.0, "dist_weighted":0.0, "times":[], "mags_arr":np.array([])}
    dists, mags, times = [], [], []
    for _, r in eq_df.iterrows():
        d = distance_to_polyline_km(r["lat"], r["lon"], fault_poly)
        if d <= buffer_km:
            dists.append(d); mags.append(r["mag"] or 0.0); times.append(r["time"])
    if not dists:
        return {"count":0, "max_mag":0.0, "b_value":0.0, "dist_weighted":0.0, "times":[], "mags_arr":np.array([])}
    mags = np.array(mags, float)
    count = len(mags); max_mag = float(np.max(mags))
    b_value = (math.log10(max(count,1))/max(max_mag,1e-6)) if count else 0.0
    dists = np.array(dists, float)
    dist_weighted = float(np.sum((10**mags)/(1.0+dists)))
    return {"count":int(count), "max_mag":round(max_mag,2), "b_value":round(b_value,3),
            "dist_weighted":dist_weighted, "times":times, "mags_arr":mags}

def seismic_rate_change(times, window_recent_days=7, window_base_days=30):
    if not times: return {"recent":0, "base_per_day":0.0, "z":0.0}
    times = pd.Series(pd.to_datetime(times)).sort_values()
    end = times.max()
    recent_start = end - pd.Timedelta(days=window_recent_days)
    base_start   = recent_start - pd.Timedelta(days=window_base_days)
    recent = times[(times > recent_start)]
    base   = times[(times > base_start) & (times <= recent_start)]
    recent_count = recent.size
    base_days = max(1.0, (recent_start - base_start).days)
    base_rate = base.size / base_days
    recent_days = max(1.0, (end - recent_start).days or window_recent_days)
    expected_recent = base_rate * recent_days
    std_recent = math.sqrt(max(1e-6, expected_recent))
    z = (recent_count - expected_recent)/std_recent if std_recent>0 else 0.0
    return {"recent": int(recent_count), "base_per_day": float(base_rate), "z": float(z)}

# =========================
# Normalize + Risk
# =========================
def normalize(value, min_val, max_val):
    if max_val <= min_val: return 0.0
    return max(0.0, min(1.0, (value - min_val)/(max_val - min_val)))

def compute_risk(seis, rateinfo, gnss, insar, gw, strength_score, weights=None):
    if weights is None:
        weights = {
            "max_mag": 0.18, "count": 0.08, "distW": 0.24, "b_inv": 0.05,
            "rateZ": 0.15, "gnss": 0.10, "insar": 0.10, "gw": 0.05, "strength": 0.05
        }
    s_maxm = normalize(seis.get("max_mag",0), 3.0, 6.5)
    s_cnt  = normalize(seis.get("count",0),   0, 200)
    s_dw   = normalize(seis.get("dist_weighted",0.0), 0, 5e6)
    s_binv = 1.0 - normalize(seis.get("b_value",0.0), 0.5, 1.5)
    s_rate = normalize(rateinfo.get("z",0.0), 0.0, 3.5)
    s_gnss = normalize(abs((gnss or {}).get("slope_per_day",0.0)), 0.0, 1.0)
    s_ins  = normalize(abs((insar or {}).get("slope_per_day",0.0)), 0.0, 0.02)
    s_gw   = normalize(abs((gw or {}).get("z_anomaly",0.0)), 0.0, 2.5)
    s_str  = 1.0 - (strength_score/100.0)

    wsum = sum(weights.values()) or 1.0
    w = {k: v/wsum for k,v in weights.items()}
    combined = (
        w["max_mag"]*s_maxm + w["count"]*s_cnt + w["distW"]*s_dw + w["b_inv"]*s_binv +
        w["rateZ"]*s_rate + w["gnss"]*s_gnss + w["insar"]*s_ins + w["gw"]*s_gw + w["strength"]*s_str
    )
    combined = max(0.0, min(1.0, combined))
    comps = {
        "MaxM": round(100*s_maxm,1), "Count": round(100*s_cnt,1),
        "DistW": round(100*s_dw,1), "b^-1": round(100*s_binv,1),
        "RateZ": round(100*s_rate,1), "GNSS": round(100*s_gnss,1),
        "InSAR": round(100*s_ins,1), "GW": round(100*s_gw,1),
        "Strength(↓)": round(100*s_str,1)
    }
    return round(100*combined,1), comps

# =========================
# Zaman penceresi kestirimi (heuristic)
# =========================
def time_window_estimate(risk, rateZ, gnss, insar, gw):
    gnss_flag = abs((gnss or {}).get("slope_per_day",0.0)) >= 0.6
    ins_flag  = abs((insar or {}).get("slope_per_day",0.0)) >= 0.008
    gw_flag   = abs((gw or {}).get("z_anomaly",0.0)) >= 1.8
    strong_signal = (rateZ >= 2.5) or sum([gnss_flag, ins_flag, gw_flag]) >= 2
    medium_signal = (rateZ >= 1.5) or gnss_flag or ins_flag or gw_flag
    if risk >= 70 and strong_signal:
        return "≈ 1 hafta içinde artmış deprem olasılığı (pilot)"
    elif risk >= 50 and medium_signal:
        return "≈ 1 ay içinde artmış deprem olasılığı (pilot)"
    else:
        return "Kısa vadede belirgin sinyal yok (pilot)"

# =========================
# Dash UI
# =========================
app = dash.Dash(__name__)
app.title = "Fay Bazlı Test Modu"

app.layout = html.Div([
    html.H2("Deprem Risk İzleme • Fay Bazlı Test Modu", style={"margin":"12px 0"}),

    html.Div(id="alert-banner", style={
        "padding":"10px 14px","borderRadius":"10px","marginBottom":"10px",
        "backgroundColor":"#20262e","color":"#fff","display":"none"
    }),

    # Kontroller
    html.Div([
        html.Div([
            html.Label("Fay seç"),
            dcc.Dropdown(id="fault-name",
                options=[{"label": k, "value": k} for k in FAULTS.keys()],
                value=list(FAULTS.keys())[0], clearable=False),
        ], style={"flex":"1", "minWidth":"260px", "marginRight":"12px"}),

        html.Div([
            html.Label("Tampon mesafe (km)"),
            dcc.Slider(id="buffer-km", min=10, max=150, step=5, value=60,
                       marks={i: str(i) for i in range(10,151,20)}),
        ], style={"flex":"2", "minWidth":"260px", "marginRight":"12px"}),

        html.Div([
            html.Label("Zaman aralığı (gün)"),
            dcc.Slider(id="days", min=7, max=90, step=1, value=30,
                       marks={7:"7", 30:"30", 60:"60", 90:"90"}),
        ], style={"flex":"2", "minWidth":"220px", "marginRight":"12px"}),

        html.Div([
            html.Label("Min magnitüd"),
            dcc.Slider(id="minmag", min=1.0, max=5.0, step=0.1, value=2.0,
                       marks={1.0:"1.0", 2.0:"2.0", 3.0:"3.0", 4.0:"4.0", 5.0:"5.0"}),
        ], style={"flex":"2", "minWidth":"220px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"8px"}),

    html.Div([
        html.Div([
            html.Label("Dayanım skoru (0=çok zayıf, 100=çok güçlü)"),
            dcc.Slider(id="strength", min=0, max=100, step=1, value=55,
                       marks={0:"0",25:"25",50:"50",75:"75",100:"100"}),
        ], style={"flex":"2", "minWidth":"300px", "marginRight":"12px"}),

        html.Div([
            html.Label("Veri yoksa otomatik genişlet"),
            dcc.Checklist(id="auto-expand", options=[{"label":" Aç", "value":"on"}], value=[]),
        ], style={"width":"220px","paddingTop":"24px"}),

        html.Div([
            html.Label("Windows bildirimi"),
            dcc.Checklist(id="notify-enable", options=[{"label":" Aç", "value":"on"}], value=[]),
        ], style={"width":"220px","paddingTop":"24px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"8px","marginTop":"8px"}),

    html.Hr(),

    # Özet & grafikler
    html.Div([
        html.Div([
            html.H3(id="risk-score", style={"margin":"0 0 8px 0"}),
            html.Div(id="seis-summary", style={"opacity":0.85}),
            html.Div(id="multi-summary", style={"opacity":0.85,"marginTop":"6px","fontSize":"13px"}),
        ], style={"flex":"1","minWidth":"260px"}),

        html.Div([ dcc.Graph(id="chart-components", style={"height":"280px"}) ],
                 style={"flex":"1","minWidth":"280px"}),

        html.Div([ dcc.Graph(id="chart-mags", style={"height":"280px"}) ],
                 style={"flex":"2","minWidth":"320px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"12px"}),

    html.Div([ dcc.Graph(id="map-plot", style={"height":"420px"}) ], style={"marginTop":"8px"}),

    html.Div(id="last-updated", style={"fontSize":"12px","opacity":0.7,"marginTop":"6px"}),

    html.Div([
        html.Button("Anlık Snapshot Kaydet", id="btn-snapshot"),
        html.Span(id="snapshot-msg", style={"marginLeft":"8px","fontSize":"12px","opacity":0.8})
    ], style={"marginTop":"8px"}),

    # Diagnostik panel
    html.Details([
        html.Summary("Diagnostik (tıkla aç/kapat)"),
        html.Pre(id="diag-text", style={"whiteSpace":"pre-wrap"})
    ], open=False, style={"marginTop":"10px"}),

    dcc.Interval(id="tick", interval=5*60*1000, n_intervals=0)
], style={"padding":"14px 18px","fontFamily":"Inter, system-ui, Arial"})

# =========================
# Callback
# =========================
@app.callback(
    Output("risk-score","children"),
    Output("seis-summary","children"),
    Output("multi-summary","children"),
    Output("chart-components","figure"),
    Output("chart-mags","figure"),
    Output("map-plot","figure"),
    Output("last-updated","children"),
    Output("snapshot-msg","children"),
    Output("alert-banner","children"),
    Output("alert-banner","style"),
    Output("diag-text","children"),
    Input("fault-name","value"),
    Input("buffer-km","value"),
    Input("days","value"),
    Input("minmag","value"),
    Input("strength","value"),
    Input("auto-expand","value"),
    Input("notify-enable","value"),
    Input("btn-snapshot","n_clicks"),
    Input("tick","n_intervals"),
    prevent_initial_call=False
)
def update(fault_name, buffer_km, days, minmag, strength, auto_expand_val, notify_val, n_clicks, n_int):
    diag_lines = []

    poly = FAULTS[fault_name]
    end = datetime.utcnow().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=int(days))
    bbox = bbox_around_polyline(poly, pad_km=max(20, buffer_km+30))

    # USGS çek + otomatik genişletme
    df, usgs_err = fetch_usgs_between(start, end, minmag=minmag, bbox=bbox)
    diag_lines.append(f"[USGS] ilk çekim: rows={len(df)} err={usgs_err}")

    def widen_once(_df, _buffer, _days, _minmag):
        changed = False
        if _minmag > AUTO_EXPAND_LIMITS["minmag"]:
            _minmag = max(AUTO_EXPAND_LIMITS["minmag"], _minmag - 0.5); changed = True
        elif _days < AUTO_EXPAND_LIMITS["days"]:
            _days = min(AUTO_EXPAND_LIMITS["days"], _days + 15); changed = True
        elif _buffer < AUTO_EXPAND_LIMITS["buffer_km"]:
            _buffer = min(AUTO_EXPAND_LIMITS["buffer_km"], _buffer + 20); changed = True
        return changed, _buffer, _days, _minmag

    expanded_info = ""
    if ("on" in (auto_expand_val or [])):
        tries = 0
        while df.empty and tries < 5:
            changed, buffer_km, days, minmag = widen_once(df, buffer_km, days, minmag)
            if not changed: break
            tries += 1
            start = end - timedelta(days=int(days))
            bbox = bbox_around_polyline(poly, pad_km=max(20, buffer_km+30))
            df, usgs_err = fetch_usgs_between(start, end, minmag=minmag, bbox=bbox)
        if tries > 0:
            expanded_info = f"[AUTO] genişletme denemesi={tries} -> minmag={minmag}, days={days}, buffer={buffer_km}"
            diag_lines.append(expanded_info)

    seis = seismic_features_near_fault(df, poly, buffer_km)
    diag_lines.append(f"[SEIS] tampon içi sayım={seis.get('count',0)}, maxM={seis.get('max_mag',0)}")
    rateinfo = seismic_rate_change(seis.get("times", []), 7, 30)
    diag_lines.append(f"[SEIS] RateZ={rateinfo.get('z',0):.2f}")

    # GNSS: önce CSV, yoksa canlı fallback
    gnss, gnss_notes = read_timeseries_latest_slope(
        GNSS_DIR, ["displacement_mm","displacement","disp_mm","disp"], days_window=GNSS_WINDOW_DAYS)
    if gnss is None:
        live, live_note = gnss_live_feature(station="PARI", period="30d")
        if live:
            gnss = live
            gnss_notes = (gnss_notes or []) + ["[GNSS] CSV bulunamadı, canlı GeoNet kullanıldı (PARI, 30d)"]
        else:
            gnss_notes = (gnss_notes or []) + [live_note or "[GNSS-LIVE] sebep bilinmiyor"]

    # InSAR / Yeraltı suyu: şimdilik CSV şart
    insar, insar_notes = read_timeseries_latest_slope(
        INSAR_DIR, ["strain","eps","defo","deformation"], days_window=INSAR_WINDOW_DAYS)
    gw, gw_notes = groundwater_anomaly(
        GW_DIR, days_recent=GW_RECENT_DAYS, days_base=GW_BASE_DAYS)

    diag_lines += [f"[GNSS] {', '.join(gnss_notes)}"] if gnss_notes else []
    diag_lines += [f"[INSAR] {', '.join(insar_notes)}"] if insar_notes else []
    diag_lines += [f"[GW] {', '.join(gw_notes)}"] if gw_notes else []

    risk, comps = compute_risk(seis, rateinfo, gnss, insar, gw, strength)

    # Bileşen grafiği
    comp_fig = go.Figure()
    comp_fig.add_bar(x=list(comps.keys()), y=list(comps.values()))
    comp_fig.update_layout(title="Bileşen Skorları (0-100)", margin=dict(l=10,r=10,t=30,b=10), yaxis=dict(range=[0,100]))

    # Tampon içindekiler
    if not df.empty:
        in_rows = []
        for _, r in df.iterrows():
            if distance_to_polyline_km(r["lat"], r["lon"], poly) <= buffer_km:
                in_rows.append(r)
        df_in = pd.DataFrame(in_rows) if in_rows else pd.DataFrame(columns=df.columns)
    else:
        df_in = pd.DataFrame(columns=["time","mag","lat","lon","depth_km"])

    # Magnitüd-zaman
    mags_fig = go.Figure()
    if not df_in.empty:
        mags_fig.add_scatter(x=df_in["time"], y=df_in["mag"], mode="markers")
    mags_fig.update_layout(title=f"Son {days} gün • tampon ≤ {buffer_km} km • minmag≥{minmag}",
                           xaxis_title="Zaman (UTC)", yaxis_title="Mw",
                           margin=dict(l=10,r=10,t=30,b=10))

    # Harita
    map_fig = go.Figure()
    map_fig.add_scattergeo(lon=[p[1] for p in poly], lat=[p[0] for p in poly],
                           mode="lines+markers", name="Fay")
    if not df.empty:
        lats, lons, mags = df["lat"].tolist(), df["lon"].tolist(), (df["mag"].fillna(0)).tolist()
        map_fig.add_scattergeo(lon=lons, lat=lats, text=[f"M {m:.1f}" for m in mags],
                               marker=dict(size=[max(4, (m or 0)*2.5) for m in mags]),
                               mode="markers", name="Depremler")
    map_fig.update_layout(geo=dict(showland=True, showcountries=True, showcoastlines=True, fitbounds="locations"),
                          margin=dict(l=10,r=10,t=10,b=10))

    risk_title = f"Risk Skoru: {risk} / 100"
    seis_summary = (f"Maks M: {seis.get('max_mag',0)} • Olay sayısı: {seis.get('count',0)} • "
                    f"b-değeri(kaba): {seis.get('b_value',0)} • Uzaklık-ağırlıklı sismisite: {int(seis.get('dist_weighted',0)):,}")
    multi_summary = (
        f"Rate Z: {rateinfo.get('z',0):.2f} (7g vs 30g) • "
        f"GNSS slope: {((gnss or {}).get('slope_per_day',0)):.3f} mm/gün • "
        f"InSAR slope: {((insar or {}).get('slope_per_day',0)):.4f} /gün • "
        f"Yeraltı suyu anomali Z: {((gw or {}).get('z_anomaly',0)):.2f}"
    )

    window_text = time_window_estimate(risk, rateinfo.get("z",0.0), gnss, insar, gw)
    color = "#d9534f" if "1 hafta" in window_text else ("#f0ad4e" if "1 ay" in window_text else "#2e7d32")
    banner_style = {
        "padding":"10px 14px","borderRadius":"10px","marginBottom":"10px",
        "color":"#fff","backgroundColor": color, "display":"block"
    }
    banner_msg = f"Uyarı (pilot): {window_text}"

    snapshot_msg = ""
    try:
        if n_clicks:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SNAP_DIR, f"snapshot_{ts}.csv")
            pd.DataFrame([{
                "fault": fault_name, "buffer_km": buffer_km, "days": days, "minmag": minmag,
                "strength": strength, "risk": risk,
                "max_mag": seis.get('max_mag',0), "count": seis.get('count',0), "b_value": seis.get('b_value',0),
                "dist_weighted": seis.get('dist_weighted',0),
                "rateZ": rateinfo.get('z',0), "gnss_slope": (gnss or {}).get('slope_per_day',None),
                "insar_slope": (insar or {}).get('slope_per_day',None), "gw_z": (gw or {}).get('z_anomaly',None),
                "window": window_text, "updated_local": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]).to_csv(fname, index=False, encoding="utf-8")
            with open(os.path.join(SNAP_DIR, "latest_alert.json"), "w", encoding="utf-8") as f:
                json.dump({"window": window_text, "risk": risk, "rateZ": rateinfo.get('z',0),
                           "time": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
            snapshot_msg = f"Kaydedildi: {os.path.abspath(fname)}"
    except Exception as e:
        snapshot_msg = f"Kaydetme hatası: {e}"

    if TOASTER and ("on" in (notify_val or [])) and ("1 hafta" in window_text or "1 ay" in window_text):
        try:
            TOASTER.show_toast("Deprem Risk İzleme (pilot uyarı)",
                               f"{fault_name}: {window_text} • Risk={risk}",
                               duration=5, threaded=True)
        except Exception:
            pass

    updated = f"Güncellendi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (yerel)"

    # Diagnostik metin
    diag = []
    diag.append(f"[FAULTS] kaynaklar: {', '.join(FAULTS.keys())}")
    diag += fault_notes
    diag.append(f"[DIR] GNSS: {', '.join(files_summary(GNSS_DIR)) or '—'}")
    diag.append(f"[DIR] INSAR: {', '.join(files_summary(INSAR_DIR)) or '—'}")
    diag.append(f"[DIR] GW   : {', '.join(files_summary(GW_DIR)) or '—'}")
    if expanded_info: diag.append(expanded_info)
    diag += diag_lines
    diag_text = "\n".join(diag)

    return (risk_title, seis_summary, multi_summary,
            comp_fig, mags_fig, map_fig, updated, snapshot_msg,
            banner_msg, banner_style, diag_text)

if __name__ == "__main__":
    app.run(debug=True)
