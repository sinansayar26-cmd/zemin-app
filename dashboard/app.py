# -*- coding: utf-8 -*-
"""
Deprem Risk İzleme - Dashboard
- Compact grid + sabit yükseklik
- Açıklama tablosu: renk kodlu ısı haritası
- KPI'lar: emoji ikonlu
- Gerçek zamanlı veri gecikmesi kartı (dakika)
"""

import os
import sys
import random
from datetime import datetime, timezone
import numpy as np

# ------------------------------------------------------------------
# Modül yolu
# ------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ------------------------------------------------------------------
# Dash / Plotly
# ------------------------------------------------------------------
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

# ------------------------------------------------------------------
# Risk modeli
# ------------------------------------------------------------------
from models.risk_model import (
    compute_risk_multiwindow,
    alarm_level_adaptive
)

# ------------------------------------------------------------------
# Boyut sabitleri
# ------------------------------------------------------------------
FIG_HEIGHT = 200          # her grafiğin yüksekliği (px)
CARD_HEIGHT = 260         # grafikli kartın toplam yüksekliği (başlık+grafik)
SPARK_HEIGHT = 110
GAP = "12px"

# ------------------------------------------------------------------
# Veri kaynakları (fallback'lı) + timestamp dönüşü
# Her fonksiyon: (values_list, last_timestamp_utc) döndürmeye çalışır.
# Eğer gerçek kaynak fonksiyonu sadece liste döndürüyorsa 'now' verilir.
# ------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)

def _fallback_magnitudes(n=20):
    mags = list(np.clip(np.random.normal(2.8, 0.5, n), 1.5, 5.2))
    if random.random() < 0.35:
        mags.append(round(random.uniform(3.8, 5.0), 1))
    return mags

def _fallback_series(n=240, start=0.0, drift=0.0, noise=0.2):
    x = start
    arr = []
    for _ in range(n):
        x += drift + np.random.normal(0, noise)
        arr.append(x)
    return arr

def _unpack_with_ts(res):
    """Kaynaklar tuple/list dönebilir. (data, ts) ya da sadece data."""
    now = _now()
    if isinstance(res, (list, tuple)):
        # Eğer (data, ts) formu ise ts datetime olabilir
        if len(res) == 2 and isinstance(res[1], (datetime,)):
            return list(res[0]), res[1].astimezone(timezone.utc) if res[1].tzinfo else res[1].replace(tzinfo=timezone.utc)
        # Düz liste gibi davran
        return list(res), now
    # bilinmeyen tip -> boş + now
    return [], now

def safe_fetch_magnitudes():
    try:
        from data_sources.usgs import fetch_usgs_last_24h
        data = fetch_usgs_last_24h()
        mags, ts = _unpack_with_ts(data)
        if len(mags) > 0:
            return mags, ts
    except Exception:
        pass
    try:
        from data_sources.afad import fetch_afad_last_24h
        data = fetch_afad_last_24h()
        mags, ts = _unpack_with_ts(data)
        if len(mags) > 0:
            return mags, ts
    except Exception:
        pass
    return _fallback_magnitudes(), _now()

def safe_fetch_gnss():
    try:
        from data_sources.gnss import fetch_gnss_series
        data = fetch_gnss_series()
        arr, ts = _unpack_with_ts(data)
        return arr, ts
    except Exception:
        return _fallback_series(n=240, start=0.0, drift=0.002, noise=0.15), _now()

def safe_fetch_insar():
    try:
        from data_sources.dummy_sensors import fetch_insar_series
        data = fetch_insar_series()
        arr, ts = _unpack_with_ts(data)
        return arr, ts
    except Exception:
        return _fallback_series(n=240, start=0.0, drift=0.0006, noise=0.03), _now()

def safe_fetch_water():
    try:
        from data_sources.dummy_sensors import fetch_water_series
        data = fetch_water_series()
        arr, ts = _unpack_with_ts(data)
        return arr, ts
    except Exception:
        base = _fallback_series(n=240, start=10.0, drift=0.001, noise=0.05)
        t = np.linspace(0, 2*np.pi, len(base))
        series = (np.array(base) + 0.2 * np.sin(2.5 * t)).tolist()
        return series, _now()

# ------------------------------------------------------------------
# Stil yardımcıları
# ------------------------------------------------------------------
app = Dash(__name__)
app.title = "Deprem Risk İzleme"

CARD_STYLE = {
    "background": "#11141a",
    "borderRadius": "16px",
    "padding": "12px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.25)",
    "border": "1px solid #202636",
}

GRID = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
    "gap": GAP,
    "alignItems": "stretch"
}

def big_number(title, id_value, id_suffix=None, emoji=None):
    return html.Div(
        style=CARD_STYLE | {"minWidth": "260px"},
        children=[
            html.Div(
                [html.Span((emoji + " ") if emoji else ""), html.Span(title)],
                style={"fontSize": "13px", "color": "#98a2b3"}
            ),
            html.Div(id=id_value, style={"fontSize": "34px", "fontWeight": "700", "color": "white"}),
            (html.Div(id=id_suffix, style={"fontSize": "12px", "color": "#cbd5e1"}) if id_suffix else None),
        ]
    )

def graph_card(title, graph_id):
    return html.Div(
        style=CARD_STYLE | {
            "height": f"{CARD_HEIGHT}px",
            "overflow": "hidden"
        },
        children=[
            html.Div(title, style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "6px", "color": "white"}),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False, "responsive": True},
                style={"height": f"{FIG_HEIGHT}px"}
            )
        ]
    )

def build_figure(y, title, height=FIG_HEIGHT):
    x = list(range(len(y)))
    return go.Figure(
        data=[go.Scatter(x=x, y=y, mode="lines", name=title)],
        layout=go.Layout(
            height=height,
            margin=dict(l=18, r=8, t=6, b=18),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
            font=dict(color="#e5e7eb")
        )
    )

def build_sparkline(scores):
    x = list(range(len(scores)))
    return go.Figure(
        data=[go.Scatter(x=x, y=scores, mode="lines")],
        layout=go.Layout(
            height=SPARK_HEIGHT,
            margin=dict(l=12, r=8, t=6, b=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(gridcolor="#1f2937", title="Skor"),
            font=dict(color="#e5e7eb")
        ),
    )

# --- Renk yardımcıları ---
def _hsl_to_rgb(h, s, l):
    # h:0-360, s/l:0-1
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs(((h/60)%2) - 1))
    m = l - c/2
    if   0 <= h < 60:   rp, gp, bp = c, x, 0
    elif 60 <= h < 120: rp, gp, bp = x, c, 0
    elif 120 <= h < 180:rp, gp, bp = 0, c, x
    elif 180 <= h < 240:rp, gp, bp = 0, x, c
    elif 240 <= h < 300:rp, gp, bp = x, 0, c
    else:               rp, gp, bp = c, 0, x
    r = int((rp + m)*255); g = int((gp + m)*255); b = int((bp + m)*255)
    return r, g, b

def heat_rgba(v):
    """0→yeşil, 0.5→sarı, 1→kırmızı; koyu tema için ~0.25 opaklık"""
    v = max(0.0, min(1.0, float(v)))
    # 120 (yeşil) → 0 (kırmızı)
    h = 120 * (1 - v)
    r, g, b = _hsl_to_rgb(h, 0.9, 0.45)
    return f"rgba({r},{g},{b},0.25)"

def explain_table(explain_dict):
    w_eff = explain_dict.get("weights_effective", {})
    subs  = explain_dict.get("subscores_multiwindow", {})
    rows = []
    for k in ["seismic", "gnss", "insar", "water"]:
        s_val = float(subs.get(k, 0.0) or 0.0)
        bg = heat_rgba(s_val)
        rows.append(
            html.Tr([
                html.Td(k.upper(), style={"padding": "6px 8px", "borderBottom": "1px solid #253046"}),
                html.Td(f"{s_val:.2f}", style={"padding": "6px 8px", "borderBottom": "1px solid #253046", "background": bg}),
                html.Td(f"{w_eff.get(k, 0):.2f}", style={"padding": "6px 8px", "borderBottom": "1px solid #253046"})
            ])
        )
    header = html.Tr([
        html.Th("Kanal", style={"textAlign": "left", "padding": "6px 8px", "borderBottom": "1px solid #334155"}),
        html.Th("Alt Skor (0–1)", style={"textAlign": "left", "padding": "6px 8px", "borderBottom": "1px solid #334155"}),
        html.Th("Ağırlık", style={"textAlign": "left", "padding": "6px 8px", "borderBottom": "1px solid #334155"})
    ])
    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "color": "#e5e7eb", "fontSize": "13px"}
    )
    return html.Div(
        table,
        style={
            "borderTop": "1px solid #253046",
            "borderBottom": "1px solid #253046",
            "padding": "6px 0"
        }
    )

def latency_badge(minutes):
    try:
        m = float(minutes)
    except Exception:
        return html.Span("N/A", style={"background":"#334155","color":"#e5e7eb","padding":"2px 6px","borderRadius":"8px","fontSize":"12px"})
    if m <= 5:
        bg = "rgba(34,197,94,0.25)"   # yeşil
        col = "#86efac"
    elif m <= 30:
        bg = "rgba(245,158,11,0.25)"  # turuncu
        col = "#fcd34d"
    else:
        bg = "rgba(239,68,68,0.25)"   # kırmızı
        col = "#fca5a5"
    return html.Span(f"{m:.0f} dk", style={"background": bg, "color": col, "padding": "2px 6px", "borderRadius": "8px", "fontSize": "12px"})

def latency_table(latencies):
    # latencies: dict {name: minutes}
    rows = []
    order = ["USGS/AFAD", "GNSS", "InSAR", "Water"]
    for k in order:
        v = latencies.get(k, None)
        rows.append(html.Tr([
            html.Td(k, style={"padding":"6px 8px", "borderBottom":"1px solid #253046"}),
            html.Td(latency_badge(v), style={"padding":"6px 8px", "borderBottom":"1px solid #253046"})
        ]))
    table = html.Table([html.Tbody(rows)],
        style={"width":"100%","borderCollapse":"collapse","color":"#e5e7eb","fontSize":"13px"})
    return table

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
app.layout = html.Div(
    style={"background": "#0b0f17", "minHeight": "100vh", "color": "white", "padding": "12px"},
    children=[
        html.H2("Deprem Risk İzleme", style={"marginBottom": "4px"}),
        html.Div("AFAD + USGS + sensör tabanlı birleşik risk skoru", style={"color": "#94a3b8", "marginBottom": GAP}),

        dcc.Store(id="score-history", data=[]),

        # KPI + Spark + Latency
        html.Div(style=GRID, children=[
            big_number("Anlık Risk Skoru", "kpi-score", "kpi-level", emoji="⚠️"),
            big_number("Adaptif Eşikler", "kpi-thresholds", "kpi-thresholds-sub", emoji="🎚️"),
            html.Div(
                style=CARD_STYLE | {"minWidth": "320px"},
                children=[
                    html.Div("Skor Sparkline (son ölçümler) 📈", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "6px"}),
                    dcc.Graph(id="sparkline", config={"displayModeBar": False}, style={"height": f"{SPARK_HEIGHT}px"})
                ]
            ),
            html.Div(
                style=CARD_STYLE | {"minWidth": "320px"},
                children=[
                    html.Div("Gerçek Zamanlı Veri Gecikmesi ⏱️", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "6px"}),
                    html.Div(id="latency-table")
                ]
            ),
        ]),

        # 4 grafik (sabit yükseklikli kartlarda)
        html.Div(style=GRID, children=[
            graph_card("GNSS Deplasman 🌐", "gnss-graph"),
            graph_card("InSAR Strain 🛰️", "insar-graph"),
            graph_card("Yeraltı Suyu Seviyesi 💧", "water-graph"),
            graph_card("Son 24s Magnitüdler 🌋", "seis-graph"),
        ]),

        # Açıklama paneli
        html.Div(style=CARD_STYLE | {"marginTop": GAP}, children=[
            html.Div("Açıklama (Neden bu skor?)", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "6px"}),
            html.Div(style=GRID, children=[
                html.Div(children=[
                    html.Div(id="explain-summary", style={"color": "#cbd5e1", "marginBottom": "6px"}),
                    html.Div(id="explain-table"),
                ]),
                html.Div(children=[
                    html.Div("Ham birleşik skor (0–1): ", style={"color": "#94a3b8"}),
                    html.Div(id="explain-combined", style={"fontSize": "24px", "fontWeight": "700"}),
                    html.Div("Pencereler: kısa/orta/uzun", style={"color": "#94a3b8", "marginTop": "6px"}),
                    html.Div(id="explain-fracs", style={"fontSize": "13px", "color": "#cbd5e1"}),
                ])
            ])
        ]),

        dcc.Interval(id="tick", interval=60_000, n_intervals=0),
    ]
)

# ------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------

@app.callback(
    [
        Output("gnss-graph", "figure"),
        Output("insar-graph", "figure"),
        Output("water-graph", "figure"),
        Output("seis-graph", "figure"),
        Output("kpi-score", "children"),
        Output("kpi-level", "children"),
        Output("kpi-thresholds", "children"),
        Output("kpi-thresholds-sub", "children"),
        Output("sparkline", "figure"),
        Output("score-history", "data"),
        Output("explain-summary", "children"),
        Output("explain-table", "children"),
        Output("explain-combined", "children"),
        Output("explain-fracs", "children"),
        Output("latency-table", "children"),
    ],
    [Input("tick", "n_intervals")],
    [State("score-history", "data")],
    prevent_initial_call=False
)
def update_dashboard(n, score_hist):
    # Verileri çek + timestamp
    gnss, ts_gnss = safe_fetch_gnss()
    insar, ts_insar = safe_fetch_insar()
    water, ts_water = safe_fetch_water()
    mags, ts_seis = safe_fetch_magnitudes()

    # Risk + açıklama
    score, explain = compute_risk_multiwindow(gnss, insar, water, mags)
    if not isinstance(score_hist, list):
        score_hist = []
    score_hist = (score_hist + [score])[-2000:]

    lvl_info = alarm_level_adaptive(score, score_hist)
    level_text = lvl_info["level"]
    thr_low = lvl_info["thresholds"]["low"]
    thr_high = lvl_info["thresholds"]["high"]

    # Grafikler
    fig_gnss = build_figure(gnss, "GNSS Deplasman", height=FIG_HEIGHT)
    fig_insar = build_figure(insar, "InSAR Strain", height=FIG_HEIGHT)
    fig_water = build_figure(water, "Yeraltı Suyu", height=FIG_HEIGHT)
    fig_seis  = build_figure(mags, "Magnitüdler", height=FIG_HEIGHT)

    # KPI metinleri
    kpi_score = f"{score:.2f}"
    kpi_level = f"Seviye: {level_text.upper()}"
    kpi_thr   = f"Alt: {thr_low:.1f}  |  Üst: {thr_high:.1f}"
    kpi_thr_sub = "Eşikler: skor geçmişi p70/p90."

    # Sparkline
    fig_spark = build_sparkline(score_hist)

    # Açıklama
    combined01 = explain.get("combined_0_1", 0.0)
    fracs = explain.get("fracs", (0.10, 0.30, 1.00))
    summary = [
        html.Div(f"Etkin Ağırlıklar: { {k: round(v, 2) for k,v in explain.get('weights_effective', {}).items()} }",
                 style={"marginBottom": "4px"}),
        html.Div(f"Alt Skorlar: { {k: round(v, 2) for k,v in explain.get('subscores_multiwindow', {}).items()} }")
    ]
    combined_view = f"{combined01:.3f}"
    fracs_view = f"Kısa {fracs[0]*100:.0f}% | Orta {fracs[1]*100:.0f}% | Uzun {fracs[2]*100:.0f}%"
    table_children = explain_table(explain)

    # Gecikme (dakika)
    now = _now()
    latencies = {
        "USGS/AFAD": (now - ts_seis).total_seconds() / 60.0 if isinstance(ts_seis, datetime) else None,
        "GNSS": (now - ts_gnss).total_seconds() / 60.0 if isinstance(ts_gnss, datetime) else None,
        "InSAR": (now - ts_insar).total_seconds() / 60.0 if isinstance(ts_insar, datetime) else None,
        "Water": (now - ts_water).total_seconds() / 60.0 if isinstance(ts_water, datetime) else None,
    }
    latency_tbl = latency_table(latencies)

    return (
        fig_gnss, fig_insar, fig_water, fig_seis,
        kpi_score, kpi_level, kpi_thr, kpi_thr_sub,
        fig_spark, score_hist,
        summary, table_children, combined_view, fracs_view,
        latency_tbl
    )

if __name__ == "__main__":
    app.run(debug=True)