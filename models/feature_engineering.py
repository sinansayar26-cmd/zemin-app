# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Dict, Any, Tuple

# ----------------------
# Yardımcılar
# ----------------------
def _as_np(x: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.Series):
        x = x.values
    return np.asarray(x, dtype=float)

def robust_scale(x: np.ndarray) -> np.ndarray:
    """Medyan ve IQR ile ölçekleme (aykırı değerlere dayanıklı)."""
    x = _as_np(x)
    if x.size == 0:
        return x
    med = np.nanmedian(x)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = max(q3 - q1, 1e-9)
    return (x - med) / iqr

def linreg_slope(y: Sequence[float], x: Sequence[float] | None = None) -> float:
    """Basit doğrusal regresyon eğimi (y ~ a*x + b). x yoksa 0..n-1."""
    y = _as_np(y)
    n = len(y)
    if n < 2 or np.all(~np.isfinite(y)):
        return 0.0
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = _as_np(x)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x, y = x[mask], y[mask]
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom <= 0:
        return 0.0
    slope = np.sum((x - xm) * (y - ym)) / denom
    return float(slope)

def last_diff(y: Sequence[float]) -> float:
    """Son değer ile bir önceki arasındaki fark."""
    y = _as_np(y)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    return float(y[-1] - y[-2])

# ----------------------
# GNSS / Strain / InSAR / Yeraltı suyu özellikleri
# ----------------------
def series_features(y: Sequence[float]) -> Dict[str, float]:
    """Zaman serisi için temel özetler: std, son_fark, eğim."""
    y = _as_np(y)
    if y.size == 0:
        return {"std": 0.0, "last_diff": 0.0, "slope": 0.0}
    std = float(np.nanstd(y))
    ld = last_diff(y)
    sl = linreg_slope(y)
    return {"std": std, "last_diff": ld, "slope": sl}

# ----------------------
# Sismik özellikler (düzeltilmiş)
# ----------------------
def seismic_features(times: Sequence[pd.Timestamp] | Sequence[str],
                     mags: Sequence[float],
                     horizon_hours: float = 24.0) -> Dict[str, float]:
    """
    Basit sismik özetler:
      - event_rate_h: Saat başına olay sayısı (son 'horizon_hours')
      - max_mag: Son 'horizon_hours' içindeki maksimum M
      - b_value_approx: log10(N) / max_mag
      - energy_proxy: Sum(10^(1.5*M + 4.8))
    """
    mags = _as_np(mags)
    # Uzunlukları hizala
    n = min(len(times), len(mags))
    if n == 0:
        return {"event_rate_h": 0.0, "max_mag": 0.0,
                "b_value_approx": 1.0, "energy_proxy": 0.0}

    times = list(times)[:n]
    mags = mags[:n]

    now = pd.Timestamp.utcnow()
    tser = pd.to_datetime(times, utc=True, errors="coerce")
    mask = (tser >= now - pd.Timedelta(hours=horizon_hours)) & np.isfinite(mags)

    if mask.sum() == 0:
        return {"event_rate_h": 0.0, "max_mag": 0.0,
                "b_value_approx": 1.0, "energy_proxy": 0.0}

    m = mags[mask]
    t = tser[mask]

    span_h = max((t.max() - t.min()).total_seconds() / 3600.0, 1e-6)
    rate = float(len(m) / span_h)
    max_mag = float(np.nanmax(m))
    b_val = float(np.log10(max(len(m), 1)) / max(max_mag, 1e-9))
    energy = float(np.nansum(10 ** (1.5 * m + 4.8)))

    return {"event_rate_h": rate, "max_mag": max_mag,
            "b_value_approx": b_val, "energy_proxy": energy}

# ----------------------
# Ana özellik derleyici
# ----------------------
def build_feature_vector(
    gnss_disp: Sequence[float],
    strain: Sequence[float],
    insar: Sequence[float],
    groundwater: Sequence[float],
    eq_times: Sequence[pd.Timestamp] | Sequence[str],
    eq_mags: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tüm kaynaklardan tek bir özellik vektörü derler."""
    gnss = series_features(gnss_disp)
    stn = series_features(strain)
    ins = series_features(insar)
    gw  = series_features(groundwater)
    seis = seismic_features(eq_times, eq_mags, horizon_hours=24.0)

    feat_names = [
        "gnss_std", "gnss_last_diff", "gnss_slope",
        "strain_std", "strain_last_diff", "strain_slope",
        "insar_std", "insar_last_diff", "insar_slope",
        "gw_std", "gw_last_diff", "gw_slope",
        "eq_event_rate_h", "eq_max_mag", "eq_b_value", "eq_energy_proxy"
    ]
    X = np.array([
        gnss["std"], gnss["last_diff"], gnss["slope"],
        stn["std"],  stn["last_diff"],  stn["slope"],
        ins["std"],  ins["last_diff"],  ins["slope"],
        gw["std"],   gw["last_diff"],   gw["slope"],
        seis["event_rate_h"], seis["max_mag"], seis["b_value_approx"], seis["energy_proxy"]
    ], dtype=float)

    meta = {
        "feat_names": feat_names,
        "summaries": {
            "gnss": gnss, "strain": stn, "insar": ins,
            "groundwater": gw, "seismic": seis
        }
    }
    return X, meta
