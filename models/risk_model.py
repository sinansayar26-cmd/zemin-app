# -*- coding: utf-8 -*-
"""
Risk skoru hesaplama modülü (v2 - multi-window + adaptif eşikler)
- Dizi/seri girişlerde güvenli istatistikler ve 0–1 alt skorlar
- Multi-window (kısa/orta/uzun) birleşimi
- Veri kalitesine göre dinamik ağırlık
- Adaptif alarm eşikleri (p70/p90)
"""

from __future__ import annotations
import math
from typing import Iterable, Union, Dict, Any, List, Tuple, Optional
import numpy as np

ArrayLike = Union[Iterable[float], np.ndarray, float, int, None]

# ============================================================
# Yardımcılar
# ============================================================

def _as_array(x: ArrayLike) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (int, float)):
        return np.array([float(x)], dtype=float)
    try:
        arr = np.asarray(list(x), dtype=float)
    except Exception:
        try:
            return np.array([float(x)], dtype=float)
        except Exception:
            return np.array([], dtype=float)
    return arr

def _safe_nanstd(a: ArrayLike) -> float:
    arr = _as_array(a)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr))

def _safe_nanmax(a: ArrayLike) -> float:
    arr = _as_array(a)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))

def _safe_count(a: ArrayLike) -> int:
    arr = _as_array(a)
    return int(np.sum(np.isfinite(arr)))

def _percentile(a: ArrayLike, q: float, fallback: float = 0.0) -> float:
    arr = _as_array(a)
    if arr.size == 0:
        return fallback
    return float(np.nanpercentile(arr, q))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))

def _robust_scale(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or not np.isfinite(low) or not np.isfinite(high):
        return 0.0
    if high <= low:
        return 0.0
    x = (value - low) / (high - low)
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)

def _tail(arr: np.ndarray, frac: float, min_n: int = 6) -> np.ndarray:
    """Serinin son kısmını döndür (ör. %10 / %30 / tümü)."""
    n = max(min_n, int(math.ceil(len(arr) * frac)))
    if n <= 0:
        return np.array([], dtype=float)
    return arr[-n:]

# ============================================================
# Alt skorlar (tek pencere)
# ============================================================

def gnss_subscore(gnss_vals: ArrayLike) -> float:
    arr = _as_array(gnss_vals)
    if arr.size == 0:
        return 0.0
    std = _safe_nanstd(arr)
    spread = _percentile(arr, 90, 0.0) - _percentile(arr, 10, 0.0)
    upper = max(1e-9, spread / 4.0)
    return _robust_scale(std, 0.0, upper)

def insar_subscore(insar_vals: ArrayLike) -> float:
    arr = _as_array(insar_vals)
    if arr.size == 0:
        return 0.0
    std = _safe_nanstd(arr)
    spread = _percentile(arr, 90, 0.0) - _percentile(arr, 10, 0.0)
    upper = max(1e-9, spread / 4.0)
    return _robust_scale(std, 0.0, upper)

def water_subscore(water_vals: ArrayLike) -> float:
    arr = _as_array(water_vals)
    if arr.size <= 2:
        return 0.0
    diffs = np.diff(arr)
    std = _safe_nanstd(diffs)
    spread = _percentile(diffs, 90, 0.0) - _percentile(diffs, 10, 0.0)
    upper = max(1e-9, spread / 4.0)
    return _robust_scale(std, 0.0, upper)

def seismic_subscore(magnitudes: ArrayLike) -> float:
    mags = _as_array(magnitudes)
    if mags.size == 0:
        return 0.0
    mags = mags[np.isfinite(mags)]
    mags = mags[mags >= 0]
    if mags.size == 0:
        return 0.0

    max_mag = float(np.nanmax(mags))
    count = int(np.sum(np.isfinite(mags)))

    E = np.sum(10.0 ** (1.5 * (mags - 5.0)))
    E_log = math.log10(E + 1.0)

    max_mag_term = _sigmoid((max_mag - 4.0) / 1.0)
    energy_term  = _sigmoid((E_log - 0.3) / 0.4)
    count_term   = _sigmoid((count - 3.0) / 2.0)

    s = 0.5 * max_mag_term + 0.35 * energy_term + 0.15 * count_term
    return _clip01(s)

# ============================================================
# Multi-window alt skorlar
# ============================================================

MW_DEFAULT_FRACS = (0.10, 0.30, 1.00)  # kısa / orta / uzun (serinin %’si)

def _multiwindow(arr: ArrayLike, fn, fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS) -> Tuple[float, float, float]:
    a = _as_array(arr)
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    short = fn(_tail(a, fracs[0]))
    mid   = fn(_tail(a, fracs[1]))
    long  = fn(_tail(a, fracs[2]))
    return (short, mid, long)

def gnss_subscore_mw(gnss_vals: ArrayLike, fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS) -> float:
    s, m, l = _multiwindow(gnss_vals, gnss_subscore, fracs)
    # Kısa pencereye biraz daha ağırlık
    return _clip01(0.5 * s + 0.3 * m + 0.2 * l)

def insar_subscore_mw(insar_vals: ArrayLike, fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS) -> float:
    s, m, l = _multiwindow(insar_vals, insar_subscore, fracs)
    return _clip01(0.4 * s + 0.35 * m + 0.25 * l)

def water_subscore_mw(water_vals: ArrayLike, fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS) -> float:
    s, m, l = _multiwindow(water_vals, water_subscore, fracs)
    return _clip01(0.45 * s + 0.35 * m + 0.20 * l)

def seismic_subscore_mw(magnitudes: ArrayLike, fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS) -> float:
    s, m, l = _multiwindow(magnitudes, seismic_subscore, fracs)
    return _clip01(0.5 * s + 0.3 * m + 0.2 * l)

# ============================================================
# Ağırlıklar & dinamik ayar
# ============================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "seismic": 0.45,
    "gnss":    0.20,
    "insar":   0.20,
    "water":   0.15,
}

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in w.values() if np.isfinite(v) and v >= 0)
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: float(v) / total for k, v in w.items()}

def _data_quality_weight(arr: ArrayLike, min_points: int = 10) -> float:
    """Veri azsa/bozuksa 0–1 arası güven katsayısı."""
    a = _as_array(arr)
    n = _safe_count(a)
    if n <= 0:
        return 0.0
    # Basit ölçek: n<min_points ise doğrusal düşür
    if n < min_points:
        return max(0.2, n / float(min_points))
    # Aşırı NaN’ler veya tekdüzelik durumunda da hafif kırpma
    std = _safe_nanstd(a)
    return _clip01(0.2 + 0.8 * _robust_scale(std, 0.0, max(1e-9, std + 1e-9)))

def dynamic_weights(
    base: Optional[Dict[str, float]],
    *,
    gnss_vals: ArrayLike,
    insar_vals: ArrayLike,
    water_vals: ArrayLike,
    magnitudes: ArrayLike
) -> Dict[str, float]:
    """Kanal veri kalitesine göre ağırlıkları yeniden dağıt."""
    w = (base or DEFAULT_WEIGHTS).copy()
    w = _normalize_weights(w)

    q = {
        "gnss":  _data_quality_weight(gnss_vals),
        "insar": _data_quality_weight(insar_vals),
        "water": _data_quality_weight(water_vals),
        "seismic": _data_quality_weight(magnitudes),
    }

    # Ağırlıkları kalite ile çarp, sonra normalize et
    for k in w.keys():
        w[k] = w[k] * (0.5 + 0.5 * q[k])  # kalite 0 ise yarıya düşür; 1 ise değişme
    w = _normalize_weights(w)
    return w

# ============================================================
# Ana skorlar
# ============================================================

def compute_risk(
    gnss_vals: ArrayLike,
    seismic_vals: ArrayLike,   # geriye uyumluluk için
    insar_vals: ArrayLike,
    water_vals: ArrayLike,
    recent_magnitudes: ArrayLike,
    *,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Geriye uyumluluk: tek pencere alt skorları ile 0–100 skor."""
    s_seis  = seismic_subscore(recent_magnitudes if recent_magnitudes is not None else seismic_vals)
    s_gnss  = gnss_subscore(gnss_vals)
    s_insar = insar_subscore(insar_vals)
    s_water = water_subscore(water_vals)

    w = _normalize_weights(weights or DEFAULT_WEIGHTS)
    combined = (
        w["seismic"] * s_seis +
        w["gnss"]    * s_gnss +
        w["insar"]   * s_insar +
        w["water"]   * s_water
    )
    if not np.isfinite(combined):
        combined = 0.0
    return round(100.0 * _clip01(combined), 2)

def compute_risk_multiwindow(
    gnss_vals: ArrayLike,
    insar_vals: ArrayLike,
    water_vals: ArrayLike,
    recent_magnitudes: ArrayLike,
    *,
    weights: Optional[Dict[str, float]] = None,
    fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS,
    use_dynamic_weights: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Multi-window alt skorlarla 0–100 risk + açıklama döndürür.
    Dönüş: (risk_score, explain_dict)
    """
    s_seis  = seismic_subscore_mw(recent_magnitudes, fracs)
    s_gnss  = gnss_subscore_mw(gnss_vals, fracs)
    s_insar = insar_subscore_mw(insar_vals, fracs)
    s_water = water_subscore_mw(water_vals, fracs)

    w_base = _normalize_weights(weights or DEFAULT_WEIGHTS)
    w = dynamic_weights(w_base,
                        gnss_vals=gnss_vals, insar_vals=insar_vals,
                        water_vals=water_vals, magnitudes=recent_magnitudes) if use_dynamic_weights else w_base

    combined = (w["seismic"] * s_seis +
                w["gnss"]    * s_gnss +
                w["insar"]   * s_insar +
                w["water"]   * s_water)
    combined = 0.0 if not np.isfinite(combined) else _clip01(float(combined))
    score = round(100.0 * combined, 2)

    explain = {
        "weights_base": w_base,
        "weights_effective": w,
        "subscores_multiwindow": {
            "seismic": s_seis,
            "gnss": s_gnss,
            "insar": s_insar,
            "water": s_water
        },
        "combined_0_1": combined,
        "risk_score_0_100": score,
        "fracs": fracs
    }
    return score, explain

# ============================================================
# Alarm seviyeleri
# ============================================================

def alarm_level(score: float) -> str:
    """Sabit eşikler: 0–33 low, 33–66 moderate, 66+ high."""
    try:
        s = float(score)
    except Exception:
        return "low"
    if not np.isfinite(s):
        return "low"
    if s < 33:
        return "low"
    elif s < 66:
        return "moderate"
    else:
        return "high"

def adaptive_thresholds(history_scores: ArrayLike, p_low: float = 70.0, p_high: float = 90.0) -> Tuple[float, float]:
    """Skor geçmişinden alt/üst eşik (p70, p90). Geçmiş azsa sabitlere döner."""
    arr = _as_array(history_scores)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        # Yeterli veri yoksa sabit eşiklere dön
        return (33.0, 66.0)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    # eşiklerin birbirine çok yakın olmasını engelle
    if hi - lo < 5.0:
        mid = (lo + hi) / 2.0
        lo, hi = mid - 2.5, mid + 2.5
    lo = float(np.clip(lo, 5.0, 95.0))
    hi = float(np.clip(hi, lo + 1.0, 99.0))
    return lo, hi

def alarm_level_adaptive(score: float, history_scores: ArrayLike) -> Dict[str, Any]:
    """Adaptif eşikler ile seviye döndür."""
    lo, hi = adaptive_thresholds(history_scores)
    try:
        s = float(score)
    except Exception:
        s = 0.0
    level = "low" if s < lo else ("moderate" if s < hi else "high")
    return {"level": level, "thresholds": {"low": lo, "high": hi}}

# ============================================================
# Açıklama (explain) yardımcıları
# ============================================================

def explain_risk_components(
    gnss_vals: ArrayLike,
    insar_vals: ArrayLike,
    water_vals: ArrayLike,
    recent_magnitudes: ArrayLike,
    *,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Tek-pencere açıklama (geriye uyumluluk)."""
    w = _normalize_weights(weights or DEFAULT_WEIGHTS)
    s = {
        "seismic_subscore": seismic_subscore(recent_magnitudes),
        "gnss_subscore":    gnss_subscore(gnss_vals),
        "insar_subscore":   insar_subscore(insar_vals),
        "water_subscore":   water_subscore(water_vals),
    }
    total = (w["seismic"] * s["seismic_subscore"] +
             w["gnss"]    * s["gnss_subscore"] +
             w["insar"]   * s["insar_subscore"] +
             w["water"]   * s["water_subscore"])
    total = _clip01(float(total))
    return {
        "weights": w,
        "subscores": s,
        "combined_0_1": total,
        "risk_score_0_100": round(100.0 * total, 2),
    }

def explain_risk_components_mw(
    gnss_vals: ArrayLike,
    insar_vals: ArrayLike,
    water_vals: ArrayLike,
    recent_magnitudes: ArrayLike,
    *,
    weights: Optional[Dict[str, float]] = None,
    fracs: Tuple[float, float, float] = MW_DEFAULT_FRACS
) -> Dict[str, Any]:
    """Multi-window açıklama (dashboard için)."""
    score, explain = compute_risk_multiwindow(
        gnss_vals, insar_vals, water_vals, recent_magnitudes,
        weights=weights, fracs=fracs, use_dynamic_weights=True
    )
    explain["risk_score_0_100"] = score
    return explain
