# deprem_risk_izleme/data_sources/gnss.py

from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from typing import Literal, Optional, Dict, List

# GeoNet Tilde API tabanı (ücretsiz, anahtarsız)
TILDE_BASE = "https://tilde.geonet.org.nz"
# GNSS için domain 'gnss', isim 'displacement', yöntem günlük '1d', aspekt: east|north|up
# Örn: /v3/data/gnss/PARI/displacement/-/1d/east/latest/30d

class GNSSFetchError(Exception):
    pass

def _tilde_url(station: str, aspect: Literal["east","north","up"], period: str = "30d") -> str:
    return f"{TILDE_BASE}/v3/data/gnss/{station}/displacement/-/1d/{aspect}/latest/{period}"

def fetch_geonet_gnss(
    station: str = "PARI",
    period: str = "30d",
    timeout: int = 15
) -> pd.DataFrame:
    """
    GeoNet Tilde API'den GNSS yer değiştirme zaman serisini (E,N,U) çeker,
    büyüklük (magnitude) hesaplar ve proje uyumlu DataFrame döner.

    Dönüş sütunları:
      - timestamp (pd.Timestamp, UTC)
      - east_m, north_m, up_m (metre)
      - displacement (metre)  -> sqrt(E^2 + N^2 + U^2)

    Not: GeoNet günlük çözümler verir; tipik olarak her sabah (NZ saatine göre) güncellenir.
    """
    aspects = ["east", "north", "up"]
    series: Dict[str, pd.Series] = {}
    meta_ok = False

    for a in aspects:
        url = _tilde_url(station, a, period)
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException as e:
            raise GNSSFetchError(f"GeoNet isteği başarısız: {e}") from e

        if r.status_code != 200:
            raise GNSSFetchError(f"GeoNet {a} yanıt kodu {r.status_code}: {r.text[:200]}")

        payload = r.json()
        if not isinstance(payload, list) or len(payload) == 0 or "data" not in payload[0]:
            raise GNSSFetchError(f"GeoNet {a} beklenmeyen yanıt yapısı.")

        data = payload[0]["data"]
        # ts: ISO8601 (UTC), val: metre cinsinden yer değiştirme (dokümanda böyle belirtiliyor)
        # Kaynak: https://www.geonet.org.nz/data/supplementary/gnss_time_series_notes
        ts = pd.to_datetime([d["ts"] for d in data], utc=True)
        vals = pd.to_numeric([d["val"] for d in data], errors="coerce")
        s = pd.Series(vals, index=ts).sort_index()
        series[a] = s
        meta_ok = True

    if not meta_ok:
        raise GNSSFetchError("GeoNet verisi alınamadı.")

    # Zaman eksenini birleştir
    df = pd.concat(series, axis=1)
    df.columns = ["east_m", "north_m", "up_m"]

    # Büyüklük (metre)
    df["displacement"] = np.sqrt(
        df["east_m"]**2 + df["north_m"]**2 + df["up_m"]**2
    )

    # index -> sütun
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df.dropna().reset_index(drop=True)

def get_gnss_dataframe(
    provider: Literal["geonet"] = "geonet",
    station: str = "PARI",
    period: str = "30d"
) -> pd.DataFrame:
    """
    Proje genelinde tek giriş noktası.
    İleride başka ücretsiz sağlayıcı eklemek istersek burayı genişletiriz.
    """
    if provider == "geonet":
        return fetch_geonet_gnss(station=station, period=period)
    raise ValueError(f"Desteklenmeyen provider: {provider}")

# CLI hızlı test:
if __name__ == "__main__":
    df = get_gnss_dataframe(station="PARI", period="30d")
    print(df.tail())