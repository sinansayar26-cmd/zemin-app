# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime, timedelta, UTC

def generate_dummy_timeseries(hours=24, seed=None):
    """Basit dummy zaman serisi (random yürüme)."""
    if seed is not None:
        np.random.seed(seed)
    now = datetime.now(UTC)
    times = [now - timedelta(hours=i) for i in reversed(range(hours))]
    values = np.random.normal(0, 1, size=hours).cumsum()
    return times, values