"""Synthetic Walmart-shaped data, reusing the original test fixture generator."""
import numpy as np
import pandas as pd

WALMART_HOLIDAYS = {
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
}

def make_synthetic_walmart(n_stores: int = 5, periods: int = 143, seed: int = 0) -> pd.DataFrame:
    """Walmart-shaped panel: Store, Date (dd-mm-YYYY), Weekly_Sales, drivers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-02-05", periods=periods, freq="W-FRI")
    woy = dates.isocalendar().week.values.astype(float)
    t = np.arange(periods)
    rows = []
    for s in range(1, n_stores + 1):
        base = rng.uniform(4e5, 2e6)
        season = 0.06 * np.sin(2 * np.pi * woy / 52.18) + 0.03 * np.cos(4 * np.pi * woy / 52.18)
        peak = np.where(woy == 47, 0.35, 0) + np.where(woy == 51, 0.55, 0) + np.where(woy == 50, 0.2, 0)
        sales = base * (1 + 0.0004 * t + season + peak + rng.normal(0, 0.025, periods))
        for i, d in enumerate(dates):
            rows.append(
                {
                    "Store": s,
                    "Date": d.strftime("%d-%m-%Y"),
                    "Weekly_Sales": round(float(sales[i]), 2),
                    "Holiday_Flag": int(d.strftime("%Y-%m-%d") in WALMART_HOLIDAYS),
                    "Temperature": round(60 + 25 * np.sin(2 * np.pi * (woy[i] - 15) / 52) + rng.normal(0, 3), 2),
                    "Fuel_Price": round(2.6 + 0.008 * i + rng.normal(0, 0.03), 3),
                    "CPI": round(210 + 0.05 * i + s * 0.3, 4),
                    "Unemployment": round(8.5 - 0.01 * i + rng.normal(0, 0.05), 3),
                }
            )
    return pd.DataFrame(rows)
