import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

dates = pd.date_range("2026-03-01", "2026-09-04", freq="D")
n = len(dates)
t = np.arange(n)

# ---------------- vessels.csv ----------------
vessels = pd.DataFrame({
    "vessel_class": ["Handysize", "Supramax", "Panamax", "Capesize"],
    "capacity_min_mt": [20000, 45000, 65000, 100000],
    "capacity_max_mt": [40000, 65000, 85000, 180000],
    "draft_m": [10.5, 12.5, 14.0, 18.0],
    "typical_daily_hire_usd": [14000, 18000, 22000, 30000],
})
vessels.to_csv("data/vessels.csv", index=False)

# ---------------- ports.csv ----------------
ports = pd.DataFrame({
    "port_name": ["Paradip", "Visakhapatnam", "Krishnapatnam"],
    "country": ["India", "India", "India"],
    "max_draft_m": [14.5, 13.5, 15.0],
    "max_length_m": [250, 230, 260],
    "avg_waiting_days": [1.8, 2.4, 1.2],
    "congestion_level": ["medium", "high", "low"],
})
ports.to_csv("data/ports.csv", index=False)

# ---------------- market_indicators.csv ----------------
fuel = 580 + 40 * np.sin(t / 45) + np.cumsum(np.random.normal(0, 1.5, n))
tension = np.clip(30 + 25 * np.sin(t / 60) + np.cumsum(np.random.normal(0, 2, n)), 0, 100)
weather = np.clip(3 + 3 * np.sin(t / 30 + 1) + np.random.normal(0, 1, n), 0, 10)
congestion = np.clip(5 + 2.5 * np.sin(t / 25) + np.random.normal(0, 0.8, n), 0, 10)

ind = pd.DataFrame({
    "date": dates,
    "fuel_price_usd": np.round(fuel, 1),
    "tension_index": np.round(tension, 1),
    "weather_risk": np.round(weather, 1),
    "congestion_index": np.round(congestion, 1),
})
ind.to_csv("data/market_indicators.csv", index=False)

# ---------------- freight_rates.csv ----------------
routes = [("Australia", "Paradip", 1.25),
          ("Indonesia", "Visakhapatnam", 0.85),
          ("Mozambique", "Krishnapatnam", 1.05)]
base_rate = {"Handysize": 24, "Supramax": 20, "Panamax": 17, "Capesize": 14}

rows = []
for origin, dest, mult in routes:
    for v in vessels["vessel_class"]:
        b = base_rate[v] * mult
        trend = np.random.uniform(-0.02, 0.03)
        rate = (b + trend * t + 1.5 * np.sin(t / 35)
                + 0.02 * (fuel - 580) + np.random.normal(0, 0.35, n))
        for d, r in zip(dates, rate):
            rows.append([d.strftime("%Y-%m-%d"), f"{origin}-{dest}",
                         origin, dest, v, round(max(r, 5), 2)])

rates = pd.DataFrame(rows, columns=["date", "route", "origin",
                                    "destination", "vessel_class",
                                    "rate_usd_per_tonne"])
rates.to_csv("data/freight_rates.csv", index=False)

print("DONE! Created CSV files in /data:")
for f in sorted(os.listdir("data")):
    print("  -", f)