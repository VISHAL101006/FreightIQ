import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="FreightIQ", layout="wide")

# ================= LOAD DATA =================
@st.cache_data
def load():
    rates = pd.read_csv("data/freight_rates.csv", parse_dates=["date"])
    ports = pd.read_csv("data/ports.csv")
    vessels = pd.read_csv("data/vessels.csv")
    ind = pd.read_csv("data/market_indicators.csv", parse_dates=["date"])
    return rates, ports, vessels, ind

rates, ports, vessels, ind = load()
ROUTES = sorted(rates["route"].unique())

# ================= USER INPUT =================
st.sidebar.title("🚢 Cargo Requirement")
route = st.sidebar.selectbox("Route (Origin → Destination)", ROUTES)
commodity = st.sidebar.selectbox("Commodity", ["Coal", "Coking Coal", "Iron Ore", "Limestone"])
qty = st.sidebar.number_input("Cargo Quantity (MT)", 1000, 300000, 70000, 1000)
horizon = st.sidebar.selectbox("Forecast Horizon (days)", [7, 14, 30], index=1)
max_date = rates["date"].max().date()
min_date = (rates["date"].min() + pd.Timedelta(days=60)).date()
sim_date = st.sidebar.date_input("📅 Simulate 'today' (data up to)",
                                 value=max_date, min_value=min_date, max_value=max_date)
rates = rates[rates["date"] <= pd.Timestamp(sim_date)]
ind = ind[ind["date"] <= pd.Timestamp(sim_date)]

origin, dest = route.split("-")
port = ports[ports["port_name"] == dest].iloc[0]
latest = ind.iloc[-1]
last_date = rates["date"].max()

st.title("FreightIQ – Intelligent Freight Forecasting & Vessel Chartering System")
st.caption(f"Route: **{origin} → {dest}** | Commodity: **{commodity}** | Quantity: **{qty:,} MT**")

# ================= FORECAST ENGINE =================
def forecast(route_name, vclass, horizon):
    df = rates[(rates["route"] == route_name) &
               (rates["vessel_class"] == vclass)].sort_values("date")
    y = df["rate_usd_per_tonne"].values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    vol = y[-14:].std()
    today = y[-1]
    days = np.arange(1, horizon + 1)
    pred = today + slope * days * 0.6
    band = 1.2 * vol * np.sqrt(days)
    return today, pred, band, vol

# ================= RISK ENGINE =================
fuel_vol = ind["fuel_price_usd"].tail(14).pct_change().dropna().std() * 100
risk = (0.35 * latest["tension_index"] / 100
        + 0.25 * latest["congestion_index"] / 10
        + 0.15 * latest["weather_risk"] / 10
        + 0.15 * min(fuel_vol / 3, 1)
        + 0.10 * 0.3)
risk_level = "Low" if risk < 0.34 else ("Medium" if risk < 0.67 else "High")

# ================= VESSEL OPTIMIZATION =================
results = []
for _, v in vessels.iterrows():
    today_rate, pred, band, vol = forecast(route, v["vessel_class"], horizon)
    best_i = int(np.argmin(pred))
    best_rate = pred[best_i]
    best_date = last_date + pd.Timedelta(days=best_i + 1)
    waiting_cost = port["avg_waiting_days"] * v["typical_daily_hire_usd"]

    util = qty / v["capacity_max_mt"] * 100
    billable = max(qty, v["capacity_min_mt"])
    total = best_rate * billable + waiting_cost

    ok, reason = True, "✅ Feasible"
    if qty > v["capacity_max_mt"]:
        ok, reason = False, "❌ Capacity too small"
    elif v["draft_m"] > port["max_draft_m"]:
        ok, reason = False, f"❌ Draft {v['draft_m']}m > port limit {port['max_draft_m']}m"
    elif billable > qty:
        reason = f"⚠️ Feasible – dead freight (paying for {int(v['capacity_min_mt']):,} MT min)"

    results.append({"Vessel": v["vessel_class"], "Status": reason,
                    "Utilisation": f"{util:.0f}%",
                    "Rate Today ($/t)": round(today_rate, 2),
                    "Best Forecast ($/t)": round(best_rate, 2),
                    "Best Date": best_date.strftime("%d %b"),
                    "Billable Tons": int(billable),
                    "Total Cost ($)": round(total),
                    "_ok": ok, "_total": total, "_best_rate": best_rate,
                    "_today": today_rate, "_date": best_date,
                    "_pred": pred, "_band": band, "_billable": billable})

feasible = [r for r in results if r["_ok"]]

# ================= SINGLE-VESSEL vs SPLIT PLAN =================
rec = min(feasible, key=lambda r: r["_total"]) if feasible else None
split = None
if not rec:
    can_enter = vessels[vessels["draft_m"] <= port["max_draft_m"]]
    if len(can_enter) > 0:
        sv = can_enter.sort_values("capacity_max_mt", ascending=False).iloc[0]
        n = math.ceil(qty / sv["capacity_max_mt"])
        t_rate, s_pred, s_band, _ = forecast(route, sv["vessel_class"], horizon)
        b_i = int(np.argmin(s_pred))
        per_voyage = s_pred[b_i] * sv["capacity_max_mt"] + port["avg_waiting_days"] * sv["typical_daily_hire_usd"]
        split = {"vessel": sv["vessel_class"], "cap": sv["capacity_max_mt"], "n": n,
                 "today": t_rate, "rate": s_pred[b_i],
                 "date": last_date + pd.Timedelta(days=b_i + 1),
                 "total": n * per_voyage, "pred": s_pred, "band": s_band}

if rec:
    k_today, k_best, k_date, k_total = rec["_today"], rec["_best_rate"], rec["_date"], rec["_total"]
    chart_v, chart_pred, chart_band = rec["Vessel"], rec["_pred"], rec["_band"]
elif split:
    k_today, k_best, k_date, k_total = split["today"], split["rate"], split["date"], split["total"]
    chart_v, chart_pred, chart_band = split["vessel"], split["pred"], split["band"]
else:
    k_today = k_best = k_total = 0
    k_date = last_date
    chart_v = None

# ================= ALERTS =================
st.subheader("⚠️ Risk & Market Alerts")
if latest["tension_index"] > 70:
    st.error(f"🌍 Geopolitical tension HIGH ({latest['tension_index']:.0f}/100) – insurance premiums rising on route.")
elif latest["tension_index"] > 45:
    st.warning(f"🌍 Geopolitical tension elevated ({latest['tension_index']:.0f}/100) – monitor route risk.")
if latest["congestion_index"] > 7:
    st.warning(f"⚓ Congestion building at {dest} – expect longer anchorage waits.")
if latest["weather_risk"] > 7:
    st.warning("🌩 Weather risk high on route – possible voyage delays.")
if ind["fuel_price_usd"].tail(7).mean() > ind["fuel_price_usd"].iloc[-28:-14].mean():
    st.info("⛽ Bunker fuel trending up – freight surcharges may increase soon.")
if risk_level == "Low" and latest["tension_index"] <= 45:
    st.success("✅ Market conditions currently stable.")

# ================= KPIs =================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rate Today", f"${k_today:.2f}/t")
c2.metric("Best Forecast Rate", f"${k_best:.2f}/t")
c3.metric("Best Charter Date", k_date.strftime("%d %b"))
c4.metric("Risk Level", risk_level, delta=f"{risk:.2f} score")

# ================= FORECAST CHART =================
if chart_v:
    st.subheader(f"📈 Freight Forecast – {chart_v} on {route}")
    df = rates[(rates["route"] == route) &
               (rates["vessel_class"] == chart_v)].sort_values("date").tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rate_usd_per_tonne"],
                             name="Historical rate", line=dict(color="#1f4e79")))
    fx = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    fig.add_trace(go.Scatter(x=fx, y=chart_pred + chart_band, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fx, y=chart_pred - chart_band, fill="tonexty",
                             fillcolor="rgba(31,78,121,0.15)", line=dict(width=0), name="Confidence band"))
    fig.add_trace(go.Scatter(x=fx, y=chart_pred, name="Forecast",
                             line=dict(dash="dash", color="#e87722")))
    fig.update_layout(yaxis_title="USD / tonne", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ================= VESSEL TABLE =================
st.subheader("🚢 Vessel Comparison")
cols = ["Vessel", "Status", "Utilisation", "Rate Today ($/t)",
        "Best Forecast ($/t)", "Best Date", "Billable Tons", "Total Cost ($)"]
st.dataframe(pd.DataFrame(results)[cols], use_container_width=True)

# ================= RECOMMENDATION =================
if rec:
    saving = (rec["_today"] - rec["_best_rate"]) * rec["_billable"]
    st.success(
        f"### ✅ RECOMMENDATION\n"
        f"**Vessel:** {rec['Vessel']}  |  **Charter around:** {rec['_date'].strftime('%d %b %Y')}\n\n"
        f"**Expected rate:** ${rec['_best_rate']:.2f}/t (today ${rec['_today']:.2f}/t)\n\n"
        f"**Billable tons:** {rec['_billable']:,.0f} MT  |  **Estimated total cost:** ${rec['_total']:,.0f}\n\n"
        f"**Projected saving vs booking today:** ${saving:,.0f}  |  **Risk level:** {risk_level}"
    )
    if rec["_best_rate"] < rec["_today"] - 0.1 and risk < 0.67:
        st.info(f"📉 Market trending down – wait and book around {rec['_date'].strftime('%d %b')} for a better rate.")
    elif rec["_best_rate"] > rec["_today"]:
        st.info("📈 Rates trending up – booking early is safer.")
elif split:
    st.warning(f"No single vessel can carry {qty:,} MT into {dest} "
               f"(Capesize needs 18 m draft; port limit is {port['max_draft_m']} m). Split-cargo plan below.")
    st.success(
        f"### ✅ RECOMMENDATION – SPLIT CARGO\n"
        f"**Optimal plan:** {split['n']} × {split['vessel']} voyages (~{split['cap']:,} MT each)\n\n"
        f"**Charter first voyage around:** {split['date'].strftime('%d %b %Y')}  |  **Expected rate:** ${split['rate']:.2f}/t\n\n"
        f"**Estimated total cost:** ${split['total']:,.0f}  |  **Risk level:** {risk_level}"
    )
    st.info(f"📦 Stagger the {split['n']} voyages ~10–15 days apart to match port berth availability.")
else:
    st.error("No vessel class can enter this port under current draft limits.")

# ================= IDLE-TIME MANAGEMENT =================
st.subheader("⏳ Idle-Time Management")
extra = 1.5 if latest["congestion_index"] > 7 else 0
st.info(f"{dest} average waiting time: **{port['avg_waiting_days'] + extra:.1f} days**. "
        f"{'Congestion is adding ~1.5 idle days – consider shifting the arrival window or using a smaller, faster-berthing vessel.' if extra else 'No abnormal idle time expected for this window.'}")

with st.expander("📊 Live Market Indicators (simulated feed)"):
    st.dataframe(ind.tail(7), use_container_width=True)