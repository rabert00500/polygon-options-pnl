# app.py — Polygon Options P&L
# Entry/Exit TIME (LA) + Peak/Trough Highlights + Underlying Adjusted Close + Full P&L Metrics
# FIX: Daily bars use UTC calendar DATE (no LA shift) so dates match your inputs exactly.
# NEW: Toggle to override DAILY option Close with RTH close (last minute bar <= 13:00 LA / 4:00 ET)
# Peak/Trough KPIs are computed AFTER entry (not before)
# KPIs:
#   - Entry→Exit: $Entry → $Exit (Entry→Exit %)
#   - Peak after entry: $High (Entry→Peak %)
#   - Before-Peak Low: $Low (Pct)
# KPI % coloring:
#   - positive => green
#   - zero => green
#   - negative => red
# Table:
#   - Highlights Peak row (green) and LOWEST-BEFORE-PEAK row (red)
# ------------------------------------------------------------------------------------------------

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

LA_TZ = ZoneInfo("America/Los_Angeles")

st.set_page_config(
    page_title="Polygon Options P&L",
    page_icon="📈",
    layout="wide",
)

# ---- Global UI tweaks (readability) ----
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }

      /* KPI cards */
      .kpi-card {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
      }
      .kpi-title { font-size: 0.85rem; color: #a3a3a3; margin-bottom: 6px; }
      .kpi-value { font-size: 2.15rem; font-weight: 850; line-height: 1.05; color: #f5f5f5; }
      .kpi-sub   { font-size: 1.02rem; margin-top: 6px; color: #d4d4d4; font-weight: 750; }

      /* KPI % colors */
      .pct-green { color: #22c55e !important; }
      .pct-red   { color: #ef4444 !important; }

      /* note under KPIs */
      .kpi-note { margin-top: 0.55rem; font-size: 0.95rem; color: #c7c7c7; }
      .kpi-note code { font-size: 0.95rem; }

      /* smaller screens */
      @media (max-width: 900px) {
        .kpi-value { font-size: 1.7rem; }
        .kpi-sub   { font-size: 0.95rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def build_option_symbol(ticker: str, expiry: str, strike: float, opt_type: str) -> str:
    dt = datetime.strptime(expiry, "%Y-%m-%d")
    return f"O:{ticker.upper()}{dt.strftime('%y%m%d')}{opt_type.upper()}{int(round(strike * 1000)):08d}"


@st.cache_data(ttl=300)
def fetch_bars(api_key: str, symbol: str, timespan: str, start: str, end: str, adjusted: bool | None = None):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start}/{end}"
    params = {"apiKey": api_key}
    if adjusted is not None:
        params["adjusted"] = "true" if adjusted else "false"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def clean_option_df(raw_results, timespan: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_results)
    if timespan == "day":
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
        df["DateTime"] = pd.to_datetime(df["DateTime"])  # naive midnight date
    else:
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(LA_TZ)

    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    keep = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].sort_values("DateTime").reset_index(drop=True)
    return df


def clean_underlying_df(raw_results, timespan: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_results)
    if timespan == "day":
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
        df["DateTime"] = pd.to_datetime(df["DateTime"])  # naive midnight date
    else:
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(LA_TZ)

    df = df.rename(columns={"c": "Underlying Adj Close"})
    df = df[["DateTime", "Underlying Adj Close"]].sort_values("DateTime").reset_index(drop=True)
    return df


def add_underlying_price(option_df: pd.DataFrame, underlying_df: pd.DataFrame) -> pd.DataFrame:
    if option_df.empty:
        return option_df
    if underlying_df.empty:
        out = option_df.copy()
        out["Underlying Adj Close"] = pd.NA
        return out

    return pd.merge_asof(
        option_df.sort_values("DateTime"),
        underlying_df.sort_values("DateTime"),
        on="DateTime",
        direction="backward",
    )


def compute_rth_close_series_from_minutes(minute_df: pd.DataFrame, rth_cutoff_hhmm: str = "13:00") -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame(columns=["Date", "RTH Close"])

    cutoff_time = pd.to_datetime(rth_cutoff_hhmm).time()
    tmp = minute_df.copy()
    if tmp["DateTime"].dt.tz is None:
        tmp["DateTime"] = tmp["DateTime"].dt.tz_localize(LA_TZ)

    tmp["LA_Date"] = tmp["DateTime"].dt.date
    tmp["LA_Time"] = tmp["DateTime"].dt.time
    tmp = tmp[tmp["LA_Time"] <= cutoff_time].sort_values("DateTime")

    rth = tmp.groupby("LA_Date", as_index=False).tail(1)[["LA_Date", "Close"]]
    rth = rth.rename(columns={"Close": "RTH Close"})
    rth["Date"] = pd.to_datetime(rth["LA_Date"])
    return rth[["Date", "RTH Close"]].sort_values("Date").reset_index(drop=True)


def apply_rth_close_override_to_daily(df_daily: pd.DataFrame, rth_close_by_date: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily

    out = df_daily.copy()
    out = out.merge(rth_close_by_date, how="left", left_on="DateTime", right_on="Date")

    if "Date" in out.columns:
        out = out.drop(columns=["Date"])

    out["Close (Polygon Daily)"] = out["Close"]
    out["Close"] = out["RTH Close"].combine_first(out["Close"])
    return out


def add_pnl_and_extremes(df: pd.DataFrame, contracts: int):
    if df.empty:
        raise ValueError("No rows returned after filtering.")

    entry_price = float(df.iloc[0]["Close"])
    exit_price = float(df.iloc[-1]["Close"])

    df["Profit%"] = (df["Close"] - entry_price) / entry_price * 100
    df["PnL ($)"] = (df["Close"] - entry_price) * contracts * 100

    peak_col = "High" if "High" in df.columns else "Close"

    if len(df) >= 2:
        df_after = df.iloc[1:].copy()
        peak_idx = df_after[peak_col].idxmax()
    else:
        peak_idx = df.index[0]

    peak_price = float(df.loc[peak_idx, peak_col])
    peak_time = df.loc[peak_idx, "DateTime"]

    entry_to_exit_pct = (exit_price - entry_price) / entry_price * 100
    entry_to_peak_pct = (peak_price - entry_price) / entry_price * 100

    return (
        df,
        entry_price,
        exit_price,
        entry_to_exit_pct,
        peak_idx,
        peak_time,
        peak_price,
        entry_to_peak_pct,
        peak_col,
    )


def lowest_low_before_peak(df: pd.DataFrame, entry_price: float, peak_idx: int) -> tuple[float, float, object, int]:
    low_col = "Low" if "Low" in df.columns else "Close"
    if len(df) < 2:
        return entry_price, 0.0, df.iloc[0]["DateTime"], int(df.index[0])

    try:
        peak_pos = df.index.get_loc(peak_idx)
    except Exception:
        peak_pos = len(df) - 1

    peak_pos = max(peak_pos, 1)
    window = df.iloc[1:peak_pos + 1].copy()
    if window.empty:
        return entry_price, 0.0, df.iloc[0]["DateTime"], int(df.index[0])

    min_idx = window[low_col].idxmin()
    min_price = float(df.loc[min_idx, low_col])
    min_time = df.loc[min_idx, "DateTime"]
    pct = (min_price - entry_price) / entry_price * 100
    return min_price, pct, min_time, int(min_idx)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def fmt_ts(x, timespan: str) -> str:
    if timespan == "day":
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(x)


def pct_class(p: float) -> str:
    # positive => green, zero => green, negative => red
    return "pct-green" if p >= 0 else "pct-red"


def kpi_card(title: str, value_line1: str, value_line2: str | None = None, pct: float | None = None):
    sub_html = ""
    if value_line2 is not None:
        cls = ""
        if pct is not None:
            cls = pct_class(pct)
        sub_html = f'<div class="kpi-sub {cls}">{value_line2}</div>'
    return f"""
      <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value_line1}</div>
        {sub_html}
      </div>
    """


# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔧 Settings")

API_KEY = st.sidebar.text_input("Polygon API Key", type="password")

st.sidebar.subheader("Contract")
ticker = st.sidebar.text_input("Underlying Ticker", "QQQ").upper()
opt_type = st.sidebar.radio("Call / Put", ["C", "P"], horizontal=True)
strike = st.sidebar.number_input("Strike", value=400.0, step=0.5, min_value=0.0)
expiry = st.sidebar.date_input("Expiration", value=date(2025, 3, 15))

st.sidebar.subheader("Window")
timespan = st.sidebar.radio("Timespan", ["minute", "day"], horizontal=True)

entry_date = st.sidebar.date_input("Entry Date", value=date(2025, 3, 12))
entry_time = st.sidebar.time_input("Entry Time (LA)", value=dtime(6, 30))

exit_date = st.sidebar.date_input("Exit Date", value=date(2025, 3, 12))
exit_time = st.sidebar.time_input("Exit Time (LA)", value=dtime(13, 0))

contracts = st.sidebar.number_input("Contracts", value=1, step=1, min_value=1)

underlying_adjusted = st.sidebar.checkbox("Underlying uses Adjusted prices", value=True)

use_rth_close_for_daily = st.sidebar.checkbox(
    "Day mode: Use RTH Close (13:00 LA) instead of Polygon Daily Close",
    value=True
)

run = st.sidebar.button("Run")


# =========================
# MAIN
# =========================
st.title("📈 Polygon Options P&L — Option + Underlying (Adjusted)")

if not run:
    st.info("Set inputs on the left and click **Run**.")
    st.stop()

if not API_KEY:
    st.error("Please enter your Polygon API key.")
    st.stop()

entry_dt = pd.Timestamp(datetime.combine(entry_date, entry_time), tz=LA_TZ)
exit_dt = pd.Timestamp(datetime.combine(exit_date, exit_time), tz=LA_TZ)

if entry_dt > exit_dt:
    st.error("Entry date/time must be before Exit date/time.")
    st.stop()

option_symbol = build_option_symbol(
    ticker=ticker,
    expiry=expiry.strftime("%Y-%m-%d"),
    strike=float(strike),
    opt_type=opt_type,
)

st.subheader(f"Symbol: `{option_symbol}`")

start_date_str = str(entry_date)
end_date_str = str(exit_date)

with st.spinner("Fetching option + underlying bars from Polygon…"):
    try:
        option_raw = fetch_bars(API_KEY, option_symbol, timespan, start_date_str, end_date_str, adjusted=None)
        underlying_raw = fetch_bars(API_KEY, ticker, timespan, start_date_str, end_date_str, adjusted=underlying_adjusted)
    except requests.HTTPError as e:
        st.error(f"Polygon API error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

if not option_raw:
    st.warning("No option bars returned. Try widening the date range or confirm the contract traded.")
    st.stop()

option_df = clean_option_df(option_raw, timespan)
underlying_df = clean_underlying_df(underlying_raw, timespan) if underlying_raw else pd.DataFrame(columns=["DateTime", "Underlying Adj Close"])

if timespan == "minute":
    option_df = option_df[(option_df["DateTime"] >= entry_dt) & (option_df["DateTime"] <= exit_dt)].reset_index(drop=True)
    underlying_df = underlying_df[(underlying_df["DateTime"] <= exit_dt)].reset_index(drop=True)

    if option_df.empty:
        st.warning("No minute option bars inside your exact entry/exit time window. Try widening the times.")
        st.stop()

if timespan == "day" and use_rth_close_for_daily:
    with st.spinner("Computing RTH Close (13:00 LA) from minute bars for day mode…"):
        try:
            option_minute_raw = fetch_bars(API_KEY, option_symbol, "minute", start_date_str, end_date_str, adjusted=None)
        except Exception as e:
            st.warning(f"Could not fetch minute bars for RTH close override; using Polygon daily close instead. ({e})")
            option_minute_raw = []

    if option_minute_raw:
        option_minute_df = clean_option_df(option_minute_raw, "minute")
        rth_by_date = compute_rth_close_series_from_minutes(option_minute_df, rth_cutoff_hhmm="13:00")
        option_df = apply_rth_close_override_to_daily(option_df, rth_by_date)

df = add_underlying_price(option_df, underlying_df)

(
    df,
    entry_price,
    exit_price,
    entry_to_exit_pct,
    peak_idx,
    peak_time,
    peak_price,
    entry_to_peak_pct,
    peak_col,
) = add_pnl_and_extremes(df, contracts)

min_before_peak_price, min_before_peak_pct, min_before_peak_time, min_before_peak_idx = lowest_low_before_peak(
    df=df,
    entry_price=entry_price,
    peak_idx=peak_idx,
)

# =========================
# KPIs (3 cards) — percent color conditional
# =========================
c1, c2, c3 = st.columns([1.7, 1.3, 1.3])

with c1:
    st.markdown(
        kpi_card(
            "Entry→Exit",
            f"${entry_price:,.2f} → ${exit_price:,.2f}",
            f"{entry_to_exit_pct:,.2f}%",
            pct=entry_to_exit_pct,
        ),
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        kpi_card(
            "Peak after entry",
            f"${peak_price:,.2f}",
            f"{entry_to_peak_pct:,.2f}%",
            pct=entry_to_peak_pct,
        ),
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        kpi_card(
            "Before-Peak Low",
            f"${min_before_peak_price:,.2f}",
            f"{min_before_peak_pct:,.2f}%",
            pct=min_before_peak_pct,
        ),
        unsafe_allow_html=True
    )

st.markdown(
    f"""
    <div class="kpi-note">
      Peak at: <code>{fmt_ts(peak_time, timespan)}</code>
      &nbsp;&nbsp;•&nbsp;&nbsp;
      Before-peak low at: <code>{fmt_ts(min_before_peak_time, timespan)}</code>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# TABLE (highlight peak + lowest-before-peak)
# =========================
st.subheader("📋 Bars (Option + Underlying Adjusted Close)")

def highlight_rows(row):
    is_peak = row.name == peak_idx
    is_before_peak_low = row.name == min_before_peak_idx

    if is_peak and is_before_peak_low:
        style = "background-color:#6d28d9;color:#ffffff;font-weight:700"
    elif is_peak:
        style = "background-color:#166534;color:#ffffff;font-weight:700"
    elif is_before_peak_low:
        style = "background-color:#991b1b;color:#ffffff;font-weight:700"
    else:
        style = ""
    return [style] * len(row)

display_df = df.copy()
if timespan == "minute":
    display_df["DateTime"] = display_df["DateTime"].dt.strftime("%Y-%m-%d %H:%M")
else:
    display_df["DateTime"] = pd.to_datetime(display_df["DateTime"]).dt.strftime("%Y-%m-%d")

styled = display_df.style.apply(highlight_rows, axis=1).format({
    "Open": "{:.2f}",
    "High": "{:.2f}",
    "Low": "{:.2f}",
    "Close": "{:.2f}",
    "Volume": "{:,.0f}",
    "Underlying Adj Close": "{:.2f}",
    "Profit%": "{:.2f}%",
    "PnL ($)": "${:.2f}",
    "RTH Close": "{:.2f}",
    "Close (Polygon Daily)": "{:.2f}",
})

st.dataframe(styled, use_container_width=True)

st.download_button(
    "Download CSV",
    data=df_to_csv_bytes(display_df),
    file_name=f"{option_symbol}_with_underlying_adj.csv",
    mime="text/csv",
)

# =========================
# CHART
# =========================
st.subheader("📊 Chart")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DateTime"], df["Close"], label="Option Close", marker="o")

ax.scatter([peak_time], [peak_price], marker="D", label=f"Peak after entry ({peak_col})", zorder=3)
ax.scatter([min_before_peak_time], [min_before_peak_price], marker="D", label="Before-peak low", zorder=3)

ax.set_ylabel("Option Price ($)")
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlabel("Los Angeles Time" if timespan == "minute" else "Date (UTC calendar day)")
st.pyplot(fig)

st.success("Done.")
