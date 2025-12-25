# app.py — Polygon Options P&L (Mobile-first UI)
# - Desktop: inputs in sidebar
# - Mobile: inputs in main-page expander (no sidebar needed)
# - Secrets-safe API key (works on Cloud + local)
# - KPIs: Entry→Exit, Peak after entry, Before-Peak Low (percent colors)
# - Table: highlight Peak row (green) + Lowest-before-peak row (red)

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

LA_TZ = ZoneInfo("America/Los_Angeles")

st.set_page_config(page_title="Polygon Options P&L", page_icon="📈", layout="wide")

# -------------------------
# MOBILE-FIRST CSS
# -------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.9rem; padding-bottom: 2.2rem; }

      /* KPI cards */
      .kpi-card {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
      }
      .kpi-title { font-size: 0.85rem; color: #a3a3a3; margin-bottom: 6px; }
      .kpi-value { font-size: 2.05rem; font-weight: 850; line-height: 1.05; color: #f5f5f5; }
      .kpi-sub   { font-size: 1.02rem; margin-top: 6px; color: #d4d4d4; font-weight: 750; }

      .pct-green { color: #22c55e !important; }
      .pct-red   { color: #ef4444 !important; }

      .kpi-note { margin-top: 0.55rem; font-size: 0.95rem; color: #c7c7c7; }
      .kpi-note code { font-size: 0.95rem; }

      /* Dataframe readability */
      div[data-testid="stDataFrame"] * { font-size: 12px !important; }
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

      /* Mobile */
      @media (max-width: 900px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        /* stack columns */
        div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        .kpi-value { font-size: 1.65rem; }
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
        peak_idx = df.iloc[1:][peak_col].idxmax()
    else:
        peak_idx = df.index[0]

    peak_price = float(df.loc[peak_idx, peak_col])
    peak_time = df.loc[peak_idx, "DateTime"]

    entry_to_exit_pct = (exit_price - entry_price) / entry_price * 100
    entry_to_peak_pct = (peak_price - entry_price) / entry_price * 100

    return df, entry_price, exit_price, entry_to_exit_pct, peak_idx, peak_time, peak_price, entry_to_peak_pct, peak_col


def lowest_low_before_peak(df: pd.DataFrame, entry_price: float, peak_idx: int):
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
    return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")


def pct_class(p: float) -> str:
    return "pct-green" if p >= 0 else "pct-red"


def kpi_card(title: str, value_line1: str, value_line2: str, pct: float):
    return f"""
      <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value_line1}</div>
        <div class="kpi-sub {pct_class(pct)}">{value_line2}</div>
      </div>
    """


# =========================
# AUTH (optional) - if you enabled APP_PASSWORD in secrets
# =========================
APP_PASSWORD = ""
try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
except Exception:
    APP_PASSWORD = ""

if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        pwd = st.text_input("App Password", type="password")
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("Unlocked")
        else:
            st.info("Enter password to continue.")
            st.stop()

# =========================
# TITLE
# =========================
st.title("📈 Polygon Options P&L — Option + Underlying (Adjusted)")

# Detect "mobile" by screen width (approx). Streamlit doesn't provide exact device info,
# so we use a simple heuristic: show main-page Settings by default on all devices,
# but keep sidebar for desktop convenience too.
# (Users on desktop can ignore the expander and use sidebar.)
show_sidebar = True

# =========================
# API KEY (secrets-safe, local-safe)
# =========================
try:
    secret_key = st.secrets.get("POLYGON_API_KEY", "")
except Exception:
    secret_key = ""

# =========================
# INPUTS
# =========================
def render_inputs(container):
    container.subheader("Contract")
    ticker_ = container.text_input("Underlying Ticker", "QQQ").upper()
    opt_type_ = container.radio("Call / Put", ["C", "P"], horizontal=True)
    strike_ = container.number_input("Strike", value=400.0, step=0.5, min_value=0.0)
    expiry_ = container.date_input("Expiration", value=date(2025, 3, 15))

    container.subheader("Window")
    timespan_ = container.radio("Timespan", ["minute", "day"], horizontal=True)

    entry_date_ = container.date_input("Entry Date", value=date(2025, 3, 12))
    entry_time_ = container.time_input("Entry Time (LA)", value=dtime(6, 30))

    exit_date_ = container.date_input("Exit Date", value=date(2025, 3, 12))
    exit_time_ = container.time_input("Exit Time (LA)", value=dtime(13, 0))

    contracts_ = container.number_input("Contracts", value=1, step=1, min_value=1)

    container.subheader("Data")
    underlying_adjusted_ = container.checkbox("Underlying uses Adjusted prices", value=True)
    use_rth_close_for_daily_ = container.checkbox(
        "Day mode: Use RTH Close (13:00 LA) instead of Polygon Daily Close",
        value=True
    )

    # API key input only if no secret present
    api_key_ = secret_key.strip()
    if not api_key_:
        api_key_ = container.text_input("Polygon API Key", type="password").strip()

    run_ = container.button("Run", use_container_width=True)
    return (ticker_, opt_type_, strike_, expiry_,
            timespan_, entry_date_, entry_time_, exit_date_, exit_time_,
            contracts_, underlying_adjusted_, use_rth_close_for_daily_,
            api_key_, run_)

# Desktop convenience: sidebar inputs
sidebar_vals = None
if show_sidebar:
    st.sidebar.header("🔧 Settings")
    sidebar_vals = render_inputs(st.sidebar)

# Mobile-first: main-page Settings expander (always available)
with st.expander("⚙️ Settings", expanded=True):
    main_vals = render_inputs(st)

# Choose which set of inputs to use:
# - If user pressed Run in sidebar, use sidebar values.
# - Else if user pressed Run in main settings, use main values.
# - Else default to main values.
use_vals = main_vals
if sidebar_vals is not None and sidebar_vals[-1]:
    use_vals = sidebar_vals
elif main_vals[-1]:
    use_vals = main_vals

(
    ticker, opt_type, strike, expiry,
    timespan, entry_date, entry_time, exit_date, exit_time,
    contracts, underlying_adjusted, use_rth_close_for_daily,
    API_KEY, run
) = use_vals

# Stop until Run is pressed (either place)
if not run:
    st.info("Set your inputs above and tap **Run**.")
    st.stop()

if not API_KEY:
    st.error("Please enter your Polygon API key.")
    st.stop()

entry_dt = pd.Timestamp(datetime.combine(entry_date, entry_time), tz=LA_TZ)
exit_dt = pd.Timestamp(datetime.combine(exit_date, exit_time), tz=LA_TZ)
if entry_dt > exit_dt:
    st.error("Entry date/time must be before Exit date/time.")
    st.stop()

option_symbol = build_option_symbol(ticker, expiry.strftime("%Y-%m-%d"), float(strike), opt_type)
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

df, entry_price, exit_price, entry_to_exit_pct, peak_idx, peak_time, peak_price, entry_to_peak_pct, peak_col = add_pnl_and_extremes(df, contracts)
min_before_peak_price, min_before_peak_pct, min_before_peak_time, min_before_peak_idx = lowest_low_before_peak(df, entry_price, peak_idx)

# KPIs
c1, c2, c3 = st.columns([1.5, 1.25, 1.25])
with c1:
    st.markdown(
        kpi_card("Entry→Exit", f"${entry_price:,.2f} → ${exit_price:,.2f}", f"{entry_to_exit_pct:,.2f}%", entry_to_exit_pct),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card("Peak after entry", f"${peak_price:,.2f}", f"{entry_to_peak_pct:,.2f}%", entry_to_peak_pct),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Before-Peak Low", f"${min_before_peak_price:,.2f}", f"{min_before_peak_pct:,.2f}%", min_before_peak_pct),
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="kpi-note">
      Peak at: <code>{fmt_ts(peak_time, timespan)}</code>
      &nbsp;&nbsp;•&nbsp;&nbsp;
      Before-peak low at: <code>{fmt_ts(min_before_peak_time, timespan)}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# Table + Chart tabs (best for mobile)
def highlight_rows(row):
    is_peak = row.name == peak_idx
    is_low = row.name == min_before_peak_idx
    if is_peak and is_low:
        style = "background-color:#6d28d9;color:#ffffff;font-weight:700"
    elif is_peak:
        style = "background-color:#166534;color:#ffffff;font-weight:700"
    elif is_low:
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

tab_table, tab_chart = st.tabs(["📋 Table", "📊 Chart"])

with tab_table:
    st.subheader("📋 Bars (Option + Underlying Adjusted Close)")
    st.dataframe(styled, use_container_width=True, height=520)
    st.download_button(
        "Download CSV",
        data=df_to_csv_bytes(display_df),
        file_name=f"{option_symbol}_with_underlying_adj.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_chart:
    st.subheader("📊 Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["DateTime"], df["Close"], label="Option Close", marker="o")
    ax.scatter([peak_time], [peak_price], marker="D", label=f"Peak after entry ({peak_col})", zorder=3)
    ax.scatter([min_before_peak_time], [min_before_peak_price], marker="D", label="Before-peak low", zorder=3)
    ax.set_ylabel("Option Price ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlabel("Los Angeles Time" if timespan == "minute" else "Date (UTC calendar day)")
    st.pyplot(fig, use_container_width=True)

st.success("Done.")
