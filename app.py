# app.py — Polygon Options P&L (Mobile-first UI)
# - Desktop: inputs in sidebar
# - Mobile: inputs in main-page expander (no sidebar needed) + sidebar auto-hidden
# - Secrets-safe API key (works on Cloud + local)
# - Optional APP_PASSWORD gate (from secrets)
# - Remembers last inputs via URL query params
# - KPIs: Entry→Exit, Peak after entry, Before-Peak Low
# - Adds Underlying % move at option peak
# - Table: highlight Peak row (green) + Lowest-before-peak row (red)
# - Chart: Entry/Peak/Low/Exit markers use distinct colors

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
# MOBILE-FIRST CSS + HIDE SIDEBAR ON MOBILE
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

        /* hide sidebar */
        section[data-testid="stSidebar"] { display: none !important; }
        button[kind="header"] { display: none !important; } /* hides sidebar toggle in some themes */
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# QUERY PARAMS (Remember last inputs)
# =========================
def qp_get_all() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def qp_set(**kwargs):
    clean = {k: str(v) for k, v in kwargs.items() if v is not None}
    try:
        st.query_params.clear()
        for k, v in clean.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**clean)

def parse_date(s: str, fallback: date) -> date:
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return fallback

def parse_time(s: str, fallback: dtime) -> dtime:
    try:
        return datetime.strptime(str(s), "%H:%M").time()
    except Exception:
        return fallback

def parse_bool(s: str, fallback: bool) -> bool:
    if s is None:
        return fallback
    return str(s).lower() in ("1", "true", "yes", "y", "t", "on")

def parse_int(s: str, fallback: int) -> int:
    try:
        return int(s)
    except Exception:
        return fallback

def parse_float(s: str, fallback: float) -> float:
    try:
        return float(s)
    except Exception:
        return fallback

def init_session_defaults_from_qp():
    qp = qp_get_all()
    defaults = {
        "ticker": "QQQ",
        "opt_type": "C",
        "strike": 400.0,
        "expiry": date(2025, 3, 15),
        "timespan": "minute",
        "entry_date": date(2025, 3, 12),
        "entry_time": dtime(6, 30),
        "exit_date": date(2025, 3, 12),
        "exit_time": dtime(13, 0),
        "contracts": 1,
        "underlying_adjusted": True,
        "use_rth_close_for_daily": True,
    }

    if "ticker" not in st.session_state:
        st.session_state.ticker = str(qp.get("ticker", defaults["ticker"])).upper()

    if "opt_type" not in st.session_state:
        ot = str(qp.get("opt_type", defaults["opt_type"])).upper()
        st.session_state.opt_type = ot if ot in ("C", "P") else defaults["opt_type"]

    if "strike" not in st.session_state:
        st.session_state.strike = parse_float(qp.get("strike"), defaults["strike"])

    if "expiry" not in st.session_state:
        st.session_state.expiry = parse_date(qp.get("expiry"), defaults["expiry"])

    if "timespan" not in st.session_state:
        ts = str(qp.get("timespan", defaults["timespan"]))
        st.session_state.timespan = ts if ts in ("minute", "day") else defaults["timespan"]

    if "entry_date" not in st.session_state:
        st.session_state.entry_date = parse_date(qp.get("entry_date"), defaults["entry_date"])

    if "entry_time" not in st.session_state:
        st.session_state.entry_time = parse_time(qp.get("entry_time"), defaults["entry_time"])

    if "exit_date" not in st.session_state:
        st.session_state.exit_date = parse_date(qp.get("exit_date"), defaults["exit_date"])

    if "exit_time" not in st.session_state:
        st.session_state.exit_time = parse_time(qp.get("exit_time"), defaults["exit_time"])

    if "contracts" not in st.session_state:
        st.session_state.contracts = parse_int(qp.get("contracts"), defaults["contracts"])

    if "underlying_adjusted" not in st.session_state:
        st.session_state.underlying_adjusted = parse_bool(qp.get("underlying_adjusted"), defaults["underlying_adjusted"])

    if "use_rth_close_for_daily" not in st.session_state:
        st.session_state.use_rth_close_for_daily = parse_bool(qp.get("use_rth_close_for_daily"), defaults["use_rth_close_for_daily"])

def save_inputs_to_qp():
    qp_set(
        ticker=st.session_state.ticker,
        opt_type=st.session_state.opt_type,
        strike=st.session_state.strike,
        expiry=st.session_state.expiry.strftime("%Y-%m-%d"),
        timespan=st.session_state.timespan,
        entry_date=st.session_state.entry_date.strftime("%Y-%m-%d"),
        entry_time=st.session_state.entry_time.strftime("%H:%M"),
        exit_date=st.session_state.exit_date.strftime("%Y-%m-%d"),
        exit_time=st.session_state.exit_time.strftime("%H:%M"),
        contracts=st.session_state.contracts,
        underlying_adjusted=str(st.session_state.underlying_adjusted).lower(),
        use_rth_close_for_daily=str(st.session_state.use_rth_close_for_daily).lower(),
    )

init_session_defaults_from_qp()

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
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    else:
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(LA_TZ)

    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    keep = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]].sort_values("DateTime").reset_index(drop=True)

def clean_underlying_df(raw_results, timespan: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_results)
    if timespan == "day":
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    else:
        df["DateTime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(LA_TZ)

    df = df.rename(columns={"c": "Underlying Adj Close"})
    return df[["DateTime", "Underlying Adj Close"]].sort_values("DateTime").reset_index(drop=True)

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

    rth = tmp.groupby("LA_Date", as_index=False).tail(1)[["LA_Date", "Close"]].rename(columns={"Close": "RTH Close"})
    rth["Date"] = pd.to_datetime(rth["LA_Date"])
    return rth[["Date", "RTH Close"]].sort_values("Date").reset_index(drop=True)

def apply_rth_close_override_to_daily(df_daily: pd.DataFrame, rth_close_by_date: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily
    out = df_daily.copy().merge(rth_close_by_date, how="left", left_on="DateTime", right_on="Date")
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
    peak_idx = df.iloc[1:][peak_col].idxmax() if len(df) >= 2 else df.index[0]
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
# TITLE
# =========================
st.title("📈 Polygon Options P&L — Option + Underlying (Adjusted)")

# =========================
# API KEY (secrets-safe)
# =========================
try:
    secret_key = st.secrets.get("POLYGON_API_KEY", "")
except Exception:
    secret_key = ""

# =========================
# INPUTS (sidebar + main expander) — prefix keys to avoid duplicates
# =========================
def render_inputs(container, prefix: str):
    container.subheader("Contract")

    ticker_ = container.text_input("Underlying Ticker", value=st.session_state.ticker, key=f"{prefix}_ticker").upper()
    opt_type_ = container.radio(
        "Call / Put", ["C", "P"],
        index=0 if st.session_state.opt_type == "C" else 1,
        key=f"{prefix}_opt_type",
        horizontal=True
    )
    strike_ = container.number_input(
        "Strike", value=float(st.session_state.strike),
        step=0.5, min_value=0.0, key=f"{prefix}_strike"
    )
    expiry_ = container.date_input("Expiration", value=st.session_state.expiry, key=f"{prefix}_expiry")

    container.subheader("Window")
    timespan_ = container.radio(
        "Timespan", ["minute", "day"],
        index=0 if st.session_state.timespan == "minute" else 1,
        key=f"{prefix}_timespan",
        horizontal=True
    )

    entry_date_ = container.date_input("Entry Date", value=st.session_state.entry_date, key=f"{prefix}_entry_date")
    entry_time_ = container.time_input("Entry Time (LA)", value=st.session_state.entry_time, key=f"{prefix}_entry_time")

    exit_date_ = container.date_input("Exit Date", value=st.session_state.exit_date, key=f"{prefix}_exit_date")
    exit_time_ = container.time_input("Exit Time (LA)", value=st.session_state.exit_time, key=f"{prefix}_exit_time")

    contracts_ = container.number_input("Contracts", value=int(st.session_state.contracts), step=1, min_value=1, key=f"{prefix}_contracts")

    container.subheader("Data")
    underlying_adjusted_ = container.checkbox(
        "Underlying uses Adjusted prices", value=bool(st.session_state.underlying_adjusted), key=f"{prefix}_underlying_adjusted"
    )
    use_rth_close_for_daily_ = container.checkbox(
        "Day mode: Use RTH Close (13:00 LA) instead of Polygon Daily Close",
        value=bool(st.session_state.use_rth_close_for_daily),
        key=f"{prefix}_use_rth_close_for_daily",
    )

    api_key_ = secret_key.strip()
    if not api_key_:
        api_key_ = container.text_input("Polygon API Key", type="password", key=f"{prefix}_manual_api_key").strip()

    run_ = container.button("Run", use_container_width=True, key=f"{prefix}_run")

    return (ticker_, opt_type_, strike_, expiry_,
            timespan_, entry_date_, entry_time_, exit_date_, exit_time_,
            contracts_, underlying_adjusted_, use_rth_close_for_daily_,
            api_key_, run_)

# Desktop sidebar (visible on desktop; hidden on mobile by CSS)
st.sidebar.header("🔧 Settings")
sidebar_vals = render_inputs(st.sidebar, prefix="s")

# Mobile-first expander
with st.expander("⚙️ Settings", expanded=True):
    main_vals = render_inputs(st, prefix="m")

# Prefer whichever Run was pressed
use_vals = main_vals
if sidebar_vals[-1]:
    use_vals = sidebar_vals
elif main_vals[-1]:
    use_vals = main_vals

(
    ticker, opt_type, strike, expiry,
    timespan, entry_date, entry_time, exit_date, exit_time,
    contracts, underlying_adjusted, use_rth_close_for_daily,
    API_KEY, run
) = use_vals

# Sync selected inputs back into session_state (so query params save the latest)
st.session_state.ticker = ticker
st.session_state.opt_type = opt_type
st.session_state.strike = float(strike)
st.session_state.expiry = expiry
st.session_state.timespan = timespan
st.session_state.entry_date = entry_date
st.session_state.entry_time = entry_time
st.session_state.exit_date = exit_date
st.session_state.exit_time = exit_time
st.session_state.contracts = int(contracts)
st.session_state.underlying_adjusted = bool(underlying_adjusted)
st.session_state.use_rth_close_for_daily = bool(use_rth_close_for_daily)

if not run:
    st.info("Set your inputs above and tap **Run**.")
    st.stop()

# Remember last inputs
save_inputs_to_qp()

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

# Underlying % move at option peak
under_entry = df.iloc[0].get("Underlying Adj Close", pd.NA)
under_peak = df.loc[peak_idx].get("Underlying Adj Close", pd.NA)
under_peak_pct = None
if pd.notna(under_entry) and pd.notna(under_peak) and float(under_entry) != 0:
    under_peak_pct = (float(under_peak) - float(under_entry)) / float(under_entry) * 100

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

extra_line = ""
if under_peak_pct is not None:
    cls = "pct-green" if under_peak_pct >= 0 else "pct-red"
    extra_line = f'&nbsp;&nbsp;•&nbsp;&nbsp; Underlying move at option peak: <span class="{cls}"><b>{under_peak_pct:,.2f}%</b></span>'

st.markdown(
    f"""
    <div class="kpi-note">
      Peak at: <code>{fmt_ts(peak_time, timespan)}</code>
      &nbsp;&nbsp;•&nbsp;&nbsp;
      Before-peak low at: <code>{fmt_ts(min_before_peak_time, timespan)}</code>
      {extra_line}
    </div>
    """,
    unsafe_allow_html=True,
)

# Table + Chart tabs
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

    entry_time_pt = df.iloc[0]["DateTime"]
    exit_time_pt = df.iloc[-1]["DateTime"]

    # Distinct marker colors for visibility
    ax.scatter([entry_time_pt], [entry_price], marker="o", s=90, color="deepskyblue", label="Entry", zorder=4)
    ax.scatter([peak_time], [peak_price], marker="D", s=120, color="orange", label=f"Peak ({peak_col})", zorder=5)
    ax.scatter([min_before_peak_time], [min_before_peak_price], marker="D", s=120, color="crimson", label="Before-peak low", zorder=5)
    ax.scatter([exit_time_pt], [exit_price], marker="s", s=90, color="violet", label="Exit", zorder=4)

    ax.set_ylabel("Option Price ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlabel("Los Angeles Time" if timespan == "minute" else "Date (UTC calendar day)")
    st.pyplot(fig, use_container_width=True)

st.success("Done.")
