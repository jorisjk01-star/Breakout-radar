"""
Micro-Cap Breakout Radar & Screener
------------------------------------
Een Streamlit dashboard dat (penny)stocks/micro-caps scant en rankt op basis
van een 'Breakout Score' (0-100), opgebouwd uit 4 pijlers:
  1. Volume & Float Score      (max 30 pt)
  2. Technische Analyse Score  (max 30 pt)
  3. Nieuws & Sentiment Score  (max 20 pt)
  4. Short Squeeze & Prijsactie Score (max 20 pt)

100% gratis databronnen (yfinance). Geen API keys nodig.
"""

import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIG & PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Micro-Cap Breakout Radar",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_TTL = 60  # seconden - ververst elke minuut

DEFAULT_TICKERS = [
    "DFNS", "CPHI", "VIVK", "KIDZ", "JUNS",
    "SOUN", "BBAI", "HOLO", "MVIS", "MULN",
]

# Kleine CSS-tweak zodat het ook prettig oogt op mobiel (iPhone/Safari)
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
        @media (max-width: 600px) {
            [data-testid="stMetricValue"] { font-size: 1.1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 2. DATA FETCHING (gecachet, robuust tegen missende data / errors)
# =============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_history(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Haalt historische koersdata op. Geeft lege DataFrame terug bij fout."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["Close", "Volume"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    """Haalt fundamentele info op (float, short interest, etc.)."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_news(ticker: str) -> list:
    """Haalt recente nieuwsberichten op via yfinance."""
    try:
        news = yf.Ticker(ticker).news
        return news if isinstance(news, list) else []
    except Exception:
        return []


# =============================================================================
# 3. INDICATOR BEREKENINGEN
# =============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50, rsi)
    return pd.Series(rsi, index=series.index).fillna(50)


def compute_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["BB_MID"] = df["Close"].rolling(window).mean()
    df["BB_STD"] = df["Close"].rolling(window).std()
    df["BB_UPPER"] = df["BB_MID"] + num_std * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - num_std * df["BB_STD"]
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]
    return df


def safe_get(d: dict, key: str, default=None):
    val = d.get(key, default)
    return val if val is not None else default


# =============================================================================
# 4. SCORING PIJLERS
# =============================================================================

def score_volume_float(df: pd.DataFrame, info: dict):
    """Pijler 1: Volume & Float — max 30 pt."""
    if df.empty or len(df) < 21:
        return 0.0, {"Status": "Onvoldoende koersdata"}

    avg_vol_20 = df["Volume"].iloc[-21:-1].mean()
    current_vol = df["Volume"].iloc[-1]
    rel_vol = (current_vol / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else 0.0

    # Volume-component (max 20 pt)
    if rel_vol >= 3.0:
        vol_score = 20.0
    elif rel_vol >= 2.5:
        vol_score = 15.0 + (rel_vol - 2.5) / 0.5 * 5.0
    elif rel_vol >= 1.5:
        vol_score = 5.0 + (rel_vol - 1.5) / 1.0 * 10.0
    else:
        vol_score = max(0.0, rel_vol / 1.5 * 5.0)

    # Float-component (max 10 pt bonus)
    float_shares = safe_get(info, "floatShares") or safe_get(info, "sharesOutstanding")
    float_m = float_shares / 1_000_000 if float_shares else None
    float_score = 0.0
    if float_m is not None:
        if float_m < 10:
            float_score = 10.0
        elif float_m < 20:
            float_score = 6.0
        elif float_m < 50:
            float_score = 3.0

    total = round(min(30.0, vol_score + float_score), 1)
    detail = {
        "RelVol (x t.o.v. 20d gem.)": round(rel_vol, 2),
        "Volume subscore": round(vol_score, 1),
        "Float (mln aandelen)": round(float_m, 2) if float_m is not None else "Onbekend",
        "Float bonus": round(float_score, 1),
    }
    return total, detail


def score_technical(df: pd.DataFrame):
    """Pijler 2: Technische Analyse (RSI + Bollinger Band squeeze/breakout) — max 30 pt."""
    if df.empty or len(df) < 25:
        return 0.0, {"Status": "Onvoldoende koersdata"}

    bb = compute_bollinger(df)
    rsi_series = compute_rsi(df["Close"])
    rsi = rsi_series.iloc[-1]

    # RSI-momentum (max 15 pt) — optimale zone 50-75
    if 50 <= rsi <= 75:
        rsi_score = 15.0
    elif 40 <= rsi < 50 or 75 < rsi <= 85:
        rsi_score = 8.0
    else:
        rsi_score = 2.0

    # Bollinger Band squeeze + breakout (max 15 pt)
    width = bb["BB_WIDTH"].dropna()
    squeeze, breakout = False, False
    bb_score = 0.0
    if len(width) > 20:
        current_width = width.iloc[-1]
        hist_window = width.iloc[-60:-1] if len(width) > 60 else width.iloc[:-1]
        hist_avg = hist_window.mean() if len(hist_window) > 0 else current_width
        squeeze = bool(current_width < hist_avg * 0.7)

        price, upper = bb["Close"].iloc[-1], bb["BB_UPPER"].iloc[-1]
        prev_price, prev_upper = bb["Close"].iloc[-2], bb["BB_UPPER"].iloc[-2]
        breakout = bool(price > upper and prev_price <= prev_upper)

        if squeeze and breakout:
            bb_score = 15.0
        elif breakout:
            bb_score = 10.0
        elif squeeze:
            bb_score = 6.0

    total = round(rsi_score + bb_score, 1)
    detail = {
        "RSI (14)": round(float(rsi), 1) if not pd.isna(rsi) else "N/A",
        "RSI subscore": round(rsi_score, 1),
        "BB Squeeze gedetecteerd": "Ja" if squeeze else "Nee",
        "BB Breakout (boven bovenband)": "Ja" if breakout else "Nee",
        "BB subscore": round(bb_score, 1),
    }
    return total, detail


def score_news_sentiment(news: list, df: pd.DataFrame):
    """Pijler 3: Nieuws & Sentiment — max 20 pt.
    Let op: er is geen gratis officiële Stocktwits/Reddit API zonder key.
    Als proxy voor sociale-media-aandacht gebruiken we (a) recente
    nieuwsfrequentie via yfinance en (b) een volume-uitschieter t.o.v.
    de afgelopen week, wat vaak samengaat met verhoogde online chatter.
    """
    now = datetime.utcnow()
    recent_count, hype_hits = 0, 0
    hype_keywords = [
        "surge", "soar", "breakout", "fda", "approval", "contract",
        "partnership", "merger", "acquisition", "patent", "record",
        "squeeze", "uplist", "offering", "buyback",
    ]

    for item in news:
        try:
            ts = item.get("providerPublishTime")
            title = (item.get("title") or "").lower()
            if ts:
                pub_dt = datetime.utcfromtimestamp(ts)
                if (now - pub_dt) <= timedelta(days=3):
                    recent_count += 1
                    if any(k in title for k in hype_keywords):
                        hype_hits += 1
        except Exception:
            continue

    # Nieuwsfrequentie (max 12 pt)
    if recent_count >= 3:
        news_score = 12.0
    elif recent_count == 2:
        news_score = 8.0
    elif recent_count == 1:
        news_score = 4.0
    else:
        news_score = 0.0

    # Hype-keyword bonus (max 8 pt)
    hype_score = min(8.0, hype_hits * 4.0)

    total = round(min(20.0, news_score + hype_score), 1)
    detail = {
        "Nieuwsitems laatste 3 dagen": recent_count,
        "Nieuwsscore": round(news_score, 1),
        "Hype-keywords gevonden": hype_hits,
        "Hype bonus": round(hype_score, 1),
        "Let op": "Proxy o.b.v. nieuws + volume (geen gratis Stocktwits/Reddit API beschikbaar)",
    }
    return total, detail


def score_short_price(df: pd.DataFrame, info: dict):
    """Pijler 4: Short Squeeze & Prijsactie — max 20 pt."""
    if df.empty or len(df) < 2:
        return 0.0, {"Status": "Onvoldoende koersdata"}

    change_pct = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100

    # Prijsactie (max 10 pt)
    if 2 <= change_pct <= 20:
        price_score = 10.0
    elif 0 < change_pct < 2:
        price_score = 4.0
    elif 20 < change_pct <= 40:
        price_score = 7.0
    else:
        price_score = 0.0

    # Short interest (max 10 pt)
    short_pct_of_float = safe_get(info, "shortPercentOfFloat")
    short_score, short_display = 0.0, "Onbekend"
    if short_pct_of_float is not None:
        short_val = short_pct_of_float * 100
        short_display = round(short_val, 1)
        if short_val >= 20:
            short_score = 10.0
        elif short_val >= 10:
            short_score = 6.0
        elif short_val >= 5:
            short_score = 3.0

    total = round(price_score + short_score, 1)
    detail = {
        "Dagverandering %": round(float(change_pct), 2),
        "Prijsactie subscore": round(price_score, 1),
        "Short % of float": short_display,
        "Short subscore": round(short_score, 1),
    }
    return total, detail


# =============================================================================
# 5. AGGREGATIE PER TICKER
# =============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_ticker(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    try:
        df = fetch_history(ticker, period="3mo", interval="1d")
        info = fetch_info(ticker)
        news = fetch_news(ticker)

        vol_score, vol_detail = score_volume_float(df, info)
        tech_score, tech_detail = score_technical(df)
        news_score, news_detail = score_news_sentiment(news, df)
        short_score, short_detail = score_short_price(df, info)

        total = round(vol_score + tech_score + news_score + short_score, 1)
        last_price = float(df["Close"].iloc[-1]) if not df.empty else None
        company_name = safe_get(info, "shortName", ticker)

        return {
            "ticker": ticker,
            "naam": company_name,
            "totaal_score": total,
            "laatste_prijs": last_price,
            "volume_float_score": vol_score,
            "technisch_score": tech_score,
            "nieuws_score": news_score,
            "short_prijs_score": short_score,
            "detail": {
                "1. Volume & Float (max 30)": vol_detail,
                "2. Technische Analyse (max 30)": tech_detail,
                "3. Nieuws & Sentiment (max 20)": news_detail,
                "4. Short Squeeze & Prijsactie (max 20)": short_detail,
            },
            "data_ok": not df.empty,
        }
    except Exception as e:
        return {
            "ticker": ticker, "naam": ticker, "totaal_score": 0.0, "laatste_prijs": None,
            "volume_float_score": 0.0, "technisch_score": 0.0, "nieuws_score": 0.0,
            "short_prijs_score": 0.0, "detail": {"Fout": {"melding": str(e)}}, "data_ok": False,
        }


def analyze_multiple(tickers: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            t = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "ticker": t, "naam": t, "totaal_score": 0.0, "laatste_prijs": None,
                    "volume_float_score": 0.0, "technisch_score": 0.0, "nieuws_score": 0.0,
                    "short_prijs_score": 0.0, "detail": {"Fout": {"melding": str(e)}}, "data_ok": False,
                })
    results.sort(key=lambda r: r["totaal_score"], reverse=True)
    return results


# =============================================================================
# 6. UI HELPERS
# =============================================================================

def fmt_price(p):
    return f"${p:,.4f}" if isinstance(p, (int, float)) and p is not None else "N/A"


def score_color(score):
    if score >= 70:
        return "🟢"
    if score >= 45:
        return "🟡"
    return "🔴"


# =============================================================================
# 7. SIDEBAR
# =============================================================================

st.sidebar.title("🚀 Breakout Radar")
st.sidebar.caption("Instellingen & tickerlijst")

if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()

selected = st.sidebar.multiselect(
    "Actieve tickers",
    options=sorted(set(st.session_state.tickers + DEFAULT_TICKERS)),
    default=st.session_state.tickers,
)

extra = st.sidebar.text_input(
    "Extra tickers toevoegen (komma-gescheiden)",
    placeholder="bv. NVOS, ATNF",
)
if extra:
    new_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]
    selected = list(dict.fromkeys(selected + new_tickers))

st.session_state.tickers = selected if selected else DEFAULT_TICKERS.copy()

if st.sidebar.button("🔄 Ververs data nu"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(
    f"Data wordt automatisch elke {CACHE_TTL} seconden herberekend "
    "(gecachet via st.cache_data). Klik 'Ververs data' om direct te forceren."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Uitsluitend educatief/informatief. Geen beleggingsadvies. "
    "Gratis data via Yahoo Finance (yfinance) kan vertraagd, onvolledig of "
    "onnauwkeurig zijn — vooral voor micro-caps."
)

# =============================================================================
# 8. HOOFDPAGINA — TABS
# =============================================================================

st.title("🚀 Micro-Cap Breakout Radar & Screener")
st.caption("Scant en rankt micro-caps op potentiële uitbraakkans (Breakout Score 0-100)")

tab1, tab2, tab3 = st.tabs(
    ["📡 Realtime Screener & Top Picks", "📈 Technische Analyse", "🧪 Backtest Simulator"]
)

# -----------------------------------------------------------------------
# TAB 1 — SCREENER
# -----------------------------------------------------------------------
with tab1:
    tickers_to_scan = st.session_state.tickers

    if not tickers_to_scan:
        st.warning("Voeg minstens één ticker toe in de zijbalk.")
    else:
        with st.spinner(f"Analyseren van {len(tickers_to_scan)} tickers..."):
            results = analyze_multiple(tickers_to_scan)

        valid_results = [r for r in results if r["data_ok"]]
        failed_results = [r for r in results if not r["data_ok"]]

        st.subheader("🏆 Top 5 Uitbraakkandidaten")
        top5 = valid_results[:5]
        if top5:
            cols = st.columns(len(top5))
            for col, r in zip(cols, top5):
                with col:
                    st.metric(
                        label=f"{score_color(r['totaal_score'])} {r['ticker']}",
                        value=f"{r['totaal_score']:.1f} / 100",
                        delta=fmt_price(r["laatste_prijs"]),
                    )
        else:
            st.info("Nog geen geldige resultaten om te tonen.")

        st.markdown("---")
        st.subheader("📋 Overzichtstabel — gesorteerd op Totaalscore")

        if valid_results:
            table_df = pd.DataFrame([
                {
                    "Ticker": r["ticker"],
                    "Naam": r["naam"],
                    "Prijs": fmt_price(r["laatste_prijs"]),
                    "Totaalscore": r["totaal_score"],
                    "Volume & Float (30)": r["volume_float_score"],
                    "Technisch (30)": r["technisch_score"],
                    "Nieuws & Sentiment (20)": r["nieuws_score"],
                    "Short & Prijsactie (20)": r["short_prijs_score"],
                }
                for r in valid_results
            ])
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Totaalscore": st.column_config.ProgressColumn(
                        "Totaalscore", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )
        if failed_results:
            with st.expander(f"⚠️ {len(failed_results)} ticker(s) konden niet geladen worden"):
                for r in failed_results:
                    st.write(f"- **{r['ticker']}**: geen (volledige) data beschikbaar via yfinance.")

        st.markdown("---")
        st.subheader("🔍 Sub-scores per aandeel")
        if valid_results:
            pick = st.selectbox(
                "Kies een ticker om de score-opbouw te bekijken",
                options=[r["ticker"] for r in valid_results],
            )
            picked = next(r for r in valid_results if r["ticker"] == pick)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volume & Float", f"{picked['volume_float_score']:.1f} / 30")
            c2.metric("Technisch", f"{picked['technisch_score']:.1f} / 30")
            c3.metric("Nieuws & Sentiment", f"{picked['nieuws_score']:.1f} / 20")
            c4.metric("Short & Prijsactie", f"{picked['short_prijs_score']:.1f} / 20")

            for pijler_naam, pijler_detail in picked["detail"].items():
                with st.expander(f"Details — {pijler_naam}"):
                    st.table(pd.DataFrame(pijler_detail.items(), columns=["Metric", "Waarde"]))

# -----------------------------------------------------------------------
# TAB 2 — TECHNISCHE ANALYSE (CANDLESTICK + BOLLINGER + VOLUME)
# -----------------------------------------------------------------------
with tab2:
    st.subheader("📈 Interactieve Candlestick Grafiek")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        chart_ticker = st.selectbox(
            "Ticker", options=st.session_state.tickers, key="chart_ticker"
        )
    with col_b:
        period_choice = st.selectbox(
            "Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=1
        )
    with col_c:
        interval_choice = st.selectbox("Interval", ["1d", "1h"], index=0)

    chart_df = fetch_history(chart_ticker, period=period_choice, interval=interval_choice)

    if chart_df.empty or len(chart_df) < 20:
        st.warning(
            f"Onvoldoende data gevonden voor {chart_ticker} met deze periode/interval. "
            "Probeer een langere periode of dagelijkse interval."
        )
    else:
        bb_df = compute_bollinger(chart_df)
        rsi_series = compute_rsi(chart_df["Close"])

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.03,
            subplot_titles=("Koers + Bollinger Bands", "Volume", "RSI (14)"),
        )

        fig.add_trace(go.Candlestick(
            x=bb_df.index, open=bb_df["Open"], high=bb_df["High"],
            low=bb_df["Low"], close=bb_df["Close"], name="Koers",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=bb_df.index, y=bb_df["BB_UPPER"], name="BB Boven",
            line=dict(color="rgba(255,99,132,0.6)", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bb_df.index, y=bb_df["BB_MID"], name="BB Midden (SMA20)",
            line=dict(color="rgba(153,153,153,0.6)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bb_df.index, y=bb_df["BB_LOWER"], name="BB Onder",
            line=dict(color="rgba(54,162,235,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(54,162,235,0.05)",
        ), row=1, col=1)

        vol_colors = np.where(bb_df["Close"] >= bb_df["Open"], "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(
            x=bb_df.index, y=bb_df["Volume"], name="Volume", marker_color=vol_colors,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=bb_df.index, y=rsi_series, name="RSI", line=dict(color="#ab47bc", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        fig.update_layout(
            height=700, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_yaxes(title_text="Prijs ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        latest = bb_df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Laatste koers", fmt_price(latest["Close"]))
        m2.metric("RSI (14)", f"{rsi_series.iloc[-1]:.1f}")
        bb_pos = "Boven bovenband 🚀" if latest["Close"] > latest["BB_UPPER"] else (
            "Onder onderband 📉" if latest["Close"] < latest["BB_LOWER"] else "Binnen band")
        m3.metric("BB-positie", bb_pos)
        m4.metric("Volume vandaag", f"{int(latest['Volume']):,}")

# -----------------------------------------------------------------------
# TAB 3 — BACKTEST SIMULATOR
# -----------------------------------------------------------------------
with tab3:
    st.subheader("🧪 Backtest Simulator")
    st.caption(
        "Test hoeveel rendement een RelVol/Float-signaal in het verleden had opgeleverd. "
        "Let op: float wordt gebruikt als huidige (statische) snapshot, historische "
        "float-data is via gratis bronnen niet beschikbaar."
    )

    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    with bt_col1:
        bt_ticker = st.text_input("Ticker", value="SOUN").strip().upper()
    with bt_col2:
        bt_period = st.selectbox("Backtest-periode", ["6mo", "1y", "2y"], index=1)
    with bt_col3:
        bt_relvol = st.number_input("Min. RelVol (x)", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    with bt_col4:
        bt_holding = st.number_input("Houdperiode (dagen)", min_value=1, max_value=30, value=5, step=1)

    bt_float_max = st.number_input(
        "Max. Float (mln aandelen, 0 = geen filter)", min_value=0, max_value=1000, value=20, step=1
    )

    if st.button("▶️ Start Backtest", type="primary"):
        if not bt_ticker:
            st.warning("Voer een ticker in.")
        else:
            with st.spinner(f"Backtest draaien voor {bt_ticker}..."):
                hist = fetch_history(bt_ticker, period=bt_period, interval="1d")
                info = fetch_info(bt_ticker)

            if hist.empty or len(hist) < 25:
                st.error(f"Onvoldoende historische data gevonden voor {bt_ticker}.")
            else:
                hist = hist.copy()
                hist["AvgVol20"] = hist["Volume"].rolling(20).mean().shift(1)
                hist["RelVol"] = hist["Volume"] / hist["AvgVol20"]

                float_shares = safe_get(info, "floatShares") or safe_get(info, "sharesOutstanding")
                float_m = float_shares / 1_000_000 if float_shares else None
                float_ok = True
                float_note = None
                if bt_float_max > 0:
                    if float_m is not None:
                        float_ok = float_m <= bt_float_max
                    else:
                        float_note = "Float onbekend voor deze ticker — floatfilter is genegeerd."

                trades = []
                n = len(hist)
                for i in range(20, n - int(bt_holding)):
                    row = hist.iloc[i]
                    if pd.isna(row["RelVol"]):
                        continue
                    if row["RelVol"] >= bt_relvol and float_ok:
                        entry_price = row["Close"]
                        exit_price = hist.iloc[i + int(bt_holding)]["Close"]
                        ret_pct = (exit_price / entry_price - 1) * 100
                        trades.append({
                            "Signaaldatum": hist.index[i].strftime("%Y-%m-%d"),
                            "RelVol": round(float(row["RelVol"]), 2),
                            "Entry ($)": round(float(entry_price), 4),
                            f"Exit na {int(bt_holding)}d ($)": round(float(exit_price), 4),
                            "Rendement %": round(float(ret_pct), 2),
                        })

                if float_note:
                    st.info(float_note)

                if not trades:
                    st.warning(
                        f"Geen signalen gevonden voor {bt_ticker} met RelVol ≥ {bt_relvol}x "
                        f"in de gekozen periode."
                    )
                else:
                    trades_df = pd.DataFrame(trades)
                    returns = trades_df["Rendement %"]

                    st.success(f"{len(trades)} signalen gevonden voor {bt_ticker}.")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Aantal signalen", len(trades))
                    s2.metric("Win rate", f"{(returns > 0).mean() * 100:.1f}%")
                    s3.metric("Gem. rendement", f"{returns.mean():.2f}%")
                    s4.metric("Max / Min", f"{returns.max():.1f}% / {returns.min():.1f}%")

                    hist_fig = go.Figure()
                    hist_fig.add_trace(go.Histogram(
                        x=returns, nbinsx=20, marker_color="#26a69a", name="Rendement per signaal"
                    ))
                    hist_fig.update_layout(
                        title=f"Verdeling rendement na {int(bt_holding)} dagen — {bt_ticker}",
                        xaxis_title="Rendement (%)", yaxis_title="Aantal signalen",
                        height=350, margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)

                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "⚠️ Dit is een vereenvoudigde signaaltest, geen volledige "
                        "portfoliosimulatie (posities overlappen mogelijk in tijd)."
                    )
