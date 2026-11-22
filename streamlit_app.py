import io
import zipfile

import pandas as pd
import streamlit as st

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
BENCH = "^CRSLDX"
RRG_N, RRG_M = 63, 10
CUT_WEAKEST = 3
MIN_SECTOR_STOCKS = 3
RS_TOP_PCT = 0.70


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_universe():
    import requests

    response = requests.get(
        NSE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df["Ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
    return df.rename(columns={"Company Name": "Name", "Industry": "Sector"})[
        ["Ticker", "Symbol", "Name", "Sector"]
    ]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_prices(tickers):
    import yfinance as yf

    raw = yf.download(
        list(tickers) + [BENCH],
        period="3y",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Could not find closing prices in Yahoo Finance data.")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()

    if BENCH not in prices.columns:
        raise ValueError(f"Benchmark {BENCH} was not returned by Yahoo Finance.")

    prices = prices.dropna(how="all").ffill(limit=5)
    benchmark = prices.pop(BENCH)
    return prices, benchmark


def build_rrg(close, bench, sector_of, n=RRG_N, m=RRG_M):
    returns = close.pct_change(fill_method=None)
    sector_returns = returns.T.groupby(sector_of.reindex(close.columns)).mean().T
    sector_index = (1 + sector_returns.fillna(0)).cumprod()
    relative_strength = sector_index.div(bench, axis=0)
    ratio = 100 * relative_strength / relative_strength.rolling(n, min_periods=n).mean()
    momentum = 100 * ratio / ratio.shift(m)
    return ratio, momentum


def quadrant(ratio, momentum):
    if ratio >= 100 and momentum >= 100:
        return "Leading"
    if ratio >= 100 and momentum < 100:
        return "Weakening"
    if ratio < 100 and momentum < 100:
        return "Lagging"
    return "Improving"


def sector_table(close, bench, sector_of, strict=False, cut=CUT_WEAKEST):
    ratio, momentum = build_rrg(close, bench, sector_of)
    counts = sector_of.reindex(close.columns).value_counts()

    table = pd.DataFrame({
        "RS_Ratio": ratio.iloc[-1],
        "RS_Mom": momentum.iloc[-1],
    })
    table["Stocks"] = counts.reindex(table.index).fillna(0).astype(int)
    table = table[table["Stocks"] >= MIN_SECTOR_STOCKS].dropna()

    table["Quadrant"] = [
        quadrant(r, m) for r, m in zip(table["RS_Ratio"], table["RS_Mom"])
    ]
    table["Score"] = (table["RS_Ratio"] - 100) + (table["RS_Mom"] - 100)
    table = table.sort_values("Score", ascending=False)

    weakest = set(table.index[-cut:]) if cut > 0 else set()
    allowed_quadrants = (
        {"Leading", "Improving"}
        if strict
        else {"Leading", "Improving", "Weakening"}
    )
    table["Allowed"] = [
        sector not in weakest and quad in allowed_quadrants
        for sector, quad in zip(table.index, table["Quadrant"])
    ]
    return table, ratio, momentum


def rs_score(close):
    ret = lambda n: close / close.shift(n) - 1
    return (
        0.4 * ret(63)
        + 0.2 * ret(126)
        + 0.2 * ret(189)
        + 0.2 * ret(252)
    )


def minervini_rules(close):
    sma = lambda n: close.rolling(n, min_periods=n).mean()
    d50, d150, d200 = sma(50), sma(150), sma(200)

    return {
        "R1": (close > d150) & (close > d200),
        "R2": d150 > d200,
        "R3": d200 > d200.shift(22),
        "R4": (d50 > d150) & (d50 > d200),
        "R5": close > d50,
        "R6": close >= 1.3 * close.rolling(250, min_periods=250).min(),
        "R7": close >= 0.75 * close.rolling(250, min_periods=250).max(),
    }


def scan(close, bench, meta, asof=None, strict=False):
    if asof is not None:
        close = close.loc[:asof]
        bench = bench.loc[:asof]

    if close.empty or bench.dropna().empty:
        raise ValueError("No price data is available for the selected date.")

    bench = bench.reindex(close.index).ffill()
    date = close.index[-1]
    bench_dma200 = bench.rolling(200, min_periods=200).mean().iloc[-1]
    market_ok = bool(
        pd.notna(bench.iloc[-1])
        and pd.notna(bench_dma200)
        and bench.iloc[-1] > bench_dma200
    )

    sector_of = meta["Sector"]
    sectors, ratio, momentum = sector_table(close, bench, sector_of, strict=strict)
    allowed_sectors = set(sectors.index[sectors["Allowed"]])

    rules = minervini_rules(close)
    result = pd.DataFrame({key: value.iloc[-1] for key, value in rules.items()})
    rs = rs_score(close).iloc[-1]
    rs_percentile = rs.rank(pct=True)

    result["RS_Score"] = rs
    result["RS_Rank"] = (rs_percentile * 100).round(0)
    result["R8"] = rs_percentile >= RS_TOP_PCT
    result["Close"] = close.iloc[-1]
    result["Pct_From_52W_High"] = (
        close.iloc[-1] / close.rolling(250, min_periods=250).max().iloc[-1] - 1
    ) * 100

    result = result.join(meta[["Symbol", "Name", "Sector"]])
    result["Sector_OK"] = result["Sector"].isin(allowed_sectors)
    result["Quadrant"] = result["Sector"].map(sectors["Quadrant"])
    rule_columns = [f"R{i}" for i in range(1, 9)]
    result["Rules_Passed"] = result[rule_columns].sum(axis=1)
    result["TREND_OK"] = result[rule_columns].all(axis=1)
    result["BUY"] = result["TREND_OK"] & result["Sector_OK"] & market_ok
    result = result.sort_values(["BUY", "RS_Rank"], ascending=False)

    funnel = [
        ("Nifty 500 universe", len(result)),
        ("Market filter: index above 200-DMA", len(result) if market_ok else 0),
        ("Sector filter: RRG", int(result["Sector_OK"].sum()) if market_ok else 0),
        (
            "Rules 1–7",
            int((result["Sector_OK"] & result[[f"R{i}" for i in range(1, 8)]].all(axis=1)).sum())
            if market_ok else 0,
        ),
        ("Rule 8 / final BUY list", int(result["BUY"].sum())),
    ]

    return {
        "date": date,
        "market_ok": market_ok,
        "benchmark": (bench.iloc[-1], bench_dma200),
        "sectors": sectors,
        "ratio": ratio,
        "momentum": momentum,
        "table": result,
        "funnel": funnel,
    }


def make_batches(buys, batch_size):
    if buys.empty:
        return buys.assign(Batch=pd.Series(dtype=int))

    result = buys.sort_values(["Sector", "RS_Rank"], ascending=[True, False]).copy()
    result["Batch"] = result.groupby("Sector", dropna=False).cumcount() // batch_size + 1
    return result


def csv_bytes(frame):
    return frame.to_csv(index=True).encode("utf-8-sig")


def batches_zip_bytes(batches):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for (sector, batch_number), group in batches.groupby(["Sector", "Batch"], dropna=False):
            safe_sector = str(sector).replace("/", "-").replace("\\", "-")
            archive.writestr(
                f"{safe_sector}_batch_{int(batch_number)}.csv",
                group.to_csv(index=True),
            )
    return buffer.getvalue()


st.set_page_config(page_title="Nifty 500 Trend Scanner", layout="wide")
st.title("Nifty 500 Monthly Trend Scanner")
st.caption("Minervini trend rules, sector RRG, market filter, and sector-wise BUY batches.")

with st.sidebar:
    st.header("Scan settings")
    asof_input = st.date_input("Scan as of", value=None)
    strict = st.checkbox("Strict sector filter", value=False)
    batch_size = st.number_input("Stocks per sector batch", 1, 100, 10, 1)
    run_scan = st.button("Run scan", type="primary", use_container_width=True)

if "scan_result" not in st.session_state:
    st.session_state["scan_result"] = None

if run_scan or st.session_state["scan_result"] is None:
    try:
        with st.spinner("Loading Nifty 500 universe and price history..."):
            universe = load_universe().set_index("Ticker")
            close, benchmark = load_prices(tuple(universe.index))
            meta = universe.reindex(close.columns).dropna(subset=["Sector"])
            close = close.reindex(columns=meta.index)

            if close.empty:
                raise ValueError("No Nifty 500 stock prices were downloaded.")

            scan_date = asof_input.isoformat() if asof_input else None
            st.session_state["scan_result"] = scan(close, benchmark, meta, scan_date, strict)
    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        st.stop()

result = st.session_state["scan_result"]
scan_date = pd.Timestamp(result["date"]).strftime("%Y-%m-%d")
benchmark_value, benchmark_dma = result["benchmark"]

left, middle, right = st.columns(3)
left.metric("Scan date", scan_date)
middle.metric("Nifty 500 benchmark", f"{benchmark_value:,.0f}" if pd.notna(benchmark_value) else "N/A")
right.metric(
    "Market filter",
    "GREEN — entries allowed" if result["market_ok"] else "RED — no new buys",
    delta=f"200-DMA: {benchmark_dma:,.0f}" if pd.notna(benchmark_dma) else "200-DMA unavailable",
)

st.subheader("Funnel")
st.dataframe(pd.DataFrame(result["funnel"], columns=["Stage", "Stocks"]), hide_index=True, use_container_width=True)

st.subheader("Sector RRG")
st.dataframe(result["sectors"].round(2), use_container_width=True)

all_results = result["table"]
buys = all_results[all_results["BUY"]].copy()
batches = make_batches(buys, int(batch_size))

st.subheader(f"BUY candidates ({len(buys)})")
if buys.empty:
    st.info("No stocks passed all filters for this scan.")
else:
    batch_columns = [
        "Symbol", "Name", "Sector", "Quadrant", "Batch",
        "Close", "RS_Rank", "Pct_From_52W_High",
    ]
    st.download_button(
        "Download all sector batches (.zip)",
        data=batches_zip_bytes(batches),
        file_name=f"nifty500_batches_{scan_date}.zip",
        mime="application/zip",
    )
    st.download_button(
        "Download all BUY candidates (.csv)",
        data=csv_bytes(batches[batch_columns]),
        file_name=f"nifty500_buys_{scan_date}.csv",
        mime="text/csv",
    )

    sectors_with_buys = sorted(batches["Sector"].dropna().unique())
    selected_sector = st.selectbox("Sector", sectors_with_buys)
    sector_batches = batches[batches["Sector"] == selected_sector]
    batch_numbers = sorted(sector_batches["Batch"].unique())
    selected_batch = st.selectbox(
        f"Batch in {selected_sector}",
        batch_numbers,
        format_func=lambda value: f"Batch {value}",
    )
    selected_stocks = sector_batches[sector_batches["Batch"] == selected_batch]

    st.dataframe(selected_stocks[batch_columns], hide_index=True, use_container_width=True)
    st.download_button(
        f"Download {selected_sector} — batch {selected_batch}",
        data=csv_bytes(selected_stocks[batch_columns]),
        file_name=f"{selected_sector}_batch_{selected_batch}_{scan_date}.csv",
        mime="text/csv",
    )

with st.expander("Full scan results"):
    st.dataframe(all_results, use_container_width=True)
    st.download_button(
        "Download full scan (.csv)",
        data=csv_bytes(all_results),
        file_name=f"nifty500_full_scan_{scan_date}.csv",
        mime="text/csv",
    )
