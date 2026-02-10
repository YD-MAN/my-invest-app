import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="투자 판단 시스템", layout="wide")
st.title("📊 실전 투자 판단 시스템")

# =========================
# 기본 설정
# =========================
ASSETS = {
    "NASDAQ": "^IXIC",
    "S&P500": "^GSPC",
    "BITCOIN": "BTC-USD"
}

PORTFOLIO_DEFAULT = {
    "NASDAQ": 50,
    "S&P500": 30,
    "BITCOIN": 20
}

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y", progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return None

    return df


# =========================
# 기술적 지표 계산
# =========================
def calculate_indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


# =========================
# 분할 매수 계산
# =========================
def split_buy_plan(total_money, price, splits):
    if splits == 3:
        ratios = [0.4, 0.35, 0.25]
        drops = [0.0, 0.05, 0.10]
    else:
        ratios = [0.3, 0.25, 0.2, 0.15, 0.1]
        drops = [0.0, 0.03, 0.06, 0.09, 0.12]

    plan = []
    for i in range(len(ratios)):
        buy_price = price * (1 - drops[i])
        invest_money = total_money * ratios[i]
        qty = int(invest_money / buy_price)

        plan.append({
            "차수": f"{i+1}차",
            "매수가": round(buy_price, 2),
            "투입금액(원)": int(invest_money),
            "매수수량": qty
        })

    return pd.DataFrame(plan)


# =========================
# 손절 / 익절 계산
# =========================
def risk_management(avg_price):
    stop_loss = avg_price * 0.93
    take_profit = avg_price * 1.15
    return stop_loss, take_profit


# =========================
# 사용자 입력
# =========================
st.sidebar.header("💼 투자 설정")

total_money = st.sidebar.number_input(
    "총 투자금 (원)",
    min_value=0,
    value=10_000_000,
    step=500_000
)

split_count = st.sidebar.selectbox(
    "분할 매수 횟수",
    [3, 5]
)

st.sidebar.subheader("📊 포트폴리오 비중 (%)")
weights = {}
total_weight = 0

for asset, default in PORTFOLIO_DEFAULT.items():
    w = st.sidebar.slider(asset, 0, 100, default)
    weights[asset] = w
    total_weight += w

if total_weight != 100:
    st.sidebar.error("❌ 비중 합계는 100%여야 합니다.")
    st.stop()

# =========================
# 메인 로직
# =========================
for asset_name, ticker in ASSETS.items():
    st.divider()
    st.header(f"📌 {asset_name}")

    df = load_data(ticker)
    if df is None:
        st.warning("데이터 로딩 실패")
        continue

    df = calculate_indicators(df)
    df = df.dropna()
    df = df.set_index("Date")

    current_price = df["Close"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma60 = df["MA60"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]

    st.line_chart(df[["Close", "MA20", "MA60"]])

    # 투자 판단 점수
    score = 0
    if ma20 > ma60:
        score += 1
    if rsi < 70:
        score += 1
    if macd > signal:
        score += 1

    if score == 3:
        st.success("✅ 매수 우위")
    elif score == 2:
        st.warning("⚠️ 관망")
    else:
        st.error("❌ 보수적 접근")

    # 자산별 투자금
    asset_money = total_money * (weights[asset_name] / 100)

    st.subheader("📦 분할 매수 계획")
    plan_df = split_buy_plan(asset_money, current_price, split_count)
    st.dataframe(plan_df, use_container_width=True)

    avg_price = plan_df["매수가"].mean()
    stop_loss, take_profit = risk_management(avg_price)

    st.subheader("🛡 리스크 관리")
    st.write(f"평균 매입가: **{avg_price:,.2f}**")
    st.write(f"손절 기준: **{stop_loss:,.2f} (-7%)**")
    st.write(f"익절 기준: **{take_profit:,.2f} (+15%)**")

