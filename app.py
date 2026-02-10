import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="내 자산 기준 투자 판단 시스템", layout="wide")
st.title("📊 내 자산 기준 투자 판단 시스템")

# ======================
# 자산 선택
# ======================
ASSETS = {
    "NASDAQ": "^IXIC",
    "S&P500": "^GSPC",
    "BITCOIN": "BTC-USD"
}

asset_name = st.selectbox("자산 선택", list(ASSETS.keys()))
ticker = ASSETS[asset_name]

# ======================
# 데이터 로딩 (완전 방어)
# ======================
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

df = load_data(ticker)

if df is None:
    st.error("❌ 데이터를 불러오지 못했습니다.")
    st.stop()

# ======================
# 기술적 지표 계산
# ======================
# 이동평균
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

chart_df = df[["Date", "Close", "MA20", "MA60", "RSI", "MACD", "Signal"]].dropna()
chart_df = chart_df.set_index("Date")

# ======================
# 가격 차트
# ======================
st.subheader("📈 가격 & 이동평균")
st.line_chart(chart_df[["Close", "MA20", "MA60"]])

# ======================
# 보조 지표
# ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("RSI")
    st.line_chart(chart_df["RSI"])

with col2:
    st.subheader("MACD")
    st.line_chart(chart_df[["MACD", "Signal"]])

# ======================
# 사용자 자산 입력
# ======================
st.subheader("💰 내 자산 입력")

total_money = st.number_input("총 투자 가능 금액 (원)", min_value=0, value=10000000, step=100000)
buy_ratio = st.slider("이번 매수 비중 (%)", min_value=10, max_value=100, value=30)

current_price = chart_df["Close"].iloc[-1]

buy_money = total_money * (buy_ratio / 100)
buy_quantity = int(buy_money / current_price)

# ======================
# 투자 판단 로직
# ======================
ma20 = chart_df["MA20"].iloc[-1]
ma60 = chart_df["MA60"].iloc[-1]
rsi = chart_df["RSI"].iloc[-1]
macd = chart_df["MACD"].iloc[-1]
signal = chart_df["Signal"].iloc[-1]

st.subheader("🧠 종합 투자 판단")

score = 0

if ma20 > ma60:
    score += 1
if rsi < 70:
    score += 1
if macd > signal:
    score += 1

if score >= 3:
    st.success("✅ 매수 신호")
elif score == 2:
    st.warning("⚠️ 관망")
else:
    st.error("❌ 매도 / 보수적 접근")

# ======================
# 매수 수량 결과
# ======================
st.subheader("📦 매수 제안")

st.write(f"- 현재 가격: **{current_price:,.2f}**")
st.write(f"- 사용 금액: **{buy_money:,.0f} 원**")
st.write(f"- 매수 가능 수량: **{buy_quantity} 단위**")

# ======================
# 데이터 확인
# ======================
with st.expander("📄 최근 데이터"):
    st.dataframe(chart_df.tail(20))
