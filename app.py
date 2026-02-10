import streamlit as st
import yfinance as yf
import pandas as pd

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
# 데이터 로딩 (완전 방어형)
# ======================
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y", progress=False)

    if df.empty:
        return None

    # MultiIndex 컬럼 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Close 컬럼 방어
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return None

    return df

df = load_data(ticker)

if df is None:
    st.error("❌ 데이터를 정상적으로 불러오지 못했습니다.")
    st.stop()

# ======================
# 이동평균 계산
# ======================
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

chart_df = df[["Date", "Close", "MA20", "MA60"]].dropna()

if chart_df.empty:
    st.error("❌ 이동평균 계산에 필요한 데이터가 부족합니다.")
    st.stop()

chart_df = chart_df.set_index("Date")

# ======================
# 차트
# ======================
st.subheader("📈 가격 & 이동평균")
st.line_chart(chart_df)

# ======================
# 투자 판단 (scalar 비교)
# ======================
ma20 = chart_df["MA20"].iloc[-1]
ma60 = chart_df["MA60"].iloc[-1]

st.subheader("🧠 투자 판단 결과")

if ma20 > ma60:
    st.success("📈 상승 추세 → 매수 고려")
else:
    st.warning("📉 하락 추세 → 관망 / 매도 고려")

# ======================
# 데이터 확인
# ======================
with st.expander("📄 최근 데이터 보기"):
    st.dataframe(chart_df.tail(20))
