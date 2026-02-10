import streamlit as st
import yfinance as yf
import pandas as pd

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="내 자산 기준 투자 판단 시스템", layout="wide")

st.title("📊 내 자산 기준 투자 판단 시스템")

# =========================
# 자산 선택
# =========================
asset_dict = {
    "NASDAQ": "^IXIC",
    "S&P500": "^GSPC",
    "BITCOIN": "BTC-USD"
}

asset_name = st.selectbox("자산 선택", list(asset_dict.keys()))
ticker = asset_dict[asset_name]

# =========================
# 데이터 불러오기
# =========================
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y")
    df = df.reset_index()
    return df

df = load_data(ticker)

# 데이터 없을 때 방어
if df.empty:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

# =========================
# 이동평균 계산
# =========================
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA60"] = df["Close"].rolling(window=60).mean()

# =========================
# 차트 표시
# =========================
st.subheader("📈 가격 & 이동평균")

chart_df = df[["Date", "Close", "MA20", "MA60"]].dropna()
chart_df = chart_df.set_index("Date")

st.line_chart(chart_df, use_container_width=True)

# =========================
# 투자 판단 로직 (⚠️ 핵심)
# =========================
latest_ma20 = chart_df["MA20"].iloc[-1]
latest_ma60 = chart_df["MA60"].iloc[-1]

st.subheader("🧠 투자 판단 결과")

if latest_ma20 > latest_ma60:
    st.success("📈 상승 추세 → **매수 고려**")
else:
    st.warning("📉 하락 추세 → **관망 / 매도 고려**")

# =========================
# 데이터 테이블 (선택)
# =========================
with st.expander("📄 원본 데이터 보기"):
    st.dataframe(df.tail(20))
