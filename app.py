import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")

st.title("📊 AI 포트폴리오 매니저 Pro")

# ---------------------------
# 입력
# ---------------------------
tickers_input = st.text_input("종목코드 (쉼표 구분)", "AAPL, MSFT, NVDA")
buy_prices_input = st.text_input("평단가", "150, 300, 400")
quantities_input = st.text_input("수량", "10, 5, 3")

def safe_float_list(s):
    result = []
    for item in s.split(","):
        try:
            result.append(float(item.strip()))
        except:
            result.append(0.0)
    return result

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
buy_prices = safe_float_list(buy_prices_input)
quantities = safe_float_list(quantities_input)

# 길이 보정
max_len = max(len(tickers), len(buy_prices), len(quantities))

while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices) < max_len:
    buy_prices.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)

# ---------------------------
# 리스크
# ---------------------------
def calculate_risk(vol):
    if vol < 0.02:
        return "낮음"
    elif vol < 0.05:
        return "보통"
    else:
        return "높음"

def calculate_ai_score(trend, vol, mom):
    score = (trend*200) + ((1-vol)*30) + (mom*100) + 20
    return int(np.clip(score, 0, 100))

# ---------------------------
# 처리
# ---------------------------
for i in range(max_len):

    ticker = tickers[i]
    if ticker == "":
        continue

    try:
        data = yf.download(ticker, period="3mo", progress=False)
    except:
        st.warning(f"{ticker} 다운로드 실패")
        continue

    if data is None:
        continue
    if len(data) == 0:
        st.warning(f"{ticker} 데이터 없음")
        continue
    if "Close" not in data.columns:
        st.warning(f"{ticker} 종가 없음")
        continue

    close = data["Close"].dropna()
    if len(close) < 5:
        st.warning(f"{ticker} 데이터 부족")
        continue

    current_price = float(close.iloc[-1])
    buy_price = float(buy_prices[i])

    if buy_price == 0:
        change_pct = 0
    else:
        change_pct = ((current_price - buy_price) / buy_price) * 100

    # 변동성
    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 0 else 0

    # 추세
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

    trend = (ma20 - ma60) / ma60 if ma60 != 0 else 0

    momentum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

    ai_score = calculate_ai_score(trend, volatility, momentum)
    risk = calculate_risk(volatility)

    # 색상 처리
    if change_pct > 0:
        color = "red"
        arrow = "▲"
    elif change_pct < 0:
        color = "blue"
        arrow = "▼"
    else:
        color = "gray"
        arrow = ""

    st.markdown(f"""
    ---
    ### {ticker}
    현재가: ${current_price:.2f}  
    <span style='color:{color}; font-size:20px; font-weight:bold;'>
    {arrow} {change_pct:.2f}%
    </span>  
    AI 점수: {ai_score}점  
    리스크: {risk}
    """, unsafe_allow_html=True)
