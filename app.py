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


def safe_float_list(s: str) -> list[float]:
    out: list[float] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            out.append(0.0)
            continue
        try:
            out.append(float(item))
        except Exception:
            out.append(0.0)
    return out


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
# 리스크/점수
# ---------------------------


def calculate_risk(vol: float) -> str:
    if vol < 0.02:
        return "낮음"
    if vol < 0.05:
        return "보통"
    return "높음"


def calculate_ai_score(trend: float, vol: float, mom: float) -> int:
    score = (trend * 200.0) + ((1.0 - vol) * 30.0) + (mom * 100.0) + 20.0
    return int(np.clip(score, 0, 100))


# ---------------------------
# 처리
# ---------------------------
for i in range(max_len):
    ticker = tickers[i]
    if ticker == "":
        continue

    try:
        data = yf.download(
            ticker,
            period="3mo",
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        continue

    if data is None or len(data) == 0:
        st.warning(f"{ticker} 데이터 없음")
        continue

    if "Close" not in data.columns:
        st.warning(f"{ticker} 종가(Close) 없음")
        continue

    # ✅ Close를 항상 1차원 Series로 정규화 (들여쓰기/타입 이슈 방어)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    if len(close) < 5:
        st.warning(f"{ticker} 데이터 부족")
        continue

    current_price = float(close.iloc[-1])
    buy_price = float(buy_prices[i])
    qty = float(quantities[i])

    # 손익률
    if buy_price == 0.0:
        change_pct = 0.0
    else:
        change_pct = ((current_price - buy_price) / buy_price) * 100.0

    # 변동성
    returns = close.pct_change().dropna()
    if len(returns) > 0:
        volatility = float(returns.std())
    else:
        volatility = 0.0

    # 추세 (ma20, ma60) + NaN/0 방어
    ma20 = close.rolling(20).mean().iloc[-1]
    if len(close) >= 60:
        ma60 = close.rolling(60).mean().iloc[-1]
    else:
        ma60 = ma20

    if pd.isna(ma20) or pd.isna(ma60) or float(ma60) == 0.0:
        trend = 0.0
    else:
        trend = float((ma20 - ma60) / ma60)

    # 모멘텀
    first_price = float(close.iloc[0])
    if first_price == 0.0:
        momentum = 0.0
    else:
        momentum = float((current_price - first_price) / first_price)

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

    # 평가손익
    if buy_price != 0.0:
        pnl = (current_price - buy_price) * qty
    else:
        pnl = 0.0

    st.markdown(
        f"""
---
### {ticker}
현재가: ${current_price:.2f}  
평단가: ${buy_price:.2f} / 수량: {qty:g}  
평가손익: ${pnl:,.2f}  

<span style='color:{color}; font-size:20px; font-weight:bold;'>
{arrow} {change_pct:.2f}%
</span>  

AI 점수: {ai_score}점  
리스크: {risk}
""",
        unsafe_allow_html=True,
    )
