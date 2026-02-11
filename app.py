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

max_len = max(len(tickers), len(buy_prices), len(quantities))
while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices) < max_len:
    buy_prices.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)


# ---------------------------
# 유틸: Close를 "항상 Series"로 뽑기 (멀티인덱스/데이터프레임 방어)
# ---------------------------
def extract_close_series(data: pd.DataFrame) -> pd.Series | None:
    if data is None or len(data) == 0:
        return None

    # yfinance가 MultiIndex 컬럼으로 주는 경우 방어
    if isinstance(data.columns, pd.MultiIndex):
        # 보통 level=0에 'Close'가 있고 level=1에 티커가 있음
        if "Close" not in data.columns.get_level_values(0):
            return None
        close_part = data.xs("Close", axis=1, level=0, drop_level=True)

        # close_part가 DataFrame(여러 컬럼)일 수 있음 -> 첫 컬럼 선택
        if isinstance(close_part, pd.DataFrame):
            if close_part.shape[1] == 0:
                return None
            close = close_part.iloc[:, 0]
        else:
            close = close_part

    else:
        if "Close" not in data.columns:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            close = close.iloc[:, 0]

    # 최종적으로 Series 보장
    if not isinstance(close, pd.Series):
        try:
            close = close.squeeze()
        except Exception:
            return None
        if not isinstance(close, pd.Series):
            return None

    close = close.dropna()
    if len(close) == 0:
        return None

    return close


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

    close = extract_close_series(data)
    if close is None or len(close) < 5:
        st.warning(f"{ticker} 데이터 부족/종가 없음")
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
    volatility = float(returns.std()) if len(returns) > 0 else 0.0

    # 추세: ma20/ma60을 "반드시 float"로 만들기
    ma20_val = close.rolling(20).mean().iloc[-1]
    if len(close) >= 60:
        ma60_val = close.rolling(60).mean().iloc[-1]
    else:
        ma60_val = ma20_val

    ma20 = float(ma20_val) if pd.notna(ma20_val) else np.nan
    ma60 = float(ma60_val) if pd.notna(ma60_val) else np.nan

    if np.isnan(ma20) or np.isnan(ma60) or ma60 == 0.0:
        trend = 0.0
    else:
        trend = (ma20 - ma60) / ma60

    # 모멘텀
    first_price = float(close.iloc[0])
    momentum = 0.0 if first_price == 0.0 else (current_price - first_price) / first_price

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

    pnl = 0.0 if buy_price == 0.0 else (current_price - buy_price) * qty

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
