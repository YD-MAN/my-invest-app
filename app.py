import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")
st.title("📊 AI 포트폴리오 매니저 Pro")

# ---------------------------
# 입력 (평단가/수량은 쉼표로)
# ---------------------------
tickers_input = st.text_input("종목코드 (쉼표 구분)", "AAPL, MSFT, NVDA, 005930.KS")
buy_prices_input = st.text_input("평단가 (원화 기준, 쉼표 구분)", "150000, 300000, 400000, 70000")
quantities_input = st.text_input("수량 (쉼표 구분)", "10, 5, 3, 10")


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
buy_prices_krw = safe_float_list(buy_prices_input)   # ✅ 전부 원화로 해석
quantities = safe_float_list(quantities_input)

# 길이 보정
max_len = max(len(tickers), len(buy_prices_krw), len(quantities))
while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices_krw) < max_len:
    buy_prices_krw.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)


# ---------------------------
# 유틸: Close를 항상 Series로 뽑기 (멀티인덱스 방어)
# ---------------------------
def extract_close_series(data: pd.DataFrame) -> pd.Series | None:
    if data is None or len(data) == 0:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return None
        close_part = data.xs("Close", axis=1, level=0, drop_level=True)
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
# 환율: USD -> KRW (yfinance "KRW=X")
# ---------------------------
def get_usdkrw_rate() -> float:
    try:
        fx = yf.download("KRW=X", period="5d", progress=False)
        fx_close = extract_close_series(fx)
        if fx_close is None:
            return 0.0
        rate = float(fx_close.iloc[-1])
        return rate if rate > 0 else 0.0
    except Exception:
        return 0.0


usdkrw = get_usdkrw_rate()
if usdkrw == 0.0:
    st.warning("USD/KRW 환율을 가져오지 못했습니다. 해외 종목의 원화 환산이 0으로 표시될 수 있어요.")


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


def is_korea_ticker(ticker: str) -> bool:
    # 국내: 코스피/코스닥 등 (필요하면 .KQ/.KS 외 추가 가능)
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def fmt_krw(x: float) -> str:
    return f"₩{x:,.0f}"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


# ---------------------------
# 처리
# ---------------------------
for i in range(max_len):
    ticker = tickers[i]
    if ticker == "":
        continue

    buy_price = float(buy_prices_krw[i])   # ✅ 원화 평단가
    qty = float(quantities[i])

    try:
        data = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        continue

    close = extract_close_series(data)
    if close is None or len(close) < 5:
        st.warning(f"{ticker} 데이터 부족/종가 없음")
        continue

    # 변동성(종목 자체 통화 기준으로 계산. 국내=KRW, 해외=USD)
    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 0 else 0.0

    # 추세/모멘텀 (종목 자체 통화 기준으로 계산해도 무방)
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

    first_price_native = float(close.iloc[0])
    current_price_native = float(close.iloc[-1])

    if first_price_native == 0.0:
        momentum = 0.0
    else:
        momentum = (current_price_native - first_price_native) / first_price_native

    ai_score = calculate_ai_score(trend, volatility, momentum)
    risk = calculate_risk(volatility)

    # ---------------------------
    # ✅ 통화 처리
    # ---------------------------
    if is_korea_ticker(ticker):
        # 국내: 현재가도 원화
        current_price_krw = current_price_native
        current_price_usd = None
        currency_label = "KRW"
    else:
        # 해외: 현재가(USD) + 원화 환산
        current_price_usd = current_price_native
        current_price_krw = current_price_usd * usdkrw if usdkrw > 0 else 0.0
        currency_label = "USD→KRW"

    # ---------------------------
    # ✅ 수익률/손익 (입력 평단가가 전부 원화이므로, 원화 기준으로 계산)
    # ---------------------------
    if buy_price == 0.0:
        change_pct = 0.0
    else:
        change_pct = ((current_price_krw - buy_price) / buy_price) * 100.0

    pnl_krw = (current_price_krw - buy_price) * qty if buy_price != 0.0 else 0.0
    eval_krw = current_price_krw * qty

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

    # 화면 표시 (국내/해외 다르게)
    if is_korea_ticker(ticker):
        price_line = f"현재가: {fmt_krw(current_price_krw)}"
        fx_line = ""
    else:
        price_line = f"현재가: {fmt_usd(current_price_usd)}  (환산 {fmt_krw(current_price_krw)})"
        fx_line = f"환율(USD/KRW): {usdkrw:,.2f}" if usdkrw > 0 else "환율(USD/KRW): 가져오기 실패"

    st.markdown(
        f"""
---
### {ticker}  <span style="font-size:14px; color:#666;">[{currency_label}]</span>

{price_line}  
{fx_line}  
평단가(원화 입력): {fmt_krw(buy_price)} / 수량: {qty:g}  
평가금액(원화): {fmt_krw(eval_krw)}  
평가손익(원화): {fmt_krw(pnl_krw)}  

<span style='color:{color}; font-size:20px; font-weight:bold;'>
{arrow} 수익률(원화 기준): {change_pct:.2f}%
</span>  

AI 점수: {ai_score}점  
리스크: {risk}
""",
        unsafe_allow_html=True,
    )
