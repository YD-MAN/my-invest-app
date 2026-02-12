# app.py
# =========================================================
# AI 포트폴리오 매니저 Pro (완성본)
# - 원화 평단가 입력 + 국내/해외 통화표시 + 원화 손익
# - RandomForest 상승확률 + 뉴스 감성(ETF/지수 키워드 대체 + NewsAPI 보강)
# - 한국어 감성(Transformer) 자동 적용(가능할 때)
# - 입력 자동 URL 저장 + Reset + 설정 JSON 저장/불러오기
# =========================================================

import json
import re
from urllib.parse import quote_plus, unquote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# Streamlit 기본
# ---------------------------
st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")
st.title("📊 AI 포트폴리오 매니저 Pro")

# =========================================================
# 0) 자동 URL 저장 입력부 (전략 B 자동 저장 + Reset + 안전 로드)
# =========================================================

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, 005930.KS, BTC-KRW"
DEFAULT_BUYPRICES = "150000, 300000, 400000, 70000, 50000000"
DEFAULT_QTYS = "10, 5, 3, 10, 0.01"


def get_qp() -> dict:
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    return st.experimental_get_query_params()


def set_qp(**kwargs):
    safe = {k: v for k, v in kwargs.items() if v is not None}
    if hasattr(st, "query_params"):
        st.query_params.clear()
        for k, v in safe.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**safe)


def qp_get_str(qp: dict, key: str, default: str) -> str:
    if key not in qp:
        return default
    v = qp[key]
    if isinstance(v, list):
        v = v[0] if len(v) else default
    if v is None or v == "":
        return default
    try:
        return unquote_plus(str(v))
    except Exception:
        return str(v)


def enc(s: str) -> str:
    return quote_plus(s)


def mark_dirty():
    st.session_state["__qp_dirty__"] = True


def request_reset():
    st.session_state["__reset_requested__"] = True


def request_apply_loaded_settings(payload: dict):
    # 콜백/업로드 타이밍에서 바로 rerun하지 말고 플래그로 처리
    st.session_state["__loaded_settings__"] = payload
    st.session_state["__apply_loaded_settings__"] = True


# 초기 플래그들
if "__qp_dirty__" not in st.session_state:
    st.session_state["__qp_dirty__"] = False
if "__reset_requested__" not in st.session_state:
    st.session_state["__reset_requested__"] = False
if "__apply_loaded_settings__" not in st.session_state:
    st.session_state["__apply_loaded_settings__"] = False
if "__loaded_settings__" not in st.session_state:
    st.session_state["__loaded_settings__"] = {}

# 초기값: URL → session_state (최초 1회)
qp = get_qp()
init_tickers = qp_get_str(qp, "tickers", DEFAULT_TICKERS)
init_buy = qp_get_str(qp, "buy", DEFAULT_BUYPRICES)
init_qty = qp_get_str(qp, "qty", DEFAULT_QTYS)

if "tickers_input" not in st.session_state:
    st.session_state["tickers_input"] = init_tickers
if "buy_prices_input" not in st.session_state:
    st.session_state["buy_prices_input"] = init_buy
if "quantities_input" not in st.session_state:
    st.session_state["quantities_input"] = init_qty

# ✅ Reset 처리: 콜백 밖에서 실행
if st.session_state["__reset_requested__"]:
    for k in ["tickers_input", "buy_prices_input", "quantities_input"]:
        if k in st.session_state:
            del st.session_state[k]

    set_qp(
        tickers=enc(DEFAULT_TICKERS),
        buy=enc(DEFAULT_BUYPRICES),
        qty=enc(DEFAULT_QTYS),
    )

    st.session_state["__qp_dirty__"] = False
    st.session_state["__reset_requested__"] = False
    st.rerun()

# ✅ JSON 업로드로 불러오기 적용: 콜백 밖에서 실행
if st.session_state["__apply_loaded_settings__"]:
    loaded = st.session_state.get("__loaded_settings__", {}) or {}
    t = str(loaded.get("tickers_input", DEFAULT_TICKERS))
    b = str(loaded.get("buy_prices_input", DEFAULT_BUYPRICES))
    q = str(loaded.get("quantities_input", DEFAULT_QTYS))

    for k in ["tickers_input", "buy_prices_input", "quantities_input"]:
        if k in st.session_state:
            del st.session_state[k]

    set_qp(tickers=enc(t), buy=enc(b), qty=enc(q))

    st.session_state["__apply_loaded_settings__"] = False
    st.session_state["__loaded_settings__"] = {}
    st.rerun()

# ---------------------------
# 입력 UI
# ---------------------------
st.subheader("입력 (자동 URL 저장)")
col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

with col1:
    st.text_input("종목코드 (쉼표 구분)", key="tickers_input", on_change=mark_dirty)

with col2:
    st.text_input("평단가 (원화 기준, 쉼표 구분)", key="buy_prices_input", on_change=mark_dirty)

with col3:
    st.text_input("수량 (쉼표 구분)", key="quantities_input", on_change=mark_dirty)

with col4:
    st.write("")
    st.write("")
    st.button("Reset", on_click=request_reset)

# ✅ 자동 URL 저장 (dirty일 때만, 다를 때만 set_qp)
if st.session_state["__qp_dirty__"]:
    desired_t = enc(st.session_state["tickers_input"])
    desired_b = enc(st.session_state["buy_prices_input"])
    desired_q = enc(st.session_state["quantities_input"])

    current = get_qp()
    cur_t = current.get("tickers", "")
    cur_b = current.get("buy", "")
    cur_q = current.get("qty", "")

    if isinstance(cur_t, list):
        cur_t = cur_t[0] if len(cur_t) else ""
    if isinstance(cur_b, list):
        cur_b = cur_b[0] if len(cur_b) else ""
    if isinstance(cur_q, list):
        cur_q = cur_q[0] if len(cur_q) else ""

    if (cur_t != desired_t) or (cur_b != desired_b) or (cur_q != desired_q):
        set_qp(tickers=desired_t, buy=desired_b, qty=desired_q)

    st.session_state["__qp_dirty__"] = False

st.caption("✅ 입력 변경 시 자동으로 URL에 저장됩니다. 새로고침/재접속/공유해도 동일 입력 유지")

# =========================================================
# 1) 입력값 파싱
# =========================================================

tickers_input = st.session_state["tickers_input"]
buy_prices_input = st.session_state["buy_prices_input"]
quantities_input = st.session_state["quantities_input"]


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
buy_prices_krw = safe_float_list(buy_prices_input)  # ✅ 전부 원화로 해석
quantities = safe_float_list(quantities_input)

max_len = max(len(tickers), len(buy_prices_krw), len(quantities))
while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices_krw) < max_len:
    buy_prices_krw.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)

# =========================================================
# 2) 유틸: Close 추출/통화 판별/표시
# =========================================================


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


def is_korea_ticker(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.endswith(".KO")


def is_crypto_ticker(ticker: str) -> bool:
    return "-" in ticker and (ticker.endswith("-USD") or ticker.endswith("-KRW"))


def fmt_krw(x: float) -> str:
    return f"₩{x:,.0f}"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


@st.cache_data(ttl=60 * 30)
def get_usdkrw_rate() -> float:
    try:
        fx = yf.download("KRW=X", period="10d", progress=False)
        fx_close = extract_close_series(fx)
        if fx_close is None:
            return 0.0
        rate = float(fx_close.iloc[-1])
        return rate if rate > 0 else 0.0
    except Exception:
        return 0.0


usdkrw = get_usdkrw_rate()
if usdkrw == 0.0:
    st.warning("USD/KRW 환율을 가져오지 못했습니다. 해외 종목 환산 KRW가 0으로 표시될 수 있어요.")

# =========================================================
# 3) 기술지표 + 점수
# =========================================================


def calculate_risk(vol: float) -> str:
    if vol < 0.02:
        return "낮음"
    if vol < 0.05:
        return "보통"
    return "높음"


def base_ai_score(trend: float, vol: float, mom: float) -> float:
    score = (trend * 200.0) + ((1.0 - vol) * 30.0) + (mom * 100.0) + 20.0
    return float(score)


def upgraded_ai_score(trend: float, vol: float, mom: float, prob_up: float, senti: float) -> int:
    # RF 확률: 0.5 기준 최대 ±25점 정도
    rf_component = (prob_up - 0.5) * 50.0
    # 뉴스 감성: 최대 ±10점 정도
    news_component = senti * 10.0
    s = base_ai_score(trend, vol, mom) + rf_component + news_component
    return int(np.clip(s, 0, 100))


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# =========================================================
# 4) RandomForest 상승확률
# =========================================================


@st.cache_data(ttl=60 * 60)
def rf_up_probability(ticker: str) -> tuple[float, dict]:
    debug = {"status": "ok"}
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
    except Exception as e:
        return 0.0, {"status": f"download_fail: {e}"}

    close = extract_close_series(df)
    if close is None or len(close) < 140:
        return 0.0, {"status": "not_enough_data"}

    s = close.copy()
    ret1 = s.pct_change()
    vol10 = ret1.rolling(10).std()
    vol20 = ret1.rolling(20).std()

    ma5 = s.rolling(5).mean()
    ma20 = s.rolling(20).mean()
    ma60 = s.rolling(60).mean()

    feat = pd.DataFrame(
        {
            "ret1": ret1,
            "vol10": vol10,
            "vol20": vol20,
            "ma_gap_5_20": (ma5 - ma20) / ma20.replace(0, np.nan),
            "ma_gap_20_60": (ma20 - ma60) / ma60.replace(0, np.nan),
            "rsi14": rsi(s, 14),
            "mom20": (s - s.shift(20)) / s.shift(20).replace(0, np.nan),
        }
    ).dropna()

    y = (s.shift(-1).loc[feat.index] > s.loc[feat.index]).astype(int)

    X = feat.iloc[:-1]
    y = y.iloc[:-1]
    X_last = feat.iloc[[-1]]

    if len(X) < 120:
        return 0.0, {"status": "not_enough_train_rows"}

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, _ = list(tscv.split(X))[-1]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    prob_up = float(model.predict_proba(X_last)[0, 1])
    debug["train_rows"] = int(len(X_train))
    return prob_up, debug

# =========================================================
# 5) 뉴스 강화: ETF/지수 키워드 대체 + NewsAPI + 한국어 감성(고성능)
# =========================================================

analyzer = SentimentIntensityAnalyzer()

# 사이드바 옵션 (무거운 기능은 사용자가 끌 수 있게)
with st.sidebar:
    st.header("⚙️ 뉴스/감성 옵션")
    ENABLE_NEWSAPI = st.toggle("NewsAPI 보강 사용", value=True)
    ENABLE_KO_TRANSFORMER = st.toggle("한국어 감성(Transformer) 사용", value=True)
    st.caption("한국어 감성은 무거울 수 있어요. 느리면 꺼도 됩니다.")

# 한국어 감성 모델(멀티링구얼, “가능할 때” 자동 사용)
KOREAN_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


def get_newsapi_key() -> str:
    try:
        return st.secrets.get("NEWSAPI_KEY", "")
    except Exception:
        return ""


NEWSAPI_KEY = get_newsapi_key()


@st.cache_data(ttl=60 * 60)
def guess_quote_type(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        qt = (info.get("quoteType") or "").upper()
        return qt if qt else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def is_index_ticker(ticker: str) -> bool:
    return ticker.startswith("^")


def is_etf_or_index(ticker: str) -> bool:
    if is_index_ticker(ticker):
        return True
    qt = guess_quote_type(ticker)
    return qt in {"ETF", "INDEX", "MUTUALFUND"}


@st.cache_data(ttl=60 * 30)
def get_keyword_for_news(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        name = str(name).strip()
        if name:
            return name
    except Exception:
        pass

    if ticker.startswith("^"):
        return ticker[1:]
    return ticker


@st.cache_data(ttl=60 * 20)
def newsapi_everything(query: str, language: str | None = None, page_size: int = 20) -> tuple[list[str], dict]:
    if not (ENABLE_NEWSAPI and NEWSAPI_KEY):
        if not NEWSAPI_KEY and ENABLE_NEWSAPI:
            return [], {"status": "no_newsapi_key"}
        return [], {"status": "newsapi_disabled"}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    if language:
        params["language"] = language

    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get("status") != "ok":
            return [], {"status": "newsapi_error", "message": data.get("message", "")}

        titles = []
        for a in data.get("articles", [])[:page_size]:
            t = (a.get("title") or "").strip()
            if t:
                titles.append(t)

        if not titles:
            return [], {"status": "newsapi_no_titles"}
        return titles, {"status": "ok", "count": len(titles)}
    except Exception as e:
        return [], {"status": "newsapi_fail", "error": str(e)}


def detect_language_simple(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "ko"
    return "en"


@st.cache_resource
def get_ko_sentiment_pipe():
    # transformers/torch가 설치되지 않았으면 예외 → 호출부에서 처리
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=KOREAN_SENTIMENT_MODEL)


def stars_to_compound(label: str) -> float:
    m = re.search(r"(\d)", label)
    if not m:
        return 0.0
    stars = int(m.group(1))  # 1~5
    return (stars - 3) / 2.0  # 1->-1, 3->0, 5->+1


def sentiment_compound_from_titles(titles: list[str]) -> tuple[float, str]:
    if not titles:
        return 0.0, "중립"

    titles = titles[:10]
    joined = " ".join(titles[:3])
    lang = detect_language_simple(joined)

    if lang == "en":
        scores = []
        for t in titles:
            s = analyzer.polarity_scores(t).get("compound", 0.0)
            scores.append(float(s))
        comp = float(np.mean(scores)) if scores else 0.0
    else:
        # 한국어/비영어: Transformer(가능할 때)
        if not ENABLE_KO_TRANSFORMER:
            comp = 0.0
        else:
            try:
                pipe = get_ko_sentiment_pipe()
                preds = pipe(titles)
                vals = []
                for p in preds:
                    label = str(p.get("label", ""))
                    vals.append(stars_to_compound(label))
                comp = float(np.mean(vals)) if vals else 0.0
            except Exception:
                # torch/transformers 미설치 또는 리소스 부족이면 안전하게 중립 처리
                comp = 0.0

    if comp > 0.05:
        lab = "긍정"
    elif comp < -0.05:
        lab = "부정"
    else:
        lab = "중립"

    return comp, lab


@st.cache_data(ttl=60 * 20)
def news_sentiment_score(ticker: str, max_items: int = 12) -> tuple[float, list[str], dict]:
    """
    return: (sentiment_score(-1~+1), titles, debug)
    - yfinance 뉴스 → 타이틀 없으면 ETF/지수는 키워드로 NewsAPI 대체
    - 그래도 없으면 titles=[]로 반환 (UI에서 숨김 처리)
    """
    # 1) yfinance 티커 뉴스 우선
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None)
    except Exception:
        news = None

    titles: list[str] = []
    if news:
        for item in news[:max_items]:
            title = (item.get("title") or "").strip()
            if title:
                titles.append(title)

    # 2) yfinance 타이틀이 없으면(= no_valid_titles) → NewsAPI fallback
    if len(titles) == 0:
        if is_etf_or_index(ticker):
            kw = get_keyword_for_news(ticker)

            titles_en, dbg_en = newsapi_everything(kw, language="en", page_size=max_items)
            if titles_en:
                comp, lab = sentiment_compound_from_titles(titles_en)
                return comp, titles_en, {"status": "fallback_newsapi_etf_index", "keyword": kw, "lang": "en", "label": lab, **dbg_en}

            titles_ko, dbg_ko = newsapi_everything(kw, language="ko", page_size=max_items)
            if titles_ko:
                comp, lab = sentiment_compound_from_titles(titles_ko)
                return comp, titles_ko, {"status": "fallback_newsapi_etf_index", "keyword": kw, "lang": "ko", "label": lab, **dbg_ko}

            return 0.0, [], {"status": "no_news_after_fallback", "keyword": kw, "quoteType": guess_quote_type(ticker)}

        # 일반 종목도 NewsAPI로 보강하고 싶으면 True
        USE_NEWSAPI_FOR_STOCKS_TOO = True
        if USE_NEWSAPI_FOR_STOCKS_TOO:
            kw = get_keyword_for_news(ticker)

            titles_en, dbg_en = newsapi_everything(kw, language="en", page_size=max_items)
            if titles_en:
                comp, lab = sentiment_compound_from_titles(titles_en)
                return comp, titles_en, {"status": "fallback_newsapi_stock", "keyword": kw, "lang": "en", "label": lab, **dbg_en}

            titles_ko, dbg_ko = newsapi_everything(kw, language="ko", page_size=max_items)
            if titles_ko:
                comp, lab = sentiment_compound_from_titles(titles_ko)
                return comp, titles_ko, {"status": "fallback_newsapi_stock", "keyword": kw, "lang": "ko", "label": lab, **dbg_ko}

            return 0.0, [], {"status": "no_news_after_fallback", "keyword": kw, "quoteType": guess_quote_type(ticker)}

        return 0.0, [], {"status": "no_valid_titles"}

    # 3) yfinance 타이틀이 있으면 감성분석 수행
    comp, lab = sentiment_compound_from_titles(titles)
    return comp, titles, {"status": "ok", "source": "yfinance", "label": lab, "count": len(titles)}

# =========================================================
# 6) 분석 결과 출력
# =========================================================

st.divider()
st.subheader("분석 결과")

for i in range(max_len):
    ticker = tickers[i]
    if ticker == "":
        continue

    buy_price_krw = float(buy_prices_krw[i])  # ✅ 원화 평단가
    qty = float(quantities[i])

    # 가격 데이터 (3개월)
    try:
        data = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        continue

    close = extract_close_series(data)
    if close is None or len(close) < 5:
        st.warning(f"{ticker} 데이터 부족/종가 없음")
        continue

    # 변동성/추세/모멘텀
    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 0 else 0.0

    ma20_val = close.rolling(20).mean().iloc[-1]
    ma60_val = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20_val

    ma20 = float(ma20_val) if pd.notna(ma20_val) else np.nan
    ma60 = float(ma60_val) if pd.notna(ma60_val) else np.nan

    if np.isnan(ma20) or np.isnan(ma60) or ma60 == 0.0:
        trend = 0.0
    else:
        trend = (ma20 - ma60) / ma60

    current_native = float(close.iloc[-1])
    first_native = float(close.iloc[0])
    momentum = 0.0 if first_native == 0.0 else (current_native - first_native) / first_native

    risk = calculate_risk(volatility)

    # 통화 처리
    if is_korea_ticker(ticker) or (is_crypto_ticker(ticker) and ticker.endswith("-KRW")):
        current_krw = current_native
        current_usd = None
        currency_label = "KRW"
        fx_line = ""
        price_line = f"현재가: {fmt_krw(current_krw)}"
    else:
        current_usd = current_native
        current_krw = current_usd * usdkrw if usdkrw > 0 else 0.0
        currency_label = "USD→KRW"
        fx_line = f"환율(USD/KRW): {usdkrw:,.2f}" if usdkrw > 0 else "환율(USD/KRW): 가져오기 실패"
        price_line = f"현재가: {fmt_usd(current_usd)}  (환산 {fmt_krw(current_krw)})"

    # 손익/수익률 (원화 기준)
    if buy_price_krw == 0.0:
        change_pct = 0.0
    else:
        change_pct = ((current_krw - buy_price_krw) / buy_price_krw) * 100.0

    eval_krw = current_krw * qty
    pnl_krw = (current_krw - buy_price_krw) * qty if buy_price_krw != 0.0 else 0.0

    if change_pct > 0:
        color = "red"
        arrow = "▲"
    elif change_pct < 0:
        color = "blue"
        arrow = "▼"
    else:
        color = "gray"
        arrow = ""

    # RF + 뉴스
    prob_up, rf_debug = rf_up_probability(ticker)
    senti, titles, news_debug = news_sentiment_score(ticker)

    ai_score = upgraded_ai_score(trend, volatility, momentum, prob_up, senti)

    prob_pct = prob_up * 100.0
    senti_label = "긍정" if senti > 0.05 else ("부정" if senti < -0.05 else "중립")

    # ✅ 뉴스가 없으면 뉴스 라인 자체를 숨김
    news_line = ""
    if titles:
        news_line = f"- 뉴스 감성(제목 기반): **{senti_label} ({senti:+.2f})**\n"

    st.markdown(
        f"""
---
### {ticker}  <span style="font-size:14px; color:#666;">[{currency_label}]</span>

{price_line}  
{fx_line}  
평단가(원화 입력): {fmt_krw(buy_price_krw)} / 수량: {qty:g}  
평가금액(원화): {fmt_krw(eval_krw)}  
평가손익(원화): {fmt_krw(pnl_krw)}  

<span style='color:{color}; font-size:20px; font-weight:bold;'>
{arrow} 수익률(원화 기준): {change_pct:.2f}%
</span>  

**AI 점수(업그레이드): {ai_score}점**  
- RF 상승확률(다음 거래일): **{prob_pct:.1f}%**
{news_line}- 리스크(변동성): **{risk}**
""",
        unsafe_allow_html=True,
    )

    with st.expander(f"🔎 {ticker} 상세(모델/뉴스) 보기"):
        st.write("**RandomForest 상태**:", rf_debug)

        # ✅ 뉴스가 있을 때만 보여주기 (요청하신 ‘깔끔하게 숨김’)
        if titles:
            st.write("**뉴스 상태**:", news_debug)
            st.write("**최근 뉴스 제목(일부)**")
            for t in titles[:10]:
                st.write(f"- {t}")

# =========================================================
# 7) 설정 JSON 저장/불러오기 (옵션: URL이 길어질 때 추천)
# =========================================================

with st.expander("💾 설정 JSON 저장/불러오기 (URL이 길어질 때 추천)"):
    current_settings = {
        "tickers_input": st.session_state["tickers_input"],
        "buy_prices_input": st.session_state["buy_prices_input"],
        "quantities_input": st.session_state["quantities_input"],
    }
    settings_json = json.dumps(current_settings, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇️ 현재 설정 JSON 다운로드",
        data=settings_json.encode("utf-8"),
        file_name="portfolio_settings.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("⬆️ 설정 JSON 업로드", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.loads(uploaded.read().decode("utf-8"))
            request_apply_loaded_settings(loaded)
            st.success("설정을 불러왔습니다. 적용을 위해 화면을 갱신합니다.")
        except Exception as e:
            st.error(f"설정 파일을 읽는 중 오류: {e}")
