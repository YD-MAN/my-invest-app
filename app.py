import streamlit as st
import json
from urllib.parse import quote_plus, unquote_plus

# ---------------------------
# 기본값
# ---------------------------
DEFAULT_TICKERS = "AAPL, MSFT, NVDA, 005930.KS, BTC-KRW"
DEFAULT_BUYPRICES = "150000, 300000, 400000, 70000, 50000000"
DEFAULT_QTYS = "10, 5, 3, 10, 0.01"

# ---------------------------
# Query Params 호환 래퍼 (Streamlit 버전 차이 대응)
# ---------------------------
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
        v = v[0] if len(v) > 0 else default
    if v is None or v == "":
        return default
    try:
        return unquote_plus(str(v))
    except Exception:
        return str(v)

def enc(s: str) -> str:
    return quote_plus(s)

# ---------------------------
# 자동 저장용 콜백 (입력 변경 시 플래그만 올림)
# ---------------------------
def mark_dirty():
    st.session_state["__qp_dirty__"] = True

# ---------------------------
# 초기 로드: URL -> session_state (최초 1회)
# ---------------------------
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

if "__qp_dirty__" not in st.session_state:
    st.session_state["__qp_dirty__"] = False

st.subheader("입력 (자동 URL 저장)")

# ---------------------------
# 입력 UI (변경되면 자동 저장 트리거)
# ---------------------------
col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

with col1:
    st.text_input(
        "종목코드 (쉼표 구분)",
        key="tickers_input",
        on_change=mark_dirty,
    )

with col2:
    st.text_input(
        "평단가 (원화 기준, 쉼표 구분)",
        key="buy_prices_input",
        on_change=mark_dirty,
    )

with col3:
    st.text_input(
        "수량 (쉼표 구분)",
        key="quantities_input",
        on_change=mark_dirty,
    )

with col4:
    st.write("")
    st.write("")
    if st.button("Reset"):
        st.session_state["tickers_input"] = DEFAULT_TICKERS
        st.session_state["buy_prices_input"] = DEFAULT_BUYPRICES
        st.session_state["quantities_input"] = DEFAULT_QTYS
        st.session_state["__qp_dirty__"] = True
        st.rerun()

# ---------------------------
# ✅ 자동 URL 저장 로직
# - dirty 플래그가 켜졌을 때만 실행
# - 현재 URL과 목표 URL이 다를 때만 set_qp() 호출 (무한 리런 방지)
# ---------------------------
if st.session_state["__qp_dirty__"]:
    desired = {
        "tickers": enc(st.session_state["tickers_input"]),
        "buy": enc(st.session_state["buy_prices_input"]),
        "qty": enc(st.session_state["quantities_input"]),
    }

    current = get_qp()
    cur_t = current.get("tickers", "")
    cur_b = current.get("buy", "")
    cur_q = current.get("qty", "")

    # 구버전은 리스트로 올 수 있어 정규화
    if isinstance(cur_t, list):
        cur_t = cur_t[0] if len(cur_t) else ""
    if isinstance(cur_b, list):
        cur_b = cur_b[0] if len(cur_b) else ""
    if isinstance(cur_q, list):
        cur_q = cur_q[0] if len(cur_q) else ""

    # 다를 때만 URL 갱신
    if (cur_t != desired["tickers"]) or (cur_b != desired["buy"]) or (cur_q != desired["qty"]):
        set_qp(**desired)

    # dirty 해제 (URL 변경으로 rerun 되더라도 다음 사이클에서 안정화)
    st.session_state["__qp_dirty__"] = False

st.caption("✅ 입력을 바꾸면 자동으로 URL에 저장됩니다. (새로고침/재접속/링크 공유 시 동일 입력 유지)")
