# app.py
import json
import secrets
import string
import time
from urllib.parse import quote_plus, unquote_plus

import requests
import streamlit as st

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")
st.title("📱📊 AI 포트폴리오 매니저 Pro (서버 저장 + 짧은 ID)")

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, 005930.KS, BTC-KRW"
DEFAULT_BUYPRICES = "150000, 300000, 400000, 70000, 50000000"
DEFAULT_QTYS = "10, 5, 3, 10, 0.01"

APP_BASE_URL_HINT = ""  # 비워도 됨. (원하면 본인 앱 URL을 넣어 안내 문구를 더 정확히 만들 수 있음)

# =========================
# URL Query Helpers
# =========================
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

# =========================
# Supabase REST Helpers
# =========================
def sb_headers():
    return {
        "apikey": st.secrets["SUPABASE_ANON_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_ANON_KEY']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def sb_table_url() -> str:
    return st.secrets["SUPABASE_URL"].rstrip("/") + "/rest/v1/portfolio_state"

def supabase_load_state(state_id: str) -> dict | None:
    url = sb_table_url()
    params = {"select": "data", "id": f"eq.{state_id}"}
    r = requests.get(url, headers=sb_headers(), params=params, timeout=10)
    if r.status_code != 200:
        return None
    arr = r.json()
    if not arr:
        return None
    data = arr[0].get("data")
    return data if isinstance(data, dict) else None

def supabase_upsert_state(state_id: str, payload: dict) -> bool:
    url = sb_table_url() + "?on_conflict=id"
    body = {"id": state_id, "data": payload}
    r = requests.post(url, headers=sb_headers(), json=body, timeout=10)
    return r.status_code in (200, 201, 204)

# =========================
# ID Generator (짧고 안전하게)
# =========================
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 혼동 문자 제거(0,O,1,I 등)
def generate_short_id(length: int = 10) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))

# =========================
# 상태(입력) 저장/복원
# =========================
def current_payload() -> dict:
    return {
        "tickers_input": st.session_state.get("tickers_input", DEFAULT_TICKERS),
        "buy_prices_input": st.session_state.get("buy_prices_input", DEFAULT_BUYPRICES),
        "quantities_input": st.session_state.get("quantities_input", DEFAULT_QTYS),
    }

def apply_payload(p: dict):
    st.session_state["tickers_input"] = str(p.get("tickers_input", DEFAULT_TICKERS))
    st.session_state["buy_prices_input"] = str(p.get("buy_prices_input", DEFAULT_BUYPRICES))
    st.session_state["quantities_input"] = str(p.get("quantities_input", DEFAULT_QTYS))

def mark_dirty():
    st.session_state["__dirty__"] = True

def ensure_state_id() -> str:
    qp = get_qp()
    sid = qp_get_str(qp, "id", "")
    if sid:
        return sid

    # id가 없는 접속(홈스크린이 base URL만 저장했거나 최초 접속) -> 임시 id 생성 후 URL에 박음
    new_id = generate_short_id(10)
    set_qp(id=new_id)
    return new_id

# =========================
# 부팅 시: id 확보 + 서버에서 복원
# =========================
if "__boot__" not in st.session_state:
    st.session_state["__boot__"] = True
    st.session_state["__dirty__"] = False
    st.session_state["__last_save_ts__"] = 0.0

    state_id = ensure_state_id()
    st.session_state["state_id"] = state_id

    # 서버에 값이 있으면 그걸 최우선으로 복원
    try:
        server = supabase_load_state(state_id)
    except Exception:
        server = None

    if isinstance(server, dict):
        apply_payload(server)
    else:
        # 서버에 아직 없으면 기본값으로 초기화하고 즉시 서버에 저장(첫 실행 고정)
        apply_payload(current_payload())
        try:
            supabase_upsert_state(state_id, current_payload())
        except Exception:
            pass

# =========================
# iOS 홈스크린 UX: 큰 CTA + 고정 링크 제공
# =========================
state_id = st.session_state.get("state_id", ensure_state_id())

st.markdown(
    f"""
**현재 설정 ID:** `{state_id}`  
- 이 ID가 “홈스크린 북마크가 들어올 때 불러오는 키”입니다.  
- 아이폰에서 URL/저장소가 초기화돼도, 이 ID로 서버에서 자동 복원됩니다.
"""
)

# 홈스크린용 링크(가장 중요)
# 실제 배포 URL을 모르더라도, 사용자는 주소창 공유→링크복사로 확보 가능
# 여기서는 “현재 페이지 URL”을 직접 읽어오는 기능은 Streamlit 기본만으로 제한이 있어 안내 중심으로 구성
home_link_hint = f"?id={state_id}"
st.success("✅ 홈스크린에 추가할 때는 반드시 `?id=...`가 포함된 링크로 들어간 뒤 추가해야 합니다.")
st.code(home_link_hint, language="text")

# 큰 버튼 UX
cA, cB, cC = st.columns([2, 2, 2])
with cA:
    if st.button("📌 홈스크린용 링크 생성/고정", use_container_width=True):
        # 현재 입력을 서버에 강제 저장
        ok = supabase_upsert_state(state_id, current_payload())
        if ok:
            st.toast("서버에 저장 완료! 이제 이 링크로 접속 후 홈 화면에 추가하세요.", icon="✅")
        else:
            st.toast("서버 저장 실패(네트워크/키 확인 필요)", icon="⚠️")

with cB:
    if st.button("🔄 지금 값 서버에서 다시 불러오기", use_container_width=True):
        server = supabase_load_state(state_id)
        if isinstance(server, dict):
            apply_payload(server)
            st.toast("서버 값으로 복원했습니다.", icon="✅")
            st.rerun()
        else:
            st.toast("서버에 저장된 값이 없습니다.", icon="⚠️")

with cC:
    if st.button("🆕 새 홈스크린 ID 만들기(별도 프로필)", use_container_width=True):
        new_id = generate_short_id(10)
        st.session_state["state_id"] = new_id
        set_qp(id=new_id)
        apply_payload(current_payload())
        supabase_upsert_state(new_id, current_payload())
        st.toast(f"새 ID 생성: {new_id}", icon="✅")
        st.rerun()

with st.expander("📱 아이폰 홈스크린 전용 안내(꼭 읽기)"):
    st.markdown(
        """
### 왜 홈스크린에서 리셋됐나?
iOS ‘홈 화면에 추가’는 **저장소(localStorage/cookie) 또는 URL 쿼리**를 상황에 따라 유지하지 않아서,
브라우저 저장 기반 방식이 깨질 수 있습니다.

### 이 버전은 왜 안 깨지나?
앱이 시작될 때 항상 **`id`로 서버에서 설정을 불러오고**, 입력 변경은 서버에 저장합니다.

### 홈스크린에서 100% 유지되게 하는 핵심 조건(딱 하나)
**홈 화면에 추가를 “id가 포함된 주소로 접속한 상태에서” 해야 합니다.**  
즉, 홈 화면 아이콘이 저장한 URL에 `?id=XXXX`가 포함되어야 합니다.
"""
    )

# =========================
# 입력 UI (변경 즉시 자동 저장)
# =========================
st.subheader("입력(자동 서버 저장)")

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input("종목코드 (쉼표 구분)", key="tickers_input", on_change=mark_dirty)
with col2:
    st.text_input("평단가 (원화 기준, 쉼표 구분)", key="buy_prices_input", on_change=mark_dirty)
with col3:
    st.text_input("수량 (쉼표 구분)", key="quantities_input", on_change=mark_dirty)

# 변경 감지 → 서버 자동 저장(디바운스)
if st.session_state.get("__dirty__", False):
    now = time.time()
    last = float(st.session_state.get("__last_save_ts__", 0.0))
    # 타이핑마다 저장 폭주 방지: 1.2초 디바운스
    if now - last > 1.2:
        ok = False
        try:
            ok = supabase_upsert_state(state_id, current_payload())
        except Exception:
            ok = False
        st.session_state["__last_save_ts__"] = now
        st.session_state["__dirty__"] = False
        if ok:
            st.caption("✅ 변경 내용이 서버에 자동 저장되었습니다. (홈스크린에서도 유지됨)")
        else:
            st.warning("서버 저장 실패: 네트워크 또는 Supabase 키/권한을 확인하세요.")

# =========================
# (데모) 현재 입력 표시
# - 여기 아래에 원래 쓰시던 RF/뉴스/환율/합산차트 로직을 그대로 붙이면 됩니다.
# =========================
st.divider()
st.subheader("현재 저장된 입력(서버 기반)")
st.write(current_payload())

with st.expander("💾 설정 JSON 백업/복원(옵션)"):
    backup = json.dumps(current_payload(), ensure_ascii=False, indent=2)
    st.download_button("⬇️ JSON 다운로드", data=backup.encode("utf-8"), file_name="portfolio_state.json", mime="application/json")

    up = st.file_uploader("⬆️ JSON 업로드", type=["json"])
    if up is not None:
        try:
            loaded = json.loads(up.read().decode("utf-8"))
            apply_payload(loaded)
            supabase_upsert_state(state_id, current_payload())
            st.success("복원 후 서버 저장 완료!")
            st.rerun()
        except Exception as e:
            st.error(f"업로드 JSON 파싱 실패: {e}")
