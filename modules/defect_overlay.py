# modules/defect_overlay.py
# 결함 오버레이 모듈
# wafer_app_global.py의 get_wafer_grid, add_wafer_outline, _wafer_layout을 import해서 사용
#
# =============================================================================
# [설계 결정 근거]
# =============================================================================
#
# ① 기존 Figure에 trace 추가 방식이 아닌 새 Figure 생성 방식 선택
#    기존 방식의 치명적 문제:
#      cached_fig = create_2d_heatmap(df_json, ...)  # @st.cache_data 반환값
#      cached_fig.add_trace(defect_scatter)           # 캐시된 객체를 직접 변경!
#      → Streamlit의 @st.cache_data는 캐시 반환 시 깊은 복사(deepcopy) 없이
#        참조를 반환할 수 있음 → 캐시된 Figure가 영구 변경됨
#      → 다음 호출 시 이미 결함 traces가 쌓인 Figure가 반환 → 중복 오버레이 버그
#    새 Figure 생성 방식:
#      create_defect_overlaid_map(@st.cache_data) {
#          get_wafer_grid(wafer_df_json, resolution)  # 하위 캐시 재사용 ✅
#          → Heatmap trace + 아웃라인 + 결함 traces 모두 새 Figure에 구성
#      }
#      → 캐시 오염 없음, 완전한 상태 제어 가능
#
# ② build_defect_traces는 @st.cache_data 미적용 내부 헬퍼로 구현
#    go.Scatter 리스트를 @st.cache_data로 캐시하면:
#      - Streamlit이 pickle 직렬화/역직렬화 → 비용 발생
#      - 반환된 trace 객체가 외부에서 변경되면 캐시 내용도 오염 위험
#    → create_defect_overlaid_map 안에서 직접 호출하는 순수 내부 헬퍼로 구현
#
# ③ 결함 클래스 수가 많을 때 심볼/컬러 할당 전략
#    클래스 수 ≤ 8:  심볼 1:1 할당 (명확히 구별)
#    클래스 수 9~N:  심볼(8종) × 색상팔레트 조합으로 최대 8×24 = 192가지 지원
#      symbol_idx = class_idx % 8
#      color_idx  = class_idx // 8 % len(COLOR_PALETTE)
#      → 심볼이 같아도 색상이 달라 구별 가능
#
# ④ 좌표계 불일치 처리 전략
#    자동 탐지: 결함 좌표 범위 vs 웨이퍼 반지름 비교
#      결함 max_coord > 5 × radius → μm vs mm 단위 불일치 가능성 → 경고
#      결함 max_coord < radius / 100 → 반대 방향 불일치 가능성 → 경고
#    사용자 처리: 좌표 스케일 팩터 selectbox (×1, ×0.001, ×25.4, 직접입력)
#
# ⑤ load_defect_file의 @st.cache_data 적용
#    full_path: str → hashable ✅
#    파일 내용 변경 시: Streamlit 1.35+의 @st.cache_data는 파일 경로 기반 캐시
#    → 파일이 변경돼도 경로가 같으면 캐시 히트 (파일 내용 hash 미지원)
#    → 엔지니어가 파일 교체 후 "재로드" 버튼을 명시적으로 누르는 UX 설계
# =============================================================================

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
import glob
import os

# ── 외부 라이브러리 ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── wafer_app_global 핵심 함수 import ────────────────────────────────────────
# add_wafer_outline: 단일 Figure 전용 (row/col 없음) → 여기서는 직접 사용 가능
# _wafer_layout: 단일 Figure 전용 → create_defect_overlaid_map에서 재사용
from app import _wafer_layout  # 원형 유지 공통 레이아웃 딕셔너리 반환
from app import add_wafer_outline  # 웨이퍼 원형 테두리 + Notch 추가
from app import get_wafer_grid  # 불규칙 산점 → 균일 그리드 보간 (@st.cache_data 적용)

# =============================================================================
# session_state 키 상수 (prefix: "def_")
# =============================================================================
# 기존 wafer_app_global 키: data_folder, show_folder_browser, browser_current,
#                           datasets, _s_file, _s_display, _s_col_map
# 다중 파라미터 모듈 키: mp_x_col, mp_y_col, mp_param_cols, mp_share_scale
# 결함 오버레이 전용 키 (충돌 없음):
_SS_FILE      = "def_file"         # 선택된 결함 파일 경로 (str)
_SS_BASE_MAP  = "def_base_map"     # 베이스 맵 타입: "Heatmap" or "Contour"
_SS_CLASSES   = "def_classes"      # 선택된 클래스 multiselect 값 (list)
_SS_OUTSIDE   = "def_show_outside" # 웨이퍼 외부 결함 포함 여부 (bool)
_SS_SCALE     = "def_coord_scale"  # 좌표 스케일 팩터 (float)


# =============================================================================
# 결함 심볼 및 컬러 팔레트 상수
# =============================================================================
# 심볼: Plotly 실선(non-outline) 8종 — 서로 모양이 명확히 다른 것만 선택
_DEFECT_SYMBOLS = [
    "circle",        # ●  가장 기본
    "square",        # ■  4각형
    "diamond",       # ◆  45° 회전 사각형
    "cross",         # +  십자
    "x",             # ×  X자 (결함 표시 전통 기호)
    "triangle-up",   # ▲  위 삼각형
    "triangle-down", # ▼  아래 삼각형
    "star",          # ★  별
]

# Plotly D3 + Safe 결합 팔레트 (가독성 높은 24색 순환)
# 인접 색상이 충분히 구별되도록 밝기/채도 다양하게 구성
_COLOR_PALETTE = [
    "#E41A1C",  # 빨강
    "#377EB8",  # 파랑
    "#4DAF4A",  # 초록
    "#FF7F00",  # 주황
    "#984EA3",  # 보라
    "#A65628",  # 갈색
    "#F781BF",  # 분홍
    "#00CED1",  # 청록
    "#FFD700",  # 금색
    "#32CD32",  # 라임그린
    "#FF69B4",  # 핫핑크
    "#1E90FF",  # 도저블루
    "#FF6347",  # 토마토
    "#7B68EE",  # 미디엄슬레이트블루
    "#00FA9A",  # 미디엄스프링그린
    "#FFA500",  # 오렌지
    "#DC143C",  # 크림슨
    "#00BFFF",  # 딥스카이블루
    "#ADFF2F",  # 그린옐로우
    "#FF4500",  # 오렌지레드
    "#9370DB",  # 미디엄퍼플
    "#20B2AA",  # 라이트시그린
    "#FF1493",  # 딥핑크
    "#228B22",  # 포레스트그린
]

# 마커 크기 정규화 범위 (픽셀)
_MARKER_SIZE_MIN = 4
_MARKER_SIZE_MAX = 18
_MARKER_SIZE_DEFAULT = 10   # "size" 컬럼 없을 때 고정 크기


# =============================================================================
# [함수 1] load_defect_file
# =============================================================================

@st.cache_data
def load_defect_file(full_path: str) -> pd.DataFrame | None:
    """
    결함 데이터 CSV/Excel 파일 로드 및 전처리.

    [@st.cache_data 적용 이유]
    full_path: str → hashable → 캐시 키로 사용 가능.
    사이드바 슬라이더, 컬러스케일 변경 등 다른 UI 조작 시 재로드 방지.

    [표준화 처리]
    1. 컬럼명 소문자 정규화 → 대소문자 무관한 인식 (X→x, Y→y 등)
    2. "class" 컬럼 없으면 "Unknown" 단일 클래스로 채움
       → build_defect_traces에서 항상 class 컬럼 존재 보장
    3. "size" 컬럼 없으면 _MARKER_SIZE_DEFAULT 고정값 채움
       → 마커 크기 정규화 로직을 단순화
    4. "description" 컬럼 없으면 빈 문자열 채움 → hover 템플릿 통일

    [필수 컬럼 검증]
    x, y 컬럼이 없으면 None 반환 (오류는 호출부에서 st.error 처리)

    인자:
        full_path: CSV/Excel 파일의 절대 경로

    반환:
        pd.DataFrame: 표준화된 결함 데이터프레임
        None        : x/y 컬럼 없거나 파일 로드 실패 시
    """
    # ── 파일 로드 ─────────────────────────────────────────────────────────────
    try:
        if full_path.lower().endswith(".csv"):
            df = pd.read_csv(full_path)
        else:
            df = pd.read_excel(full_path, sheet_name=0)
    except Exception as e:
        # @st.cache_data 내부에서 st.error를 직접 호출하면 안 됨
        # → None 반환 후 호출부에서 처리
        return None

    if df.empty:
        return None

    # ── 컬럼명 소문자 정규화 ──────────────────────────────────────────────────
    # 원본 컬럼명을 보존하면서 매핑 딕셔너리 생성
    # 예: "X" → "x", "Defect_Type" → "defect_type", "SIZE" → "size"
    col_lower_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower_map)

    # ── 필수 컬럼 검증: x, y ──────────────────────────────────────────────────
    # 소문자 정규화 후에도 x, y가 없으면 결함 위치를 특정할 수 없음 → None 반환
    if "x" not in df.columns or "y" not in df.columns:
        return None

    # ── x, y 컬럼 숫자형 변환 ────────────────────────────────────────────────
    # CSV에서 문자열로 읽힌 경우 대비 (예: "100.5" → 100.5)
    # errors="coerce": 변환 불가 값은 NaN으로 처리
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # x 또는 y가 NaN인 행 제거 (위치 불명 결함은 시각화 불가)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    if df.empty:
        return None

    # ── 선택 컬럼 기본값 설정 ─────────────────────────────────────────────────
    # "class" 컬럼: 없으면 모든 결함을 "Unknown" 단일 클래스로 취급
    # → build_defect_traces에서 항상 groupby("class") 가능
    if "class" not in df.columns:
        # 다른 가능한 클래스 컬럼명 자동 탐색 (Type, Category, Defect_Type 등)
        class_candidates = ["type", "defect_type", "category",
                            "defect", "label", "kind", "group"]
        found_class = next((c for c in class_candidates if c in df.columns), None)
        if found_class:
            # 발견된 컬럼을 "class"로 rename
            df = df.rename(columns={found_class: "class"})
        else:
            df["class"] = "Unknown"

    # "class" 값의 NA를 "Unknown"으로 채움
    df["class"] = df["class"].fillna("Unknown").astype(str)

    # "size" 컬럼: 없으면 고정 기본값, 있으면 숫자형 변환
    if "size" not in df.columns:
        df["size"] = float(_MARKER_SIZE_DEFAULT)
    else:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        # 변환 실패한 size 값은 기본값으로 채움
        df["size"] = df["size"].fillna(float(_MARKER_SIZE_DEFAULT))

    # "description" 컬럼: hover 툴팁에 사용
    if "description" not in df.columns:
        df["description"] = ""
    else:
        df["description"] = df["description"].fillna("").astype(str)

    return df


# =============================================================================
# [함수 2] _assign_class_styles (내부 헬퍼)
# =============================================================================

def _assign_class_styles(classes: list[str]) -> dict[str, dict]:
    """
    결함 클래스 목록에 심볼과 색상을 할당하여 딕셔너리 반환.

    [할당 전략]
    클래스 수 ≤ 8: 심볼 1:1 할당 (색상도 팔레트 순서대로)
    클래스 수 9+: 심볼(8종) × 색상(24종) 조합으로 최대 192가지 지원
      symbol_idx = class_idx % 8         → 심볼 8종 순환
      color_idx  = class_idx // 8 % 24   → 색상 24종 순환 (심볼 1바퀴마다 전환)
      예: 클래스 9번 → 심볼 1번(circle)로 돌아오지만 색상은 다음 팔레트 색 사용

    인자:
        classes: 고유 클래스명 리스트 (순서가 할당 순서가 됨)

    반환:
        {class_name: {"symbol": str, "color": str}} 딕셔너리
    """
    styles = {}
    for i, cls in enumerate(classes):
        symbol_idx = i % len(_DEFECT_SYMBOLS)
        # 심볼 1바퀴(8개) 마다 다음 색상 그룹으로 전환
        color_idx  = (i // len(_DEFECT_SYMBOLS) + i) % len(_COLOR_PALETTE)
        styles[cls] = {
            "symbol": _DEFECT_SYMBOLS[symbol_idx],
            "color":  _COLOR_PALETTE[color_idx],
        }
    return styles


# =============================================================================
# [함수 3] _normalize_marker_sizes (내부 헬퍼)
# =============================================================================

def _normalize_marker_sizes(sizes: np.ndarray) -> np.ndarray:
    """
    결함 크기(size) 컬럼 값을 마커 픽셀 크기 범위[MIN, MAX]로 MinMax 정규화.

    [정규화 공식]
    size_px = (size - size_min) / (size_max - size_min) × (MAX - MIN) + MIN

    [엣지 케이스 처리]
    - 모든 size가 동일한 값: 분모 = 0 → 정규화 불가 → 기본값 사용
    - size < 0: 물리적으로 불가능 → 절댓값 사용 (부호 오류 가능성)
    - NaN: _MARKER_SIZE_DEFAULT로 대체

    인자:
        sizes: 결함 크기 값 배열 (원본 단위, 양수 기대)

    반환:
        마커 픽셀 크기 배열 [_MARKER_SIZE_MIN, _MARKER_SIZE_MAX] 범위
    """
    # NaN을 기본값으로 대체
    sizes = np.where(np.isnan(sizes), float(_MARKER_SIZE_DEFAULT), sizes)
    # 음수 크기는 절댓값으로 처리
    sizes = np.abs(sizes)

    s_min = sizes.min()
    s_max = sizes.max()

    if s_max - s_min < 1e-10:
        # 모든 값이 동일 → 정규화 불가 → 중간값으로 고정
        return np.full_like(sizes, float((_MARKER_SIZE_MIN + _MARKER_SIZE_MAX) / 2))

    normalized = (sizes - s_min) / (s_max - s_min)
    return normalized * (_MARKER_SIZE_MAX - _MARKER_SIZE_MIN) + _MARKER_SIZE_MIN


# =============================================================================
# [함수 4] _build_defect_traces (내부 헬퍼, 캐시 미적용)
# =============================================================================

def _build_defect_traces(
    df_defect: pd.DataFrame,
    selected_classes: tuple,
    wafer_radius: float,
    show_outside: bool,
    coord_scale: float = 1.0,
) -> list[go.Scatter]:
    """
    결함 데이터프레임에서 클래스별 go.Scatter trace 리스트 생성.

    [@st.cache_data 미적용 이유]
    go.Scatter 리스트를 캐시하면 pickle 직렬화 오버헤드 + 반환된 trace 객체가
    외부에서 수정될 위험이 있음.
    → create_defect_overlaid_map(@st.cache_data) 내부에서만 호출하는 헬퍼로 설계.
    → 전체 Figure 캐시(상위)가 미스일 때만 이 함수가 실행됨.

    [마커 크기 처리]
    all_sizes_same = (df_defect["size"].nunique() == 1)
    → 모두 같으면 정규화 의미 없음 → 클래스별로 고정 크기 사용

    [좌표 스케일 팩터]
    coord_scale != 1.0이면 x, y에 곱한 후 웨이퍼 좌표계와 맞춤
    예: μm 단위 결함 파일에서 mm 웨이퍼 좌표계로 변환 시 coord_scale=0.001

    인자:
        df_defect       : load_defect_file 반환값 (표준화된 결함 DataFrame)
        selected_classes: 표시할 클래스명 tuple
        wafer_radius    : 웨이퍼 반지름 (mm), 원 밖 필터링에 사용
        show_outside    : True이면 웨이퍼 원 밖 결함도 포함
        coord_scale     : 결함 좌표에 곱할 스케일 팩터

    반환:
        list[go.Scatter]: fig.add_traces()로 일괄 추가 가능한 trace 리스트
    """
    if df_defect is None or df_defect.empty:
        return []

    # 스케일 팩터 적용 (1.0이 아닐 때만 실제 변환)
    if abs(coord_scale - 1.0) > 1e-10:
        df_defect = df_defect.copy()
        df_defect["x"] = df_defect["x"] * coord_scale
        df_defect["y"] = df_defect["y"] * coord_scale

    # 선택된 클래스만 필터링
    df_filtered = df_defect[df_defect["class"].isin(selected_classes)].copy()

    if df_filtered.empty:
        return []

    # 웨이퍼 원 밖 결함 필터링 (show_outside=False일 때)
    if not show_outside:
        inside_mask = (df_filtered["x"] ** 2 + df_filtered["y"] ** 2 <= wafer_radius ** 2)
        df_filtered = df_filtered[inside_mask].copy()

    if df_filtered.empty:
        return []

    # 클래스 스타일 할당 (selected_classes 순서로 일관성 유지)
    styles = _assign_class_styles(list(selected_classes))

    # 전체 크기 정규화 여부 판단
    # 모든 size 값이 동일하면 정규화 의미 없음 → 고정 크기 사용
    all_same_size = (df_defect["size"].nunique() == 1)

    traces = []
    for cls in selected_classes:
        # 이 클래스에 해당하는 결함만 추출
        cls_df = df_filtered[df_filtered["class"] == cls]
        if cls_df.empty:
            continue

        style = styles.get(cls, {"symbol": "x", "color": "#E41A1C"})

        # ── 마커 크기 계산 ────────────────────────────────────────────────────
        if all_same_size:
            # 모든 결함이 같은 크기 → 클래스 인덱스로 약간 차별화 (8~12px)
            idx       = list(selected_classes).index(cls)
            sizes_px  = np.full(len(cls_df), float(_MARKER_SIZE_DEFAULT + idx % 4))
        else:
            # size 컬럼 값을 픽셀 범위로 MinMax 정규화
            # ★ 클래스별 독립 정규화가 아닌 이 클래스의 원본 크기를 직접 정규화
            # → 클래스 내 상대적 크기 차이를 마커로 표현 (큰 결함 = 큰 마커)
            cls_size_raw = cls_df["size"].values
            sizes_px = _normalize_marker_sizes(cls_size_raw)

        # ── hover 템플릿 구성 ─────────────────────────────────────────────────
        # %{text}: go.Scatter의 text 인자로 전달되는 추가 정보
        # customdata: description 컬럼 (빈 문자열 가능)
        has_desc = cls_df["description"].ne("").any()
        if has_desc:
            hover_template = (
                f"<b>클래스: {cls}</b><br>"
                "위치: (%{x:.1f}, %{y:.1f}) mm<br>"
                "크기: %{marker.size:.1f}<br>"
                "설명: %{customdata}<extra></extra>"
            )
        else:
            hover_template = (
                f"<b>클래스: {cls}</b><br>"
                "위치: (%{x:.1f}, %{y:.1f}) mm<br>"
                "크기: %{marker.size:.1f}<extra></extra>"
            )

        traces.append(
            go.Scatter(
                x=cls_df["x"].values,
                y=cls_df["y"].values,
                mode="markers",
                name=f"결함: {cls}",          # 범례에 표시
                text=[cls] * len(cls_df),      # hover용 텍스트 (클래스명)
                customdata=cls_df["description"].values,
                marker=dict(
                    symbol=style["symbol"],
                    size=sizes_px.tolist(),
                    color=style["color"],
                    opacity=0.85,
                    line=dict(
                        width=1.5,
                        color="rgba(0,0,0,0.6)",   # 반투명 검정 테두리 → 배경과 구별
                    ),
                ),
                hovertemplate=hover_template,
                showlegend=True,
            )
        )

    return traces


# =============================================================================
# [함수 5] create_defect_overlaid_map (@st.cache_data 적용)
# =============================================================================

@st.cache_data
def create_defect_overlaid_map(
    wafer_df_json: str,
    defect_df_json: str,
    selected_classes: tuple,   # tuple 필수: list는 @st.cache_data 해시 불가
    resolution: int,
    colorscale: str,
    base_map_type: str,        # "heatmap" 또는 "contour"
    show_outside: bool,
    coord_scale: float = 1.0,
    n_contours: int = 20,
) -> go.Figure:
    """
    웨이퍼 맵(Heatmap/Contour) 위에 결함 오버레이된 통합 Figure 생성.

    [캐시 키 구성 요소]
    (wafer_df_json, defect_df_json, selected_classes, resolution,
     colorscale, base_map_type, show_outside, coord_scale, n_contours)
    - 결함 파일 변경 → defect_df_json 달라짐 → 자동 캐시 갱신
    - 클래스 필터 변경 → selected_classes tuple 달라짐 → 자동 캐시 갱신
    - 웨이퍼 데이터 편집 → wafer_df_json 달라짐 → 자동 캐시 갱신

    [새 Figure 생성 방식 선택 이유]
    기존 create_2d_heatmap 캐시에서 반환된 Figure를 직접 수정하면
    캐시 오염(mutation) 발생 → 다음 호출 시 결함 traces가 중복 누적되는 버그.
    → 이 함수 내에서 get_wafer_grid(하위 캐시 재사용)로 새 Figure를 구성.

    [z-order (trace 순서)]
    1. Heatmap/Contour (베이스 맵) — 맨 아래
    2. 웨이퍼 아웃라인 Scatter     — 맵 위에 테두리
    3. 결함 Scatter traces         — 최상위 (클릭/호버 가능)

    인자:
        wafer_df_json   : 웨이퍼 측정 데이터 JSON (x, y, data 컬럼 필수)
        defect_df_json  : 결함 데이터 JSON (표준화 완료 상태)
        selected_classes: 표시할 클래스명 tuple
        resolution      : 보간 그리드 해상도 (30~200)
        colorscale      : Plotly 컬러스케일 이름
        base_map_type   : "heatmap" or "contour"
        show_outside    : True이면 웨이퍼 원 밖 결함도 표시
        coord_scale     : 결함 좌표 스케일 팩터 (기본 1.0)
        n_contours      : Contour 등고선 수 (base_map_type="contour" 시만 사용)

    반환:
        go.Figure: 오버레이 완성된 Figure
    """
    # ── 웨이퍼 그리드 보간 (하위 캐시 재사용) ──────────────────────────────
    # get_wafer_grid는 wafer_app_global에서 @st.cache_data 적용됨
    # → 이전에 동일 wafer_df_json + resolution으로 호출됐으면 캐시 히트
    XI, YI, ZI, radius = get_wafer_grid(wafer_df_json, resolution)

    # ── 새 Figure 생성 ──────────────────────────────────────────────────────
    fig = go.Figure()

    # ── 베이스 맵 trace 추가 ────────────────────────────────────────────────
    if base_map_type == "contour":
        fig.add_trace(go.Contour(
            x=XI[0],
            y=YI[:, 0],
            z=ZI,
            colorscale=colorscale,
            ncontours=n_contours,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.8, color="rgba(0,0,0,0.5)"),
            colorbar=dict(
                thickness=12,
                len=0.75,
                title=dict(text="Data", side="right"),
            ),
            connectgaps=False,
            name="측정값",
            showlegend=False,    # 베이스 맵은 범례에서 제외 (결함 클래스만 표시)
        ))
    else:
        # 기본: Heatmap
        fig.add_trace(go.Heatmap(
            x=XI[0],
            y=YI[:, 0],
            z=ZI,
            colorscale=colorscale,
            zsmooth="best",
            colorbar=dict(
                thickness=12,
                len=0.75,
                title=dict(text="Data", side="right"),
            ),
            connectgaps=False,
            name="측정값",
            showlegend=False,    # 베이스 맵은 범례에서 제외
        ))

    # ── 웨이퍼 아웃라인 추가 ────────────────────────────────────────────────
    # add_wafer_outline: 단일 Figure 전용 (row/col 없음) → 여기서는 직접 사용 가능
    # 결함 scatter보다 아래 z-order에 오도록 outline을 먼저 추가
    add_wafer_outline(fig, radius)

    # ── 결함 traces 추가 ────────────────────────────────────────────────────
    # @st.cache_data 함수 내부에서 pd.read_json으로 역직렬화
    df_defect = pd.read_json(defect_df_json)

    defect_traces = _build_defect_traces(
        df_defect=df_defect,
        selected_classes=selected_classes,
        wafer_radius=radius,
        show_outside=show_outside,
        coord_scale=coord_scale,
    )

    # fig.add_traces(): trace 리스트를 한 번에 추가 (개별 add_trace 루프보다 효율적)
    if defect_traces:
        fig.add_traces(defect_traces)

    # ── 레이아웃 설정 ────────────────────────────────────────────────────────
    # _wafer_layout: 원형 유지 공통 레이아웃 (scaleanchor="y", 범위 설정 등)
    layout = _wafer_layout(radius, height=520)
    # 결함 범례 추가 설정 (베이스 맵은 showlegend=False이므로 결함만 범례 표시)
    layout.update({
        "showlegend": True,
        "legend": dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",   # 마커 크기와 무관하게 범례 아이콘 크기 일정
            x=1.02,                  # 차트 오른쪽 외부에 배치
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        "margin": dict(l=35, r=150, t=30, b=35),  # r=150: 범례 공간 확보
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# [함수 6] _check_coord_mismatch (내부 헬퍼)
# =============================================================================

def _check_coord_mismatch(
    df_defect: pd.DataFrame,
    wafer_radius: float,
    coord_scale: float,
) -> str | None:
    """
    결함 좌표 범위와 웨이퍼 반지름을 비교하여 단위 불일치 경고 생성.

    [판단 기준]
    스케일 적용 후 결함 좌표의 최대 절댓값을 wafer_radius와 비교:
    - max_abs > 5 × radius: 결함 좌표가 너무 큼 → μm vs mm 불일치 가능성
    - max_abs < radius / 100: 결함 좌표가 너무 작음 → 반대 방향 불일치 가능성
    - 0.01 × radius ≤ max_abs ≤ 3 × radius: 정상 범위

    인자:
        df_defect   : 표준화된 결함 DataFrame
        wafer_radius: 웨이퍼 반지름 (mm)
        coord_scale : 현재 적용 중인 스케일 팩터

    반환:
        str  : 경고 메시지 (문제 있을 때)
        None : 정상 범위 (경고 없음)
    """
    if df_defect is None or df_defect.empty:
        return None

    scaled_x = df_defect["x"] * coord_scale
    scaled_y = df_defect["y"] * coord_scale
    max_abs   = max(scaled_x.abs().max(), scaled_y.abs().max())

    if max_abs > 5.0 * wafer_radius:
        ratio = max_abs / wafer_radius
        return (
            f"⚠️ 결함 좌표 범위({max_abs:.1f} mm)가 웨이퍼 반지름({wafer_radius:.1f} mm)의 "
            f"{ratio:.0f}배입니다. 단위가 다를 수 있습니다. "
            f"(예: μm 단위라면 스케일 팩터 × 0.001 사용)"
        )
    elif max_abs < wafer_radius / 100.0 and max_abs > 0:
        ratio = wafer_radius / max_abs if max_abs > 0 else float("inf")
        return (
            f"⚠️ 결함 좌표 범위({max_abs:.4f} mm)가 웨이퍼 반지름({wafer_radius:.1f} mm)의 "
            f"1/{ratio:.0f}입니다. 단위가 다를 수 있습니다. "
            f"(예: m 단위라면 스케일 팩터 × 1000 사용)"
        )
    return None


# =============================================================================
# [함수 7] render_defect_tab (UI 렌더러)
# =============================================================================

def render_defect_tab(
    wafer_df_json: str,
    wafer_radius: float,
    resolution: int,
    colorscale: str,
    data_folder: str,
) -> None:
    """
    결함 오버레이 탭의 전체 UI를 렌더링.

    [레이아웃 구조]
    ┌─────────────────────────┬─────────────────────────┐
    │ 파일 선택 영역            │ 결함 통계 영역             │
    │  • 결함 파일 selectbox   │  • 총 결함 수 metric       │
    │  • 스케일 팩터 selectbox  │  • 클래스별 분포 metric    │
    │  • [파일 로드] 버튼        │  • 웨이퍼 내/외부 비율     │
    └─────────────────────────┴─────────────────────────┘
    ┌─────────────────────────────────────────────────────┐
    │ 베이스 맵 타입: ◉ Heatmap  ○ Contour                 │
    └─────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────┐
    │ 클래스 필터: [multiselect — 기본값: 전체 선택]         │
    └─────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────┐
    │ [☐] 웨이퍼 외부 결함 포함                             │
    └─────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────┐
    │  결함 오버레이 맵 (Plotly 차트)                        │
    └─────────────────────────────────────────────────────┘
    ▾ 결함 데이터 테이블 (expander)

    [session_state 흐름]
    1. 사용자가 selectbox에서 결함 파일 선택
    2. "파일 로드" 버튼 클릭 → def_file 업데이트 → st.rerun()
    3. load_defect_file(def_file) → 캐시 또는 신규 로드
    4. 클래스 목록 추출 → multiselect 옵션 업데이트
    5. create_defect_overlaid_map 호출 → Figure 렌더링

    인자:
        wafer_df_json : 현재 웨이퍼 측정 데이터 JSON (x, y, data 컬럼)
        wafer_radius  : 웨이퍼 반지름 (mm) — get_wafer_grid로 계산된 값
        resolution    : 보간 해상도 (사이드바 슬라이더)
        colorscale    : 컬러스케일 이름 (사이드바 selectbox)
        data_folder   : 데이터 폴더 경로 (결함 파일 목록 수집용)
    """
    # ── 결함 파일 목록 수집 ──────────────────────────────────────────────────
    # data_folder에서 CSV, XLSX, XLS 파일 목록 수집
    csv_files  = glob.glob(os.path.join(data_folder, "*.csv"))
    xlsx_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
    xls_files  = glob.glob(os.path.join(data_folder, "*.xls"))
    all_files  = sorted(csv_files + xlsx_files + xls_files)

    # 표시용 파일명 → 전체 경로 매핑 딕셔너리
    file_options: dict[str, str] = {os.path.basename(f): f for f in all_files}

    # ── 2컬럼 레이아웃: 파일 선택 | 통계 ─────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("##### 📁 결함 파일 선택")

        if not file_options:
            st.warning(
                f"⚠️ `{data_folder}` 폴더에 CSV/Excel 파일이 없습니다. "
                "사이드바에서 데이터 폴더를 변경하거나 결함 파일을 추가하세요."
            )
            # 파일이 없어도 탭 자체가 오류 없이 표시되도록 return하지 않음
            loaded_df = None
        else:
            # 현재 선택된 파일이 목록에 없으면 초기화
            current_file = st.session_state.get(_SS_FILE, "")
            current_basename = os.path.basename(current_file) if current_file else ""
            default_idx = (
                list(file_options.keys()).index(current_basename)
                if current_basename in file_options
                else 0
            )

            selected_basename = st.selectbox(
                "결함 파일",
                options=list(file_options.keys()),
                index=default_idx,
                key="def_file_select",
                help="결함 좌표(x, y)가 포함된 CSV/Excel 파일을 선택하세요.",
            )

            # ── 좌표 스케일 팩터 selectbox ─────────────────────────────────
            scale_options = {
                "× 1.0   (mm, 기본값)":    1.0,
                "× 0.001 (μm → mm)":       0.001,
                "× 1000  (m → mm)":        1000.0,
                "× 25.4  (inch → mm)":     25.4,
                "× 10.0  (cm → mm)":       10.0,
            }
            selected_scale_label = st.selectbox(
                "결함 좌표 단위 변환",
                options=list(scale_options.keys()),
                index=0,
                key=_SS_SCALE,
                help=(
                    "결함 파일의 좌표 단위가 웨이퍼 맵과 다를 때 스케일을 조정합니다.\n"
                    "웨이퍼 맵은 mm 단위를 사용합니다."
                ),
            )
            coord_scale = scale_options[selected_scale_label]

            # ── 파일 로드 버튼 ──────────────────────────────────────────────
            # 버튼 클릭 시에만 session_state 업데이트 → 불필요한 재계산 방지
            if st.button(
                "📂 파일 로드",
                key="def_load_btn",
                use_container_width=True,
                type="primary",
            ):
                selected_path = file_options[selected_basename]
                st.session_state[_SS_FILE] = selected_path
                # 파일 변경 시 클래스 필터 초기화 (이전 파일의 클래스 목록 제거)
                if _SS_CLASSES in st.session_state:
                    del st.session_state[_SS_CLASSES]
                st.rerun()

            # ── 현재 로드된 파일 처리 ────────────────────────────────────
            current_path = st.session_state.get(_SS_FILE, "")
            if not current_path or not os.path.exists(current_path):
                st.info("📋 위에서 결함 파일을 선택하고 [파일 로드] 버튼을 눌러주세요.")
                loaded_df = None
            else:
                loaded_df = load_defect_file(current_path)

                if loaded_df is None:
                    st.error(
                        f"❌ 파일 로드 실패 또는 x/y 컬럼을 찾을 수 없습니다.\n\n"
                        f"**파일**: `{os.path.basename(current_path)}`\n\n"
                        "결함 파일에 x와 y 좌표 컬럼이 있어야 합니다. "
                        "컬럼명은 대소문자 무관합니다. (예: X, x, X_pos 미지원 → X만 지원)"
                    )
                else:
                    st.success(
                        f"✅ `{os.path.basename(current_path)}` 로드 완료 "
                        f"— 총 {len(loaded_df):,}개 결함"
                    )

                    # 좌표 불일치 경고
                    mismatch_warning = _check_coord_mismatch(
                        loaded_df, wafer_radius, coord_scale
                    )
                    if mismatch_warning:
                        st.warning(mismatch_warning)

    # ── 오른쪽: 결함 통계 ────────────────────────────────────────────────────
    with col_right:
        if loaded_df is not None and not loaded_df.empty:
            st.markdown("##### 📊 결함 통계")

            total_count  = len(loaded_df)
            class_counts = loaded_df["class"].value_counts()
            n_classes    = len(class_counts)

            # 웨이퍼 내/외부 비율 계산
            inside_mask  = (
                (loaded_df["x"] * coord_scale) ** 2 +
                (loaded_df["y"] * coord_scale) ** 2
                <= wafer_radius ** 2
            )
            n_inside  = int(inside_mask.sum())
            n_outside = total_count - n_inside
            inside_pct = (n_inside / total_count * 100) if total_count > 0 else 0.0

            # 총 결함 수 + 클래스 수 메트릭
            m1, m2 = st.columns(2)
            m1.metric("총 결함", f"{total_count:,}개")
            m2.metric("클래스 수", f"{n_classes}종")

            # 웨이퍼 내/외부 비율 메트릭
            m3, m4 = st.columns(2)
            m3.metric(
                "웨이퍼 내부",
                f"{n_inside:,}개",
                delta=f"{inside_pct:.1f}%",
                delta_color="off",
            )
            m4.metric(
                "웨이퍼 외부",
                f"{n_outside:,}개",
                delta=f"{(100 - inside_pct):.1f}%",
                delta_color="off",
            )

            # 클래스별 분포 (최대 5개 표시)
            if n_classes <= 8:
                st.markdown("**클래스별 결함 수:**")
                class_metric_cols = st.columns(min(n_classes, 4))
                for j, (cls_name, cnt) in enumerate(class_counts.head(8).items()):
                    col_idx = j % len(class_metric_cols)
                    pct     = cnt / total_count * 100
                    class_metric_cols[col_idx].metric(
                        label=cls_name[:12] if len(cls_name) > 12 else cls_name,
                        value=f"{cnt:,}",
                        delta=f"{pct:.1f}%",
                        delta_color="off",
                    )
            else:
                # 클래스가 많으면 테이블로 표시
                st.dataframe(
                    class_counts.rename("개수")
                    .reset_index()
                    .rename(columns={"index": "클래스"}),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── 결함 데이터 없으면 이후 렌더링 중단 ─────────────────────────────────
    if loaded_df is None or loaded_df.empty:
        st.info(
            "ℹ️ 결함 파일을 선택하고 [파일 로드] 버튼을 눌러 결함 오버레이 맵을 생성하세요."
        )
        return

    # ── 베이스 맵 타입 선택 ──────────────────────────────────────────────────
    st.markdown("---")
    base_map_type = st.radio(
        "베이스 맵 타입",
        options=["Heatmap", "Contour"],
        horizontal=True,
        key=_SS_BASE_MAP,
        help=(
            "**Heatmap**: 연속적인 색상 그라데이션 → 두께/저항 분포 파악에 유리\n\n"
            "**Contour**: 등고선 표시 → 레벨 경계 명확히 구분"
        ),
    )

    # ── 클래스 필터 multiselect ───────────────────────────────────────────────
    all_classes = sorted(loaded_df["class"].unique().tolist())

    # 기본값: 전체 클래스 선택
    # 파일이 바뀌었을 때 session_state에 이전 파일의 클래스가 남아있을 수 있음
    # → 현재 파일의 클래스 목록에서만 유효한 값을 기본값으로 사용
    previous_selection = st.session_state.get(_SS_CLASSES, all_classes)
    valid_default      = [c for c in previous_selection if c in all_classes]
    if not valid_default:
        valid_default  = all_classes  # 유효한 이전 선택이 없으면 전체 선택

    selected_classes_list: list[str] = st.multiselect(
        f"결함 클래스 필터 (전체 {len(all_classes)}종)",
        options=all_classes,
        default=valid_default,
        key=_SS_CLASSES,
        help=(
            "표시할 결함 클래스를 선택합니다.\n"
            "선택 해제하면 해당 클래스의 결함이 맵에서 숨겨집니다.\n"
            "**전체 선택**: 모든 클래스 표시"
        ),
    )

    if not selected_classes_list:
        st.warning("⚠️ 클래스를 1개 이상 선택해야 결함이 표시됩니다.")
        return

    # ── 웨이퍼 외부 결함 포함 checkbox ──────────────────────────────────────
    show_outside: bool = st.checkbox(
        "🌐 웨이퍼 외부 결함 포함",
        value=False,
        key=_SS_OUTSIDE,
        help=(
            "✅ 체크: 웨이퍼 경계(x²+y² > r²) 밖에 위치한 결함도 표시\n\n"
            "☐ 해제: 웨이퍼 내부 결함만 표시 (기본값)\n"
            "→ 좌표 오류나 다이 외부 결함을 구분할 때 활용"
        ),
    )

    # ── defect_df_json 생성 ──────────────────────────────────────────────────
    # @st.cache_data 함수에 DataFrame을 직접 전달 불가 → JSON 직렬화
    defect_df_json: str = loaded_df.to_json()

    # selected_classes를 tuple로 변환
    # list는 hash() 불가 → @st.cache_data TypeError 발생
    selected_classes_tuple: tuple[str, ...] = tuple(sorted(selected_classes_list))

    # ── 오버레이 맵 생성 및 렌더링 ───────────────────────────────────────────
    with st.spinner("결함 오버레이 맵 생성 중..."):
        fig = create_defect_overlaid_map(
            wafer_df_json=wafer_df_json,
            defect_df_json=defect_df_json,
            selected_classes=selected_classes_tuple,
            resolution=resolution,
            colorscale=colorscale,
            base_map_type=base_map_type.lower(),
            show_outside=show_outside,
            coord_scale=coord_scale,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── 결함 데이터 테이블 (expander) ────────────────────────────────────────
    with st.expander("📋 결함 데이터 테이블", expanded=False):
        # 표시할 컬럼: x, y, class, size, description (있는 것만)
        display_cols = ["x", "y", "class"]
        if "size" in loaded_df.columns:
            display_cols.append("size")
        if "description" in loaded_df.columns and loaded_df["description"].ne("").any():
            display_cols.append("description")

        # 선택된 클래스만 필터링하여 표시
        df_display = loaded_df[
            loaded_df["class"].isin(selected_classes_list)
        ][display_cols].reset_index(drop=True)

        # 스케일 적용 후 좌표로 표시 (원본 단위가 아닌 mm 단위로 표시)
        if abs(coord_scale - 1.0) > 1e-10:
            df_display = df_display.copy()
            df_display["x"] = (df_display["x"] * coord_scale).round(3)
            df_display["y"] = (df_display["y"] * coord_scale).round(3)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "x":           st.column_config.NumberColumn("X (mm)",  format="%.3f"),
                "y":           st.column_config.NumberColumn("Y (mm)",  format="%.3f"),
                "class":       st.column_config.TextColumn("클래스"),
                "size":        st.column_config.NumberColumn("크기",    format="%.2f"),
                "description": st.column_config.TextColumn("설명"),
            },
        )

        # 현재 표시 중인 결함 수 요약
        n_shown = len(df_display)
        n_total = len(loaded_df)
        st.caption(
            f"표시 중: {n_shown:,}개 / 전체: {n_total:,}개 "
            f"({n_shown / n_total * 100:.1f}%)"
        )

        # CSV 다운로드 버튼
        st.download_button(
            label="📥 결함 데이터 CSV 다운로드",
            data=df_display.to_csv(index=False),
            file_name="defect_data_filtered.csv",
            mime="text/csv",
            key="def_download_btn",
        )