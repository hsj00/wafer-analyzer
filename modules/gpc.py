# modules/gpc.py
# GPC (Growth Per Cycle) 분석 모듈
# ALD 공정의 사이클당 성장률 계산, 반경별 프로파일, 구역별 박스플롯 시각화
#
# =============================================================================
# [설계 결정 근거]
# =============================================================================
#
# ① compute_gpc_column 반환 타입: str (df.to_json())
#    이유: "x","y","data" 컬럼 구조로 반환하면
#         create_2d_heatmap, calculate_stats 등 기존 함수와 직접 체인 호출 가능.
#         @st.cache_data는 DataFrame 해시 불가 → JSON 문자열로 직렬화.
#
# ② cycle_mode="column" 시 0 나눔 방지
#    df[cycle_col].replace(0, np.nan): 0인 사이클을 NaN으로 대체
#    → pandas 나눗셈에서 NaN 나누기는 NaN 반환 (ZeroDivisionError 없음)
#    음수 사이클: 물리적으로 불가능 → .where(cycles > 0, np.nan)으로 NaN 처리
#
# ③ cycle_mode="fixed" 시 fixed_cycles 검증
#    fixed_cycles <= 0이면 캐시 함수 내부에서 처리 불가 (st.error 호출 불가)
#    → render_gpc_tab에서 사전 검증 → 함수 미호출
#    캐시 함수 내부에서도 방어적 처리: fixed_cycles <= 0 → NaN 반환
#
# ④ 반경별 이동 평균 (rolling 방식)
#    sort_values("r") 후 rolling(window, center=True, min_periods=1)
#    - center=True: 현재 점 기준 앞뒤 window/2 범위 → 인과성 위반이지만
#      공간 데이터에서는 인과성 개념 없음 → 더 자연스러운 스무딩
#    - min_periods=1: 양 끝에서 window 미만이어도 NaN 없이 계산
#    - window 자동 조정: 포인트 수에 따라 5~20 범위로 자동 선택 (과도 스무딩 방지)
#
# ⑤ 3구역 정의 (Center/Mid/Edge)
#    Center: r < radius × 0.3  → 가스 주입부 직접 노출 영역
#    Mid:    r in [0.3R, 0.7R) → 전이 영역
#    Edge:   r ≥ radius × 0.7  → 로딩 효과(loading effect) 민감 영역
#    → 반도체 공정 표준 3구역 분류 (공정 가이드라인 반영)
#
# ⑥ create_2d_heatmap 재사용
#    compute_gpc_column이 "x","y","data"(=GPC) 구조를 반환하므로
#    기존 create_2d_heatmap에 그대로 전달 가능 → 코드 중복 없음
# =============================================================================

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
# (없음)

# ── 외부 라이브러리 ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── wafer_app_global 핵심 함수 import ────────────────────────────────────────
from app import _default_col_index  # 컬럼 기본값 탐색 헬퍼
from app import \
    calculate_stats  # GPC 통계: Mean, Std, Uniformity(%), Range, No.Sites
from app import create_2d_heatmap  # GPC Heatmap 시각화 (data 컬럼에 GPC 값 전달)

# =============================================================================
# session_state 키 상수 (prefix: "gpc_")
# =============================================================================
# 기존 키: data_folder, datasets, _s_display 등 (충돌 없음)
# multi_param 키: mp_x_col, mp_y_col 등 (충돌 없음)
# defect_overlay 키: def_file 등 (충돌 없음)
_SS_THICKNESS = "gpc_thickness_col"  # 두께 컬럼 selectbox 선택값
_SS_MODE      = "gpc_cycle_mode"     # 사이클 방식 radio 선택값
_SS_CYCLE_COL = "gpc_cycle_col"      # 사이클 수 컬럼 selectbox (column 모드)
_SS_FIXED_N   = "gpc_fixed_cycles"   # 고정 사이클 수 number_input (fixed 모드)
_SS_UNIT      = "gpc_unit"           # 단위 표기 text_input
_SS_WINDOW    = "gpc_smooth_window"  # 이동 평균 window 크기 slider
_SS_EDGE_R    = "gpc_edge_ratio"     # 엣지 기준 반지름 비율 slider


# =============================================================================
# 3구역 경계 상수 (반도체 공정 표준)
# =============================================================================
_CENTER_RATIO = 0.3   # r < radius × 0.3 → Center Zone
_MID_RATIO    = 0.7   # radius×0.3 ≤ r < radius×0.7 → Mid Zone
                       # r ≥ radius × 0.7 → Edge Zone

# 구역별 배경 색상 (add_vrect fillcolor)
_ZONE_COLORS = {
    "Center": "rgba(100, 200, 100, 0.12)",   # 연초록
    "Mid":    "rgba(255, 220, 50,  0.10)",   # 연노랑
    "Edge":   "rgba(255, 100, 100, 0.12)",   # 연빨강
}

# 구역별 평균 수평선 색상
_ZONE_LINE_COLORS = {
    "Center": "#2e7d32",  # 짙은 초록
    "Mid":    "#f57f17",  # 짙은 노랑
    "Edge":   "#c62828",  # 짙은 빨강
}


# =============================================================================
# [함수 1] compute_gpc_column
# =============================================================================

@st.cache_data
def compute_gpc_column(
    df_json: str,
    x_col: str,
    y_col: str,
    thickness_col: str,
    cycle_mode: str,           # "column" 또는 "fixed"
    cycle_col: str | None,     # cycle_mode="column"일 때 사용
    fixed_cycles: int | None,  # cycle_mode="fixed"일 때 사용
) -> str | None:
    """
    두께 데이터에서 GPC(Growth Per Cycle) 컬럼을 계산하고
    표준 컬럼 구조("x","y","data"=GPC) JSON을 반환.

    [반환 컬럼 구조]
    "x":    X 좌표 (wafer_app_global 표준)
    "y":    Y 좌표 (wafer_app_global 표준)
    "data": GPC 값 (Å/cycle 또는 nm/cycle)
    → create_2d_heatmap, calculate_stats 등 기존 함수와 직접 호환

    [GPC 계산식]
    cycle_mode="column": GPC = thickness / cycle_count (포인트별 다른 사이클 수)
    cycle_mode="fixed":  GPC = thickness / fixed_cycles (전체 동일 사이클 수)

    [음수/비정상 GPC 처리]
    물리적으로 GPC는 반드시 양수여야 함.
    음수 GPC = 측정 오류 또는 참조층 문제 → NaN으로 처리하여 맵에서 제외.

    [캐시 키 구성]
    (df_json, x_col, y_col, thickness_col, cycle_mode, cycle_col, fixed_cycles)
    - 모든 인자가 str, int, None → hashable → @st.cache_data 정상 작동
    - cycle_col: str | None → None도 hashable ✅
    - fixed_cycles: int | None → None도 hashable ✅

    인자:
        df_json       : 원본 데이터 JSON (x_col, y_col, thickness_col 포함)
        x_col         : X 좌표 컬럼명
        y_col         : Y 좌표 컬럼명
        thickness_col : 두께 측정값 컬럼명 (Å 또는 nm 단위)
        cycle_mode    : "column" (컬럼으로 나누기) 또는 "fixed" (고정값으로 나누기)
        cycle_col     : cycle_mode="column" 시 사용할 사이클 수 컬럼명
        fixed_cycles  : cycle_mode="fixed" 시 사용할 고정 사이클 수 (양의 정수)

    반환:
        str  : "x","y","data" 컬럼 구조의 df.to_json() 문자열
        None : 계산 실패 (cycle_col 없음, fixed_cycles ≤ 0 등)
    """
    # ── 캐시 함수 진입: JSON → DataFrame 역직렬화 ───────────────────────────
    df = pd.read_json(df_json)

    # ── 필수 컬럼 존재 확인 ──────────────────────────────────────────────────
    required = [x_col, y_col, thickness_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        # @st.cache_data 내부에서는 st.error 호출 불가 → None 반환으로 오류 전달
        return None

    # ── 두께 값 추출 및 숫자형 변환 ─────────────────────────────────────────
    thickness = pd.to_numeric(df[thickness_col], errors="coerce")

    # ── GPC 계산 분기 ─────────────────────────────────────────────────────────
    if cycle_mode == "column":
        # 컬럼으로 나누기: 포인트마다 다른 사이클 수 (멀티-사이클 실험)
        if cycle_col is None or cycle_col not in df.columns:
            return None

        cycles_raw = pd.to_numeric(df[cycle_col], errors="coerce")

        # ★ 0 나눔 방지: replace(0, np.nan)
        #   pandas 나눗셈에서 NaN 나누기 = NaN (ZeroDivisionError 없음)
        # ★ 음수 사이클 방지: 물리적으로 사이클 수는 양수만 유효
        cycles_safe = cycles_raw.where(cycles_raw > 0, other=np.nan)

        gpc = thickness / cycles_safe

    elif cycle_mode == "fixed":
        # 고정값으로 나누기: 표준 ALD (전체 웨이퍼에 동일 사이클 수 적용)
        if fixed_cycles is None or fixed_cycles <= 0:
            # fixed_cycles=0: ZeroDivisionError 방지
            # fixed_cycles<0: 물리적 불가
            return None

        gpc = thickness / float(fixed_cycles)

    else:
        # 알 수 없는 cycle_mode
        return None

    # ── 음수/비정상 GPC 값 NaN 처리 ──────────────────────────────────────────
    # GPC는 물리적으로 반드시 양수 (음수 = 측정 오류 또는 참조층 편차)
    # np.nan은 get_wafer_grid에서 NaN 마스크와 함께 처리됨 → 맵에서 자동 제외
    gpc = gpc.where(gpc > 0, other=np.nan)

    # ── 표준 컬럼 구조 DataFrame 생성 ────────────────────────────────────────
    # "x","y","data" 구조: wafer_app_global의 모든 시각화 함수와 호환
    result_df = pd.DataFrame({
        "x":    pd.to_numeric(df[x_col], errors="coerce").values,
        "y":    pd.to_numeric(df[y_col], errors="coerce").values,
        "data": gpc.values,
    }).dropna(subset=["x", "y"]).reset_index(drop=True)
    # data(GPC) NaN은 dropna에서 제외하지 않음 → get_wafer_grid가 처리

    if result_df.empty:
        return None

    return result_df.to_json()


# =============================================================================
# [내부 헬퍼] _compute_zone_stats
# =============================================================================

def _compute_zone_stats(
    r: np.ndarray,
    gpc: np.ndarray,
    radius: float,
) -> dict[str, dict]:
    """
    3구역(Center/Mid/Edge)별 GPC 통계를 계산.

    [구역 정의]
    Center: r < radius × _CENTER_RATIO (0.3)
    Mid:    radius×0.3 ≤ r < radius×_MID_RATIO (0.7)
    Edge:   r ≥ radius × _MID_RATIO (0.7)

    인자:
        r      : 각 측정 포인트의 반경 배열 (mm)
        gpc    : 각 측정 포인트의 GPC 값 배열
        radius : 웨이퍼 최대 반경 (mm)

    반환:
        {"Center": {"mean": ..., "std": ..., "data": ...}, "Mid": ..., "Edge": ...}
    """
    center_mask = r <  radius * _CENTER_RATIO
    mid_mask    = (r >= radius * _CENTER_RATIO) & (r < radius * _MID_RATIO)
    edge_mask   = r >= radius * _MID_RATIO

    zones = {}
    for zone_name, mask in [("Center", center_mask),
                              ("Mid",    mid_mask),
                              ("Edge",   edge_mask)]:
        zone_gpc = gpc[mask & ~np.isnan(gpc)]
        zones[zone_name] = {
            "data":  zone_gpc,
            "mean":  float(np.nanmean(zone_gpc)) if len(zone_gpc) > 0 else np.nan,
            "std":   float(np.nanstd(zone_gpc))  if len(zone_gpc) > 0 else np.nan,
            "count": int(len(zone_gpc)),
        }

    return zones


# =============================================================================
# [함수 2] create_gpc_radial_profile
# =============================================================================

@st.cache_data
def create_gpc_radial_profile(
    df_json: str,
    window: int = 20,
    unit: str = "Å/cycle",
) -> go.Figure:
    """
    반경별 GPC 프로파일 차트 생성.

    [시각화 구성]
    1. 배경 구역 색상 (add_vrect):
       Center(0~0.3R): 연초록, Mid(0.3R~0.7R): 연노랑, Edge(0.7R~R): 연빨강
    2. 원본 산점도 (반투명 회색): 측정 노이즈 포함된 원본 데이터
    3. 이동 평균 추세선 (진한 색): rolling 스무딩으로 전반적 경향 표시
    4. 구역별 평균 수평 점선: 각 구역의 대표 GPC 값

    [이동 평균 처리]
    df.sort_values("r") 후 rolling(window, center=True, min_periods=1)
    - center=True: 각 점의 앞뒤를 동등하게 반영 (공간 데이터에서 적합)
    - min_periods=1: 양 끝에서 window 미만이어도 NaN 없이 계산
    - window 자동 조정: len(df) × 0.1 기준, 5~25 클리핑

    [add_vrect 선택 이유]
    add_shape으로 fillrect를 그리는 것보다 add_vrect이 더 간결하고
    paper/data 좌표 자동 처리로 x축 범위 변경 시에도 자동 반영됨.

    인자:
        df_json: "x","y","data"(=GPC) 컬럼의 JSON (compute_gpc_column 반환값)
        window : 이동 평균 window 크기 (측정 포인트 수 기준 자동 조정)
        unit   : GPC 단위 표기 (Y축 라벨에 사용)

    반환:
        go.Figure: 반경별 GPC 프로파일 Figure
    """
    df = pd.read_json(df_json)

    # ── 반경 계산 ─────────────────────────────────────────────────────────────
    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    radius  = df["r"].max()

    # GPC 값 (NaN 포함 가능)
    gpc_vals = df["data"].values
    r_vals   = df["r"].values

    # ── 구역별 통계 계산 ──────────────────────────────────────────────────────
    zone_stats = _compute_zone_stats(r_vals, gpc_vals, radius)

    # ── 이동 평균 계산 ────────────────────────────────────────────────────────
    # 반드시 r 기준 정렬 후 rolling → 정렬 없이 rolling하면 의미 없는 순서 평균
    df_sorted = df.sort_values("r").reset_index(drop=True)

    # window 자동 조정: 포인트 수 × 10% 기준, [5, 25] 클리핑
    # 포인트가 적으면 window가 크면 전체가 하나의 평균으로 뭉개짐 → 자동 축소
    auto_window = max(5, min(25, int(len(df_sorted) * 0.1), window))

    rolling_mean = (
        df_sorted["data"]
        .rolling(window=auto_window, center=True, min_periods=1)
        .mean()
    )

    # ── Figure 생성 ───────────────────────────────────────────────────────────
    fig = go.Figure()

    # ── 구역 배경 (add_vrect) ─────────────────────────────────────────────────
    # x0, x1은 data 좌표 (반경, mm)
    # layer="below": Heatmap보다 아래에 배치 → scatter가 배경 위에 렌더링
    zone_boundaries = [
        ("Center", 0,                       radius * _CENTER_RATIO),
        ("Mid",    radius * _CENTER_RATIO,   radius * _MID_RATIO),
        ("Edge",   radius * _MID_RATIO,      radius),
    ]

    for zone_name, x0, x1 in zone_boundaries:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=_ZONE_COLORS[zone_name],
            opacity=1.0,       # fillcolor 자체에 투명도 포함 (rgba)
            line_width=0,      # 구역 경계선 없음 (자연스러운 전환)
            layer="below",     # 데이터 포인트 아래에 배경으로 배치
            annotation_text=zone_name,
            annotation_position="top left",
            annotation=dict(
                font=dict(size=10, color="gray"),
                opacity=0.7,
            ),
        )

    # ── 원본 산점도 (반투명): 측정 노이즈 포함 원본 ──────────────────────────
    # 모든 포인트를 반투명 회색으로 → "이 아래에 데이터가 있다"는 맥락 제공
    valid_mask = ~np.isnan(gpc_vals)
    fig.add_trace(go.Scatter(
        x=r_vals[valid_mask],
        y=gpc_vals[valid_mask],
        mode="markers",
        name="원본 데이터",
        marker=dict(
            size=4,
            color="rgba(150, 150, 150, 0.40)",   # 반투명 회색
            line=dict(width=0),
        ),
        showlegend=True,
        hovertemplate=(
            "반경: %{x:.2f} mm<br>"
            f"GPC: %{{y:.4f}} {unit}<extra>원본</extra>"
        ),
    ))

    # ── 이동 평균 추세선: 전반적 반경-GPC 경향 ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_sorted["r"].values,
        y=rolling_mean.values,
        mode="lines",
        name=f"이동평균 (window={auto_window})",
        line=dict(color="royalblue", width=2.5),
        showlegend=True,
        hovertemplate=(
            "반경: %{x:.2f} mm<br>"
            f"이동평균 GPC: %{{y:.4f}} {unit}<extra>이동평균</extra>"
        ),
    ))

    # ── 구역별 평균 수평 점선 ─────────────────────────────────────────────────
    # 각 구역의 대표 GPC 값을 수평 점선으로 표시 → 구역 간 차이 시각화
    for zone_name, (x0, x1) in zip(
        ["Center", "Mid", "Edge"],
        [(0, radius * _CENTER_RATIO),
         (radius * _CENTER_RATIO, radius * _MID_RATIO),
         (radius * _MID_RATIO, radius)],
    ):
        zone_mean = zone_stats[zone_name]["mean"]
        if not np.isnan(zone_mean):
            fig.add_shape(
                type="line",
                x0=x0, x1=x1,
                y0=zone_mean, y1=zone_mean,
                line=dict(
                    color=_ZONE_LINE_COLORS[zone_name],
                    width=1.8,
                    dash="dot",
                ),
                layer="above",
            )
            # 구역 평균값 텍스트 라벨 (수평선 오른쪽 끝에 표시)
            fig.add_annotation(
                x=x1,
                y=zone_mean,
                text=f"μ={zone_mean:.3f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=9, color=_ZONE_LINE_COLORS[zone_name]),
                bgcolor="rgba(255,255,255,0.7)",
                xref="x",
                yref="y",
            )

    # ── 웨이퍼 경계 수직선 ───────────────────────────────────────────────────
    fig.add_vline(
        x=radius,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text=f"Edge ({radius:.1f} mm)",
        annotation_position="top right",
        annotation_font=dict(size=9),
    )

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="반경별 GPC 프로파일", x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="반경 (mm)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            range=[0, radius * 1.05],
        ),
        yaxis=dict(
            title=f"GPC ({unit})",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=80, t=50, b=50),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.5)",
            borderwidth=1,
        ),
    )

    return fig


# =============================================================================
# [함수 3] create_gpc_uniformity_summary
# =============================================================================

@st.cache_data
def create_gpc_uniformity_summary(
    df_json: str,
    unit: str = "Å/cycle",
) -> go.Figure:
    """
    구역별(Center/Mid/Edge) GPC 박스플롯 + 전체 평균 기준선.

    [박스플롯 선택 이유]
    구역별 평균/표준편차 바 차트보다 박스플롯이:
    - 중위수, 사분위수, 이상치를 동시에 표현
    - 분포 비대칭성 시각화 가능 (ALD 공정 이상 진단에 유용)
    - 구역 내 측정 산포를 직관적으로 비교

    [3구역 go.Box trace 구성]
    각 구역을 별도 go.Box trace로 → 색상/레이블 개별 지정 가능
    boxmean=True: 박스 내부에 평균 기호(◇) 추가 → 중위수와 차이 시각화

    인자:
        df_json: "x","y","data"(=GPC) 컬럼의 JSON
        unit   : GPC 단위 표기 (Y축 라벨)

    반환:
        go.Figure: 구역별 GPC 박스플롯 Figure
    """
    df = pd.read_json(df_json)

    # ── 반경 계산 및 구역 분류 ────────────────────────────────────────────────
    df["r"]    = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    radius     = df["r"].max()
    gpc_vals   = df["data"].values
    r_vals     = df["r"].values

    zone_stats = _compute_zone_stats(r_vals, gpc_vals, radius)

    # ── 전체 평균 GPC (기준선용) ─────────────────────────────────────────────
    valid_gpc  = gpc_vals[~np.isnan(gpc_vals)]
    global_mean = float(np.nanmean(valid_gpc)) if len(valid_gpc) > 0 else np.nan

    # ── Figure 생성: 구역별 go.Box trace ────────────────────────────────────
    fig = go.Figure()

    # 구역별 박스플롯 색상
    box_colors = {
        "Center": "rgba(46, 125, 50, 0.7)",    # 초록 계열
        "Mid":    "rgba(245, 127, 23, 0.7)",   # 주황 계열
        "Edge":   "rgba(198, 40, 40, 0.7)",    # 빨강 계열
    }
    box_line_colors = {
        "Center": _ZONE_LINE_COLORS["Center"],
        "Mid":    _ZONE_LINE_COLORS["Mid"],
        "Edge":   _ZONE_LINE_COLORS["Edge"],
    }

    for zone_name in ["Center", "Mid", "Edge"]:
        zone_data  = zone_stats[zone_name]["data"]
        zone_count = zone_stats[zone_name]["count"]

        if zone_count == 0:
            # 해당 구역에 데이터 없으면 빈 trace (레이아웃 일관성 유지)
            continue

        fig.add_trace(go.Box(
            y=zone_data,
            name=f"{zone_name}<br><sub>({zone_count}pts)</sub>",
            boxmean=True,           # 박스 내부에 평균(◇) 표시
            boxpoints="outliers",   # 이상치 점만 표시 (모든 점은 너무 많음)
            jitter=0.3,             # 이상치 점들이 겹치지 않도록 가로로 분산
            pointpos=0,             # 이상치 점 위치: 박스 중앙
            marker=dict(
                color=box_colors[zone_name],
                size=4,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            fillcolor=box_colors[zone_name],
            line=dict(color=box_line_colors[zone_name], width=1.5),
            hovertemplate=(
                f"<b>{zone_name} Zone</b><br>"
                f"GPC: %{{y:.4f}} {unit}<extra></extra>"
            ),
        ))

    # ── 전체 평균 기준선 ─────────────────────────────────────────────────────
    if not np.isnan(global_mean):
        fig.add_hline(
            y=global_mean,
            line_dash="dash",
            line_color="rgba(50, 50, 50, 0.7)",
            line_width=1.5,
            annotation_text=f"전체 평균 {global_mean:.4f} {unit}",
            annotation_position="right",
            annotation_font=dict(size=9, color="gray"),
        )

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="구역별 GPC 분포 (박스플롯)", x=0.5, font=dict(size=14)),
        yaxis=dict(
            title=f"GPC ({unit})",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            zeroline=False,
        ),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=60, r=80, t=50, b=40),
        showlegend=False,     # X축 라벨이 범례 역할
    )

    return fig


# =============================================================================
# [함수 4] render_gpc_tab (UI 렌더러)
# =============================================================================

def render_gpc_tab(
    df_raw: pd.DataFrame,
    all_cols: list,
    resolution: int,
    colorscale: str,
) -> None:
    """
    GPC 분석 탭의 전체 UI를 렌더링.

    [레이아웃 구조]
    ┌─────────────┬─────────────────────────────────────────────────────────┐
    │  컨트롤 패널 │  GPC 계산 결과 요약 지표                                  │
    │  • 두께 컬럼 │  • 전체 평균 GPC, Uniformity(%), 중심-가장자리 편차        │
    │  • 사이클   │                                                           │
    │  • 단위     │                                                           │
    └─────────────┴─────────────────────────────────────────────────────────┘
    ┌──────────────────────────┬──────────────────────────┐
    │  GPC Heatmap              │  반경별 GPC 프로파일        │
    │  (create_2d_heatmap 재사용)│  (산점도 + 이동평균)       │
    └──────────────────────────┴──────────────────────────┘
    ┌──────────────────────────┬──────────────────────────┐
    │  구역별 GPC 박스플롯        │  상세 통계 + CSV 다운로드   │
    └──────────────────────────┴──────────────────────────┘

    [핵심 데이터 흐름]
    df_raw → compute_gpc_column() → gpc_df_json (str)
    gpc_df_json → create_2d_heatmap()        → GPC Heatmap Figure
    gpc_df_json → create_gpc_radial_profile() → 반경별 프로파일 Figure
    gpc_df_json → create_gpc_uniformity_summary() → 박스플롯 Figure
    gpc_df_json → calculate_stats()          → 통계 dict

    인자:
        df_raw     : 원본 DataFrame (파일 로딩 직후 상태)
        all_cols   : df_raw의 전체 컬럼명 리스트
        resolution : 보간 해상도 (사이드바 슬라이더)
        colorscale : 컬러스케일 이름 (사이드바 selectbox)
    """
    # ── 컨트롤 패널 + 요약 지표 (2컬럼 레이아웃) ─────────────────────────────
    col_ctrl, col_summary = st.columns([1, 2])

    with col_ctrl:
        st.markdown("##### ⚙️ GPC 계산 설정")

        # ── X, Y 좌표 컬럼 selectbox ─────────────────────────────────────────
        # GPC 탭은 단일 모드에서 이미 선택된 x_col, y_col을 재사용하는 것이 이상적이나,
        # 독립 탭으로 설계하여 별도 선택 가능하게 구성
        x_col_sel: str = st.selectbox(
            "X 좌표 컬럼",
            options=all_cols,
            index=_default_col_index(all_cols, "x", 0),
            key="gpc_x_col",
        )
        y_col_sel: str = st.selectbox(
            "Y 좌표 컬럼",
            options=all_cols,
            index=_default_col_index(all_cols, "y", 1),
            key="gpc_y_col",
        )

        # ── 두께 컬럼 selectbox ───────────────────────────────────────────────
        # x, y로 선택된 컬럼은 두께 후보에서 제외 (좌표를 두께로 나누는 것 방지)
        thickness_candidates = [c for c in all_cols if c not in (x_col_sel, y_col_sel)]

        if not thickness_candidates:
            st.warning("⚠️ 두께 컬럼으로 사용 가능한 컬럼이 없습니다. X/Y 컬럼을 확인하세요.")
            return

        # "thickness", "thick", "thk", "t" 등 일반 두께 컬럼명 자동 탐색
        thickness_keywords = ["thickness", "thick", "thk", "film", "t", "depth"]
        def_thickness_idx = next(
            (i for i, c in enumerate(thickness_candidates)
             if any(kw in c.lower() for kw in thickness_keywords)),
            0,    # 자동 탐색 실패 시 첫 번째 컬럼
        )

        thickness_col: str = st.selectbox(
            "두께 컬럼",
            options=thickness_candidates,
            index=def_thickness_idx,
            key=_SS_THICKNESS,
            help="GPC = 이 컬럼 ÷ 사이클 수로 계산됩니다.",
        )

        st.markdown("---")

        # ── 사이클 방식 radio ─────────────────────────────────────────────────
        cycle_mode_label: str = st.radio(
            "사이클 수 입력 방식",
            options=["컬럼으로 나누기", "고정값으로 나누기"],
            index=1,    # 기본값: 고정값 (표준 ALD에서 더 일반적)
            key=_SS_MODE,
            help=(
                "**컬럼으로 나누기**: 측정 포인트마다 다른 사이클 수가 있는 경우\n\n"
                "**고정값으로 나누기**: 전체 웨이퍼에 동일한 사이클 수를 적용하는 경우 (표준 ALD)"
            ),
        )

        # 사이클 입력 방식에 따라 분기 UI
        cycle_col  = None
        fixed_cycles = None

        if cycle_mode_label == "컬럼으로 나누기":
            # 사이클 컬럼 후보: x, y, 두께 컬럼 제외
            cycle_candidates = [
                c for c in all_cols
                if c not in (x_col_sel, y_col_sel, thickness_col)
            ]
            if not cycle_candidates:
                st.warning("⚠️ 사이클 컬럼으로 사용 가능한 컬럼이 없습니다.")
                return

            # "cycle", "cycles", "n_cycles" 등 자동 탐색
            cycle_keywords = ["cycle", "cycles", "n_cycle", "ncycle",
                              "n", "count", "number"]
            def_cycle_idx = next(
                (i for i, c in enumerate(cycle_candidates)
                 if any(kw in c.lower() for kw in cycle_keywords)),
                0,
            )

            cycle_col = st.selectbox(
                "사이클 수 컬럼",
                options=cycle_candidates,
                index=def_cycle_idx,
                key=_SS_CYCLE_COL,
                help="각 측정 포인트의 ALD 사이클 수가 담긴 컬럼을 선택하세요.",
            )
            cycle_mode = "column"

        else:
            # 고정값 입력: min=1로 0 및 음수 입력 원천 차단
            fixed_cycles_input = st.number_input(
                "ALD 사이클 수",
                min_value=1,
                max_value=10000,
                value=st.session_state.get(_SS_FIXED_N, 100),
                step=10,
                key=_SS_FIXED_N,
                help=(
                    "ALD 공정에서 사용한 총 사이클 수를 입력하세요.\n\n"
                    "예: 100사이클 → 기대 GPC ≈ 두께 / 100\n\n"
                    "최소값 1로 ZeroDivisionError 방지."
                ),
            )
            fixed_cycles = int(fixed_cycles_input)
            cycle_mode   = "fixed"

        st.markdown("---")

        # ── 단위 표기 text_input ─────────────────────────────────────────────
        unit: str = st.text_input(
            "GPC 단위 표기",
            value=st.session_state.get(_SS_UNIT, "Å/cycle"),
            key=_SS_UNIT,
            help="차트 Y축과 통계 라벨에 표시될 단위 문자열.",
        )

        # ── 이동 평균 window 슬라이더 ────────────────────────────────────────
        smooth_window: int = st.slider(
            "반경 프로파일 이동평균 window",
            min_value=3,
            max_value=50,
            value=st.session_state.get(_SS_WINDOW, 20),
            step=1,
            key=_SS_WINDOW,
            help=(
                "클수록 더 부드러운 추세선 (노이즈 제거 강함).\n"
                "작을수록 원본 데이터의 국소 변동 반영.\n"
                "포인트 수의 10~20%가 적절합니다."
            ),
        )

    # ── GPC 계산 실행 ─────────────────────────────────────────────────────────
    # compute_gpc_column 호출 전 df_raw를 JSON으로 변환
    # (all_cols 전체를 포함하는 원본 df → 컬럼 선택은 함수 내부)
    df_raw_json = df_raw.to_json()

    # ★ compute_gpc_column: @st.cache_data 적용 → 동일 인자이면 재계산 없음
    gpc_df_json = compute_gpc_column(
        df_json=df_raw_json,
        x_col=x_col_sel,
        y_col=y_col_sel,
        thickness_col=thickness_col,
        cycle_mode=cycle_mode,
        cycle_col=cycle_col,
        fixed_cycles=fixed_cycles,
    )

    if gpc_df_json is None:
        st.error(
            "❌ GPC 계산에 실패했습니다.\n\n"
            "가능한 원인:\n"
            "- 두께 컬럼 또는 사이클 컬럼에 숫자가 아닌 값이 있음\n"
            "- 사이클 수가 0 또는 음수인 행이 있음 (NaN으로 처리됨)\n"
            "- 유효한 데이터 포인트가 없음"
        )
        st.session_state["gpc_result"] = None
        return

    # ── 통계 계산 (계산 후 즉시 요약 지표 표시) ──────────────────────────────
    # calculate_stats: @st.cache_data 적용 → gpc_df_json 동일하면 캐시 히트
    stats = calculate_stats(gpc_df_json)

    fig_gpc_heatmap = create_2d_heatmap(
        df_json=gpc_df_json,
        resolution=resolution,
        colorscale=colorscale,
        show_points=False,
    )
    st.session_state["gpc_result"] = {
        "stats": stats,
        "heatmap_fig": fig_gpc_heatmap,
        "df_json": gpc_df_json,
    }

    with col_summary:
        st.markdown("##### 📊 GPC 계산 결과 요약")

        # 전체 균일도 기반 등급 결정
        uniformity = stats.get("Uniformity (%)", float("nan"))
        if not pd.isna(uniformity):
            if uniformity < 1.0:
                unif_grade, unif_color = "우수 ▲", "normal"
            elif uniformity < 2.0:
                unif_grade, unif_color = "양호 ●", "off"
            else:
                unif_grade, unif_color = "주의 ▼", "inverse"
        else:
            unif_grade, unif_color = "N/A", "off"

        # 구역별 통계 (Center-Edge 편차 계산용)
        gpc_df_for_zone = pd.read_json(gpc_df_json)
        gpc_df_for_zone["r"] = np.sqrt(
            gpc_df_for_zone["x"] ** 2 + gpc_df_for_zone["y"] ** 2
        )
        radius_val = gpc_df_for_zone["r"].max()
        zone_stats = _compute_zone_stats(
            gpc_df_for_zone["r"].values,
            gpc_df_for_zone["data"].values,
            radius_val,
        )

        center_mean = zone_stats["Center"]["mean"]
        edge_mean   = zone_stats["Edge"]["mean"]
        ce_delta    = edge_mean - center_mean if not (pd.isna(center_mean) or pd.isna(edge_mean)) else np.nan
        ce_pct      = (ce_delta / center_mean * 100) if (not pd.isna(ce_delta) and center_mean != 0) else np.nan

        # 4개 요약 지표
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            label=f"전체 평균 GPC ({unit})",
            value=f"{stats.get('Mean', 0):.4f}",
            help=f"N={stats.get('No. Sites', 0)} 측정 포인트 평균",
        )
        m2.metric(
            label="Uniformity (%)",
            value=f"{uniformity:.3f} %" if not pd.isna(uniformity) else "N/A",
            delta=unif_grade,
            delta_color=unif_color,
            help="σ/μ × 100. ALD 양호 기준: < 2%",
        )
        m3.metric(
            label=f"중심 평균 ({unit})",
            value=f"{center_mean:.4f}" if not pd.isna(center_mean) else "N/A",
            help=f"Center Zone (r < {_CENTER_RATIO*100:.0f}%R) 평균 GPC",
        )
        m4.metric(
            label="중심-가장자리 편차",
            value=f"{ce_pct:+.2f} %" if not pd.isna(ce_pct) else "N/A",
            delta=(f"Δ={ce_delta:+.4f}" if not pd.isna(ce_delta) else None),
            delta_color="inverse" if not pd.isna(ce_pct) and abs(ce_pct) > 1.0 else "off",
            help="(Edge평균 - Center평균) / Center평균 × 100. ALD 양호 기준: < 1%",
        )

    st.markdown("---")

    # ── 4분할 차트 레이아웃 ───────────────────────────────────────────────────
    # 상단: GPC Heatmap | 반경별 프로파일
    row1_left, row1_right = st.columns([1, 1])

    with row1_left:
        st.markdown("##### 🗺️ GPC Heatmap")
        # ★ create_2d_heatmap 재사용: gpc_df_json의 "data" 컬럼 = GPC 값
        #   compute_gpc_column이 "x","y","data" 구조를 반환하므로 바로 전달 가능
        fig_heatmap = create_2d_heatmap(
            df_json=gpc_df_json,
            resolution=resolution,
            colorscale=colorscale,
            show_points=False,    # GPC 맵에서 측정점은 오히려 가독성 저하
        )
        # 컬러바 제목을 단위로 업데이트
        fig_heatmap.data[0].colorbar.title = dict(text=unit, side="right")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with row1_right:
        st.markdown("##### 📈 반경별 GPC 프로파일")
        fig_radial = create_gpc_radial_profile(
            df_json=gpc_df_json,
            window=smooth_window,
            unit=unit,
        )
        st.plotly_chart(fig_radial, use_container_width=True)

    # 하단: 구역별 박스플롯 | 상세 통계
    row2_left, row2_right = st.columns([1, 1])

    with row2_left:
        st.markdown("##### 📦 구역별 GPC 분포 (박스플롯)")
        fig_box = create_gpc_uniformity_summary(
            df_json=gpc_df_json,
            unit=unit,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with row2_right:
        st.markdown("##### 📋 상세 통계")

        # 전체 통계 테이블
        stats_df = pd.DataFrame([
            {"항목": k, "값": str(v) + (f" {unit}" if "Mean" in k or "Dev" in k or "Range" in k or "Min" in k or "Max" in k else "")}
            for k, v in stats.items()
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 구역별 통계 테이블
        st.markdown("**구역별 GPC 통계:**")
        zone_rows = []
        for zone_name in ["Center", "Mid", "Edge"]:
            zs = zone_stats[zone_name]
            zone_rows.append({
                "구역":       zone_name,
                f"평균 ({unit})": f"{zs['mean']:.4f}" if not pd.isna(zs["mean"]) else "N/A",
                f"표준편차":   f"{zs['std']:.4f}"  if not pd.isna(zs["std"])  else "N/A",
                "포인트 수":  zs["count"],
            })
        st.dataframe(
            pd.DataFrame(zone_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── GPC 데이터 CSV 다운로드 버튼 ─────────────────────────────────────
        gpc_download_df = pd.read_json(gpc_df_json)
        gpc_download_df = gpc_download_df.rename(columns={"data": f"GPC_{unit.replace('/', '_per_')}"})
        gpc_download_df["r_mm"] = np.sqrt(
            gpc_download_df["x"] ** 2 + gpc_download_df["y"] ** 2
        ).round(4)

        st.download_button(
            label=f"📥 GPC 데이터 CSV 다운로드",
            data=gpc_download_df.to_csv(index=False),
            file_name="gpc_data.csv",
            mime="text/csv",
            key="gpc_download_btn",
            help="계산된 GPC 값, 좌표, 반경을 포함한 CSV 파일을 다운로드합니다.",
        )

        # 계산 조건 요약 표시 (재현성 확인용)
        with st.expander("🔧 계산 조건 확인", expanded=False):
            cond_rows = [
                ("두께 컬럼",     thickness_col),
                ("사이클 방식",   cycle_mode_label),
            ]
            if cycle_mode == "column":
                cond_rows.append(("사이클 컬럼", cycle_col))
            else:
                cond_rows.append(("고정 사이클 수", str(fixed_cycles)))
            cond_rows += [
                ("단위", unit),
                ("보간 해상도", str(resolution)),
                ("이동평균 window", str(smooth_window)),
            ]
            st.dataframe(
                pd.DataFrame(cond_rows, columns=["항목", "값"]),
                use_container_width=True,
                hide_index=True,
            )