# modules/multi_param.py
# 다중 파라미터 서브플롯 모듈
# wafer_app_global.py의 get_wafer_grid, add_wafer_outline,
# _wafer_layout, calculate_stats를 import해서 사용
#
# =============================================================================
# [설계 결정 근거 — 읽기 전에 반드시 이해할 것]
# =============================================================================
#
# ① add_wafer_outline 재사용 불가
#    wafer_app_global.add_wafer_outline(fig, radius)는 row/col 인자가 없음.
#    make_subplots 컨텍스트에서 row/col 없이 add_trace() 호출하면
#    Plotly 내부적으로 첫 번째 subplot(row=1, col=1)에만 쌓임.
#    → 2번째 이후 subplot에는 아웃라인 없이 Heatmap만 남는 버그 발생.
#    → 로컬 _add_outline_to_subplot(fig, radius, row, col) 구현 필수.
#
# ② param_cols: tuple 강제 사용 이유
#    @st.cache_data는 함수 인자를 hash()로 캐시 키 생성.
#    list는 mutable → hash() 불가 → TypeError 발생.
#    tuple은 immutable → hash() 가능 → 캐시 키로 정상 작동.
#    호출부(render_multi_param_tab)에서 반드시 tuple(sel_params) 변환 필수.
#
# ③ 2단계 캐시 설계 (캐시 효율 최대화)
#    상위 캐시: create_multi_param_subplots(@st.cache_data)
#      → 전체 조합(df+x+y+params+resolution+colorscale+share_scale)이 동일하면
#        함수 진입 자체를 건너뜀 (get_wafer_grid 호출 0회)
#    하위 캐시: get_wafer_grid(@st.cache_data, wafer_app_global에 정의됨)
#      → 상위 캐시 미스 시에도 "변경되지 않은 파라미터"는 하위 캐시 히트
#    효과: 파라미터 1개만 추가/제거 시 나머지 파라미터 재보간 없음 (성능↑)
#
# ④ _wafer_layout 재사용 불가
#    _wafer_layout(radius, height)은 단일 Figure용으로
#    "xaxis", "yaxis" 키를 하드코딩하여 반환.
#    make_subplots에서 col=2 이상은 "xaxis2", "yaxis2" 등 동적 키 필요.
#    → update_layout(**{f"xaxis{suffix}": ...}) 패턴으로 로컬 구현.
#
# ⑤ sub_json 생성 공식 통일 (캐시 키 충돌 방지)
#    create_multi_param_subplots 내부와 render_multi_param_tab(통계 계산부)에서
#    sub_json을 생성하는 방식이 완전히 동일해야 get_wafer_grid 하위 캐시가 히트됨.
#    공식: df[[x, y, param]].rename(...).dropna().reset_index(drop=True).to_json()
#    순서가 조금이라도 달라지면 JSON 문자열이 달라져 캐시 미스 발생 → 성능 저하.
# =============================================================================

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
# (없음)

# ── 외부 라이브러리 ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── wafer_app_global 핵심 함수 import ────────────────────────────────────────
# 주의: _wafer_layout, add_wafer_outline은 설계 이유 ①④에 의해 여기서 사용 불가.
#       대신 아래 로컬 헬퍼(_add_outline_to_subplot, _apply_subplot_axes) 사용.
from app import _default_col_index  # 컬럼 기본값 탐색 (이름 매칭 실패 시 fallback 인덱스 반환)
from app import \
    calculate_stats  # 통계 계산: Mean, Std, Uniformity(%), Range, No.Sites
from app import get_wafer_grid  # 불규칙 산점 → 균일 그리드 보간 (@st.cache_data 적용됨)

# =============================================================================
# session_state 키 상수 (prefix: "mp_")
# =============================================================================
# 기존 wafer_app_global 키: data_folder, show_folder_browser, browser_current,
#                           datasets, _s_file, _s_display, _s_col_map
# 새 기능은 반드시 "mp_" prefix → 기존 키와 충돌 없음 보장
_SS_X_COL     = "mp_x_col"       # X 좌표 컬럼 selectbox 선택값
_SS_Y_COL     = "mp_y_col"       # Y 좌표 컬럼 selectbox 선택값
_SS_PARAMS    = "mp_param_cols"  # 파라미터 컬럼 multiselect 선택값
_SS_SHARE     = "mp_share_scale" # 전체 통일 스케일 checkbox 값


# =============================================================================
# 파라미터 선택 제약 상수
# =============================================================================
_MIN_PARAMS = 2   # 서브플롯은 최소 2개 비교가 의미 있음
_MAX_PARAMS = 6   # 6개 초과는 subplot당 너비가 너무 좁아 가독성 저하


# =============================================================================
# subplot 간격 계산 상수
# =============================================================================
# 기준 0.06 → 파라미터 1개씩 증가마다 0.005 감소 → 최소 0.02 보장
# 예: n=2: 0.05, n=3: 0.045, n=4: 0.04, n=5: 0.035, n=6: 0.03
_SPACING_BASE = 0.06
_SPACING_STEP = 0.005
_SPACING_MIN  = 0.02


# =============================================================================
# [내부 헬퍼 함수들]
# =============================================================================

def _add_outline_to_subplot(
    fig: go.Figure,
    radius: float,
    row: int,
    col: int,
) -> None:
    """
    make_subplots 컨텍스트에서 특정 셀(row, col)에 웨이퍼 아웃라인 추가.

    [wafer_app_global.add_wafer_outline과의 차이점]
    단 하나: fig.add_trace(..., row=row, col=col) 인자가 추가됨.
    원형 테두리(360점), Notch(반지름 3%, 아래 반원, 흰색 채움) 로직은 동일.

    [row/col을 반드시 지정해야 하는 이유]
    make_subplots로 생성된 Figure는 subplot 메타데이터를 내부 관리.
    row=None, col=None으로 add_trace() 호출 → Plotly가 (row=1, col=1)로 fallback
    → 2번째 이후 subplot에는 아웃라인 trace가 추가되지 않는 버그.
    """
    # ── 원형 테두리: 360개 점으로 부드러운 원 근사 ──────────────────────────
    theta = np.linspace(0, 2 * np.pi, 360)
    fig.add_trace(
        go.Scatter(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",    # 아웃라인 위에서 마우스 오버 시 툴팁 표시 안 함
        ),
        row=row,
        col=col,
    )

    # ── Notch: 하단(6시 방향) 반원 V자형 홈 ─────────────────────────────────
    # np.linspace(π, 2π): 아래 반원 180°~360°만 사용
    # nr = radius × 0.03: 실제 웨이퍼 Notch 크기 비율 (약 0.5mm / 150mm 웨이퍼)
    # y 중심 = -radius: 원의 맨 아래 지점에 Notch 위치
    nt = np.linspace(np.pi, 2 * np.pi, 60)
    nr = radius * 0.03
    fig.add_trace(
        go.Scatter(
            x=nr * np.cos(nt),
            y=-radius + nr * np.sin(nt),   # y = -radius가 Notch 반원의 중심
            mode="lines",
            line=dict(color="black", width=2),
            fill="toself",       # 경로 내부를 채움
            fillcolor="white",   # 흰색 채움 = "잘라낸" 시각 효과
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _calc_colorbar_x(col_idx: int, n_cols: int, spacing: float) -> float:
    """
    make_subplots에서 각 subplot의 colorbar x 위치(paper 좌표 0.0~1.0+) 계산.

    [make_subplots의 subplot domain 계산식]
    subplot domain을 n등분 시 각 subplot의 너비:
      col_width = (1.0 - (n - 1) × spacing) / n
    col_idx번째 subplot의 domain 우측 끝:
      domain_end = (col_idx - 1) × (col_width + spacing) + col_width

    colorbar는 해당 domain 우측 끝에서 소량 여백(+0.012) 우측에 배치.
    마지막 subplot(col_idx=n)의 colorbar는 paper 좌표를 약간 초과하나
    Plotly가 자동으로 처리함.

    예시 (n=3, spacing=0.045):
      col_width ≈ 0.303
      col=1 → domain [0.000, 0.303] → colorbar_x ≈ 0.315
      col=2 → domain [0.348, 0.651] → colorbar_x ≈ 0.663
      col=3 → domain [0.697, 1.000] → colorbar_x ≈ 1.012

    인자:
        col_idx: 1-based 컬럼 인덱스
        n_cols : 전체 subplot 컬럼 수
        spacing: horizontal_spacing 값
    """
    col_width  = (1.0 - (n_cols - 1) * spacing) / n_cols
    domain_end = (col_idx - 1) * (col_width + spacing) + col_width
    return round(domain_end + 0.012, 4)


def _apply_subplot_axes(
    fig: go.Figure,
    radius: float,
    col_idx: int,
) -> None:
    """
    특정 subplot 셀(col_idx)의 x/y 축에 1:1 비율 및 범위 설정 적용.

    [_wafer_layout을 재사용하지 않는 이유]
    _wafer_layout은 단일 Figure 전용으로 "xaxis", "yaxis" 키를 하드코딩.
    make_subplots에서:
      col=1: "xaxis",  "yaxis"    (suffix 없음)
      col=2: "xaxis2", "yaxis2"   (suffix="2")
      col=N: "xaxisN", "yaxisN"
    → update_layout에 동적 키 딕셔너리를 **언패킹으로 전달하여 해결.

    [scaleanchor 설정의 중요성]
    scaleanchor=f"y{suffix}": 해당 x축의 단위 길이를 같은 subplot의 y축에 고정.
    이 설정 없이는 컨테이너 너비에 따라 원형 웨이퍼가 타원으로 찌그러짐.
    subplot마다 독립적인 y축 참조(y, y2, y3...)를 사용해야 정확히 동작.

    인자:
        fig     : make_subplots로 생성된 Figure
        radius  : 웨이퍼 반지름 (mm 단위, get_wafer_grid에서 반환)
        col_idx : 1-based 컬럼 인덱스
    """
    # col=1 → suffix="" (xaxis, yaxis)
    # col=2 → suffix="2" (xaxis2, yaxis2)
    ax_suffix = "" if col_idx == 1 else str(col_idx)

    # 웨이퍼 표시 여백 계산
    r_side   = radius * 1.15   # 좌우 여백: 테두리에서 15% 여유
    r_bottom = radius * 1.20   # 하단 여백: Notch 돌출 공간 추가 확보
    r_top    = radius * 1.15   # 상단 여백: 좌우와 동일

    # update_layout에 동적 키 dict를 **로 언패킹
    # f-string으로 col_idx에 따라 "xaxis", "xaxis2", "xaxis3"... 동적 생성
    fig.update_layout(
        **{
            f"xaxis{ax_suffix}": dict(
                scaleanchor=f"y{ax_suffix}",  # ★ 1:1 비율 유지의 핵심
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-r_side, r_side],
            ),
            f"yaxis{ax_suffix}": dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-r_bottom, r_top],
            ),
        }
    )


# =============================================================================
# [핵심 함수: create_multi_param_subplots]
# =============================================================================

@st.cache_data
def create_multi_param_subplots(
    df_json: str,
    x_col: str,
    y_col: str,
    param_cols: tuple,     # ★ tuple 필수: list는 hash() 불가 → @st.cache_data TypeError
    resolution: int,
    colorscale: str,
    share_scale: bool,
) -> go.Figure:
    """
    다중 파라미터를 1행 N열 서브플롯 Heatmap으로 시각화.

    [캐시 키 구성 요소]
    (df_json, x_col, y_col, param_cols, resolution, colorscale, share_scale)
    - param_cols 변경(추가/제거/순서 변경) → tuple 달라짐 → 자동 캐시 갱신
    - df 편집 → df_json 달라짐 → 자동 캐시 갱신
    - resolution, colorscale 변경 → 자동 캐시 갱신
    ★ 모든 인자가 hashable 타입임을 보장해야 @st.cache_data 정상 동작:
      df_json    : str ✅
      x_col      : str ✅
      y_col      : str ✅
      param_cols : tuple (list 불가) ✅
      resolution : int ✅
      colorscale : str ✅
      share_scale: bool ✅

    [2단계 캐시 전략]
    이 함수(상위 캐시) 미스 시:
      → get_wafer_grid(sub_json, resolution) 호출 (하위 캐시)
      → 하위 캐시가 이전에 동일 sub_json으로 호출된 적 있으면 히트
      → 파라미터 1개만 추가/제거해도 나머지 파라미터는 하위 캐시 히트 → 재보간 없음

    [share_scale 동작]
    True:  전체 파라미터 값의 통합 min/max를 zmin/zmax로 설정
           → 파라미터 간 절대값 크기 비교 가능
           → colorbar는 마지막 subplot에만 1개 표시
    False: 각 파라미터 자체 범위 사용 (zmin=None, zmax=None → Plotly 자동)
           → 각 파라미터 내부의 공간 분포 패턴 비교에 유리
           → 각 subplot에 개별 colorbar 표시 (x 위치 수동 계산)

    인자:
        df_json    : x_col, y_col, 모든 param_cols 컬럼을 포함한 DataFrame의 JSON
        x_col      : X 좌표 컬럼명
        y_col      : Y 좌표 컬럼명
        param_cols : 분석할 파라미터 컬럼명 tuple (최소 2개, 최대 6개)
        resolution : 보간 그리드 해상도 (30~200)
        colorscale : Plotly 컬러스케일 이름 (예: "Rainbow", "Viridis")
        share_scale: True=전체 통일 스케일, False=파라미터별 개별 스케일

    반환:
        go.Figure: make_subplots로 구성된 1행 N열 Heatmap Figure
    """
    # ── 캐시 함수 진입: df_json 역직렬화 ─────────────────────────────────────
    # @st.cache_data 적용 함수는 DataFrame을 인자로 받을 수 없으므로
    # 항상 함수 진입 즉시 pd.read_json()으로 복원
    df = pd.read_json(df_json)
    n  = len(param_cols)  # subplot 컬럼 수

    # ── subplot 간격 계산: 파라미터 수가 많을수록 좁게 ──────────────────────
    # n=2: 0.05, n=3: 0.045, n=4: 0.04, n=5: 0.035, n=6: 0.03
    spacing = max(_SPACING_MIN, _SPACING_BASE - _SPACING_STEP * n)

    # ── 통일 스케일 계산 (share_scale=True일 때만) ────────────────────────
    # pd.concat으로 모든 파라미터 값을 한 번에 합산 → 벡터 연산 (빠름)
    # Python 루프로 extend 하는 것보다 NumPy 레벨 연산으로 처리
    global_zmin: float | None = None
    global_zmax: float | None = None
    if share_scale:
        all_vals = pd.concat(
            [df[col].dropna() for col in param_cols],
            ignore_index=True,
        )
        if len(all_vals) > 0:
            global_zmin = float(all_vals.min())
            global_zmax = float(all_vals.max())

    # ── make_subplots 생성 ───────────────────────────────────────────────────
    # shared_yaxes=False: 각 subplot이 독립 y축 보유
    #   → _apply_subplot_axes에서 scaleanchor로 각 subplot 독립적 1:1 비율 보장
    #   → shared_yaxes=True이면 scaleanchor가 첫 번째 y축에만 적용되어
    #     2번째 이후 subplot에서 원형이 타원으로 찌그러지는 버그 발생
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=list(param_cols),   # 각 subplot 상단 파라미터명 표시
        shared_yaxes=False,                # 독립 y축 → scaleanchor 정상 작동
        horizontal_spacing=spacing,
    )

    # ── 각 파라미터 처리 (1-based 인덱스) ────────────────────────────────────
    for i, param_col in enumerate(param_cols, start=1):

        # ── 파라미터별 표준 sub_df 생성 ─────────────────────────────────────
        # ★ 주의: 이 공식은 render_multi_param_tab의 calculate_stats 호출부와
        #   반드시 동일해야 get_wafer_grid 하위 캐시가 히트됨.
        # 순서 통일: [[x,y,param]] → rename → dropna → reset_index → to_json
        # 순서가 조금이라도 달라지면 JSON 문자열이 달라져 캐시 미스 발생.
        sub_df = (
            df[[x_col, y_col, param_col]]
            .rename(columns={x_col: "x", y_col: "y", param_col: "data"})
            .dropna()
            .reset_index(drop=True)
        )
        sub_json = sub_df.to_json()

        # ── 그리드 보간 (2단계 캐시의 하위 캐시 활용) ───────────────────────
        # get_wafer_grid는 wafer_app_global에서 @st.cache_data 적용됨.
        # sub_json이 이전과 같으면 → 캐시 히트 → 재보간 없음 (성능↑)
        XI, YI, ZI, radius = get_wafer_grid(sub_json, resolution)

        # ── colorbar 설정 결정 ───────────────────────────────────────────────
        if share_scale:
            # 통일 스케일 모드:
            # - 마지막 subplot에만 colorbar 1개 표시 (전체 스케일 기준)
            # - 나머지 subplot은 showscale=False로 colorbar 숨김
            show_scale   = (i == n)
            zmin, zmax   = global_zmin, global_zmax
            colorbar_cfg = (
                dict(thickness=12, len=0.80, xanchor="left",
                     title=dict(text="", side="right"))
                if show_scale else None
            )
        else:
            # 개별 스케일 모드:
            # - 각 subplot에 자체 colorbar 표시
            # - x 위치를 수동 계산하여 겹침 방지
            show_scale   = True
            zmin, zmax   = None, None      # Plotly가 자동으로 min/max 결정
            cb_x         = _calc_colorbar_x(i, n, spacing)
            # colorbar 제목: 컬럼명이 너무 길면 잘라서 표시 (공간 절약)
            cb_title = param_col if len(param_col) <= 8 else param_col[:7] + "…"
            colorbar_cfg = dict(
                thickness=10,
                len=0.75,
                x=cb_x,
                xanchor="left",
                title=dict(text=cb_title, side="right", font=dict(size=9)),
            )

        # ── Heatmap trace 추가 ───────────────────────────────────────────────
        # row=1, col=i: 1행 i열 subplot에 정확히 배치 (필수)
        # XI[0]   : x축 좌표 벡터 (모든 행에서 x값이 동일 → 첫 행만 추출)
        # YI[:,0] : y축 좌표 벡터 (모든 열에서 y값이 동일 → 첫 열만 추출)
        # zsmooth="best": 보간 그리드를 추가 스무딩 → 시각적 품질 향상
        # connectgaps=False: NaN(원 밖 마스크 영역)을 투명으로 유지
        fig.add_trace(
            go.Heatmap(
                x=XI[0],
                y=YI[:, 0],
                z=ZI,
                colorscale=colorscale,
                zsmooth="best",
                zmin=zmin,
                zmax=zmax,
                showscale=show_scale,
                colorbar=colorbar_cfg if show_scale else None,
                connectgaps=False,
                name=param_col,
            ),
            row=1,
            col=i,
        )

        # ── 웨이퍼 아웃라인 추가 ────────────────────────────────────────────
        # ★ add_wafer_outline(fig, radius) 대신 로컬 헬퍼 사용
        #   이유: add_wafer_outline은 row/col 인자 없음 → 모두 (1,1)에 쌓임
        #   로컬 헬퍼: row=1, col=i를 명시적으로 전달 → 올바른 subplot에 배치
        _add_outline_to_subplot(fig, radius, row=1, col=i)

        # ── 축 비율 설정 (scaleanchor로 원형 유지) ───────────────────────────
        # ★ _wafer_layout 대신 로컬 헬퍼 사용
        #   이유: _wafer_layout은 "xaxis"/"yaxis" 하드코딩 → col=2 이상 부적용
        #   로컬 헬퍼: col_idx에 따라 "xaxis", "xaxis2", "xaxis3"... 동적 생성
        _apply_subplot_axes(fig, radius, col_idx=i)

    # ── 전체 레이아웃 설정 ────────────────────────────────────────────────────
    # height 공식:
    #   파라미터 ≤3개: 400px (subplot이 충분히 넓어 높이 400으로도 원형 유지)
    #   파라미터 ≥4개: 700px (subplot 너비 감소를 height 증가로 보완)
    height = max(400, 350 * (1 if n <= 3 else 2))

    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        # r=80: 마지막 개별 colorbar가 잘리지 않도록 오른쪽 여백 확보
        margin=dict(l=10, r=80, t=50, b=10),
    )

    # ── subplot 제목 폰트 크기 조정 ──────────────────────────────────────────
    # 파라미터 수가 많을수록 subplot이 좁아지므로 제목도 작게 표시
    # n=2: 12px, n=3: 11px, n=4: 10px, n=5: 9px, n=6: 9px (최소 9px 보장)
    title_font_size = max(9, 14 - n)
    for annotation in fig.layout.annotations:
        annotation.font.size  = title_font_size
        annotation.font.color = "#333333"

    return fig


# =============================================================================
# [UI 렌더러: render_multi_param_tab]
# =============================================================================

def render_multi_param_tab(
    # df_raw: pd.DataFrame,
    df_json: str,
    all_cols: list,
    resolution: int,
    colorscale: str,
) -> None:
    df_raw = pd.read_json(df_json)
    """
    다중 파라미터 서브플롯 탭의 전체 UI를 렌더링.

    [레이아웃 구조]
    ┌─────────────────────┬─────────────────────┐
    │ X 좌표 컬럼 selectbox │ Y 좌표 컬럼 selectbox │  ← 2열
    └─────────────────────┴─────────────────────┘
    ┌─────────────────────────────────────────────┐
    │ 파라미터 컬럼 multiselect (2~6개)             │
    └─────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────┐
    │ [☐] 전체 통일 스케일 checkbox                 │
    └─────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────┐
    │  서브플롯 Heatmap (1행 N열, Plotly 차트)       │
    └─────────────────────────────────────────────┘
    ┌───────┬───────┬───────┬── ...                │
    │ U% ▲  │ U% ●  │ U% ▼  │ ...  (N개 metric)   │  ← 파라미터별 Uniformity%
    └───────┴───────┴───────┴────                  │

    [session_state 사용 이유]
    Streamlit은 매 상호작용(버튼, 슬라이더 등)마다 스크립트 전체 재실행.
    selectbox/multiselect의 선택값은 key로 session_state에 자동 저장되어
    재실행 후에도 유지됨.
    prefix "mp_"로 wafer_app_global의 기존 키와 충돌 없이 독립 관리.

    [df_raw vs df_subset 구분]
    df_raw    : 파일에서 로드된 원본 전체 DataFrame (모든 컬럼 포함)
    df_subset : 선택된 x, y, param 컬럼만 추출한 서브셋
              → JSON 크기 최소화 (불필요한 컬럼 제외)
              → create_multi_param_subplots에 전달

    인자:
        df_raw     : 원본 DataFrame (파일 로딩 직후 상태, apply_col_mapping 전)
        all_cols   : df_raw의 전체 컬럼명 리스트 (selectbox 옵션으로 사용)
        resolution : 보간 해상도 (사이드바 슬라이더 값 전달받음)
        colorscale : 컬러스케일 이름 (사이드바 selectbox 값 전달받음)
    """
    # ── X, Y 좌표 컬럼 selectbox (2열 배치) ──────────────────────────────────
    col_x_ui, col_y_ui = st.columns(2)

    with col_x_ui:
        # _default_col_index: 컬럼명 "x"(대소문자 무관)로 기본값 탐색
        # 없으면 fallback 인덱스 0(첫 번째 컬럼) 사용
        sel_x: str = st.selectbox(
            "X 좌표 컬럼",
            options=all_cols,
            index=_default_col_index(all_cols, "x", 0),
            key=_SS_X_COL,
        )

    with col_y_ui:
        sel_y: str = st.selectbox(
            "Y 좌표 컬럼",
            options=all_cols,
            index=_default_col_index(all_cols, "y", 1),
            key=_SS_Y_COL,
        )

    # ── 파라미터 후보 컬럼 계산 ──────────────────────────────────────────────
    # x, y로 선택된 컬럼은 파라미터 후보에서 제외 (좌표를 파라미터로 보간하는 것 방지)
    param_candidates = [c for c in all_cols if c not in (sel_x, sel_y)]

    if not param_candidates:
        st.warning(
            "⚠️ 파라미터로 사용 가능한 컬럼이 없습니다. "
            "X/Y 컬럼 설정을 확인하거나 더 많은 컬럼이 있는 파일을 사용하세요."
        )
        return

    # ── 파라미터 컬럼 multiselect ─────────────────────────────────────────────
    # 기본값: 후보 중 앞의 2개 선택 (후보가 1개뿐이면 1개만 → 아래 검증에서 안내)
    default_params = param_candidates[:min(2, len(param_candidates))]

    sel_params: list[str] = st.multiselect(
        f"파라미터 컬럼 선택 ({_MIN_PARAMS}~{_MAX_PARAMS}개)",
        options=param_candidates,
        default=default_params,
        key=_SS_PARAMS,
        help=(
            f"동시에 Heatmap으로 시각화할 측정 파라미터 컬럼을 선택합니다.\n\n"
            f"• 최소 {_MIN_PARAMS}개 이상 선택해야 서브플롯이 생성됩니다.\n"
            f"• 최대 {_MAX_PARAMS}개까지 선택 가능합니다.\n"
            f"• 예: Thickness, Rs (면저항), Stress, GPC 등\n\n"
            f"💡 파라미터 순서가 서브플롯 배치 순서가 됩니다."
        ),
    )

    # ── 선택 개수 검증 ────────────────────────────────────────────────────────
    if len(sel_params) < _MIN_PARAMS:
        # st.info: 오류가 아닌 안내 메시지
        # → 사용자가 선택을 완료하기 전의 자연스러운 상태 (경고 아님)
        st.info(
            f"📊 파라미터를 {_MIN_PARAMS}개 이상 선택하면 서브플롯이 생성됩니다. "
            f"(현재 {len(sel_params)}개 선택됨)"
        )
        return  # 아래 차트 렌더링 코드 실행 안 함

    # 최대 초과 시: 앞의 _MAX_PARAMS개만 사용하고 경고
    if len(sel_params) > _MAX_PARAMS:
        st.warning(
            f"⚠️ 최대 {_MAX_PARAMS}개까지 선택 가능합니다. "
            f"처음 {_MAX_PARAMS}개 파라미터만 표시합니다."
        )
        sel_params = sel_params[:_MAX_PARAMS]

    # ── 통일 스케일 선택 ──────────────────────────────────────────────────────
    share_scale: bool = st.checkbox(
        "🔒 전체 통일 스케일 (파라미터 간 절대값 크기 비교)",
        value=False,
        key=_SS_SHARE,
        help=(
            "✅ 체크 시: 모든 서브플롯이 동일한 색상 범위를 사용합니다.\n"
            "   → 파라미터 간 절대값 크기 차이를 색상으로 직접 비교 가능\n"
            "   → 예: Thickness 100Å vs 200Å의 차이가 색상으로 표현됨\n\n"
            "☐ 해제 시: 각 파라미터가 자체 min~max 범위 사용\n"
            "   → 각 파라미터 내부의 공간적 분포 패턴 비교에 유리\n"
            "   → 예: 모든 파라미터의 Edge-Center 편차 패턴 비교"
        ),
    )

    # ── df_subset 생성 ────────────────────────────────────────────────────────
    # 선택된 컬럼만 포함하여 JSON 크기 최소화 (불필요한 컬럼 제외)
    # dict.fromkeys로 중복 제거하면서 순서 유지
    # ([x, y] + params 중 중복 가능성: x 또는 y가 param으로도 선택된 엣지 케이스)
    needed_cols = list(dict.fromkeys([sel_x, sel_y] + sel_params))

    try:
        df_subset = df_raw[needed_cols].dropna().reset_index(drop=True)
    except KeyError as e:
        st.error(f"❌ 선택한 컬럼을 데이터에서 찾을 수 없습니다: {e}")
        return

    if len(df_subset) < 3:
        st.warning(
            "⚠️ 유효한 데이터 포인트가 3개 미만입니다. "
            "파일과 컬럼 설정을 확인하세요."
        )
        return

    # df_json 생성: 선택된 컬럼만 포함한 서브셋을 JSON으로 직렬화
    # create_multi_param_subplots 내부에서 sub_df를 추출하므로
    # 여기서는 전체 서브셋을 전달 (컬럼 선택은 함수 내부에서 처리)
    df_json = df_subset.to_json()

    # ── param_cols tuple 변환 ─────────────────────────────────────────────────
    # ★ 반드시 tuple로 변환: st.multiselect는 list를 반환하나
    #   @st.cache_data 함수에 list를 넘기면 hash() 불가 → TypeError
    param_cols_tuple: tuple[str, ...] = tuple(sel_params)

    # ── 서브플롯 생성 및 렌더링 ──────────────────────────────────────────────
    with st.spinner(f"서브플롯 생성 중... ({len(param_cols_tuple)}개 파라미터)"):
        fig = create_multi_param_subplots(
            df_json=df_json,
            x_col=sel_x,
            y_col=sel_y,
            param_cols=param_cols_tuple,
            resolution=resolution,
            colorscale=colorscale,
            share_scale=share_scale,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── 파라미터별 Uniformity(%) 통계 지표 ───────────────────────────────────
    st.markdown("##### 📊 파라미터별 통계 요약")
    metric_cols = st.columns(len(sel_params))

    for metric_col_widget, param_col in zip(metric_cols, sel_params):
        # ★ sub_json을 create_multi_param_subplots 내부와 완전히 동일한 방식으로 생성
        #   이 공식이 달라지면 get_wafer_grid / calculate_stats 캐시 미스 발생
        #   공식: [[x,y,param]] → rename → dropna → reset_index → to_json
        sub_df = (
            df_subset[[sel_x, sel_y, param_col]]
            .rename(columns={sel_x: "x", sel_y: "y", param_col: "data"})
            .dropna()
            .reset_index(drop=True)
        )
        sub_json = sub_df.to_json()

        # calculate_stats: @st.cache_data 적용됨
        # → sub_json이 create 함수 내부와 동일 → 캐시 히트 → 0 계산 비용
        stats = calculate_stats(sub_json)

        uniformity = stats.get("Uniformity (%)", float("nan"))
        mean_val   = stats.get("Mean",           float("nan"))
        n_sites    = stats.get("No. Sites",      0)

        # ── Uniformity(%) 기준 색상(delta_color) 결정 ─────────────────────
        # 반도체 공정 균일도 일반 기준:
        #   < 1.0%: 우수 (normal = 초록 화살표 ↑)
        #   1.0~2.0%: 양호 (off = 회색 화살표 →)
        #   > 2.0%: 주의 (inverse = 빨강 화살표 ↓)
        if isinstance(uniformity, (int, float)) and not pd.isna(uniformity):
            if uniformity < 1.0:
                delta_color = "normal"    # Streamlit st.metric: 초록색
                grade_label = "▲ 우수"
            elif uniformity < 2.0:
                delta_color = "off"       # Streamlit st.metric: 회색
                grade_label = "● 양호"
            else:
                delta_color = "inverse"   # Streamlit st.metric: 빨강색
                grade_label = "▼ 주의"
        else:
            delta_color = "off"
            grade_label = "N/A"

        # 컬럼명이 길면 잘라서 metric 라벨로 사용 (UI 레이아웃 유지)
        short_name = param_col if len(param_col) <= 14 else param_col[:12] + "…"

        metric_col_widget.metric(
            label=short_name,
            value=f"{uniformity:.3f} %" if not pd.isna(uniformity) else "N/A",
            # delta: 등급 라벨 + Mean + 측정 사이트 수를 한 줄에 압축
            delta=f"{grade_label} | μ={mean_val:.4g} | N={n_sites}",
            delta_color=delta_color,
            help=(
                f"파라미터: {param_col}\n"
                f"Uniformity(%) = σ/μ × 100\n\n"
                f"  Mean     : {stats.get('Mean',      'N/A')}\n"
                f"  Std Dev  : {stats.get('Std Dev',   'N/A')}\n"
                f"  Minimum  : {stats.get('Minimum',   'N/A')}\n"
                f"  Maximum  : {stats.get('Maximum',   'N/A')}\n"
                f"  Range    : {stats.get('Range',     'N/A')}\n"
                f"  No. Sites: {n_sites}"
            ),
        )