# core.py
# =============================================================================
# 공유 핵심 함수 모듈 (Streamlit Community Cloud 배포용)
#
# [왜 이 파일이 필요한가]
# 기존 구조: 모든 모듈이 "from app import ..." 으로 app.py의 함수를 import.
# 문제점:
#   1. app.py를 import하면 Streamlit이 스크립트 전체를 재실행 → 순환 실행 버그
#   2. Streamlit Community Cloud에서 "from app import X" 시 st.set_page_config()
#      중복 호출 에러 발생
#   3. 모듈 간 의존성이 app.py에 집중 → 테스트/재사용 불가
#
# 해결: app.py에서 순수 계산/시각화 함수만 추출 → core.py로 분리
#       모든 모듈은 "from core import X"로 변경
#       app.py는 UI 로직만 담당 (함수 정의 없음, core를 재export)
# =============================================================================

# ── 외부 라이브러리 ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata


# =============================================================================
# 데이터 처리 함수
# =============================================================================

@st.cache_data
def load_file_cached(full_path: str, sheet_name=None) -> pd.DataFrame:
    """CSV/Excel 파일 로드 (캐시 적용)."""
    if full_path.lower().endswith(".csv"):
        return pd.read_csv(full_path)
    effective_sheet = 0 if sheet_name is None else sheet_name
    return pd.read_excel(full_path, sheet_name=effective_sheet)


@st.cache_data
def get_sheet_names(full_path: str) -> list:
    """Excel 시트 목록 반환 (CSV는 빈 리스트)."""
    if full_path.lower().endswith(".csv"):
        return []
    try:
        with pd.ExcelFile(full_path) as xf:
            return xf.sheet_names
    except Exception:
        return []


@st.cache_data
def get_wafer_grid(df_json: str, resolution: int):
    """
    불규칙 산점(x,y,z) → 균일 그리드(XI, YI, ZI) 보간.
    3단계 폴백: linear → nearest → NaN 배열.
    """
    df = pd.read_json(df_json)
    x, y, z = df["x"].values, df["y"].values, df["data"].values

    radius = np.sqrt(x**2 + y**2).max()
    xi = np.linspace(-radius, radius, resolution)
    yi = np.linspace(-radius, radius, resolution)
    XI, YI = np.meshgrid(xi, yi)

    try:
        ZI = griddata((x, y), z, (XI, YI), method="linear")
    except Exception:
        try:
            ZI = griddata((x, y), z, (XI, YI), method="nearest")
        except Exception:
            ZI = np.full_like(XI, np.nan)

    ZI[XI**2 + YI**2 > radius**2] = np.nan
    return XI, YI, ZI, radius


def add_wafer_outline(fig: go.Figure, radius: float) -> None:
    """웨이퍼 원형 테두리 + Notch 추가."""
    theta = np.linspace(0, 2 * np.pi, 360)
    fig.add_trace(go.Scatter(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip"
    ))
    nt = np.linspace(np.pi, 2 * np.pi, 60)
    nr = radius * 0.03
    fig.add_trace(go.Scatter(
        x=nr * np.cos(nt),
        y=-radius + nr * np.sin(nt),
        mode="lines",
        line=dict(color="black", width=2),
        fill="toself", fillcolor="white",
        showlegend=False, hoverinfo="skip"
    ))


def _wafer_layout(radius: float, height: int) -> dict:
    """반복 사용되는 공통 레이아웃 설정."""
    return dict(
        xaxis=dict(
            scaleanchor="y", showgrid=False, zeroline=False,
            range=[-radius * 1.15, radius * 1.15]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            range=[-radius * 1.2, radius * 1.15]
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height,
        margin=dict(l=35, r=15, t=10, b=35)
    )


# =============================================================================
# 시각화 함수
# =============================================================================

@st.cache_data
def create_2d_heatmap(df_json: str, resolution: int, colorscale: str,
                      show_points: bool, compact: bool = False,
                      zmin=None, zmax=None) -> go.Figure:
    """2D Heatmap 생성."""
    df = pd.read_json(df_json)
    x, y = df["x"].values, df["y"].values
    XI, YI, ZI, radius = get_wafer_grid(df_json, resolution)
    height = 300 if compact else 460

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=XI[0], y=YI[:, 0], z=ZI,
        colorscale=colorscale, zsmooth="best",
        zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=10 if compact else 14, len=0.75),
        connectgaps=False
    ))
    add_wafer_outline(fig, radius)

    if show_points:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=3 if compact else 4, color="black", opacity=0.5),
            showlegend=False
        ))

    fig.update_layout(**_wafer_layout(radius, height))
    return fig


@st.cache_data
def create_contour_map(df_json: str, resolution: int, colorscale: str,
                       n_contours: int, show_points: bool,
                       compact: bool = False,
                       zmin=None, zmax=None) -> go.Figure:
    """Contour 맵 생성."""
    df = pd.read_json(df_json)
    x, y = df["x"].values, df["y"].values
    XI, YI, ZI, radius = get_wafer_grid(df_json, resolution)
    height = 300 if compact else 460

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=XI[0], y=YI[:, 0], z=ZI,
        colorscale=colorscale, ncontours=n_contours,
        contours=dict(coloring="heatmap", showlines=True),
        line=dict(width=0.8, color="rgba(0,0,0,0.6)"),
        zmin=zmin, zmax=zmax,
        colorbar=dict(
            thickness=10 if compact else 14,
            len=0.75 if compact else 0.85
        ),
        connectgaps=False
    ))
    add_wafer_outline(fig, radius)

    if show_points:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=3 if compact else 4, color="black", opacity=0.5),
            showlegend=False
        ))

    fig.update_layout(**_wafer_layout(radius, height))
    return fig


@st.cache_data
def create_3d_surface(df_json: str, resolution: int,
                      colorscale: str) -> go.Figure:
    """3D Surface 맵 생성."""
    XI, YI, ZI, _ = get_wafer_grid(df_json, resolution)
    fig = go.Figure(data=go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=colorscale,
        colorbar=dict(title="Value", thickness=14)
    ))
    fig.update_layout(
        title=dict(text="3D Surface", x=0.5),
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Data",
            bgcolor="white",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor="white", height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


@st.cache_data
def create_line_scan(df_json: str, angle_deg: int,
                     resolution: int) -> go.Figure:
    """특정 각도 방향 단면 프로파일 (Line Scan)."""
    df = pd.read_json(df_json)
    x, y, z = df["x"].values, df["y"].values, df["data"].values
    radius = np.sqrt(x**2 + y**2).max()
    angle_rad = np.deg2rad(angle_deg)

    positions = np.linspace(-radius, radius, resolution)
    px = positions * np.cos(angle_rad)
    py = positions * np.sin(angle_rad)

    profile = griddata((x, y), z, (px, py), method="linear")
    profile[~(px**2 + py**2 <= radius**2)] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions, y=profile,
        mode="lines+markers",
        line=dict(color="royalblue", width=2),
        marker=dict(size=4, color="royalblue")
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=dict(text=f"Line Scan — {angle_deg}°", x=0.5),
        xaxis=dict(title="Position (mm)", showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(title="Data", showgrid=True, gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380,
        margin=dict(l=60, r=20, t=50, b=50)
    )
    return fig


@st.cache_data
def calculate_stats(df_json: str) -> dict:
    """측정 데이터 통계. Uniformity(%) = (Std / Mean) × 100."""
    df = pd.read_json(df_json)
    d = df["data"].dropna()
    mean = d.mean()
    std = d.std()
    d_max = d.max()
    d_min = d.min()
    return {
        "Mean":           round(mean, 4),
        "Maximum":        round(d_max, 4),
        "Minimum":        round(d_min, 4),
        "Std Dev":        round(std, 4),
        "Uniformity (%)": round((std / mean) * 100, 4) if mean != 0 else float("nan"),
        "Range":          round(d_max - d_min, 4),
        "No. Sites":      int(len(d))
    }


# =============================================================================
# 유틸리티 함수
# =============================================================================

def apply_col_mapping(df_raw: pd.DataFrame,
                      x_col: str, y_col: str, data_col: str) -> pd.DataFrame:
    """사용자 선택 컬럼 → 내부 표준명(x, y, data)으로 매핑."""
    all_cols = df_raw.columns.tolist()
    x_idx = all_cols.index(x_col)
    y_idx = all_cols.index(y_col)
    data_idx = all_cols.index(data_col)
    return (
        pd.DataFrame({
            "x":    df_raw.iloc[:, x_idx].values,
            "y":    df_raw.iloc[:, y_idx].values,
            "data": df_raw.iloc[:, data_idx].values,
        })
        .dropna()
        .reset_index(drop=True)
    )


def _default_col_index(columns: list, name: str, fallback: int) -> int:
    """컬럼 기본값 탐색. name이 없으면 min(fallback, len-1) 반환."""
    if name in columns:
        return columns.index(name)
    return min(fallback, len(columns) - 1)
