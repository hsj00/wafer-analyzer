# app.py
# 웨이퍼 맵 분석기 v4.0 — 통합 메인 앱
# 실행: streamlit run app.py
#
# =============================================================================
# [v4.0 변경 이력]
# - [요청 1] 파일·폴더 선택 — @st.dialog 기반 팝업 구현
# - [요청 2] 비교 모드 데이터셋 이름 자동 입력
# - [요청 3] Raw Data 행 추가·수정·삭제 + 편집 내용 session_state 유지
# - [요청 4] 한글/영문 언어 선택 + README 버튼
# - [요청 5] 성능·UX·에러 핸들링 자유 개선
#
# 디렉토리 구조:
# wafer_analysis/
# ├── app.py
# ├── i18n.py
# ├── folder_picker_helper.py
# └── modules/
#     ├── __init__.py
#     ├── defect_overlay.py
#     ├── gpc.py
#     ├── report.py
#     └── ml_anomaly.py
# =============================================================================

# ── 표준 라이브러리 ────────────────────────────────────────────────────────────
import glob
import os
import platform
import subprocess
import sys
import time

# ── 외부 라이브러리 ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata

# ── i18n 모듈 ──────────────────────────────────────────────────────────────────
# [요청 4] 다국어 지원
from i18n import t, get_lang

# =============================================================================
# 모듈 import (graceful degradation)
# =============================================================================

try:
    from modules.defect_overlay import render_defect_tab
    DEFECT_AVAILABLE = True
except ImportError:
    DEFECT_AVAILABLE = False

try:
    from modules.gpc import render_gpc_tab
    GPC_AVAILABLE = True
except ImportError:
    GPC_AVAILABLE = False

try:
    from modules.report import render_report_tab
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

try:
    from modules.ml_anomaly import render_anomaly_tab
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# =============================================================================
# [1] 페이지 설정
# =============================================================================
st.set_page_config(page_title="Wafer Map Analyzer", layout="wide")


# =============================================================================
# [요청 1] 환경 감지 함수
# =============================================================================

def _is_cloud_env() -> bool:
    """Cloud 환경 자동 감지."""
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") is not None
        or os.path.exists("/.dockerenv")
        or os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
    )


IS_CLOUD = _is_cloud_env()


# =============================================================================
# [2] 폴더 선택 함수
# =============================================================================

def try_native_folder_dialog() -> str | None:
    """OS 네이티브 폴더 선택창 호출 (subprocess 방식)."""
    if IS_CLOUD:
        return None
    try:
        helper = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "folder_picker_helper.py"
        )
        if not os.path.exists(helper):
            return None

        initial = st.session_state.get("data_folder", os.path.expanduser("~"))

        result = subprocess.run(
            [sys.executable, helper, initial],
            capture_output=True,
            text=True,
            timeout=60
        )

        folder = result.stdout.strip()
        return folder if folder and os.path.isdir(folder) else None

    except Exception:
        return None


# [요청 1] @st.dialog 기반 인라인 폴더 브라우저 팝업
@st.dialog(t("fb_title") if "app_lang" in st.session_state else "📂 폴더 선택", width="large")
def folder_browser_dialog() -> None:
    """@st.dialog 기반 팝업 폴더 브라우저."""
    lang = get_lang()

    if platform.system() == "Darwin":
        st.info(t("fb_macos_tip"))

    if "browser_current" not in st.session_state:
        st.session_state.browser_current = os.path.expanduser("~")

    current = st.session_state.browser_current

    st.markdown(t("fb_current_path", current))

    manual = st.text_input(
        t("sidebar_path_input"), value=current,
        label_visibility="collapsed",
        placeholder=t("fb_path_placeholder")
    )
    if manual != current and os.path.isdir(manual):
        st.session_state.browser_current = manual
        st.rerun()

    st.markdown("---")

    parent = os.path.dirname(current)
    if parent != current:
        if st.button(t("fb_parent"), use_container_width=True, key="fb_up"):
            st.session_state.browser_current = parent
            st.rerun()

    home = os.path.expanduser("~")
    favorites: dict[str, str] = {
        t("fb_home"):       home,
        "🖥️ Desktop":      os.path.join(home, "Desktop"),
        "📄 Documents":     os.path.join(home, "Documents"),
    }
    if platform.system() == "Windows":
        favorites["💾 C:\\"] = "C:\\"

    valid_favs = {label: path for label, path in favorites.items()
                  if os.path.exists(path)}
    if valid_favs:
        fav_cols = st.columns(len(valid_favs))
        for col, (label, path) in zip(fav_cols, valid_favs.items()):
            if col.button(label, key=f"fav_{label}", use_container_width=True):
                st.session_state.browser_current = path
                st.rerun()

    st.markdown("---")

    try:
        entries = sorted([
            d for d in os.listdir(current)
            if os.path.isdir(os.path.join(current, d))
            and not d.startswith(".")
        ])

        if not entries:
            st.info(t("fb_no_subfolders"))
        else:
            st.markdown(t("fb_subfolders", len(entries)))
            for i in range(0, min(len(entries), 30), 3):
                row = st.columns(3)
                for j, col in enumerate(row):
                    if i + j < len(entries):
                        d = entries[i + j]
                        label = d if len(d) <= 18 else d[:16] + "…"
                        if col.button(
                            f"📁 {label}",
                            key=f"fb_dir_{i+j}",
                            use_container_width=True,
                            help=d
                        ):
                            st.session_state.browser_current = \
                                os.path.join(current, d)
                            st.rerun()

    except PermissionError:
        st.error(t("fb_permission_denied"))

    st.markdown("---")

    n_csv   = len(glob.glob(os.path.join(current, "*.csv")))
    n_xls   = len(glob.glob(os.path.join(current, "*.xls*")))
    n_total = n_csv + n_xls
    if n_total:
        st.success(t("fb_file_count", n_csv, n_xls, n_total))
    else:
        st.warning(t("fb_no_data_files"))

    if st.button(t("fb_confirm"), type="primary",
                 use_container_width=True, key="fb_confirm"):
        st.session_state.data_folder = current
        # [요청 1] session_state flag로 dialog 닫기 + rerun 예외 회피
        st.session_state._dialog_folder_selected = True
        st.rerun()


# [요청 1] Cloud 환경용 파일 업로더 다이얼로그
@st.dialog(t("file_dialog_title") if "app_lang" in st.session_state else "📄 파일 선택", width="large")
def file_upload_dialog() -> None:
    """Cloud 환경에서 st.file_uploader를 팝업으로 제공."""
    uploaded = st.file_uploader(
        t("file_dialog_upload_label"),
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help=t("file_dialog_upload_help"),
        key="cloud_file_uploader",
    )
    if uploaded:
        upload_dir = st.session_state.data_folder
        os.makedirs(upload_dir, exist_ok=True)
        for f in uploaded:
            fpath = os.path.join(upload_dir, f.name)
            with open(fpath, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded)} files saved")
        st.session_state._dialog_file_uploaded = True
        st.rerun()


# =============================================================================
# [3] 데이터 처리 함수
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
    """불규칙 산점(x,y,z) → 균일 그리드(XI, YI, ZI) 보간."""
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
        showlegend=False,
        hoverinfo="skip"
    ))

    nt = np.linspace(np.pi, 2 * np.pi, 60)
    nr = radius * 0.03
    fig.add_trace(go.Scatter(
        x=nr * np.cos(nt),
        y=-radius + nr * np.sin(nt),
        mode="lines",
        line=dict(color="black", width=2),
        fill="toself",
        fillcolor="white",
        showlegend=False,
        hoverinfo="skip"
    ))


def _wafer_layout(radius: float, height: int) -> dict:
    """반복 사용되는 공통 레이아웃 설정."""
    return dict(
        xaxis=dict(
            scaleanchor="y",
            showgrid=False,
            zeroline=False,
            range=[-radius * 1.15, radius * 1.15]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            range=[-radius * 1.2, radius * 1.15]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=35, r=15, t=10, b=35)
    )


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
        x=XI[0],
        y=YI[:, 0],
        z=ZI,
        colorscale=colorscale,
        zsmooth="best",
        zmin=zmin,
        zmax=zmax,
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
        colorscale=colorscale,
        ncontours=n_contours,
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
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Data",
            bgcolor="white",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor="white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


@st.cache_data
def create_line_scan(df_json: str, angle_deg: int,
                     resolution: int) -> go.Figure:
    """특정 각도 방향 단면 프로파일 (Line Scan)."""
    df = pd.read_json(df_json)
    x, y, z   = df["x"].values, df["y"].values, df["data"].values
    radius    = np.sqrt(x**2 + y**2).max()
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=60, r=20, t=50, b=50)
    )
    return fig


@st.cache_data
def calculate_stats(df_json: str) -> dict:
    """측정 데이터 통계 계산."""
    df   = pd.read_json(df_json)
    d    = df["data"].dropna()
    mean = d.mean()
    std  = d.std()
    d_max = d.max()
    d_min = d.min()
    return {
        "Mean":           round(mean, 4),
        "Maximum":        round(d_max, 4),
        "Minimum":        round(d_min, 4),
        "Std Dev":        round(std, 4),
        "Uniformity (%)": round((std / mean) * 100, 4) if mean != 0 else 0.0,
        "Range":          round(d_max - d_min, 4),
        "No. Sites":      int(len(d))
    }


def apply_col_mapping(df_raw: pd.DataFrame,
                      x_col: str, y_col: str, data_col: str) -> pd.DataFrame:
    """사용자가 선택한 컬럼명 → 내부 표준명(x, y, data)으로 통일."""
    all_cols = df_raw.columns.tolist()
    # [요청 5] 에러 핸들링 개선: 컬럼 존재 확인
    for col_name, col_val in [("X", x_col), ("Y", y_col), ("Data", data_col)]:
        if col_val not in all_cols:
            raise ValueError(f"{col_name} column '{col_val}' not found in data")
    x_idx    = all_cols.index(x_col)
    y_idx    = all_cols.index(y_col)
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


# =============================================================================
# [4] UI 헬퍼 함수
# =============================================================================

def _default_col_index(columns: list, name: str, fallback: int) -> int:
    """컬럼 기본값 탐색 헬퍼."""
    if name in columns:
        return columns.index(name)
    return min(fallback, len(columns) - 1)


def wafer_title_banner(fname: str, prefix: str = "") -> None:
    """편집 가능한 파란 제목 배너."""
    key = f"title_{prefix}{fname}"
    if key not in st.session_state:
        st.session_state[key] = os.path.splitext(fname)[0]

    new_title = st.text_input(
        t("compare_title_input"),
        value=st.session_state[key],
        key=f"input_{prefix}{fname}",
        label_visibility="collapsed"
    )
    st.session_state[key] = new_title

    st.markdown(
        f"<div style='text-align:center;padding:7px;background:#1a6bbf;"
        f"color:white;border-radius:7px;font-size:14px;font-weight:bold;"
        f"margin-bottom:6px;'>📊 {new_title}</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# [5] 비교 모드 헬퍼 (웨이퍼 맵 탭 내부에서 사용)
# =============================================================================

def dataset_id() -> str:
    """데이터셋 고유 ID 생성."""
    return f"ds_{time.time_ns()}"


def _render_compare_dataset_manager() -> None:
    """비교 서브탭: 등록된 데이터셋 목록 + 순서 변경(▲/▼) + 삭제(✕)."""
    datasets = st.session_state.get("wm_datasets", [])
    if not datasets:
        return

    st.markdown(t("compare_dataset_list"))

    for i, ds in enumerate(datasets):
        ds_id = ds["id"]
        c_name, c_up, c_dn, c_del = st.columns([5, 1, 1, 1])

        label = ds["name"] if len(ds["name"]) <= 20 else ds["name"][:18] + "…"
        c_name.markdown(
            f"<div style='padding-top:5px;font-size:12px;'>"
            f"<b>{i+1}.</b> {label}</div>",
            unsafe_allow_html=True
        )

        if c_up.button("▲", key=f"cmpup_{ds_id}", disabled=(i == 0)):
            datasets[i], datasets[i-1] = datasets[i-1], datasets[i]
            st.session_state.wm_datasets = datasets
            st.rerun()

        if c_dn.button("▼", key=f"cmpdn_{ds_id}", disabled=(i == len(datasets)-1)):
            datasets[i], datasets[i+1] = datasets[i+1], datasets[i]
            st.session_state.wm_datasets = datasets
            st.rerun()

        if c_del.button("✕", key=f"cmpdel_{ds_id}"):
            st.session_state.wm_datasets.pop(i)
            st.rerun()


def _render_compare_dataset_adder(file_names: list, data_folder: str) -> None:
    """비교 서브탭: 파일 또는 수동 입력으로 데이터셋 추가."""
    with st.expander(t("compare_add_expander"), expanded=(len(st.session_state.get("wm_datasets", [])) < 2)):

        add_file_tab, add_manual_tab = st.tabs([t("compare_from_file"), t("compare_manual_input")])

        # ── [탭 A] 파일에서 추가 ──────────────────────────────────────────
        with add_file_tab:
            if not file_names:
                st.info(t("compare_no_files"))
            else:
                fa_col, sh_col = st.columns([3, 2])
                with fa_col:
                    sel_file = st.selectbox(t("compare_file_label"), file_names, key="cmp_add_file")
                full_path = os.path.join(data_folder, sel_file)

                try:
                    sheets = get_sheet_names(full_path)
                except Exception:
                    sheets = []

                with sh_col:
                    if sheets:
                        sel_sheet = st.selectbox(t("compare_sheet_label"), sheets, key="cmp_add_sheet")
                    else:
                        sel_sheet = None
                        st.markdown(
                            f"<div style='padding-top:28px;font-size:12px;color:#888;'>"
                            f"{t('compare_csv_no_sheet')}</div>",
                            unsafe_allow_html=True,
                        )

                try:
                    df_preview = load_file_cached(full_path, sel_sheet)
                    all_cols = df_preview.columns.tolist()
                except Exception as exc:
                    st.error(t("compare_file_load_fail", exc))
                    return

                cx, cy, cd = st.columns(3)
                with cx:
                    x_col = st.selectbox(
                        t("sidebar_x_col"), all_cols,
                        index=_default_col_index(all_cols, "x", 0),
                        key="cmp_add_x",
                    )
                with cy:
                    y_col = st.selectbox(
                        t("sidebar_y_col"), all_cols,
                        index=_default_col_index(all_cols, "y", 1),
                        key="cmp_add_y",
                    )
                with cd:
                    data_col = st.selectbox(
                        t("sidebar_data_col"), all_cols,
                        index=_default_col_index(all_cols, "data", 2),
                        key="cmp_add_data",
                    )

                # [요청 2] 데이터셋 이름 자동 생성 — data_col 반영
                sheet_tag = f"[{sel_sheet}]" if sel_sheet else ""
                auto_name = f"{os.path.splitext(sel_file)[0]}{sheet_tag}·{data_col}"

                # [요청 2] 수동 수정 감지를 위한 session_state 패턴
                _prev_auto_key = "_cmp_prev_auto_name"
                _user_edited_key = "_cmp_name_user_edited"

                # data_col 변경 시 자동 이름 갱신 (사용자 수동 수정이 아닌 경우에만)
                if st.session_state.get(_prev_auto_key) != auto_name:
                    st.session_state[_prev_auto_key] = auto_name
                    if not st.session_state.get(_user_edited_key, False):
                        st.session_state["cmp_add_name"] = auto_name

                ds_name = st.text_input(
                    t("compare_ds_name"), value=auto_name, key="cmp_add_name",
                    on_change=lambda: st.session_state.update({_user_edited_key: True})
                )

                existing_names = [d.get("name") for d in st.session_state.get("wm_datasets", [])]
                btn_disabled = ds_name in existing_names
                if btn_disabled:
                    st.warning(t("compare_name_exists", ds_name))

                if st.button(t("compare_add_btn"), type="primary", key="cmp_add_btn",
                             disabled=btn_disabled, use_container_width=True):
                    try:
                        with st.spinner("..."):
                            df_mapped = apply_col_mapping(df_preview, x_col, y_col, data_col)
                        new_ds = {
                            "id":       dataset_id(),
                            "name":     ds_name,
                            "file":     full_path,
                            "sheet":    sel_sheet,
                            "x_col":    x_col,
                            "y_col":    y_col,
                            "data_col": data_col,
                            "df_json":  df_mapped.to_json(),
                        }
                        if "wm_datasets" not in st.session_state:
                            st.session_state.wm_datasets = []
                        st.session_state.wm_datasets.append(new_ds)
                        # [요청 2] 추가 후 자동이름 플래그 초기화
                        st.session_state[_user_edited_key] = False
                        st.success(t("compare_add_success", ds_name))
                        st.rerun()
                    except Exception as exc:
                        st.error(t("compare_add_fail", exc))

        # ── [탭 B] 수동 입력 ─────────────────────────────────────────────
        with add_manual_tab:
            st.caption(t("compare_manual_caption"))

            manual_name = st.text_input(
                t("compare_ds_name"),
                value=t("compare_manual_ds_name", len(st.session_state.get('wm_datasets', [])) + 1),
                key="cmp_manual_name",
            )

            _CMP_MANUAL_KEY = "cmp_manual_df"
            if _CMP_MANUAL_KEY not in st.session_state:
                st.session_state[_CMP_MANUAL_KEY] = pd.DataFrame({
                    "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                })

            edited_manual = st.data_editor(
                st.session_state[_CMP_MANUAL_KEY],
                num_rows="dynamic",
                key="cmp_manual_editor",
                column_config={
                    "x":    st.column_config.NumberColumn(t("col_x_mm"),  format="%.2f"),
                    "y":    st.column_config.NumberColumn(t("col_y_mm"),  format="%.2f"),
                    "data": st.column_config.NumberColumn(t("col_value"), format="%.4f"),
                },
                use_container_width=True,
                hide_index=False,
            )
            st.session_state[_CMP_MANUAL_KEY] = edited_manual

            df_valid = edited_manual.dropna(subset=["x", "y", "data"]).copy()
            for col in ["x", "y", "data"]:
                df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
            df_valid = df_valid.dropna().reset_index(drop=True)
            n_valid = len(df_valid)

            if n_valid > 0:
                st.success(t("compare_manual_valid", n_valid))
            else:
                st.info(t("compare_manual_hint"))

            existing_names = [d.get("name") for d in st.session_state.get("wm_datasets", [])]
            name_dup = manual_name in existing_names
            if name_dup:
                st.warning(t("compare_name_exists", manual_name))

            btn_col, reset_col = st.columns([3, 1])
            with btn_col:
                if st.button(
                    t("compare_manual_add_btn"), type="primary",
                    key="cmp_manual_add_btn",
                    disabled=(n_valid < 3 or name_dup),
                    use_container_width=True,
                ):
                    new_ds = {
                        "id":       dataset_id(),
                        "name":     manual_name,
                        "file":     None,
                        "sheet":    None,
                        "x_col":    "x",
                        "y_col":    "y",
                        "data_col": "data",
                        "df_json":  df_valid.to_json(),
                    }
                    if "wm_datasets" not in st.session_state:
                        st.session_state.wm_datasets = []
                    st.session_state.wm_datasets.append(new_ds)
                    st.session_state[_CMP_MANUAL_KEY] = pd.DataFrame({
                        "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                    })
                    st.success(t("compare_add_success", manual_name))
                    st.rerun()

            with reset_col:
                if st.button("🗑️", key="cmp_manual_reset", help=t("manual_reset_btn")):
                    st.session_state[_CMP_MANUAL_KEY] = pd.DataFrame({
                        "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                    })
                    st.rerun()

def _render_compare_card(ds: dict, resolution: int, colorscale: str,
                         n_contours: int, show_points: bool,
                         global_zmin, global_zmax) -> None:
    """비교 서브탭: 데이터셋 1개를 카드로 렌더링 (Heatmap + Contour + 통계)."""
    ds_id = ds["id"]

    # 제목 배너
    title_key = f"cmp_title_{ds_id}"
    if title_key not in st.session_state:
        st.session_state[title_key] = ds["name"]

    new_title = st.text_input(
        t("compare_title_label"), value=st.session_state[title_key],
        key=f"cmp_input_{ds_id}", label_visibility="collapsed"
    )
    st.session_state[title_key] = new_title
    st.markdown(
        f"<div style='text-align:center;padding:6px;background:#1a6bbf;"
        f"color:white;border-radius:6px;font-size:13px;font-weight:bold;"
        f"margin-bottom:6px;'>📊 {new_title}</div>",
        unsafe_allow_html=True
    )

    try:
        if "df_json" in ds and ds["df_json"]:
            df_json = ds["df_json"]
        else:
            df_raw = load_file_cached(ds["file"], ds["sheet"])
            df_mapped = apply_col_mapping(
                df_raw, ds["x_col"], ds["y_col"], ds["data_col"]
            )
            df_json = df_mapped.to_json()

        # Heatmap
        fig_hm = create_2d_heatmap(
            df_json, resolution, colorscale, show_points,
            compact=True, zmin=global_zmin, zmax=global_zmax
        )
        st.plotly_chart(fig_hm, use_container_width=True, key=f"cmp_hm_{ds_id}")

        # Contour
        fig_ct = create_contour_map(
            df_json, resolution, colorscale, n_contours, show_points,
            compact=True, zmin=global_zmin, zmax=global_zmax
        )
        st.plotly_chart(fig_ct, use_container_width=True, key=f"cmp_ct_{ds_id}")

        # 통계
        stats = calculate_stats(df_json)
        stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["값" if get_lang() == "ko" else "Value"])
        stats_df.index.name = "항목" if get_lang() == "ko" else "Metric"
        st.dataframe(stats_df, use_container_width=True)

        # Raw Data (접힘)
        # [요청 3] 비교 카드에서도 편집 가능한 data_editor 사용
        with st.expander(t("compare_raw_data"), expanded=False):
            _edit_key = f"cmp_edited_{ds_id}"
            df_disp = pd.read_json(df_json)

            edited_cmp = st.data_editor(
                st.session_state.get(_edit_key, df_disp),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key=f"cmp_editor_{ds_id}",
                column_config={
                    "x":    st.column_config.NumberColumn(t("col_x_mm"),  format="%.2f"),
                    "y":    st.column_config.NumberColumn(t("col_y_mm"),  format="%.2f"),
                    "data": st.column_config.NumberColumn(t("col_value"), format="%.3f")
                },
            )
            # [요청 3] 편집 내용 session_state에 저장
            st.session_state[_edit_key] = edited_cmp

    except Exception as e:
        st.error(f"❌ {e}")


# =============================================================================
# [6] 신규 탭 공통 가드 헬퍼
# =============================================================================

def _check_module_available(available: bool, module_name: str) -> bool:
    """모듈 import 실패 시 에러 메시지 표시."""
    if not available:
        st.error(t("module_unavailable", module_name))
        return False
    return True


def _check_shared_data() -> bool:
    """shared_df_json이 None이면 로드 안내 메시지 표시."""
    if st.session_state.get("shared_df_json") is None:
        st.info(t("load_data_first"))
        return False
    return True


# =============================================================================
# [7] 수동 입력 모드 헬퍼
# =============================================================================

_MANUAL_EMPTY_ROWS = 20

def _create_empty_wafer_df(n_rows: int = _MANUAL_EMPTY_ROWS) -> pd.DataFrame:
    """수동 입력용 빈 DataFrame 생성 (x, y, data 컬럼, NaN으로 채움)."""
    return pd.DataFrame({
        "x":    [None] * n_rows,
        "y":    [None] * n_rows,
        "data": [None] * n_rows,
    })


# =============================================================================
# [요청 4] README 표시 함수
# =============================================================================
def _show_readme():
    """현재 언어에 맞는 README를 메인 영역에 표시."""
    lang = get_lang()
    readme_file = "README_EN.md" if lang == "en" else "README.md"
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), readme_file)
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    else:
        st.info(f"README file not found: {readme_file}")


# =============================================================================
# [8] 메인 앱
# =============================================================================

# ── SIDEBAR: 언어 선택 + README 버튼 [요청 4] ──────────────────────────────
lang_options = {"한국어": "ko", "English": "en"}
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "ko"

selected_lang_label = st.sidebar.radio(
    "🌐 Language",
    list(lang_options.keys()),
    index=0 if st.session_state.app_lang == "ko" else 1,
    horizontal=True,
    key="_lang_radio",
)
new_lang = lang_options[selected_lang_label]
if new_lang != st.session_state.app_lang:
    st.session_state.app_lang = new_lang
    st.rerun()

# README 버튼
if "show_readme" not in st.session_state:
    st.session_state.show_readme = False

if st.sidebar.button(t("sidebar_readme_btn"), use_container_width=True, key="readme_btn"):
    st.session_state.show_readme = not st.session_state.show_readme
    st.rerun()

st.sidebar.markdown("---")

# ── 앱 제목 ──────────────────────────────────────────────────────────────────
st.title(t("app_title"))

# README 표시 모드
if st.session_state.show_readme:
    _show_readme()
    st.stop()

st.markdown("---")


# ── SIDEBAR: 데이터 관리 ────────────────────────────────────────────────────
st.sidebar.header(t("sidebar_data_mgmt"))

for key, val in [("data_folder", "./wafer_data/"),
                 ("show_folder_browser", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# [요청 1] dialog 플래그 체크 (rerun 후 dialog 닫기)
if st.session_state.pop("_dialog_folder_selected", False):
    pass  # data_folder는 이미 dialog 내에서 설정됨
if st.session_state.pop("_dialog_file_uploaded", False):
    pass  # 파일은 이미 저장됨

btn_col, path_col = st.sidebar.columns([1, 3])

with btn_col:
    if st.button("📂", help=t("sidebar_folder_btn_help"), use_container_width=True):
        if IS_CLOUD:
            # [요청 1] Cloud: file_uploader dialog
            file_upload_dialog()
        else:
            # [요청 1] Local: 네이티브 → 실패 시 @st.dialog 팝업
            folder = try_native_folder_dialog()
            if folder:
                st.session_state.data_folder = folder
                st.rerun()
            else:
                folder_browser_dialog()

with path_col:
    manual_input = st.text_input(
        t("sidebar_path_input"),
        value=st.session_state.data_folder,
        label_visibility="collapsed",
        key="folder_text_input"
    )
    if manual_input != st.session_state.data_folder:
        st.session_state.data_folder = manual_input
        st.rerun()

data_folder = st.session_state.data_folder

st.sidebar.markdown(
    f"<div style='font-size:11px;color:#888;word-break:break-all;"
    f"padding:4px 6px;background:#f4f4f4;border-radius:4px;'>"
    f"📂 {data_folder}</div>",
    unsafe_allow_html=True
)

# ── 파일 목록 수집 ─────────────────────────────────────────────────────────
all_files, file_names = [], []
if os.path.exists(data_folder):
    all_files = sorted(
        glob.glob(os.path.join(data_folder, "*.csv")) +
        glob.glob(os.path.join(data_folder, "*.xls*"))
    )
    file_names = [os.path.basename(f) for f in all_files]

# ── 샘플 데이터 생성 버튼 ───────────────────────────────────────────────────
if not file_names:
    st.sidebar.warning(t("sidebar_no_files"))
    if st.sidebar.button(t("sidebar_sample_btn"), type="primary"):
        os.makedirs(data_folder, exist_ok=True)
        for idx in range(1, 6):
            np.random.seed(idx * 7)
            n, R = 400, 100
            r_pts = np.sqrt(np.random.uniform(0, 1, n)) * R * 0.95
            t_pts = np.random.uniform(0, 2 * np.pi, n)
            x_pts = r_pts * np.cos(t_pts)
            y_pts = r_pts * np.sin(t_pts)
            cx, cy = np.random.uniform(-25, 25), np.random.uniform(-25, 25)
            vals = (500 + idx * 8
                    + 35 * np.exp(-((x_pts-cx)**2 + (y_pts-cy)**2) / (2*38**2))
                    - 10 * np.exp(-((x_pts+cx)**2 + (y_pts+cy)**2) / (2*20**2))
                    + np.random.normal(0, idx * 1.5, n))
            pd.DataFrame({"x": x_pts, "y": y_pts, "data": vals}).to_csv(
                os.path.join(data_folder, f"wafer_{idx:02d}.csv"), index=False
            )
        st.sidebar.success(t("sidebar_sample_done"))
        st.rerun()


# ── SIDEBAR: 파일 선택 (파일이 있을 때만) ──────────────────────────────────
has_files = bool(file_names)

if has_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader(t("sidebar_file_select"))

    selected_file = st.sidebar.selectbox(t("sidebar_analysis_file"), file_names)
    full_path     = os.path.join(data_folder, selected_file)

    sheets = get_sheet_names(full_path)
    if sheets:
        selected_sheet = st.sidebar.selectbox(t("sidebar_sheet_select"), options=sheets, key="s_sheet")
    else:
        selected_sheet = None

    # 파일/시트 변경 감지
    current_key = (selected_file, selected_sheet)
    if st.session_state.get("_s_file") != current_key:
        st.session_state._s_file    = current_key
        st.session_state._s_display = None
        st.session_state._s_col_map = None
        # [요청 3] 편집 데이터도 초기화
        st.session_state.pop("edited_df", None)
        # 공유 데이터 초기화
        for sk in ["shared_df_json", "shared_stats", "shared_fig_heatmap",
                    "shared_fig_contour", "shared_fig_linescan", "shared_fig_3d",
                    "shared_df_raw", "shared_all_cols", "shared_wafer_radius",
                    "shared_filename", "shared_raw_df_json",
                    "shared_df_raw_original"]:
            st.session_state[sk] = None
else:
    selected_file  = None
    full_path      = None
    selected_sheet = None


# ── SIDEBAR: 시각화 설정 ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header(t("sidebar_vis_settings"))

COLORSCALES = {
    "Rainbow":       "Rainbow",
    "RdYlBu (반전)": "RdYlBu_r",
    "Jet":           "Jet",
    "Hot":           "Hot",
    "Viridis":       "Viridis",
    "Plasma":        "Plasma",
    "Spectral":      "Spectral_r"
}
cs_name    = st.sidebar.selectbox(t("sidebar_colorscale"), list(COLORSCALES.keys()), index=0)
colorscale = COLORSCALES[cs_name]

resolution  = st.sidebar.slider(t("sidebar_resolution"), 30, 200, 100, 10)
show_points = st.sidebar.checkbox(t("sidebar_show_points"), value=False)
n_contours  = st.sidebar.slider(t("sidebar_contour_levels"), 5, 40, 20)
line_angle  = st.sidebar.slider(t("sidebar_linescan_angle"), 0, 175, 0, 5)


# =============================================================================
# [9] 데이터 소스 결정 (파일 or 수동 입력)
# =============================================================================
if "data_source_mode" not in st.session_state:
    st.session_state.data_source_mode = "file" if has_files else "manual"


# =============================================================================
# [10] 탭 구성 — 5개 탭
# =============================================================================

tab_labels = [
    t("tab_wafer"),
    t("tab_defect") + ("" if DEFECT_AVAILABLE else " ⚠️"),
    t("tab_gpc")    + ("" if GPC_AVAILABLE    else " ⚠️"),
    t("tab_report") + ("" if REPORT_AVAILABLE else " ⚠️"),
    t("tab_ml")     + ("" if ML_AVAILABLE     else " ⚠️"),
]

(tab_wafer,
 tab_defect,
 tab_gpc,
 tab_report,
 tab_ml) = st.tabs(tab_labels)


# =============================================================================
# [TAB 1] 웨이퍼 맵 — 단일 분석 + 비교 분석 서브탭
# =============================================================================
with tab_wafer:
    sub_single, sub_compare = st.tabs([t("subtab_single"), t("subtab_compare")])

    # =====================================================================
    # [서브탭 A] 단일 분석
    # =====================================================================
    with sub_single:
        source_options = []
        if has_files:
            source_options.append(t("source_file"))
        source_options.append(t("source_manual"))

        if len(source_options) > 1:
            data_source = st.radio(
                t("source_label"),
                options=source_options,
                horizontal=True,
                key="wm_data_source_radio",
            )
        else:
            data_source = source_options[0]

        use_manual = (data_source == t("source_manual"))

        if use_manual:
            # ── 수동 입력 모드 ──────────────────────────────────────────────
            st.markdown(t("manual_title"))
            st.caption(t("manual_caption"))

            if "manual_df" not in st.session_state:
                st.session_state.manual_df = _create_empty_wafer_df()

            edited_manual = st.data_editor(
                st.session_state.manual_df,
                num_rows="dynamic",
                key="manual_editor",
                column_config={
                    "x":    st.column_config.NumberColumn(t("col_x_mm"),  format="%.2f"),
                    "y":    st.column_config.NumberColumn(t("col_y_mm"),  format="%.2f"),
                    "data": st.column_config.NumberColumn(t("col_value"), format="%.4f"),
                },
                use_container_width=True,
                hide_index=False,
            )
            st.session_state.manual_df = edited_manual

            df_valid = (
                edited_manual
                .dropna(subset=["x", "y", "data"])
                .reset_index(drop=True)
            )
            for col in ["x", "y", "data"]:
                df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
            df_valid = df_valid.dropna().reset_index(drop=True)

            n_valid = len(df_valid)

            reset_col, info_col = st.columns([1, 3])
            with reset_col:
                if st.button(t("manual_reset_btn"), key="manual_reset"):
                    st.session_state.manual_df = _create_empty_wafer_df()
                    st.rerun()
            with info_col:
                if n_valid >= 3:
                    st.success(t("manual_valid_points", n_valid))
                elif n_valid > 0:
                    st.warning(t("manual_min_warning", n_valid))
                else:
                    st.info(t("manual_input_hint"))

            if n_valid < 3:
                st.session_state["shared_df_json"] = None
                st.session_state["shared_filename"] = t("manual_label")
                st.session_state["shared_all_cols"] = ["x", "y", "data"]
            else:
                df_display = df_valid.copy()
                df_json    = df_display.to_json()

                # [요청 5] st.spinner 추가
                with st.spinner("..."):
                    stats      = calculate_stats(df_json)
                    fig_heatmap  = create_2d_heatmap(df_json, resolution, colorscale, show_points)
                    fig_contour  = create_contour_map(df_json, resolution, colorscale,
                                                      n_contours, show_points)
                    fig_linescan = create_line_scan(df_json, line_angle, resolution)
                    fig_3d       = create_3d_surface(df_json, resolution, colorscale)
                    _, _, _, wafer_radius = get_wafer_grid(df_json, resolution)

                # 공유 데이터 저장
                st.session_state["shared_df_json"]      = df_json
                st.session_state["shared_stats"]        = stats
                st.session_state["shared_fig_heatmap"]  = fig_heatmap
                st.session_state["shared_fig_contour"]  = fig_contour
                st.session_state["shared_fig_linescan"] = fig_linescan
                st.session_state["shared_fig_3d"]       = fig_3d
                st.session_state["shared_df_raw"]       = df_display
                st.session_state["shared_all_cols"]     = ["x", "y", "data"]
                st.session_state["shared_wafer_radius"] = float(wafer_radius)
                st.session_state["shared_filename"]     = t("manual_label")
                st.session_state["shared_raw_df_json"]  = df_json
                st.session_state["shared_df_raw_original"] = df_display

                st.markdown(
                    f"<div style='text-align:center;padding:7px;background:#1a6bbf;"
                    f"color:white;border-radius:7px;font-size:14px;font-weight:bold;"
                    f"margin-bottom:6px;'>📊 {t('manual_data_title')}</div>",
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns([5, 5, 2])
                with c1:
                    st.markdown(t("chart_heatmap"))
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                with c2:
                    st.markdown(t("chart_contour"))
                    st.plotly_chart(fig_contour, use_container_width=True)
                with c3:
                    st.markdown(t("chart_statistics"))
                    for k, v in stats.items():
                        st.markdown(f"**{k}**")
                        st.code(str(v), language=None)

                c4, c5 = st.columns([2, 3])
                with c4:
                    st.plotly_chart(fig_linescan, use_container_width=True)
                with c5:
                    st.plotly_chart(fig_3d, use_container_width=True)

                st.download_button(
                    t("csv_download"),
                    df_display.to_csv(index=False),
                    "manual_wafer_export.csv",
                    "text/csv",
                    key="manual_csv_dl",
                )

        else:
            # ── 파일 데이터 모드 ──────────────────────────────────────────────
            if not has_files:
                st.info(t("file_no_files_info"))
            else:
                try:
                    df_raw = load_file_cached(full_path, selected_sheet)
                except Exception as e:
                    st.error(t("file_load_fail", e))
                    st.stop()

                all_cols = df_raw.columns.tolist()

                st.sidebar.markdown("---")
                st.sidebar.subheader(t("sidebar_col_mapping"))

                def_x = _default_col_index(all_cols, "x",    0)
                def_y = _default_col_index(all_cols, "y",    1)
                def_d = _default_col_index(all_cols, "data", 2)

                x_col    = st.sidebar.selectbox(t("sidebar_x_col"),    all_cols, index=def_x, key="s_xcol")
                y_col    = st.sidebar.selectbox(t("sidebar_y_col"),    all_cols, index=def_y, key="s_ycol")
                data_col = st.sidebar.selectbox(t("sidebar_data_col"), all_cols, index=def_d, key="s_dcol")

                col_key = f"{x_col}|{y_col}|{data_col}"

                # [요청 3] 컬럼 매핑 변경 시 편집 데이터도 초기화
                if st.session_state.get("_s_col_map") != col_key:
                    st.session_state._s_col_map = col_key
                    st.session_state._s_display = apply_col_mapping(df_raw, x_col, y_col, data_col)
                    st.session_state.pop("edited_df", None)

                # [요청 3] 편집된 DataFrame이 있으면 우선 사용
                if "edited_df" in st.session_state and st.session_state["edited_df"] is not None:
                    df_display = st.session_state["edited_df"]
                elif st.session_state.get("_s_display") is not None:
                    df_display = st.session_state._s_display
                else:
                    df_display = apply_col_mapping(df_raw, x_col, y_col, data_col)
                    st.session_state._s_display = df_display

                df_json = df_display.to_json()

                # [요청 5] st.spinner 추가
                with st.spinner("..."):
                    stats      = calculate_stats(df_json)
                    fig_heatmap  = create_2d_heatmap(df_json, resolution, colorscale, show_points)
                    fig_contour  = create_contour_map(df_json, resolution, colorscale,
                                                      n_contours, show_points)
                    fig_linescan = create_line_scan(df_json, line_angle, resolution)
                    fig_3d       = create_3d_surface(df_json, resolution, colorscale)
                    _, _, _, wafer_radius = get_wafer_grid(df_json, resolution)

                # 공유 데이터 저장
                st.session_state["shared_df_json"]      = df_json
                st.session_state["shared_stats"]        = stats
                st.session_state["shared_fig_heatmap"]  = fig_heatmap
                st.session_state["shared_fig_contour"]  = fig_contour
                st.session_state["shared_fig_linescan"] = fig_linescan
                st.session_state["shared_fig_3d"]       = fig_3d
                st.session_state["shared_df_raw"]       = df_display
                st.session_state["shared_all_cols"]     = all_cols
                st.session_state["shared_wafer_radius"] = float(wafer_radius)
                st.session_state["shared_filename"]     = selected_file
                st.session_state["shared_raw_df_json"]  = df_raw.to_json()
                st.session_state["shared_df_raw_original"] = df_raw

                # 제목 배너
                wafer_title_banner(selected_file, prefix="single_")

                # ── 차트 레이아웃 ──────────────────────────────────────────
                c1, c2, c3 = st.columns([5, 5, 2])
                with c1:
                    st.markdown(t("chart_heatmap"))
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                with c2:
                    st.markdown(t("chart_contour"))
                    st.plotly_chart(fig_contour, use_container_width=True)
                with c3:
                    st.markdown(t("chart_statistics"))
                    for k, v in stats.items():
                        st.markdown(f"**{k}**")
                        st.code(str(v), language=None)

                c4, c5 = st.columns([2, 3])
                with c4:
                    st.plotly_chart(fig_linescan, use_container_width=True)
                with c5:
                    st.plotly_chart(fig_3d, use_container_width=True)

                # ── [요청 3] Raw Data 편집 (행 추가·수정·삭제 + 편집 유지) ──
                st.markdown("---")
                st.subheader(t("raw_data_header"))

                edited_df = st.data_editor(
                    df_display,
                    num_rows="dynamic",
                    key=f"single_editor_{selected_file}_{col_key}",
                    column_config={
                        "x":    st.column_config.NumberColumn(t("col_x_mm"),  format="%.2f"),
                        "y":    st.column_config.NumberColumn(t("col_y_mm"),  format="%.2f"),
                        "data": st.column_config.NumberColumn(t("col_value"), format="%.3f")
                    },
                    use_container_width=True,
                    hide_index=False
                )
                # [요청 3] 편집된 DataFrame을 session_state에 저장 → rerun 후에도 유지
                st.session_state["edited_df"] = edited_df
                st.session_state._s_display = edited_df

                # [요청 3] 편집 초기화 버튼
                reset_col, dl_col = st.columns([1, 2])
                with reset_col:
                    if st.button(t("raw_data_reset_btn"), key="reset_edit_btn"):
                        st.session_state.pop("edited_df", None)
                        st.session_state._s_display = apply_col_mapping(
                            df_raw, x_col, y_col, data_col
                        )
                        st.success(t("raw_data_reset_done"))
                        st.rerun()
                with dl_col:
                    st.download_button(
                        t("csv_download"),
                        edited_df.to_csv(index=False),
                        "wafer_export.csv",
                        "text/csv"
                    )

    # =====================================================================
    # [서브탭 B] 비교 분석
    # =====================================================================
    with sub_compare:
        st.markdown(t("compare_title"))
        st.caption(t("compare_caption"))

        if "wm_datasets" not in st.session_state:
            st.session_state.wm_datasets = []

        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            cols_per_row = st.radio(
                t("compare_cols_per_row"), [2, 3, 4, 5, 6], index=1,
                horizontal=True,
                key="cmp_cols_per_row"
            )
        with ctrl_col2:
            lock_scale = st.checkbox(
                t("compare_lock_scale"), value=False,
                key="cmp_lock_scale"
            )

        st.markdown("---")

        _render_compare_dataset_manager()
        _render_compare_dataset_adder(file_names, data_folder)

        datasets = st.session_state.get("wm_datasets", [])

        if len(datasets) < 2:
            st.info(t("compare_min_info"))
        else:
            st.markdown("---")
            n_sel = len(datasets)
            st.markdown(t("compare_comparing", n_sel))

            global_zmin, global_zmax = None, None
            if lock_scale:
                series_list = []
                for ds in datasets:
                    try:
                        if "df_json" in ds and ds["df_json"]:
                            df_tmp = pd.read_json(ds["df_json"])
                        else:
                            dfc = load_file_cached(ds["file"], ds["sheet"])
                            df_tmp = apply_col_mapping(
                                dfc, ds["x_col"], ds["y_col"], ds["data_col"]
                            )
                        series_list.append(df_tmp["data"].dropna())
                    except Exception:
                        pass

                if series_list:
                    all_vals = pd.concat(series_list, ignore_index=True)
                    global_zmin = float(all_vals.min())
                    global_zmax = float(all_vals.max())
                    st.info(t("compare_scale_info").format(global_zmin, global_zmax))

            for batch_start in range(0, n_sel, cols_per_row):
                batch = datasets[batch_start : batch_start + cols_per_row]
                cols  = st.columns(len(batch))
                for col, ds in zip(cols, batch):
                    with col:
                        _render_compare_card(
                            ds, resolution, colorscale,
                            n_contours, show_points,
                            global_zmin, global_zmax
                        )
                if batch_start + cols_per_row < n_sel:
                    st.markdown(
                        "<hr style='border:1px solid #ddd;'>",
                        unsafe_allow_html=True
                    )


# =============================================================================
# [TAB 2] 결함 오버레이
# =============================================================================
with tab_defect:
    if not _check_module_available(DEFECT_AVAILABLE, "defect_overlay"):
        pass
    elif not _check_shared_data():
        pass
    else:
        render_defect_tab(
            wafer_df_json=st.session_state["shared_df_json"],
            wafer_radius=st.session_state["shared_wafer_radius"],
            resolution=resolution,
            colorscale=colorscale,
            data_folder=data_folder,
        )


# =============================================================================
# [TAB 3] GPC 분석
# =============================================================================
with tab_gpc:
    if not _check_module_available(GPC_AVAILABLE, "gpc"):
        pass
    elif not _check_shared_data():
        pass
    else:
        render_gpc_tab(
            df_raw=st.session_state.get("shared_df_raw_original", pd.DataFrame()),
            all_cols=st.session_state.get("shared_all_cols", []),
            resolution=resolution,
            colorscale=colorscale,
        )


# =============================================================================
# [TAB 4] 보고서 생성
# =============================================================================
with tab_report:
    if not _check_module_available(REPORT_AVAILABLE, "report"):
        pass
    elif not _check_shared_data():
        pass
    else:
        gpc_data = st.session_state.get("gpc_result", None)
        render_report_tab(
            filename=st.session_state.get("shared_filename", "unknown"),
            stats=st.session_state["shared_stats"],
            df_display=st.session_state["shared_df_raw"],
            fig_heatmap=st.session_state["shared_fig_heatmap"],
            fig_contour=st.session_state["shared_fig_contour"],
            fig_linescan=st.session_state["shared_fig_linescan"],
            fig_3d=st.session_state["shared_fig_3d"],
            gpc_data=gpc_data,
        )


# =============================================================================
# [TAB 5] ML 이상 탐지
# =============================================================================
with tab_ml:
    if not _check_module_available(ML_AVAILABLE, "ml_anomaly"):
        pass
    else:
        app_initial_datasets = []

        current_df_json  = st.session_state.get("shared_df_json")
        current_filename = st.session_state.get("shared_filename", "")
        if current_df_json is not None and current_filename:
            app_initial_datasets.append({
                "name":    os.path.splitext(current_filename)[0] if current_filename != t("manual_label") else t("manual_label"),
                "df_json": current_df_json,
            })

        for ds in st.session_state.get("wm_datasets", []):
            if any(m["name"] == ds["name"] for m in app_initial_datasets):
                continue
            ds_json = ds.get("df_json")
            if ds_json:
                app_initial_datasets.append({
                    "name":    ds["name"],
                    "df_json": ds_json,
                })
            else:
                try:
                    df_ml = apply_col_mapping(
                        load_file_cached(ds["file"], ds["sheet"]),
                        ds["x_col"], ds["y_col"], ds["data_col"],
                    )
                    app_initial_datasets.append({
                        "name":    ds["name"],
                        "df_json": df_ml.to_json(),
                    })
                except Exception:
                    pass

        render_anomaly_tab(
            datasets=app_initial_datasets,
            resolution=resolution,
            data_folder=data_folder,
        )
