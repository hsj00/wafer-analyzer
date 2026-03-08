# app.py
# 웨이퍼 맵 분석기 v3.0 — 통합 메인 앱
# 실행: streamlit run app.py
#
# =============================================================================
# [v3.0 변경 이력]
# - 비교 모드를 웨이퍼 맵 탭 내부 서브탭으로 통합
# - 사이드바 비교 모드 토글/데이터셋 관리 UI 제거
# - 다중 파라미터(multi_param) 모듈 제거 → 비교 서브탭에서 대체
#   (같은 파일의 다른 Data 컬럼을 데이터셋으로 추가)
# - 데이터 없이도 웨이퍼 맵 사용 가능 (수동 입력 모드)
#   빈 테이블에 x, y, data를 직접 붙여넣어 맵 생성
# - 탭 구조: 웨이퍼 맵(단일+비교) | 결함 오버레이 | GPC | 보고서 | ML
#
# 디렉토리 구조:
# wafer_analysis/
# ├── app.py
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
st.set_page_config(page_title="웨이퍼 맵 분석기", layout="wide")


# =============================================================================
# [2] 폴더 선택 함수
# =============================================================================

def try_native_folder_dialog() -> str | None:
    """OS 네이티브 폴더 선택창 호출 (subprocess 방식)."""
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


def render_folder_browser() -> None:
    """메인 화면에 직접 렌더링하는 인라인 폴더 브라우저."""
    with st.container(border=True):
        hd_col, close_col = st.columns([8, 1])
        hd_col.markdown("### 📂 폴더 선택")

        if close_col.button("✕", key="fb_close", use_container_width=True):
            st.session_state.show_folder_browser = False
            st.rerun()

        if platform.system() == "Darwin":
            st.info("💡 네이티브 창: `brew install python-tk@3.14` 설치 후 재시작")

        if "browser_current" not in st.session_state:
            st.session_state.browser_current = os.path.expanduser("~")

        current = st.session_state.browser_current
        st.markdown(f"**현재 위치:** `{current}`")

        manual = st.text_input(
            "경로 입력", value=current,
            label_visibility="collapsed",
            placeholder="경로를 직접 입력하세요..."
        )
        if manual != current and os.path.isdir(manual):
            st.session_state.browser_current = manual
            st.rerun()

        st.markdown("---")

        parent = os.path.dirname(current)
        if parent != current:
            if st.button("⬆️ 상위 폴더", use_container_width=True, key="fb_up"):
                st.session_state.browser_current = parent
                st.rerun()

        home = os.path.expanduser("~")
        favorites: dict[str, str] = {
            "🏠 홈":         home,
            "🖥️ Desktop":   os.path.join(home, "Desktop"),
            "📄 Documents":  os.path.join(home, "Documents"),
        }
        if platform.system() == "Windows":
            favorites["💾 C:\\"] = "C:\\"

        valid_favs = {label: path for label, path in favorites.items()
                      if os.path.exists(path)}
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
                st.info("하위 폴더 없음")
            else:
                st.markdown(f"**하위 폴더 ({len(entries)}개)**")
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
            st.error("⛔ 접근 권한 없음")

        st.markdown("---")

        n_csv   = len(glob.glob(os.path.join(current, "*.csv")))
        n_xls   = len(glob.glob(os.path.join(current, "*.xls*")))
        n_total = n_csv + n_xls
        if n_total:
            st.success(f"✅ CSV {n_csv}개 · XLS {n_xls}개 (총 {n_total}개)")
        else:
            st.warning("⚠️ CSV/XLS 파일 없음")

        if st.button("✅ 이 폴더 선택", type="primary",
                     use_container_width=True, key="fb_confirm"):
            st.session_state.data_folder        = current
            st.session_state.show_folder_browser = False
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
        "제목",
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

    st.markdown("**📋 데이터셋 목록**")

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
    with st.expander("➕ 데이터셋 추가", expanded=(len(st.session_state.get("wm_datasets", [])) < 2)):

        add_file_tab, add_manual_tab = st.tabs(["📁 파일에서 추가", "✏️ 수동 입력"])

        # ── [탭 A] 파일에서 추가 (기존 로직 그대로) ──────────────────────
        with add_file_tab:
            if not file_names:
                st.info("ℹ️ 데이터 폴더에 파일이 없습니다. '✏️ 수동 입력' 탭을 사용하세요.")
            else:
                # === 기존 파일 선택 로직 전체 (sel_file ~ st.button까지) ===
                fa_col, sh_col = st.columns([3, 2])
                with fa_col:
                    sel_file = st.selectbox("파일", file_names, key="cmp_add_file")
                full_path = os.path.join(data_folder, sel_file)

                try:
                    sheets = get_sheet_names(full_path)
                except Exception:
                    sheets = []

                with sh_col:
                    if sheets:
                        sel_sheet = st.selectbox("시트", sheets, key="cmp_add_sheet")
                    else:
                        sel_sheet = None
                        st.markdown(
                            "<div style='padding-top:28px;font-size:12px;color:#888;'>"
                            "CSV (시트 없음)</div>",
                            unsafe_allow_html=True,
                        )

                try:
                    df_preview = load_file_cached(full_path, sel_sheet)
                    all_cols = df_preview.columns.tolist()
                except Exception as exc:
                    st.error(f"❌ 파일 읽기 실패: {exc}")
                    return

                cx, cy, cd = st.columns(3)
                with cx:
                    x_col = st.selectbox(
                        "X 컬럼", all_cols,
                        index=_default_col_index(all_cols, "x", 0),
                        key="cmp_add_x",
                    )
                with cy:
                    y_col = st.selectbox(
                        "Y 컬럼", all_cols,
                        index=_default_col_index(all_cols, "y", 1),
                        key="cmp_add_y",
                    )
                with cd:
                    data_col = st.selectbox(
                        "Data 컬럼", all_cols,
                        index=_default_col_index(all_cols, "data", 2),
                        key="cmp_add_data",
                    )

                sheet_tag = f"[{sel_sheet}]" if sel_sheet else ""
                auto_name = f"{os.path.splitext(sel_file)[0]}{sheet_tag}·{data_col}"
                ds_name = st.text_input("데이터셋 이름", value=auto_name, key="cmp_add_name")

                existing_names = [d.get("name") for d in st.session_state.get("wm_datasets", [])]
                btn_disabled = ds_name in existing_names
                if btn_disabled:
                    st.warning(f"⚠️ '{ds_name}' 이름이 이미 존재합니다.")

                if st.button("✅ 추가", type="primary", key="cmp_add_btn",
                             disabled=btn_disabled, use_container_width=True):
                    try:
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
                        st.success(f"✅ '{ds_name}' 추가됨")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ 추가 실패: {exc}")

        # ── [탭 B] 수동 입력 (신규) ─────────────────────────────────────
        with add_manual_tab:
            st.caption(
                "X, Y 좌표와 측정값을 직접 입력하거나 "
                "스프레드시트에서 복사(Ctrl+V)해 붙여넣으세요."
            )

            # 수동 입력 이름
            manual_name = st.text_input(
                "데이터셋 이름",
                value=f"수동입력_{len(st.session_state.get('wm_datasets', [])) + 1}",
                key="cmp_manual_name",
            )

            # 빈 테이블 (session_state로 유지)
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
                    "x":    st.column_config.NumberColumn("X (mm)",  format="%.2f"),
                    "y":    st.column_config.NumberColumn("Y (mm)",  format="%.2f"),
                    "data": st.column_config.NumberColumn("측정값",  format="%.4f"),
                },
                use_container_width=True,
                hide_index=False,
            )
            st.session_state[_CMP_MANUAL_KEY] = edited_manual

            # 유효 데이터 추출
            df_valid = edited_manual.dropna(subset=["x", "y", "data"]).copy()
            for col in ["x", "y", "data"]:
                df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
            df_valid = df_valid.dropna().reset_index(drop=True)
            n_valid = len(df_valid)

            if n_valid > 0:
                st.success(f"✅ 유효 포인트: {n_valid}개")
            else:
                st.info("ℹ️ 데이터를 입력하면 추가할 수 있습니다.")

            # 중복 이름 체크
            existing_names = [d.get("name") for d in st.session_state.get("wm_datasets", [])]
            name_dup = manual_name in existing_names
            if name_dup:
                st.warning(f"⚠️ '{manual_name}' 이름이 이미 존재합니다.")

            btn_col, reset_col = st.columns([3, 1])
            with btn_col:
                if st.button(
                    "✅ 수동 데이터 추가", type="primary",
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
                    # 테이블 초기화 (추가 후 새 입력 준비)
                    st.session_state[_CMP_MANUAL_KEY] = pd.DataFrame({
                        "x": [None] * 10, "y": [None] * 10, "data": [None] * 10,
                    })
                    st.success(f"✅ '{manual_name}' 추가됨")
                    st.rerun()

            with reset_col:
                if st.button("🗑️", key="cmp_manual_reset", help="테이블 초기화"):
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
        "이름", value=st.session_state[title_key],
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
        # df_json이 이미 저장되어 있으면 사용, 아니면 파일에서 로드
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
        stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["값"])
        stats_df.index.name = "항목"
        st.dataframe(stats_df, use_container_width=True)

        # Raw Data (접힘)
        with st.expander("📋 Raw Data", expanded=False):
            df_disp = pd.read_json(df_json)
            st.dataframe(
                df_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "x":    st.column_config.NumberColumn("X (mm)",  format="%.2f"),
                    "y":    st.column_config.NumberColumn("Y (mm)",  format="%.2f"),
                    "data": st.column_config.NumberColumn("측정값",  format="%.3f")
                },
            )

    except Exception as e:
        st.error(f"❌ {e}")


# =============================================================================
# [6] 신규 탭 공통 가드 헬퍼
# =============================================================================

def _check_module_available(available: bool, module_name: str) -> bool:
    """모듈 import 실패 시 에러 메시지 표시."""
    if not available:
        st.error(
            f"⚠️ **{module_name} 모듈을 불러올 수 없습니다.**\n\n"
            f"`modules/` 폴더와 필요 패키지를 확인하세요."
        )
        return False
    return True


def _check_shared_data() -> bool:
    """shared_df_json이 None이면 로드 안내 메시지 표시."""
    if st.session_state.get("shared_df_json") is None:
        st.info(
            "ℹ️ **먼저 '📊 웨이퍼 맵' 탭에서 파일을 로드하거나 데이터를 입력해주세요.**"
        )
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
# [8] 메인 앱
# =============================================================================
st.title("🔬 웨이퍼 맵 분석기")
st.markdown("---")


# ── SIDEBAR: 데이터 관리 ────────────────────────────────────────────────────
st.sidebar.header("📁 데이터 관리")

for key, val in [("data_folder", "./wafer_data/"),
                 ("show_folder_browser", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

btn_col, path_col = st.sidebar.columns([1, 3])

with btn_col:
    if st.button("📂", help="폴더 선택", use_container_width=True):
        folder = try_native_folder_dialog()
        if folder:
            st.session_state.data_folder = folder
            st.rerun()
        else:
            st.session_state.show_folder_browser = True
            st.rerun()

with path_col:
    manual_input = st.text_input(
        "경로",
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

# ── 샘플 데이터 생성 버튼 (파일 없을 때 표시하되, 앱 중단 안 함) ───────────
if not file_names:
    st.sidebar.warning("폴더에 파일 없음")
    if st.sidebar.button("🎯 샘플 5개 생성", type="primary"):
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
        st.sidebar.success("✅ 샘플 생성!")
        st.rerun()

# ── 폴더 브라우저 ─────────────────────────────────────────────────────────
if st.session_state.show_folder_browser:
    render_folder_browser()
    st.stop()


# ── SIDEBAR: 파일 선택 (파일이 있을 때만) ──────────────────────────────────
# v3.0: 파일 없어도 수동 입력으로 사용 가능하므로 st.stop() 호출 안 함
has_files = bool(file_names)

if has_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 파일 선택")

    selected_file = st.sidebar.selectbox("분석 파일", file_names)
    full_path     = os.path.join(data_folder, selected_file)

    sheets = get_sheet_names(full_path)
    if sheets:
        selected_sheet = st.sidebar.selectbox("시트 선택", options=sheets, key="s_sheet")
    else:
        selected_sheet = None

    # 파일/시트 변경 감지
    current_key = (selected_file, selected_sheet)
    if st.session_state.get("_s_file") != current_key:
        st.session_state._s_file    = current_key
        st.session_state._s_display = None
        st.session_state._s_col_map = None
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
st.sidebar.header("⚙️ 시각화 설정")

COLORSCALES = {
    "Rainbow":       "Rainbow",
    "RdYlBu (반전)": "RdYlBu_r",
    "Jet":           "Jet",
    "Hot":           "Hot",
    "Viridis":       "Viridis",
    "Plasma":        "Plasma",
    "Spectral":      "Spectral_r"
}
cs_name    = st.sidebar.selectbox("컬러 스케일", list(COLORSCALES.keys()), index=0)
colorscale = COLORSCALES[cs_name]

resolution  = st.sidebar.slider("해상도", 30, 200, 100, 10)
show_points = st.sidebar.checkbox("데이터 포인트 표시", value=False)
n_contours  = st.sidebar.slider("Contour 단계 수", 5, 40, 20)
line_angle  = st.sidebar.slider("Line Scan 각도 (°)", 0, 175, 0, 5)


# =============================================================================
# [9] 데이터 소스 결정 (파일 or 수동 입력)
# =============================================================================
# v3.0: 데이터 소스 = "file" (사이드바에서 파일 선택) 또는 "manual" (수동 입력)
# 파일이 있으면 기본적으로 파일 모드, 웨이퍼 맵 탭 내부에서 수동 입력 전환 가능

# 데이터 소스 모드 초기화
if "data_source_mode" not in st.session_state:
    st.session_state.data_source_mode = "file" if has_files else "manual"


# =============================================================================
# [10] 탭 구성 — 5개 탭 (비교 모드는 웨이퍼 맵 탭 서브탭으로 통합)
# =============================================================================

tab_labels = [
    "📊 웨이퍼 맵",
    "🔍 결함 오버레이" + ("" if DEFECT_AVAILABLE else " ⚠️"),
    "⚗️ GPC 분석"      + ("" if GPC_AVAILABLE    else " ⚠️"),
    "📄 보고서 생성"   + ("" if REPORT_AVAILABLE else " ⚠️"),
    "🤖 ML 이상 탐지"  + ("" if ML_AVAILABLE     else " ⚠️"),
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
    # ── 서브탭: 단일 분석 | 비교 분석 ──────────────────────────────────────
    sub_single, sub_compare = st.tabs(["🔬 단일 분석", "🔀 비교 분석"])

    # =====================================================================
    # [서브탭 A] 단일 분석
    # =====================================================================
    with sub_single:
        # ── 데이터 소스 선택 (파일 / 수동 입력) ────────────────────────────
        source_options = []
        if has_files:
            source_options.append("📁 파일 데이터")
        source_options.append("✏️ 수동 입력")

        if len(source_options) > 1:
            data_source = st.radio(
                "데이터 소스",
                options=source_options,
                horizontal=True,
                key="wm_data_source_radio",
            )
        else:
            data_source = source_options[0]

        use_manual = (data_source == "✏️ 수동 입력")

        if use_manual:
            # ── 수동 입력 모드 ──────────────────────────────────────────────
            st.markdown("##### ✏️ 수동 데이터 입력")
            st.caption(
                "아래 테이블에 X, Y 좌표와 측정값을 직접 입력하거나, "
                "스프레드시트에서 복사(Ctrl+V)해 붙여넣으세요."
            )

            # session_state에 수동 입력 데이터 유지
            if "manual_df" not in st.session_state:
                st.session_state.manual_df = _create_empty_wafer_df()

            edited_manual = st.data_editor(
                st.session_state.manual_df,
                num_rows="dynamic",
                key="manual_editor",
                column_config={
                    "x":    st.column_config.NumberColumn("X (mm)",  format="%.2f"),
                    "y":    st.column_config.NumberColumn("Y (mm)",  format="%.2f"),
                    "data": st.column_config.NumberColumn("측정값",  format="%.4f"),
                },
                use_container_width=True,
                hide_index=False,
            )
            st.session_state.manual_df = edited_manual

            # 유효 데이터 추출 (NaN 행 제거)
            df_valid = (
                edited_manual
                .dropna(subset=["x", "y", "data"])
                .reset_index(drop=True)
            )
            # 숫자형 변환
            for col in ["x", "y", "data"]:
                df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
            df_valid = df_valid.dropna().reset_index(drop=True)

            n_valid = len(df_valid)

            reset_col, info_col = st.columns([1, 3])
            with reset_col:
                if st.button("🗑️ 초기화", key="manual_reset"):
                    st.session_state.manual_df = _create_empty_wafer_df()
                    st.rerun()
            with info_col:
                if n_valid >= 3:
                    st.success(f"✅ 유효 포인트: {n_valid}개")
                elif n_valid > 0:
                    st.warning(f"⚠️ 유효 포인트 {n_valid}개 — 최소 3개 필요")
                else:
                    st.info("ℹ️ 테이블에 데이터를 입력하면 웨이퍼 맵이 생성됩니다.")

            if n_valid < 3:
                # 데이터 부족 — 차트 렌더링 안 함
                # 공유 데이터도 None으로 설정
                st.session_state["shared_df_json"] = None
                # 수동 입력 시 다른 탭에서 사용할 원본 정보도 저장
                st.session_state["shared_filename"] = "수동 입력"
                st.session_state["shared_all_cols"] = ["x", "y", "data"]
            else:
                # ── 유효 데이터로 차트 생성 ──────────────────────────────────
                df_display = df_valid.copy()
                df_json    = df_display.to_json()
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
                st.session_state["shared_filename"]     = "수동 입력"
                st.session_state["shared_raw_df_json"]  = df_json
                st.session_state["shared_df_raw_original"] = df_display

                # 제목 배너
                st.markdown(
                    "<div style='text-align:center;padding:7px;background:#1a6bbf;"
                    "color:white;border-radius:7px;font-size:14px;font-weight:bold;"
                    "margin-bottom:6px;'>📊 수동 입력 데이터</div>",
                    unsafe_allow_html=True
                )

                # ── 차트 레이아웃 (기존과 동일) ──────────────────────────────
                c1, c2, c3 = st.columns([5, 5, 2])
                with c1:
                    st.markdown("#### 2D Wafer Map")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                with c2:
                    st.markdown("#### Contour Map")
                    st.plotly_chart(fig_contour, use_container_width=True)
                with c3:
                    st.markdown("### 📊 Statistics")
                    for k, v in stats.items():
                        st.markdown(f"**{k}**")
                        st.code(str(v), language=None)

                c4, c5 = st.columns([2, 3])
                with c4:
                    st.plotly_chart(fig_linescan, use_container_width=True)
                with c5:
                    st.plotly_chart(fig_3d, use_container_width=True)

                st.download_button(
                    "📥 CSV 다운로드",
                    df_display.to_csv(index=False),
                    "manual_wafer_export.csv",
                    "text/csv",
                    key="manual_csv_dl",
                )

        else:
            # ── 파일 데이터 모드 ──────────────────────────────────────────────
            if not has_files:
                st.info("ℹ️ 데이터 폴더에 파일이 없습니다. 사이드바에서 샘플 데이터를 생성하거나 '✏️ 수동 입력'을 선택하세요.")
            else:
                try:
                    df_raw = load_file_cached(full_path, selected_sheet)
                except Exception as e:
                    st.error(f"❌ 파일 로드 실패: {e}")
                    st.stop()

                all_cols = df_raw.columns.tolist()

                # ── 컬럼 매핑 (사이드바에서 이미 설정된 값 또는 탭 내부에서 선택) ──
                st.sidebar.markdown("---")
                st.sidebar.subheader("🔗 컬럼 매핑")

                def_x = _default_col_index(all_cols, "x",    0)
                def_y = _default_col_index(all_cols, "y",    1)
                def_d = _default_col_index(all_cols, "data", 2)

                x_col    = st.sidebar.selectbox("X 컬럼",    all_cols, index=def_x, key="s_xcol")
                y_col    = st.sidebar.selectbox("Y 컬럼",    all_cols, index=def_y, key="s_ycol")
                data_col = st.sidebar.selectbox("Data 컬럼", all_cols, index=def_d, key="s_dcol")

                col_key = f"{x_col}|{y_col}|{data_col}"
                if st.session_state.get("_s_col_map") != col_key or \
                   st.session_state.get("_s_display") is None:
                    st.session_state._s_col_map = col_key
                    st.session_state._s_display = apply_col_mapping(df_raw, x_col, y_col, data_col)

                df_display = st.session_state._s_display
                df_json    = df_display.to_json()
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
                    st.markdown("#### 2D Wafer Map")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                with c2:
                    st.markdown("#### Contour Map")
                    st.plotly_chart(fig_contour, use_container_width=True)
                with c3:
                    st.markdown("### 📊 Statistics")
                    for k, v in stats.items():
                        st.markdown(f"**{k}**")
                        st.code(str(v), language=None)

                c4, c5 = st.columns([2, 3])
                with c4:
                    st.plotly_chart(fig_linescan, use_container_width=True)
                with c5:
                    st.plotly_chart(fig_3d, use_container_width=True)

                # ── Raw Data 편집 ────────────────────────────────────────
                st.markdown("---")
                st.subheader("📋 Raw Data (셀 편집 즉시 반영)")
                edited_df = st.data_editor(
                    df_display,
                    num_rows="dynamic",
                    key=f"single_editor_{selected_file}_{col_key}",
                    column_config={
                        "x":    st.column_config.NumberColumn("X (mm)",  format="%.2f"),
                        "y":    st.column_config.NumberColumn("Y (mm)",  format="%.2f"),
                        "data": st.column_config.NumberColumn("측정값",  format="%.3f")
                    },
                    use_container_width=True,
                    hide_index=False
                )
                st.session_state._s_display = edited_df

                st.download_button(
                    "📥 CSV 다운로드",
                    edited_df.to_csv(index=False),
                    "wafer_export.csv",
                    "text/csv"
                )

    # =====================================================================
    # [서브탭 B] 비교 분석
    # =====================================================================
    with sub_compare:
        st.markdown("#### 🔀 다중 웨이퍼 비교 분석")
        st.caption(
            "여러 웨이퍼를 나란히 비교합니다. "
            "같은 파일의 다른 Data 컬럼을 추가하면 다중 파라미터 비교도 가능합니다."
        )

        # 데이터셋 목록 초기화
        if "wm_datasets" not in st.session_state:
            st.session_state.wm_datasets = []

        # ── 설정 패널 ────────────────────────────────────────────────────
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            cols_per_row = st.radio(
                "행당 맵 개수", [2, 3, 4, 5, 6], index=1,
                horizontal=True,
                key="cmp_cols_per_row"
            )
        with ctrl_col2:
            lock_scale = st.checkbox(
                "🔒 컬러 스케일 통일", value=False,
                key="cmp_lock_scale"
            )

        st.markdown("---")

        # ── 데이터셋 관리 + 추가 ──────────────────────────────────────────
        _render_compare_dataset_manager()
        _render_compare_dataset_adder(file_names, data_folder)

        datasets = st.session_state.get("wm_datasets", [])

        if len(datasets) < 2:
            st.info(
                "ℹ️ 데이터셋을 **2개 이상** 추가하면 비교 분석이 시작됩니다.\n\n"
                "**💡 팁:** 같은 파일에서 Data 컬럼만 다르게 추가하면 "
                "다중 파라미터 비교가 가능합니다."
            )
        else:
            st.markdown("---")
            n_sel = len(datasets)
            st.markdown(f"**비교 중: {n_sel}개 데이터셋**")

            # ── 컬러 스케일 통일 범위 계산 ──────────────────────────────────
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
                    st.info(f"🔒 스케일 고정: {global_zmin:.2f} ~ {global_zmax:.2f}")

            # ── 카드 렌더링 ──────────────────────────────────────────────────
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
        # ── 앱 제공 초기 데이터셋 구성 ──────────────────────────────────────
        app_initial_datasets = []

        # 현재 단일 모드 웨이퍼
        current_df_json  = st.session_state.get("shared_df_json")
        current_filename = st.session_state.get("shared_filename", "")
        if current_df_json is not None and current_filename:
            app_initial_datasets.append({
                "name":    os.path.splitext(current_filename)[0] if current_filename != "수동 입력" else "수동 입력",
                "df_json": current_df_json,
            })

        # 비교 서브탭에서 추가된 데이터셋
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