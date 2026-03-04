# app.py
# 웨이퍼 맵 분석기 — Streamlit Community Cloud 배포용
# 실행: streamlit run app.py
#
# [Cloud 배포 주요 변경점]
# ① 폴더 브라우저(tkinter/subprocess) 제거 → st.file_uploader 사용
# ② core.py에 공유 함수 분리 → 순환 import 해결
# ③ folder_picker_helper.py 제거 (Cloud에서 tkinter 사용 불가)
# ④ 샘플 데이터 생성: tempfile 사용 → BytesIO로 메모리에서 처리
# ⑤ 비교 모드: 로컬 파일 → 업로드 파일 기반으로 변경
# =============================================================================

# ── 표준 라이브러리 ────────────────────────────────────────────────────────────
import io
import os
import time

# ── 외부 라이브러리 ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── 핵심 함수 (core.py에서 import) ─────────────────────────────────────────────
from core import (
    get_wafer_grid,
    add_wafer_outline,
    _wafer_layout,
    create_2d_heatmap,
    create_contour_map,
    create_3d_surface,
    create_line_scan,
    calculate_stats,
    apply_col_mapping,
    _default_col_index,
    load_file_cached,
    get_sheet_names,
)

# =============================================================================
# 신규 모듈 import (graceful degradation)
# =============================================================================
try:
    from modules.multi_param import render_multi_param_tab
    MULTI_PARAM_AVAILABLE = True
except ImportError:
    MULTI_PARAM_AVAILABLE = False

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
# [2] 파일 업로드 헬퍼 (Cloud 전용 — 로컬 폴더 브라우저 대체)
# =============================================================================

def _save_uploaded_file(uploaded_file) -> str:
    """업로드된 파일을 임시 디렉토리에 저장하고 경로 반환."""
    import tempfile
    tmp_dir = os.path.join(tempfile.gettempdir(), "wafer_data")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _generate_sample_data() -> dict[str, pd.DataFrame]:
    """샘플 웨이퍼 데이터 5개 생성 (메모리 내)."""
    samples = {}
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
        samples[f"wafer_{idx:02d}.csv"] = pd.DataFrame({
            "x": x_pts, "y": y_pts, "data": vals
        })
    return samples


# =============================================================================
# [3] UI 헬퍼 함수
# =============================================================================

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
# [4] 데이터셋 관리 함수 (비교 모드용)
# =============================================================================

def dataset_id() -> str:
    """데이터셋 고유 ID 생성."""
    return f"ds_{time.time_ns()}"


def render_dataset_manager() -> None:
    """사이드바: 데이터셋 목록 + 순서 변경(▲/▼) + 삭제(✕)."""
    datasets = st.session_state.get("datasets", [])
    if not datasets:
        st.sidebar.info("데이터셋 없음. 아래에서 추가하세요.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 데이터셋 목록**")

    for i, ds in enumerate(datasets):
        ds_id = ds["id"]
        c_name, c_up, c_dn, c_del = st.sidebar.columns([5, 1, 1, 1])

        label = ds["name"] if len(ds["name"]) <= 16 else ds["name"][:14] + "…"
        c_name.markdown(
            f"<div style='padding-top:5px;font-size:11px;'>"
            f"<b>{i+1}.</b> {label}</div>",
            unsafe_allow_html=True
        )

        if c_up.button("▲", key=f"dsup_{ds_id}", disabled=(i == 0)):
            datasets[i], datasets[i-1] = datasets[i-1], datasets[i]
            st.session_state.datasets = datasets
            st.rerun()

        if c_dn.button("▼", key=f"dsdn_{ds_id}", disabled=(i == len(datasets)-1)):
            datasets[i], datasets[i+1] = datasets[i+1], datasets[i]
            st.session_state.datasets = datasets
            st.rerun()

        if c_del.button("✕", key=f"dsdel_{ds_id}"):
            st.session_state.datasets.pop(i)
            st.rerun()


def render_dataset_creator_cloud() -> None:
    """사이드바: 업로드 파일 → X/Y/Data 컬럼 선택 → 데이터셋 추가 (Cloud용)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ 데이터셋 추가")

    uploaded = st.sidebar.file_uploader(
        "파일 업로드",
        type=["csv", "xlsx", "xls"],
        key="dc_file_uploader",
    )

    if uploaded is None:
        st.sidebar.info("CSV 또는 Excel 파일을 업로드하세요.")
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df_preview = pd.read_csv(uploaded)
            sel_sheet = None
        else:
            xf = pd.ExcelFile(uploaded)
            sheets = xf.sheet_names
            sel_sheet = st.sidebar.selectbox("시트", sheets, key="dc_sheet")
            df_preview = pd.read_excel(uploaded, sheet_name=sel_sheet)
        all_cols = df_preview.columns.tolist()
    except Exception:
        st.sidebar.error("❌ 파일 읽기 실패")
        return

    def_x = _default_col_index(all_cols, "x",    0)
    def_y = _default_col_index(all_cols, "y",    1)
    def_d = _default_col_index(all_cols, "data", 2)

    x_col    = st.sidebar.selectbox("X 컬럼",    all_cols, index=def_x, key="dc_x")
    y_col    = st.sidebar.selectbox("Y 컬럼",    all_cols, index=def_y, key="dc_y")
    data_col = st.sidebar.selectbox("Data 컬럼", all_cols, index=def_d, key="dc_d")

    sheet_tag = f"[{sel_sheet}]" if sel_sheet else ""
    auto_name = f"{os.path.splitext(uploaded.name)[0]}{sheet_tag} · {data_col}"
    ds_name   = st.sidebar.text_input("데이터셋 이름", value=auto_name, key="dc_name")

    if st.sidebar.button("✅ 데이터셋 추가", type="primary",
                          use_container_width=True, key="dc_add"):
        if "datasets" not in st.session_state:
            st.session_state.datasets = []

        df_mapped = apply_col_mapping(df_preview, x_col, y_col, data_col)
        new_ds = {
            "id":       dataset_id(),
            "name":     ds_name,
            "df_json":  df_mapped.to_json(),
            "x_col":    x_col,
            "y_col":    y_col,
            "data_col": data_col,
        }
        st.session_state.datasets.append(new_ds)
        st.rerun()


def render_compare_card_cloud(ds: dict, resolution: int, colorscale: str,
                              n_contours: int, show_points: bool,
                              global_zmin, global_zmax) -> None:
    """데이터셋 1개를 비교 카드 1개로 렌더링 (Cloud용 — df_json 직접 사용)."""
    ds_id = ds["id"]

    title_key = f"title_{ds_id}"
    if title_key not in st.session_state:
        st.session_state[title_key] = ds["name"]

    new_title = st.text_input(
        "이름",
        value=st.session_state[title_key],
        key=f"input_{ds_id}",
        label_visibility="collapsed"
    )
    st.session_state[title_key] = new_title
    st.markdown(
        f"<div style='text-align:center;padding:7px;background:#1a6bbf;"
        f"color:white;border-radius:7px;font-size:13px;font-weight:bold;"
        f"margin-bottom:6px;'>📊 {new_title}</div>",
        unsafe_allow_html=True
    )

    try:
        df_json = ds["df_json"]

        fig = create_contour_map(
            df_json, resolution, colorscale, n_contours, show_points,
            compact=True, zmin=global_zmin, zmax=global_zmax
        )
        st.plotly_chart(fig, use_container_width=True)

        stats    = calculate_stats(df_json)
        stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["값"])
        stats_df.index.name = "항목"
        st.dataframe(stats_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ {e}")


# =============================================================================
# [5] 탭 공통 가드 헬퍼
# =============================================================================

def _check_module_available(available: bool, module_name: str) -> bool:
    """모듈 import 실패 시 에러 메시지를 표시하고 False 반환."""
    if not available:
        st.error(
            f"⚠️ **{module_name} 모듈을 불러올 수 없습니다.**\n\n"
            f"`modules/{module_name.lower().replace(' ', '_')}.py` 파일이 "
            f"`modules/` 폴더에 있는지 확인하세요.\n\n"
            f"필요 패키지가 설치되어 있는지도 확인하세요:\n"
            f"```\npip install scikit-learn openpyxl kaleido\n```"
        )
        return False
    return True


def _check_shared_data() -> bool:
    """shared_df_json이 None이면 로드 안내 메시지."""
    if st.session_state.get("shared_df_json") is None:
        st.info(
            "ℹ️ **먼저 '📊 웨이퍼 맵' 탭에서 파일을 로드해주세요.**\n\n"
            "웨이퍼 맵 탭을 방문하면 데이터가 자동으로 이 탭에도 공유됩니다."
        )
        return False
    return True


# =============================================================================
# [6] 메인 앱
# =============================================================================
st.title("🔬 Wafer")
st.markdown("---")

# ── SIDEBAR: 데이터 관리 ────────────────────────────────────────────────────
st.sidebar.header("📁 데이터 관리")

# ── 파일 업로드 (Cloud 전용) ──────────────────────────────────────────────────
uploaded_files = st.sidebar.file_uploader(
    "웨이퍼 데이터 파일",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="main_uploader",
    help="CSV 또는 Excel 파일을 1개 이상 업로드하세요.",
)

# ── 업로드 파일을 session_state에 저장 ────────────────────────────────────────
# uploaded_dfs: {파일명: DataFrame} — 사이드바 파일 선택용 (첫 시트)
# uploaded_raw: {파일명: bytes} — Excel 멀티시트 접근용 원본 바이트
# uploaded_sheets: {파일명: [시트명, ...]} — Excel 시트 목록
if "uploaded_dfs" not in st.session_state:
    st.session_state.uploaded_dfs = {}
if "uploaded_raw" not in st.session_state:
    st.session_state.uploaded_raw = {}
if "uploaded_sheets" not in st.session_state:
    st.session_state.uploaded_sheets = {}

for uf in uploaded_files:
    if uf.name not in st.session_state.uploaded_dfs:
        try:
            raw_bytes = uf.getvalue()
            if uf.name.lower().endswith(".csv"):
                st.session_state.uploaded_dfs[uf.name] = pd.read_csv(uf)
                st.session_state.uploaded_sheets[uf.name] = []
            else:
                # Excel: 원본 바이트 저장 → 나중에 시트별 접근 가능
                st.session_state.uploaded_raw[uf.name] = raw_bytes
                xf = pd.ExcelFile(io.BytesIO(raw_bytes))
                st.session_state.uploaded_sheets[uf.name] = xf.sheet_names
                # 기본 표시용으로 첫 시트 로드
                st.session_state.uploaded_dfs[uf.name] = pd.read_excel(
                    io.BytesIO(raw_bytes), sheet_name=0
                )
        except Exception:
            st.sidebar.error(f"❌ {uf.name} 읽기 실패")

file_names = list(st.session_state.uploaded_dfs.keys())

# ── 샘플 데이터 생성 ──────────────────────────────────────────────────────────
if not file_names:
    st.sidebar.warning("⚠️ 업로드된 파일 없음")
    if st.sidebar.button("🎯 샘플 5개 생성", type="primary"):
        samples = _generate_sample_data()
        st.session_state.uploaded_dfs.update(samples)
        st.rerun()

if not file_names and not st.session_state.uploaded_dfs:
    st.info(
        "👋 왼쪽 사이드바에서 웨이퍼 데이터 파일(CSV/Excel)을 업로드하거나 "
        "**🎯 샘플 5개 생성** 버튼을 클릭하세요."
    )
    st.stop()

# 샘플 생성 후 file_names 갱신
file_names = list(st.session_state.uploaded_dfs.keys())
if not file_names:
    st.stop()


# ── SIDEBAR: 분석 모드 ───────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔀 분석 모드")

compare_mode = st.sidebar.toggle("비교 모드 활성화", value=False)

if compare_mode:
    render_dataset_manager()
    render_dataset_creator_cloud()
    st.sidebar.markdown("---")
    cols_per_row = st.sidebar.radio(
        "행당 맵 개수", [2, 3, 4], index=1,
        horizontal=True, key="cpr_compare"
    )
    lock_scale = st.sidebar.checkbox(
        "🔒 컬러 스케일 통일", value=False, key="lock_scale_compare"
    )
else:
    selected_file = st.sidebar.selectbox("분석 파일 선택", file_names)

    # Excel 시트 처리 — 멀티시트 지원
    sheets = st.session_state.get("uploaded_sheets", {}).get(selected_file, [])
    selected_sheet = None
    if len(sheets) > 1:
        selected_sheet = st.sidebar.selectbox(
            "시트 선택", sheets, key="s_sheet_sel"
        )
        raw_bytes = st.session_state.uploaded_raw.get(selected_file)
        if raw_bytes:
            df_raw = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=selected_sheet)
        else:
            df_raw = st.session_state.uploaded_dfs[selected_file]
    else:
        df_raw = st.session_state.uploaded_dfs[selected_file]

    current_key = (selected_file, selected_sheet)
    if st.session_state.get("_s_file") != current_key:
        st.session_state._s_file    = current_key
        st.session_state._s_display = None
        st.session_state._s_col_map = None
        for k in ["shared_df_json", "shared_stats", "shared_fig_heatmap",
                   "shared_fig_contour", "shared_fig_linescan", "shared_fig_3d",
                   "shared_df_raw", "shared_all_cols", "shared_wafer_radius",
                   "shared_filename"]:
            st.session_state[k] = None


# ── SIDEBAR: 시각화 설정 ──────────────────────────────────────────────────────
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
if not compare_mode:
    line_angle = st.sidebar.slider("Line Scan 각도 (°)", 0, 175, 0, 5)


# =============================================================================
# [단일 분석 모드]
# =============================================================================
if not compare_mode:
    all_cols = df_raw.columns.tolist()

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
       st.session_state._s_display is None:
        st.session_state._s_col_map = col_key
        st.session_state._s_display = apply_col_mapping(df_raw, x_col, y_col, data_col)

    df_display = st.session_state._s_display
    df_json    = df_display.to_json()
    stats      = calculate_stats(df_json)

    # ── 차트 사전 계산 ─────────────────────────────────────────────────────
    fig_heatmap  = create_2d_heatmap(df_json, resolution, colorscale, show_points)
    fig_contour  = create_contour_map(df_json, resolution, colorscale,
                                      n_contours, show_points)
    fig_linescan = create_line_scan(df_json, line_angle, resolution)
    fig_3d       = create_3d_surface(df_json, resolution, colorscale)

    _, _, _, wafer_radius = get_wafer_grid(df_json, resolution)

    # ── 탭 간 공유 데이터 저장 ───────────────────────────────────────────────
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

    # ── 제목 배너 ─────────────────────────────────────────────────────────────
    wafer_title_banner(selected_file, prefix="single_")

    # ── 탭 레이블 ─────────────────────────────────────────────────────────────
    tab_labels = [
        "📊 웨이퍼 맵",
        "📐 다중 파라미터" + ("" if MULTI_PARAM_AVAILABLE else " ⚠️"),
        "🔍 결함 오버레이" + ("" if DEFECT_AVAILABLE      else " ⚠️"),
        "⚗️ GPC 분석"      + ("" if GPC_AVAILABLE         else " ⚠️"),
        "📄 보고서 생성"   + ("" if REPORT_AVAILABLE      else " ⚠️"),
        "🤖 ML 이상 탐지"  + ("" if ML_AVAILABLE          else " ⚠️"),
    ]

    (tab_wafer, tab_multi, tab_defect,
     tab_gpc, tab_report, tab_ml) = st.tabs(tab_labels)

    # =========================================================================
    # tab_wafer: 기존 단일 모드 시각화
    # =========================================================================
    with tab_wafer:
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

    # =========================================================================
    # tab_multi: 다중 파라미터 서브플롯
    # =========================================================================
    with tab_multi:
        if not _check_module_available(MULTI_PARAM_AVAILABLE, "multi_param"):
            pass
        elif not _check_shared_data():
            pass
        else:
            render_multi_param_tab(
                df_json=st.session_state["shared_raw_df_json"],
                all_cols=st.session_state["shared_all_cols"],
                resolution=resolution,
                colorscale=colorscale,
            )

    # =========================================================================
    # tab_defect: 결함 오버레이
    # =========================================================================
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
                data_folder="",  # Cloud에서는 폴더 미사용
            )

    # =========================================================================
    # tab_gpc: GPC 분석
    # =========================================================================
    with tab_gpc:
        if not _check_module_available(GPC_AVAILABLE, "gpc"):
            pass
        elif not _check_shared_data():
            pass
        else:
            render_gpc_tab(
                df_raw=st.session_state["shared_df_raw_original"],
                all_cols=st.session_state["shared_all_cols"],
                resolution=resolution,
                colorscale=colorscale,
            )

    # =========================================================================
    # tab_report: Excel 보고서 생성
    # =========================================================================
    with tab_report:
        if not _check_module_available(REPORT_AVAILABLE, "report"):
            pass
        elif not _check_shared_data():
            pass
        else:
            gpc_data = st.session_state.get("gpc_result", None)
            render_report_tab(
                filename=st.session_state["shared_filename"],
                stats=st.session_state["shared_stats"],
                df_display=st.session_state["shared_df_raw"],
                fig_heatmap=st.session_state["shared_fig_heatmap"],
                fig_contour=st.session_state["shared_fig_contour"],
                fig_linescan=st.session_state["shared_fig_linescan"],
                fig_3d=st.session_state["shared_fig_3d"],
                gpc_data=gpc_data,
            )

    # =========================================================================
    # tab_ml: ML 이상 탐지
    # =========================================================================
    with tab_ml:
        if not _check_module_available(ML_AVAILABLE, "ml_anomaly"):
            pass
        else:
            app_initial_datasets = []

            current_df_json  = st.session_state.get("shared_df_json")
            current_filename = st.session_state.get("shared_filename", "")
            if current_df_json is not None and current_filename:
                app_initial_datasets.append({
                    "name":    os.path.splitext(current_filename)[0],
                    "df_json": current_df_json,
                })

            for ds in st.session_state.get("datasets", []):
                if any(m["name"] == ds["name"] for m in app_initial_datasets):
                    continue
                if "df_json" in ds:
                    app_initial_datasets.append({
                        "name":    ds["name"],
                        "df_json": ds["df_json"],
                    })

            render_anomaly_tab(
                datasets=app_initial_datasets,
                resolution=resolution,
                data_folder="",
            )


# =============================================================================
# [비교 분석 모드]
# =============================================================================
else:
    datasets = st.session_state.get("datasets", [])

    if len(datasets) < 2:
        st.warning("⚠️ 사이드바 하단에서 데이터셋을 2개 이상 추가하세요.")
        with st.expander("💡 데이터셋 추가 방법", expanded=True):
            st.markdown("""
            1. 사이드바 **➕ 데이터셋 추가** 에서 파일 업로드
            2. X / Y / Data 컬럼 지정
            3. 이름 설정 후 **✅ 데이터셋 추가** 클릭
            4. **같은 파일에서 여러 번 추가** 가능 (Data 컬럼만 다르게)
            """)
        st.stop()

    n_sel = len(datasets)
    st.subheader(f"🔀 데이터셋 비교 — {n_sel}개")

    # ── 컬러 스케일 통일 ──────────────────────────────────────────────────────
    global_zmin, global_zmax = None, None
    if lock_scale:
        series_list = []
        for ds in datasets:
            try:
                df_tmp = pd.read_json(ds["df_json"])
                series_list.append(df_tmp["data"].dropna())
            except Exception:
                pass

        if series_list:
            all_vals = pd.concat(series_list, ignore_index=True)
            global_zmin = float(all_vals.min())
            global_zmax = float(all_vals.max())
            st.info(f"🔒 스케일 고정: {global_zmin:.2f} ~ {global_zmax:.2f}")

    # ── cols_per_row씩 묶어서 행 단위 렌더링 ─────────────────────────────────
    for batch_start in range(0, n_sel, cols_per_row):
        batch = datasets[batch_start : batch_start + cols_per_row]
        cols  = st.columns(len(batch))
        for col, ds in zip(cols, batch):
            with col:
                render_compare_card_cloud(
                    ds, resolution, colorscale,
                    n_contours, show_points, global_zmin, global_zmax
                )
        if batch_start + cols_per_row < n_sel:
            st.markdown(
                "<hr style='border:1px solid #ddd;'>",
                unsafe_allow_html=True
            )