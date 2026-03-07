# modules/__init__.py
# =============================================================================
# modules 패키지 진입점
#
# [v3.0 변경]
# - multi_param 모듈 제거 (비교 서브탭에서 대체)
# - 나머지 모듈: defect_overlay, gpc, report, ml_anomaly 유지
#
# [session_state 키 prefix 충돌 방지 규칙]
#   defect_overlay.py → "def_"
#   gpc.py          → "gpc_"
#   report.py       → "rep_"
#   ml_anomaly.py   → "ml_"
# =============================================================================

MODULES_STATUS: dict[str, bool] = {}

# ── [1] defect_overlay.py ─────────────────────────────────────────────────────
try:
    from modules.defect_overlay import render_defect_tab
    MODULES_STATUS["defect_overlay"] = True
except ImportError as e:
    MODULES_STATUS["defect_overlay"] = False
    def render_defect_tab(wafer_df_json, wafer_radius, resolution, colorscale, data_folder):
        import streamlit as st
        st.warning(f"⚠️ 결함 오버레이 모듈 로드 실패: {e}")

# ── [2] gpc.py ────────────────────────────────────────────────────────────────
try:
    from modules.gpc import render_gpc_tab
    MODULES_STATUS["gpc"] = True
except ImportError as e:
    MODULES_STATUS["gpc"] = False
    def render_gpc_tab(df_json, allcols, resolution, colorscale):
        import streamlit as st
        st.warning(f"⚠️ GPC 분석 모듈 로드 실패: {e}")

# ── [3] report.py ─────────────────────────────────────────────────────────────
try:
    from modules.report import render_report_tab
    MODULES_STATUS["report"] = True
except ImportError as e:
    MODULES_STATUS["report"] = False
    def render_report_tab(filename, stats, df_display,
                          fig_heatmap, fig_contour, fig_linescan, fig_3d,
                          gpc_data=None):
        import streamlit as st
        st.warning(f"⚠️ 보고서 모듈 로드 실패: {e}")

# ── [4] ml_anomaly.py ────────────────────────────────────────────────────────
try:
    from modules.ml_anomaly import render_anomaly_tab
    MODULES_STATUS["ml_anomaly"] = True
except ImportError as e:
    MODULES_STATUS["ml_anomaly"] = False
    def render_anomaly_tab(datasets, resolution, data_folder):
        import streamlit as st
        st.warning(f"⚠️ ML 이상 탐지 모듈 로드 실패: {e}")


__all__ = [
    "render_defect_tab",
    "render_gpc_tab",
    "render_report_tab",
    "render_anomaly_tab",
    "MODULES_STATUS",
]