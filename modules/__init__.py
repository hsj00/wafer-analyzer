# modules/__init__.py
# =============================================================================
# modules 패키지 진입점
#
# [역할]
# 1. Python이 modules/ 디렉터리를 패키지로 인식하게 함
# 2. app.py에서 각 모듈을 개별 import하는 대신,
#    "from modules import render_multi_param_tab" 형태로 단일 진입점 제공
# 3. 각 모듈 import 실패 시 graceful fallback 처리
#    → 특정 모듈만 파일이 없거나 의존성 문제가 있어도 앱 전체가 죽지 않음
#
# [사용 패턴]
# app.py에서:
#   from modules import (
#       render_multi_param_tab,
#       render_defect_tab,
#       render_gpc_tab,
#       render_report_tab,
#       render_anomaly_tab,
#       MODULES_STATUS,  # 각 모듈 로드 성공/실패 여부 확인용
#   )
#
# [session_state 키 prefix 충돌 방지 규칙]
# 각 모듈은 고유 prefix를 사용하여 키 충돌을 방지합니다:
#   multi_param.py  → "mp_"
#   defect_overlay.py → "def_"
#   gpc.py          → "gpc_"
#   report.py       → "rep_"
#   ml_anomaly.py   → "ml_"
# =============================================================================

# 각 모듈의 로드 성공/실패 여부를 추적
# app.py에서 MODULES_STATUS를 확인해 탭 비활성화 등 처리 가능
MODULES_STATUS: dict[str, bool] = {}

# ── [1] multi_param.py ────────────────────────────────────────────────────────
# render_multi_param_tab(df_json: str, allcols: list, resolution: int, colorscale: str)
try:
    from modules.multi_param import render_multi_param_tab
    MODULES_STATUS["multi_param"] = True
except ImportError as e:
    MODULES_STATUS["multi_param"] = False
    # fallback: 호출 시 에러 대신 경고 메시지를 출력하는 더미 함수
    def render_multi_param_tab(df_json, allcols, resolution, colorscale):
        import streamlit as st
        st.warning(f"⚠️ 다중 파라미터 모듈 로드 실패: {e}")

# ── [2] defect_overlay.py ─────────────────────────────────────────────────────
# render_defect_tab(
#     wafer_df_json: str, wafer_radius: float,
#     resolution: int, colorscale: str, data_folder: str
# )
try:
    from modules.defect_overlay import render_defect_tab
    MODULES_STATUS["defect_overlay"] = True
except ImportError as e:
    MODULES_STATUS["defect_overlay"] = False
    def render_defect_tab(wafer_df_json, wafer_radius, resolution, colorscale, data_folder):
        import streamlit as st
        st.warning(f"⚠️ 결함 오버레이 모듈 로드 실패: {e}")

# ── [3] gpc.py ────────────────────────────────────────────────────────────────
# render_gpc_tab(
#     df_json: str, allcols: list,
#     resolution: int, colorscale: str
# )
try:
    from modules.gpc import render_gpc_tab
    MODULES_STATUS["gpc"] = True
except ImportError as e:
    MODULES_STATUS["gpc"] = False
    def render_gpc_tab(df_json, allcols, resolution, colorscale):
        import streamlit as st
        st.warning(f"⚠️ GPC 분석 모듈 로드 실패: {e}")

# ── [4] report.py ─────────────────────────────────────────────────────────────
# render_report_tab(
#     filename: str, stats: dict, df_display: pd.DataFrame,
#     fig_heatmap: go.Figure, fig_contour: go.Figure,
#     fig_linescan: go.Figure, fig_3d: go.Figure,
#     gpc_ dict | None = None
# )
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

# ── [5] ml_anomaly.py ────────────────────────────────────────────────────────
# render_anomaly_tab(
#     datasets: list, resolution: int, data_folder: str
# )
try:
    from modules.ml_anomaly import render_anomaly_tab
    MODULES_STATUS["ml_anomaly"] = True
except ImportError as e:
    MODULES_STATUS["ml_anomaly"] = False
    def render_anomaly_tab(datasets, resolution, data_folder):
        import streamlit as st
        st.warning(f"⚠️ ML 이상 탐지 모듈 로드 실패: {e}")


# =============================================================================
# 패키지 공개 API 명시 (__all__)
# "from modules import *" 사용 시 아래 목록만 노출됨
# =============================================================================
__all__ = [
    "render_multi_param_tab",
    "render_defect_tab",
    "render_gpc_tab",
    "render_report_tab",
    "render_anomaly_tab",
    "MODULES_STATUS",
]
