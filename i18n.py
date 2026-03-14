# i18n.py
# =============================================================================
# 다국어 지원 모듈 (한국어 / English)
# [요청 4] 언어 선택 기능을 위한 번역 사전
#
# 사용법:
#   from i18n import t, get_lang
#   lang = get_lang()      # "ko" 또는 "en"
#   t("app_title")         # 현재 언어에 맞는 문자열 반환
# =============================================================================

import streamlit as st

# =============================================================================
# 번역 사전
# =============================================================================
_TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── 앱 전역 ──────────────────────────────────────────────────────────────
    "app_title":                 {"ko": "🔬 웨이퍼 맵 분석기", "en": "🔬 Wafer Map Analyzer"},
    "page_title":                {"ko": "웨이퍼 맵 분석기", "en": "Wafer Map Analyzer"},

    # ── 사이드바 ─────────────────────────────────────────────────────────────
    "sidebar_lang_label":        {"ko": "🌐 언어 / Language", "en": "🌐 Language / 언어"},
    "sidebar_readme_btn":        {"ko": "📖 README 보기", "en": "📖 View README"},
    "sidebar_data_mgmt":         {"ko": "📁 데이터 관리", "en": "📁 Data Management"},
    "sidebar_folder_btn_help":   {"ko": "폴더 선택", "en": "Select Folder"},
    "sidebar_no_files":          {"ko": "폴더에 파일 없음", "en": "No files in folder"},
    "sidebar_sample_btn":        {"ko": "🎯 샘플 5개 생성", "en": "🎯 Generate 5 Samples"},
    "sidebar_sample_done":       {"ko": "✅ 샘플 생성!", "en": "✅ Samples generated!"},
    "sidebar_file_select":       {"ko": "📄 파일 선택", "en": "📄 File Selection"},
    "sidebar_analysis_file":     {"ko": "분석 파일", "en": "Analysis File"},
    "sidebar_sheet_select":      {"ko": "시트 선택", "en": "Select Sheet"},
    "sidebar_vis_settings":      {"ko": "⚙️ 시각화 설정", "en": "⚙️ Visualization Settings"},
    "sidebar_colorscale":        {"ko": "컬러 스케일", "en": "Color Scale"},
    "sidebar_resolution":        {"ko": "해상도", "en": "Resolution"},
    "sidebar_show_points":       {"ko": "데이터 포인트 표시", "en": "Show Data Points"},
    "sidebar_contour_levels":    {"ko": "Contour 단계 수", "en": "Contour Levels"},
    "sidebar_linescan_angle":    {"ko": "Line Scan 각도 (°)", "en": "Line Scan Angle (°)"},
    "sidebar_col_mapping":       {"ko": "🔗 컬럼 매핑", "en": "🔗 Column Mapping"},
    "sidebar_x_col":             {"ko": "X 컬럼", "en": "X Column"},
    "sidebar_y_col":             {"ko": "Y 컬럼", "en": "Y Column"},
    "sidebar_data_col":          {"ko": "Data 컬럼", "en": "Data Column"},
    "sidebar_path_input":        {"ko": "경로", "en": "Path"},
    "sidebar_path_placeholder":  {"ko": "경로를 직접 입력하세요...", "en": "Enter path directly..."},

    # ── 탭 라벨 ──────────────────────────────────────────────────────────────
    "tab_wafer":                 {"ko": "📊 웨이퍼 맵", "en": "📊 Wafer Map"},
    "tab_defect":                {"ko": "🔍 결함 오버레이", "en": "🔍 Defect Overlay"},
    "tab_gpc":                   {"ko": "⚗️ GPC 분석", "en": "⚗️ GPC Analysis"},
    "tab_report":                {"ko": "📄 보고서 생성", "en": "📄 Report Generation"},
    "tab_ml":                    {"ko": "🤖 ML 이상 탐지", "en": "🤖 ML Anomaly Detection"},

    # ── 서브탭 ───────────────────────────────────────────────────────────────
    "subtab_single":             {"ko": "🔬 단일 분석", "en": "🔬 Single Analysis"},
    "subtab_compare":            {"ko": "🔀 비교 분석", "en": "🔀 Compare Analysis"},

    # ── 데이터 소스 ──────────────────────────────────────────────────────────
    "source_file":               {"ko": "📁 파일 데이터", "en": "📁 File Data"},
    "source_manual":             {"ko": "✏️ 수동 입력", "en": "✏️ Manual Input"},
    "source_label":              {"ko": "데이터 소스", "en": "Data Source"},

    # ── 수동 입력 ────────────────────────────────────────────────────────────
    "manual_title":              {"ko": "##### ✏️ 수동 데이터 입력", "en": "##### ✏️ Manual Data Entry"},
    "manual_caption":            {"ko": "아래 테이블에 X, Y 좌표와 측정값을 직접 입력하거나, 스프레드시트에서 복사(Ctrl+V)해 붙여넣으세요.",
                                  "en": "Enter X, Y coordinates and measurement values in the table below, or paste from a spreadsheet (Ctrl+V)."},
    "manual_reset_btn":          {"ko": "🗑️ 초기화", "en": "🗑️ Reset"},
    "manual_valid_points":       {"ko": "✅ 유효 포인트: {}개", "en": "✅ Valid points: {}"},
    "manual_min_warning":        {"ko": "⚠️ 유효 포인트 {}개 — 최소 3개 필요", "en": "⚠️ {} valid points — at least 3 required"},
    "manual_input_hint":         {"ko": "ℹ️ 테이블에 데이터를 입력하면 웨이퍼 맵이 생성됩니다.", "en": "ℹ️ Enter data in the table to generate a wafer map."},
    "manual_label":              {"ko": "수동 입력", "en": "Manual Input"},
    "manual_data_title":         {"ko": "📊 수동 입력 데이터", "en": "📊 Manual Input Data"},

    # ── 차트 / 통계 ─────────────────────────────────────────────────────────
    "chart_heatmap":             {"ko": "#### 2D Wafer Map", "en": "#### 2D Wafer Map"},
    "chart_contour":             {"ko": "#### Contour Map", "en": "#### Contour Map"},
    "chart_statistics":          {"ko": "### 📊 Statistics", "en": "### 📊 Statistics"},
    "col_x_mm":                  {"ko": "X (mm)", "en": "X (mm)"},
    "col_y_mm":                  {"ko": "Y (mm)", "en": "Y (mm)"},
    "col_value":                 {"ko": "측정값", "en": "Value"},

    # ── Raw Data 편집 ────────────────────────────────────────────────────────
    "raw_data_header":           {"ko": "📋 Raw Data (셀 편집 즉시 반영)", "en": "📋 Raw Data (edit cells in real-time)"},
    "raw_data_reset_btn":        {"ko": "🔄 편집 초기화", "en": "🔄 Reset Edits"},
    "raw_data_reset_done":       {"ko": "✅ 원본 데이터로 복원되었습니다.", "en": "✅ Restored to original data."},
    "csv_download":              {"ko": "📥 CSV 다운로드", "en": "📥 Download CSV"},

    # ── 비교 분석 ────────────────────────────────────────────────────────────
    "compare_title":             {"ko": "#### 🔀 다중 웨이퍼 비교 분석", "en": "#### 🔀 Multi-Wafer Compare Analysis"},
    "compare_caption":           {"ko": "여러 웨이퍼를 나란히 비교합니다. 같은 파일의 다른 Data 컬럼을 추가하면 다중 파라미터 비교도 가능합니다.",
                                  "en": "Compare multiple wafers side by side. Adding different Data columns from the same file enables multi-parameter comparison."},
    "compare_cols_per_row":      {"ko": "행당 맵 개수", "en": "Maps per row"},
    "compare_lock_scale":        {"ko": "🔒 컬러 스케일 통일", "en": "🔒 Unified Color Scale"},
    "compare_dataset_list":      {"ko": "**📋 데이터셋 목록**", "en": "**📋 Dataset List**"},
    "compare_add_expander":      {"ko": "➕ 데이터셋 추가", "en": "➕ Add Dataset"},
    "compare_from_file":         {"ko": "📁 파일에서 추가", "en": "📁 Add from File"},
    "compare_manual_input":      {"ko": "✏️ 수동 입력", "en": "✏️ Manual Input"},
    "compare_no_files":          {"ko": "ℹ️ 데이터 폴더에 파일이 없습니다. '✏️ 수동 입력' 탭을 사용하세요.",
                                  "en": "ℹ️ No files in data folder. Use the '✏️ Manual Input' tab."},
    "compare_file_label":        {"ko": "파일", "en": "File"},
    "compare_sheet_label":       {"ko": "시트", "en": "Sheet"},
    "compare_csv_no_sheet":      {"ko": "CSV (시트 없음)", "en": "CSV (no sheets)"},
    "compare_ds_name":           {"ko": "데이터셋 이름", "en": "Dataset Name"},
    "compare_name_exists":       {"ko": "⚠️ '{}' 이름이 이미 존재합니다.", "en": "⚠️ Name '{}' already exists."},
    "compare_add_btn":           {"ko": "✅ 추가", "en": "✅ Add"},
    "compare_add_success":       {"ko": "✅ '{}' 추가됨", "en": "✅ '{}' added"},
    "compare_add_fail":          {"ko": "❌ 추가 실패: {}", "en": "❌ Add failed: {}"},
    "compare_file_load_fail":    {"ko": "❌ 파일 읽기 실패: {}", "en": "❌ File load failed: {}"},
    "compare_min_info":          {"ko": "ℹ️ 데이터셋을 **2개 이상** 추가하면 비교 분석이 시작됩니다.\n\n**💡 팁:** 같은 파일에서 Data 컬럼만 다르게 추가하면 다중 파라미터 비교가 가능합니다.",
                                  "en": "ℹ️ Add **2 or more** datasets to start compare analysis.\n\n**💡 Tip:** Adding the same file with different Data columns enables multi-parameter comparison."},
    "compare_comparing":         {"ko": "**비교 중: {}개 데이터셋**", "en": "**Comparing: {} datasets**"},
    "compare_scale_info":        {"ko": "🔒 스케일 고정: {:.2f} ~ {:.2f}", "en": "🔒 Scale locked: {:.2f} ~ {:.2f}"},
    "compare_manual_ds_name":    {"ko": "수동입력_{}", "en": "Manual_{}"},
    "compare_manual_caption":    {"ko": "X, Y 좌표와 측정값을 직접 입력하거나 스프레드시트에서 복사(Ctrl+V)해 붙여넣으세요.",
                                  "en": "Enter X, Y coordinates and values directly, or paste from a spreadsheet (Ctrl+V)."},
    "compare_manual_valid":      {"ko": "✅ 유효 포인트: {}개", "en": "✅ Valid points: {}"},
    "compare_manual_hint":       {"ko": "ℹ️ 데이터를 입력하면 추가할 수 있습니다.", "en": "ℹ️ Enter data to add a dataset."},
    "compare_manual_add_btn":    {"ko": "✅ 수동 데이터 추가", "en": "✅ Add Manual Data"},
    "compare_raw_data":          {"ko": "📋 Raw Data", "en": "📋 Raw Data"},
    "compare_title_label":       {"ko": "이름", "en": "Name"},
    "compare_title_input":       {"ko": "제목", "en": "Title"},

    # ── 폴더 브라우저 ────────────────────────────────────────────────────────
    "fb_title":                  {"ko": "### 📂 폴더 선택", "en": "### 📂 Select Folder"},
    "fb_macos_tip":              {"ko": "💡 네이티브 창: `brew install python-tk@3.14` 설치 후 재시작",
                                  "en": "💡 Native dialog: Install `brew install python-tk@3.14` and restart"},
    "fb_current_path":           {"ko": "**현재 위치:** `{}`", "en": "**Current path:** `{}`"},
    "fb_path_placeholder":       {"ko": "경로를 직접 입력하세요...", "en": "Enter path directly..."},
    "fb_parent":                 {"ko": "⬆️ 상위 폴더", "en": "⬆️ Parent Folder"},
    "fb_home":                   {"ko": "🏠 홈", "en": "🏠 Home"},
    "fb_no_subfolders":          {"ko": "하위 폴더 없음", "en": "No subfolders"},
    "fb_subfolders":             {"ko": "**하위 폴더 ({}개)**", "en": "**Subfolders ({})**"},
    "fb_permission_denied":      {"ko": "⛔ 접근 권한 없음", "en": "⛔ Permission denied"},
    "fb_file_count":             {"ko": "✅ CSV {}개 · XLS {}개 (총 {}개)", "en": "✅ CSV {} · XLS {} (Total {})"},
    "fb_no_data_files":          {"ko": "⚠️ CSV/XLS 파일 없음", "en": "⚠️ No CSV/XLS files"},
    "fb_confirm":                {"ko": "✅ 이 폴더 선택", "en": "✅ Select This Folder"},

    # ── 파일 선택 다이얼로그 ─────────────────────────────────────────────────
    "file_dialog_title":         {"ko": "📄 파일 선택", "en": "📄 Select File"},
    "file_dialog_upload_label":  {"ko": "파일 업로드", "en": "Upload Files"},
    "file_dialog_upload_help":   {"ko": "CSV 또는 Excel 파일을 드래그하거나 선택하세요.", "en": "Drag or select CSV/Excel files."},

    # ── 모듈 가드 ────────────────────────────────────────────────────────────
    "module_unavailable":        {"ko": "⚠️ **{} 모듈을 불러올 수 없습니다.**\n\n`modules/` 폴더와 필요 패키지를 확인하세요.",
                                  "en": "⚠️ **Cannot load {} module.**\n\nCheck the `modules/` folder and required packages."},
    "load_data_first":           {"ko": "ℹ️ **먼저 '📊 웨이퍼 맵' 탭에서 파일을 로드하거나 데이터를 입력해주세요.**",
                                  "en": "ℹ️ **Please load a file or enter data in the '📊 Wafer Map' tab first.**"},

    # ── 파일 모드 ────────────────────────────────────────────────────────────
    "file_no_files_info":        {"ko": "ℹ️ 데이터 폴더에 파일이 없습니다. 사이드바에서 샘플 데이터를 생성하거나 '✏️ 수동 입력'을 선택하세요.",
                                  "en": "ℹ️ No files in data folder. Generate sample data from the sidebar or select '✏️ Manual Input'."},
    "file_load_fail":            {"ko": "❌ 파일 로드 실패: {}", "en": "❌ File load failed: {}"},
}


def get_lang() -> str:
    """현재 선택된 언어 코드 반환 ('ko' 또는 'en')."""
    return st.session_state.get("app_lang", "ko")


def t(key: str, *args) -> str:
    """
    번역 키에 대한 현재 언어 문자열을 반환.
    포맷 인자가 있으면 .format()으로 적용.
    키가 없으면 키 자체를 반환.
    """
    lang = get_lang()
    entry = _TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("ko", key))
    if args:
        try:
            text = text.format(*args)
        except (IndexError, KeyError):
            pass
    return text
