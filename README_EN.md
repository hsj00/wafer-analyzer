
# 🔬 Wafer Map Analyzer v4.0

A Streamlit-based web application for analyzing semiconductor process data directly in a browser.

Load CSV or Excel files — or **enter data manually** — to perform **wafer map visualization · multi-wafer comparison · GPC analysis · defect overlay · ML anomaly detection · Excel report generation** all in one place.

---

## Table of Contents

1. [Who This Is For](#1-who-this-is-for)
2. [What's New in v4.0](#2-whats-new-in-v40)
3. [How It All Fits Together](#3-how-it-all-fits-together)
4. [Project Structure](#4-project-structure)
5. [Installation & Launch](#5-installation--launch)
6. [Input Data Format](#6-input-data-format)
7. [UI Layout](#7-ui-layout)
8. [Tab-by-Tab Guide](#8-tab-by-tab-guide)
9. [Workflow Examples](#9-workflow-examples)
10. [FAQ](#10-faq)
11. [requirements.txt](#11-requirementstxt)

---

## 1. Who This Is For

* Process engineers who need to **visualize wafer-level spatial distribution** data (film thickness, etch depth, sheet resistance, etc.)
* Researchers analyzing **GPC (Growth Per Cycle)** uniformity in ALD processes
* QC engineers who want to **overlay defect inspection coordinates** on top of process measurement maps
* Engineers who need to **compare multiple wafers** or automatically **screen for anomalous wafers** across a lot
* Engineers who want to **manually enter measurement values** to quickly check a wafer map without needing a file

---

## 2. What's New in v4.0

### ① Korean / English Language Selection (New)

Select **한국어 / English** at the top of the sidebar to switch the entire app UI to your preferred language.

| Feature              | Description                                          |
| -------------------- | ---------------------------------------------------- |
| 🌐 Language radio    | Top of sidebar, switches between Korean and English  |
| 📖 View README button | Displays the full README in the selected language    |

### ② File & Folder Selection — Unified Dialog Popups

The inline folder browser has been replaced with **@st.dialog**-based popups for a consistent UX across Local and Cloud environments.

| Environment | File Selection                    | Folder Selection                                 |
| ----------- | --------------------------------- | ------------------------------------------------ |
| Local       | @st.dialog inline browser         | Native (tkinter) → falls back to dialog on fail  |
| Cloud       | @st.dialog with st.file_uploader  | N/A (file upload workflow)                       |

### ③ Enhanced Raw Data Editing

| Feature           | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| Add/delete rows   | `num_rows="dynamic"` → +/- buttons at table bottom            |
| Persistent edits  | Saved to `session_state["edited_df"]` → survives tab switches |
| Reset button      | 🔄 button instantly restores original data                     |
| Live chart update | Charts and statistics recalculate from edited DataFrame        |
| Compare card edit | Each compare card also supports data_editor editing            |

### ④ Auto-fill Dataset Name in Compare Mode

When adding a dataset, the **dataset name automatically updates** when you change the Data column selection. If you manually edit the name, it won't be overwritten.

### ⑤ Performance & UX Improvements

* Added `st.spinner` to chart generation sections
* Improved error handling in `apply_col_mapping` with column existence checks
* Auto-detection of Cloud environment (`STREAMLIT_SHARING_MODE`, `/.dockerenv`)

---

## 3. How It All Fits Together

```
CSV / Excel files  or  Manual input (typing / paste)
        │
        ▼
① 📊 Wafer Map ─┬─ 🔬 Single Analysis  ← Start with file load or manual entry
                 └─ 🔀 Compare Analysis ← Side-by-side multi-wafer comparison
        │
        ├──► ② 🔍 Defect Overlay    Overlay defect CSV on wafer map
        │
        ├──► ③ ⚗️ GPC Analysis      Thickness ÷ cycles, radial uniformity profile
        │
        ├──► ④ 📄 Report Generation Stats + chart images + raw data → xlsx download
        │
        └──► ⑤ 🤖 ML Anomaly       Auto-classify wafers via PCA + IsolationForest
```

> **Tabs ②–④** require data to be loaded in Single Analysis first.
>
> **Tab ⑤** has its own dataset panel — you can add files directly within the tab.

---

## 4. Project Structure

```
wafer_analysis/
│
├── app.py                   # Main app entry point — run this file
├── i18n.py                  # Internationalization module (Korean/English)
├── folder_picker_helper.py  # OS-native folder picker helper (auto-invoked)
│
└── modules/
    ├── __init__.py          # Safe module loader (no edits needed)
    ├── defect_overlay.py    # Tab ② Defect Overlay
    ├── gpc.py               # Tab ③ GPC Analysis
    ├── report.py            # Tab ④ Excel Report Generation
    └── ml_anomaly.py        # Tab ⑤ ML Anomaly Detection
```

> **⚠️ Note:** Both the `modules/` folder and `i18n.py` must be present.

---

## 5. Installation & Launch

### 5-1. Check Python Version

Python **3.10 or newer** is required.

```bash
python --version
```

### 5-2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 5-3. Install Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy plotly scipy openpyxl scikit-learn kaleido
```

| Package      | Purpose                             | If Missing                |
| ------------ | ----------------------------------- | ------------------------- |
| streamlit    | Web UI framework                    | App won't start           |
| pandas       | CSV/Excel I/O and data processing   | App won't start           |
| numpy        | Numerical computation               | App won't start           |
| plotly        | Interactive charts                  | App won't start           |
| scipy        | Wafer grid interpolation (`griddata`) | App won't start         |
| openpyxl     | Excel file read/write               | App won't start           |
| scikit-learn | ML anomaly detection (PCA, IF)      | 🤖 ML tab disabled only  |
| kaleido      | Chart → PNG conversion              | Reports without images    |

### 5-4. Launch

```bash
streamlit run app.py
```

Your browser will auto-open at `http://localhost:8501`.

### 5-5. First Run

No data files needed to get started:

**Option A — Generate Samples:**

1. Click **"🎯 Generate 5 Samples"** in the sidebar.
2. Five sample CSV files will be created in `./wafer_data/`.

**Option B — Manual Input:**

1. Wafer Map tab → Single Analysis → select **"✏️ Manual Input"**.
2. Enter X, Y, Data values directly or paste from a spreadsheet (Ctrl+V).
3. Charts are generated automatically when 3+ valid points are entered.

---

## 6. Input Data Format

### 6-1. Basic Wafer Map Data (CSV / Excel)

Any file with X/Y coordinate columns and a measurement value column.

| x     | y     | data  |
| ----- | ----- | ----- |
| 0.0   | 100.0 | 512.3 |
| -50.0 | 86.6  | 498.7 |

* **Auto-detected column names:** `x`, `y`, `data` (case-insensitive). Other names can be mapped via **🔗 Column Mapping** in the sidebar.
* **Units:** mm. 200 mm wafer → radius ~100 mm, 300 mm wafer → radius ~150 mm
* **Minimum points:** 10+ recommended for quality interpolation (3 minimum for manual input)

### 6-2. GPC Analysis Data

Requires additional thickness and cycle count columns.

### 6-3. Defect Overlay Data (Separate File)

| x     | y     | class    | size | description     |
| ----- | ----- | -------- | ---- | --------------- |
| 10.5  | -20.3 | Particle | 5.0  | Large particle  |

* **Defect class column:** Auto-detects `class`, `type`, `category`, `defecttype`, or `label`.

---

## 7. UI Layout

### Sidebar

| Section             | Items                                                |
| ------------------- | ---------------------------------------------------- |
| 🌐 Language         | Korean / English toggle (v4.0 new)                   |
| 📖 README           | Display full README for selected language             |
| 📁 Data Management  | Folder selection, path input, file list, sample gen   |
| ⚙️ Visualization    | Color scale, resolution, contour levels, scan angle   |
| 🔗 Column Mapping   | X / Y / Data column assignment                       |

### Main Tabs

| Tab                    | Core Function                                        |
| ---------------------- | ---------------------------------------------------- |
| 📊 Wafer Map          | Single + Compare analysis sub-tabs                   |
| 🔍 Defect Overlay     | Overlay defect coordinates on wafer maps             |
| ⚗️ GPC Analysis       | ALD growth-per-cycle calculation and spatial profile  |
| 📄 Report Generation  | Stats + charts → Excel download                      |
| 🤖 ML Anomaly Detection | PCA + IsolationForest auto anomaly screening       |

---

## 8. Tab-by-Tab Guide

### Tab ① — 📊 Wafer Map

#### 🔬 Single Analysis Sub-tab

**Data source:** 📁 File Data or ✏️ Manual Input

**Charts:** 2D Heatmap · Contour · Line Scan · 3D Surface

**Raw Data Editing (v4.0 Enhanced):**
* Add/delete rows: +/- buttons at table bottom
* Edit cells: Click to modify values → charts update immediately
* Reset edits: 🔄 button restores original data
* Edits persist across tab switches and reruns

#### 🔀 Compare Analysis Sub-tab

Compare multiple wafers or multiple parameters from the same wafer side by side.

**v4.0 improvement:** Dataset name auto-updates based on Data column selection.

### Tab ② — 🔍 Defect Overlay

Overlay defect coordinates on wafer maps to analyze spatial correlation with process data.

### Tab ③ — ⚗️ GPC Analysis

Calculate GPC = Thickness(nm) ÷ Cycle count and analyze spatial distribution.

### Tab ④ — 📄 Report Generation

Export analysis results to Excel (.xlsx) files.

### Tab ⑤ — 🤖 ML Anomaly Detection

Auto-detect anomalous wafer patterns using PCA + Isolation Forest. Requires 3+ wafer datasets.

---

## 9. Workflow Examples

### Example A: First Run — Quick Start with Manual Input

```
1. streamlit run app.py
2. Wafer Map tab → Single Analysis → ✏️ Manual Input
3. Enter coordinates and values in the empty table
4. 3+ points → Heatmap, Contour, Statistics auto-generated
```

### Example B: Switch to Korean UI

```
1. Select "한국어" at 🌐 Language in the sidebar
2. All UI switches to Korean
3. Click 📖 README 보기 for Korean documentation
```

### Example C: Edit Raw Data and Re-analyze

```
1. Load file → Single Analysis → scroll to Raw Data table
2. Add rows (+): Enter additional measurement points
3. Delete rows: Remove outlier data
4. Charts and statistics update immediately
5. Use 🔄 Reset Edits to restore original data
```

---

## 10. FAQ

| Question                                  | Answer                                                           |
| ----------------------------------------- | ---------------------------------------------------------------- |
| Can I use it in Korean?                   | Select 한국어 at 🌐 Language in the sidebar                      |
| My edits disappear when switching tabs    | Fixed in v4.0 — edits are auto-saved to session_state            |
| Folder selection doesn't work on Cloud    | On Cloud, the 📂 button opens a file upload popup instead        |
| No test data available                    | Use 🎯 Generate 5 Samples button or ✏️ Manual Input             |
| ⚠️ icon next to tab names                | Module file missing or required package not installed             |

---

## 11. requirements.txt

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scipy>=1.10.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
kaleido>=0.2.1
```

```bash
pip install -r requirements.txt
```

---

*Tested on Python 3.10+ · Streamlit 1.35+*
