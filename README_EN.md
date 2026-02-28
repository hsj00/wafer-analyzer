
# 🔬 Wafer Map Analyzer — Cloud Edition

A Streamlit-based web application for **instant semiconductor process data analysis** in the browser.

Upload a CSV or Excel file and get **wafer map visualization · multi-parameter comparison · defect overlay · GPC analysis · Excel reports · ML anomaly detection** — all in one place.

> 🌐 **This edition is optimized for Streamlit Community Cloud deployment.**
>
> It uses file upload instead of local filesystem access, so anyone can use it from a browser with zero installation.

---

## Table of Contents

1. [Who Is This For?](#1-who-is-this-for)
2. [30-Second Quick Start](#2-30-second-quick-start)
3. [Feature Overview](#3-feature-overview)
4. [Project Structure](#4-project-structure)
5. [Installation & Running](#5-installation--running)
6. [Input Data Formats](#6-input-data-formats)
7. [UI Layout](#7-ui-layout)
8. [Tab-by-Tab Guide](#8-tab-by-tab-guide)
9. [Usage Examples](#9-usage-examples)
10. [FAQ](#10-faq)

---

## 1. Who Is This For?

* **Process engineers** who need to visualize wafer-level distribution data — film thickness, etch depth, sheet resistance, etc.
* **Researchers** analyzing **GPC (Growth Per Cycle)** uniformity for ALD processes
* **QC engineers** who want to **overlay defect inspection data** on top of process measurements
* **Engineers** looking to compare multiple wafers side-by-side or **automatically detect anomalous wafers**
* **Anyone** who needs to package analysis results into a neat **Excel report** for sharing

---

## 2. 30-Second Quick Start

New here? Just follow these three steps.

```
1.  Open the app — you'll see a sidebar on the left.
2.  Click the "🎯 Generate 5 Samples" button.
     → Virtual wafer data is created instantly for testing.
3.  Select wafer_01.csv from the file list.
     → Heatmap, Contour, 3D Surface, and statistics appear immediately!
```

> 💡 **Want to skip samples?** Just drag your own CSV/Excel file into the file uploader in the sidebar.

---

## 3. Feature Overview

```
CSV / Excel File Upload
       │
       ▼
① 📊 Wafer Map     ←── Starting point for all tabs. Load your file here first.
       │
       ├──► ② 📐 Multi-Param     Compare multiple columns as side-by-side heatmaps
       │
       ├──► ③ 🔍 Defect Overlay  Overlay defect coordinates on the wafer map
       │
       ├──► ④ ⚗️ GPC Analysis    Thickness ÷ Cycles, radial uniformity profile
       │
       ├──► ⑤ 📄 Report Export   Stats + chart images + raw data → xlsx download
       │
       └──► ⑥ 🤖 ML Anomaly     Classify wafers via PCA + IsolationForest
```

> **Tabs ②–⑤** require data to be loaded in Tab ① first.
>
> **Tab ⑥** has its own dataset panel — you can add files directly within the tab.

---

## 4. Project Structure

```
wafer_cloud/
│
├── .streamlit/
│   └── config.toml          # App config (upload limit, theme colors)
│
├── app.py                   # Main entry point — run this file
├── core.py                  # Shared core functions (interpolation, plots, stats)
├── requirements.txt         # Python package list
│
└── modules/
    ├── __init__.py          # Safe module loader
    ├── multi_param.py       # Tab ② Multi-parameter subplots
    ├── defect_overlay.py    # Tab ③ Defect overlay
    ├── gpc.py               # Tab ④ GPC analysis
    ├── report.py            # Tab ⑤ Excel report generation
    └── ml_anomaly.py        # Tab ⑥ ML-based anomaly detection
```

> **⚠️ Important:** Both the `modules/` folder and `core.py` must be present.
> If only `app.py` exists, Tabs ②–⑥ will all appear with a `⚠️` disabled status.

---

## 5. Installation & Running

### Option A: Deploy to Streamlit Community Cloud (Recommended)

No installation needed — just push to GitHub!

```
1.  Push this entire project folder to a GitHub repository.
2.  Go to https://share.streamlit.io
3.  Click [New app].
4.  Select your repository, branch, and main file path (app.py).
5.  Click [Deploy!] — deployment finishes in a few minutes.
6.  A unique URL is generated — anyone can access it from a browser.
```

### Option B: Run Locally

**Step 1 — Check Python version** (3.10 or higher required)

```bash
python --version
```

**Step 2 — Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install packages**

```bash
pip install -r requirements.txt
```

**Step 4 — Launch the app**

```bash
streamlit run app.py
```

Your browser will automatically open `http://localhost:8501`.

### Package List

| Package      | Purpose                                  | If Missing               |
| ------------ | ---------------------------------------- | ------------------------ |
| streamlit    | Web UI framework                         | App will not launch      |
| pandas       | CSV/Excel I/O and data processing        | App will not launch      |
| numpy        | Numerical operations                     | App will not launch      |
| plotly       | Interactive charts                       | App will not launch      |
| scipy        | Wafer grid interpolation (`griddata`)    | App will not launch      |
| openpyxl     | Excel file read/write                    | App will not launch      |
| scikit-learn | ML anomaly detection (PCA, IsolationForest) | 🤖 ML tab disabled only |
| kaleido      | Chart → PNG export (for report images)  | Report generated without images |

> `scikit-learn` and `kaleido` are optional. All other tabs work fine without them.

---

## 6. Input Data Formats

### 6-1. Basic Wafer Map Data (CSV / Excel)

Any file with X/Y coordinates and a measurement value column will work.

| x     | y     | data  |
| ----- | ----- | ----- |
| 0.0   | 100.0 | 512.3 |
| -50.0 | 86.6  | 498.7 |
| -86.6 | 50.0  | 503.1 |

* **Auto-detected column names:** `x`, `y`, `data` (case-insensitive). For other names, use the **🔗 Column Mapping** section in the sidebar.
* **Units:** mm-based. 200 mm wafer → radius ~100 mm
* **Minimum points:** 10 or more recommended for good interpolation quality

### 6-2. GPC Analysis Data

Requires additional thickness and cycle count columns.

| x     | y     | thickness_nm | n_cycles |
| ----- | ----- | ------------ | -------- |
| 0.0   | 100.0 | 51.2         | 100      |
| -50.0 | 86.6  | 49.8         | 100      |

* If cycle count is the same for all points, you can enter the number directly in the tab instead of adding a column.

### 6-3. Defect Overlay Data (Separate File)

Upload separately within the Defect Overlay tab.

| x     | y     | class    | size | description    |
| ----- | ----- | -------- | ---- | -------------- |
| 10.5  | -20.3 | Particle | 5.0  | Large particle |
| -33.2 | 41.8  | Scratch  | 12.0 | Linear scratch |

* **Defect category column:** Auto-detects `class`, `type`, or `category`.
* **Different coordinate units?** Use the unit conversion option in the tab (mm / μm / cm / inch).

---

## 7. UI Layout

### Sidebar

The left sidebar is organized into four sections.

#### 📁 Data Management

| Item               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| File Upload        | Drag or click to upload CSV/Excel files              |
| 🎯 Generate Samples | Instantly create 5 virtual wafer datasets for testing |

> **You can upload multiple files at once.** Useful for Compare Mode.

#### 🔀 Analysis Mode

Switch between single-file analysis and side-by-side multi-wafer comparison.

#### ⚙️ Visualization Settings

| Setting          | Default | Description                                  |
| ---------------- | ------- | -------------------------------------------- |
| Color Scale      | Rainbow | Chart color theme                            |
| Resolution       | 100     | Interpolation grid size (higher = sharper, slower) |
| Contour Levels   | 20      | Number of contour line levels                |
| Line Scan Angle  | 0°      | Cross-section profile direction              |

#### 🔗 Column Mapping

After loading a file, assign which columns represent X, Y, and the measurement value. Columns named `x`, `y`, `data` are selected automatically.

### Main Area (Tabs)

In single-analysis mode, six tabs are displayed at the top.

```
📊 Wafer Map │ 📐 Multi-Param │ 🔍 Defect Overlay │ ⚗️ GPC Analysis │ 📄 Report │ 🤖 ML Anomaly
```

---

## 8. Tab-by-Tab Guide

### Tab ① — 📊 Wafer Map

**The starting point for all tabs.** Upload a file and charts + statistics are generated automatically.

**Charts:** 2D Heatmap · Contour · Line Scan · 3D Surface

**Statistics:** Mean · Maximum · Minimum · Std Dev · Uniformity (%) · Range · Number of Sites

**Raw Data Editing:** Edit cells directly in the table at the bottom — charts update instantly. Download the modified data as CSV.

---

### Tab ② — 📐 Multi-Parameter

**Compare multiple columns from the same file as side-by-side wafer heatmaps.**

```
1. Load a data file in Tab ①.
2. Switch to the 📐 Multi-Param tab.
3. Set X/Y columns and select 2–6 parameter columns to compare.
4. Enable "Shared Scale" to unify the color range across all maps.
```

---

### Tab ③ — 🔍 Defect Overlay

**Overlay defect coordinates on top of the wafer map** to analyze spatial correlation between process data and defects.

```
1. Load wafer map data in Tab ①.
2. Switch to the 🔍 Defect Overlay tab.
3. Upload a defect CSV/Excel file.
4. Select which defect classes to display.
5. If coordinate units differ, adjust the unit conversion option.
```

---

### Tab ④ — ⚗️ GPC Analysis

Calculate **GPC (Growth Per Cycle)** for ALD processes and analyze spatial distribution.

> **GPC = Thickness (nm) ÷ Number of Cycles**

```
1. Load a file containing thickness data in Tab ①.
2. Switch to the ⚗️ GPC tab.
3. Select the thickness column and cycle input mode.
4. GPC Heatmap and radial profile are generated automatically.
```

**Results:** GPC Heatmap · Radial GPC Profile · Center / Mid / Edge zone statistics

---

### Tab ⑤ — 📄 Report Export

Export analysis results as an **Excel (.xlsx) file**.

| Sheet      | Contents                                              |
| ---------- | ----------------------------------------------------- |
| Summary    | File name, timestamp, statistical summary             |
| Statistics | Detailed statistics                                   |
| Maps       | Heatmap · Contour · Line Scan · 3D Surface images    |
| Raw Data   | Original measurement data (up to 5,000 rows)          |
| GPC        | GPC analysis results (included only if GPC tab was run) |

> Chart images require the `kaleido` package. Without it, the report is generated without images.

---

### Tab ⑥ — 🤖 ML Anomaly Detection

**Automatically detect anomalous wafers** using PCA + IsolationForest. Requires at least 3 wafer datasets.

```
1. Switch to the 🤖 ML Anomaly tab.
2. Upload wafer files in the 📋 Analysis Datasets panel.
3. Once 3+ datasets are added, configure parameters.
4. Click [🤖 Run Anomaly Detection].
```

**Anomaly Pattern Classification**

| Pattern          | Characteristics       | Possible Process Cause                        |
| ---------------- | --------------------- | --------------------------------------------- |
| Ring             | Donut-shaped radial   | Gas flow center concentration, diffusion non-uniformity |
| Edge Degradation | Edge thickness drop   | Edge exclusion, loading effect                |
| X/Y-Gradient     | Directional gradient  | Substrate tilt, gas directionality            |
| Hotspot          | Localized anomaly     | Particle, scratch, measurement error          |
| Global Shift     | Overall level offset  | Recipe change, process drift                  |
| Normal           | Normal                | —                                             |

---

### Compare Mode

Toggle **🔀 Analysis Mode → Enable Compare Mode** in the sidebar to view multiple wafers side by side.

* Use **➕ Add Dataset** to upload files and assign columns for each wafer.
* You can compare different data columns from the same file by adding it multiple times.
* **🔒 Lock Color Scale** unifies the color range across all comparison cards.

---

## 9. Usage Examples

### Example A: First Launch — Try Everything with Sample Data

```
1. Open the app (Cloud URL or localhost:8501)
2. Sidebar → Click 🎯 Generate 5 Samples
3. Select wafer_01.csv from the file list
4. Tab ① → Check Heatmap, Contour, and statistics
5. Tab ② → Try multi-column comparison
6. Tab ⑥ → Upload 5 wafer files → Run anomaly detection
```

### Example B: ALD Process GPC Analysis

```
1. Upload a CSV with x, y, thickness_nm, n_cycles columns
2. Sidebar column mapping: X=x, Y=y, Data=thickness_nm
3. Tab ④ → Thickness column: thickness_nm / Mode: Column / Cycle column: n_cycles
4. Review GPC Heatmap and Center/Mid/Edge statistics
5. Tab ⑤ → Check "Include GPC" → Download Excel report
```

### Example C: Defect–Process Correlation Analysis

```
1. Prepare two files: thickness CSV + defect coordinates CSV
2. Tab ① → Upload thickness CSV → Check Heatmap
3. Tab ③ → Upload defect CSV → Select classes (e.g., Particle, Scratch)
4. Analyze spatial correlation between low-thickness regions and defect distribution
```

### Example D: Automated Anomaly Detection Across a Lot

```
1. Prepare measurement CSVs for 10 wafers from the same process
2. Tab ⑥ → Upload and add all 10 files
3. Set Contamination to 0.10, Resolution to 40 → Run anomaly detection
4. Check the PCA scatter plot for outlier wafers
5. Review the results table → Investigate top-scoring wafers
```

---

## 10. FAQ

| Question                                  | Answer                                                                                    |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| I don't have any data to test with        | Click the **🎯 Generate 5 Samples** button in the sidebar                                 |
| A tab shows ⚠️ next to its name          | The module file is missing or a required package is not installed                          |
| The 🤖 ML tab is disabled                 | `scikit-learn` is needed (already included in requirements.txt for Cloud deployment)      |
| Report is missing chart images            | `kaleido` is needed (already included in requirements.txt for Cloud deployment)           |
| Defect coordinates don't align            | Use the **unit conversion** option in the 🔍 Defect Overlay tab                           |
| I see an interpolation failure warning    | Too few data points or all points lie on a single line. Auto-fallback is applied           |
| Is there a file upload size limit?        | Default is 200 MB. Adjust in `.streamlit/config.toml`                                      |
| I deployed to Cloud but get errors        | Verify that `requirements.txt` and `core.py` are in the project root                      |

---

*Tested with Python 3.10+ · Streamlit 1.35+*
