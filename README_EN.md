
# 🔬 Wafer Map Analyzer v3.0

A Streamlit-based web application for analyzing semiconductor process data directly in a browser.

Load CSV or Excel files — or **enter data manually** — to perform **wafer map visualization · multi-wafer comparison · GPC analysis · defect overlay · ML anomaly detection · Excel report generation** all in one place.

---

## Table of Contents

1. [Who This Is For](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#1-who-this-is-for)
2. [What&#39;s New in v3.0](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#2-whats-new-in-v30)
3. [How It All Fits Together](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#3-how-it-all-fits-together)
4. [Project Structure](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#4-project-structure)
5. [Installation &amp; Launch](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#5-installation--launch)
6. [Input Data Format](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#6-input-data-format)
7. [UI Layout](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#7-ui-layout)
8. [Tab-by-Tab Guide](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#8-tab-by-tab-guide)
9. [Workflow Examples](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#9-workflow-examples)
10. [FAQ](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#10-faq)
11. [requirements.txt](https://claude.ai/chat/4247ea22-24e9-4413-805e-69f1ae3d6061#11-requirementstxt)

---

## 1. Who This Is For

* Process engineers who need to **visualize wafer-level spatial distribution** data (film thickness, etch depth, sheet resistance, etc.)
* Researchers analyzing **GPC (Growth Per Cycle)** uniformity in ALD processes
* QC engineers who want to **overlay defect inspection coordinates** on top of process measurement maps
* Engineers who need to **compare multiple wafers** or automatically **screen for anomalous wafers** across a lot
* Engineers who want to **manually enter measurement values** to quickly check a wafer map without needing a file

---

## 2. What's New in v3.0

### ① Compare Mode → Integrated into Wafer Map Tab

The sidebar compare mode toggle has been removed. Comparison is now a  **sub-tab inside the Wafer Map tab** .

| v2.0                                           | v3.0                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| Sidebar `Enable Compare Mode`toggle          | Wafer Map tab →`🔬 Single Analysis`/`🔀 Compare Analysis`sub-tabs |
| Compare mode replaces all tab views with cards | Freely switch between single and compare analysis                      |
| Dataset management UI in sidebar               | Dataset management UI inside compare sub-tab                           |

### ② Manual Input Mode Added

**The app now works without any data files.** Select "✏️ Manual Input" in the single analysis sub-tab to get an empty table where you can type X/Y coordinates and measurement values directly, or paste data from a spreadsheet (Ctrl+V). Charts are generated automatically when 3+ valid points are entered.

### ③ Multi-Parameter Tab Removed → Merged into Compare Sub-tab

The `📐 Multi-Parameter` tab has been removed. To compare multiple measurement columns from the same file,  **add the same file multiple times in the compare sub-tab with different Data columns** .

---

## 3. How It All Fits Together

```
CSV / Excel file(s)  or  Manual input (typing / paste)
        │
        ▼
① 📊 Wafer Map ─┬─ 🔬 Single Analysis  ← Start here: load a file or enter data
                 └─ 🔀 Compare Analysis  ← Compare multiple wafers/parameters side by side
        │
        ├──► ② 🔍 Defect Overlay    Overlay a defect CSV on top of the wafer map
        │
        ├──► ③ ⚗️ GPC Analysis      Thickness ÷ Cycle count, radial uniformity profile
        │
        ├──► ④ 📄 Report            Export stats + chart images + raw data as .xlsx
        │
        └──► ⑤ 🤖 ML Anomaly        Auto-classify wafers via PCA + IsolationForest
```

> Tabs ②–④ require data to be loaded in Single Analysis first.
>
> Tab ⑤ has its own built-in dataset panel — files can be added directly inside the tab.
>
> Datasets added in the Compare sub-tab are automatically available in the ML tab.

---

## 4. Project Structure

```
wafer_analysis/
│
├── app.py                   # Main entry point — run this file
├── folder_picker_helper.py  # OS native folder picker helper (called automatically)
│
└── modules/
    ├── __init__.py          # Safe module loader — do not modify
    ├── defect_overlay.py    # Tab ② Defect overlay
    ├── gpc.py               # Tab ③ GPC analysis
    ├── report.py            # Tab ④ Excel report generation
    └── ml_anomaly.py        # Tab ⑤ ML-based anomaly detection
```

> **⚠️ Important:** The `modules/` folder and all files inside it must be present.
>
> If only `app.py` exists, tabs ②–⑤ will all appear as `⚠️` (disabled).
>
> `multi_param.py` was removed in v3.0. If the file still exists, it is safely ignored.

---

## 5. Installation & Launch

### Step 1 — Check Python Version

Python **3.10 or higher** is required.

```bash
python --version
# Must show Python 3.10.x or higher
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Packages

If a `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Otherwise, install directly:

```bash
pip install streamlit pandas numpy plotly scipy openpyxl scikit-learn kaleido
```

| Package      | Purpose                                     | Without It                       |
| ------------ | ------------------------------------------- | -------------------------------- |
| streamlit    | Web UI framework                            | App won't start                  |
| pandas       | CSV/Excel reading and data processing       | App won't start                  |
| numpy        | Numerical computation                       | App won't start                  |
| plotly       | Interactive charts                          | App won't start                  |
| scipy        | Wafer grid interpolation (`griddata`)     | App won't start                  |
| openpyxl     | Reading and writing Excel files             | App won't start                  |
| scikit-learn | ML anomaly detection (PCA, IsolationForest) | 🤖 ML tab disabled only          |
| kaleido      | Chart → PNG conversion for reports         | Reports generated without images |

> `scikit-learn` and `kaleido` are optional. All other tabs work without them.

### Step 4 — Run the Application

```bash
streamlit run app.py
```

A browser tab opens automatically at `http://localhost:8501`.

### Step 5 — First Launch

No data files yet? No problem. You have two options:

**Option A — Generate Sample Data:**

1. After launching, find the **"🎯 Generate 5 Samples"** button in the left sidebar.
2. Click it to auto-create `wafer_01.csv` through `wafer_05.csv` in `./wafer_data/`.

**Option B — Manual Input (new in v3.0):**

1. Go to Wafer Map tab → Single Analysis → select  **"✏️ Manual Input"** .
2. Type X, Y, Data values directly in the empty table, or paste from a spreadsheet.
3. Charts appear automatically once 3+ valid points are entered.

---

## 6. Input Data Format

### 6-1. Wafer Map Data (CSV / Excel)

Any file containing X/Y coordinates and a measurement value column will work.

| x     | y     | data  |
| ----- | ----- | ----- |
| 0.0   | 100.0 | 512.3 |
| -50.0 | 86.6  | 498.7 |
| -86.6 | 50.0  | 503.1 |

* **Auto-detected column names:** `x`, `y`, `data` (case-insensitive). Other names can be mapped in the sidebar under  **🔗 Column Mapping** .
* **Unit:** Millimeters (mm). Typical: radius ~100 mm for a 200 mm wafer, ~150 mm for a 300 mm wafer.
* **Minimum points:** 10+ recommended for reliable interpolation (3+ for manual input).

### 6-2. Manual Input Data (new in v3.0)

Enter data directly in the Wafer Map tab without any files.

* Select "✏️ Manual Input" in the Single Analysis sub-tab
* A 20-row empty table (x, y, data) is displayed
* Type values directly in cells, or copy a range from a spreadsheet and paste with **Ctrl+V**
* Rows can be added/removed (use the `+` button below the table)
* NaN rows are automatically excluded; charts appear once 3+ valid points exist
* Use the 🗑️ Reset button to clear back to an empty table

### 6-3. GPC Analysis Data

Requires an additional thickness column and a cycle count column.

| x     | y     | thickness_nm | n_cycles |
| ----- | ----- | ------------ | -------- |
| 0.0   | 100.0 | 51.2         | 100      |
| -50.0 | 86.6  | 49.8         | 100      |

* If all sites share the same cycle count, you can enter it as a single number in the GPC tab.

### 6-4. Defect Overlay Data (Separate File)

Place a separate CSV/Excel file in the same data folder as your wafer map files.

| x     | y     | class    | size | description    |
| ----- | ----- | -------- | ---- | -------------- |
| 10.5  | -20.3 | Particle | 5.0  | Large particle |
| -33.2 | 41.8  | Scratch  | 12.0 | Linear scratch |

* **Defect class column:** Any of `class`, `type`, `category`, `defecttype`, `label` is recognized automatically.
* `size` and `description` are optional.
* **Unit mismatch** (e.g., defect file in μm, wafer file in mm): use the unit conversion selector in the tab.

---

## 7. UI Layout

### Sidebar

The left sidebar is divided into three sections.

#### 📁 Data Management

| Item                  | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| 📂 Folder button      | Open OS file dialog or built-in browser to select a data folder |
| Path text input       | Type a folder path directly                                     |
| File list             | CSV/Excel files in the selected folder are listed automatically |
| 🎯 Generate 5 Samples | Creates 5 sample CSVs in `./wafer_data/`for immediate testing |

> **Default folder:** The app looks for `./wafer_data/` on startup.

#### ⚙️ Visualization Settings

Controls for chart appearance.

| Setting         | Default | Description                                        |
| --------------- | ------- | -------------------------------------------------- |
| Color scale     | Rainbow | Chart color theme                                  |
| Resolution      | 100     | Interpolation grid size (higher = sharper, slower) |
| Contour levels  | 20      | Number of contour lines                            |
| Line Scan angle | 0°     | Cross-section direction                            |

#### 🔗 Column Mapping

After loading a file, assign the X-axis, Y-axis, and measurement value columns. Columns named `x`, `y`, and `data` are selected automatically.

### Main Panel (Tabs)

Five tabs appear along the top. A tab shows `⚠️` if its module file is missing or a required package is not installed.

```
📊 Wafer Map │ 🔍 Defect Overlay │ ⚗️ GPC Analysis │ 📄 Report │ 🤖 ML Anomaly
```

The Wafer Map tab contains two sub-tabs:

```
🔬 Single Analysis │ 🔀 Compare Analysis
```

---

## 8. Tab-by-Tab Guide

### Tab ① — 📊 Wafer Map

The  **starting point for everything** .

#### 🔬 Single Analysis Sub-tab

**Data Source Selection**

| Mode              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| 📁 File Data      | Analyze the CSV/Excel file selected in the sidebar (default) |
| ✏️ Manual Input | Type or paste data directly into an empty table              |

> When no files exist in the data folder, only "✏️ Manual Input" is shown.

**Available Charts**

| Chart      | Description                                       |
| ---------- | ------------------------------------------------- |
| 2D Heatmap | Color-coded spatial distribution map              |
| Contour    | Iso-value lines showing distribution boundaries   |
| Line Scan  | Cross-section profile along a user-selected angle |
| 3D Surface | Measurement values rendered as a 3D height map    |

**Statistics:** Mean · Maximum · Minimum · Std Dev · Uniformity(%) · Range · Site count

**Live Data Editing:** Edit cells in the Raw Data table at the bottom — charts update instantly. Use **📥 Download CSV** to save modified data.

#### 🔀 Compare Analysis Sub-tab

Compare multiple wafers or multiple parameters from the same wafer  **side by side** .

> This sub-tab  **replaces both the v2.0 Compare Mode and the Multi-Parameter tab** .

**How to Use**

1. Switch to the Compare Analysis sub-tab.
2. Expand **➕ Add Dataset** and select file, sheet, and X/Y/Data columns.
3. Add 2+ datasets and a card-based comparison view appears.
4. Each card shows  **Heatmap + Contour + Statistics** .

**Features**

* **🔒 Lock Color Scale** unifies the colorbar range across all cards.
* **Cards per row** can be set to 2, 3, or 4.
* **Add the same file with different Data columns** to compare multiple parameters from one wafer.

**Multi-Parameter Comparison Example:**

From the same `wafer_01.xlsx`:

* Dataset 1: X=x, Y=y, Data=**thickness**
* Dataset 2: X=x, Y=y, Data=**sheet_resistance**
* Dataset 3: X=x, Y=y, Data=**stress**

→ Three cards appear side by side, showing three properties from the same wafer at a glance.

---

### Tab ② — 🔍 Defect Overlay

**Overlay defect locations on the wafer map** to analyze spatial correlation between process results and defect distribution.

**How to Use**

1. Place a defect CSV/Excel file in the data folder.
2. In the 🔍 Defect Overlay tab, select the defect file.
3. Choose which defect classes to display.
4. If coordinate units differ from the wafer map, set the unit conversion option.

**Features:** Scatter overlay on Heatmap or Contour · Per-class color/marker assignment (up to 24 classes) · Unit conversion (mm / cm / m / inch) · Show/hide out-of-wafer defects

---

### Tab ③ — ⚗️ GPC Analysis

Calculate and visualize **GPC (Growth Per Cycle)** for ALD processes.

> **GPC = Thickness (nm) ÷ Cycle count**

**Cycle Count Modes**

| Mode         | Description                                               |
| ------------ | --------------------------------------------------------- |
| Fixed Cycle  | Apply one cycle count to all sites (enter a number)       |
| Column Cycle | Use per-site values from a cycle count column in the file |

**How to Use**

1. Load a thickness file in Single Analysis.
2. Go to the ⚗️ GPC tab.
3. Select the thickness column and cycle mode.
4. GPC Heatmap and radial profile are generated automatically.

**Results:** GPC Heatmap · Radial GPC profile · Center / Mid / Edge zone statistics · Center-to-Edge uniformity

---

### Tab ④ — 📄 Report

Export analysis results as an  **Excel (.xlsx) file** .

**Excel Sheet Layout**

| Sheet      | Contents                                                                 |
| ---------- | ------------------------------------------------------------------------ |
| Summary    | Filename, generation timestamp, statistics (Mean, Std, Uniformity, etc.) |
| Statistics | Detailed statistical data                                                |
| Maps       | Heatmap · Contour · Line Scan · 3D Surface chart images               |
| Raw Data   | Original measurement data (up to 5,000 rows)                             |
| GPC        | GPC results and chart (included only if GPC analysis was run first)      |

**How to Use**

1. Load data in Single Analysis.
2. (Optional) Run GPC analysis in the GPC tab to include the GPC sheet.
3. Go to the 📄 Report tab.
4. Check the items to include (chart images / raw data / GPC).
5. Click **Generate Excel Report** — the file downloads immediately.

> Chart images require the `kaleido` package.

---

### Tab ⑤ — 🤖 ML Anomaly Detection

Automatically detect anomalous wafers among a group using  **PCA + Isolation Forest** .

Requires at least 3 wafer datasets.

**How to Use**

1. Go to the 🤖 ML Anomaly Detection tab.
2. In the **📋 Analysis Datasets** panel, add your wafer files.| Action                     | Description                                       |
   | -------------------------- | ------------------------------------------------- |
   | **➕ Add Dataset**   | Select file, sheet, and columns, then click Add   |
   | **🔄 Sync from App** | Import datasets from Compare sub-tab in one click |
   | **✕ button**        | Remove an individual dataset                      |
   | **🗑️ Reset All**   | Clear all datasets and analysis results           |
3. Once 3+ datasets are registered, configure parameters.| Parameter          | Default | Description                           |
   | ------------------ | ------- | ------------------------------------- |
   | Contamination      | 10%     | Expected fraction of anomalous wafers |
   | Feature Resolution | 40      | Grid size for feature extraction      |
4. Click  **🤖 Run Anomaly Detection** .

**Results**

* **PCA Scatter Plot:** Anomalous wafers (red) vs. normal wafers (blue)
* **Anomaly Score Bar Chart:** Per-wafer score ranked from high to low
* **Results Table:** Anomaly score ranking + pattern classification
* **Anomaly Wafer Heatmap Preview:** Wafer maps for detected anomalies

**Anomaly Pattern Types**

| Pattern          | Characteristics                  | Possible Process Cause                             |
| ---------------- | -------------------------------- | -------------------------------------------------- |
| Ring             | Donut-shaped radial distribution | Non-uniform gas flow (center-heavy or edge-heavy)  |
| Edge Degradation | Thickness drop near wafer edge   | Edge exclusion effect, loading effect              |
| X/Y-Gradient     | Directional thickness gradient   | Substrate tilt, directional gas flow               |
| Hotspot          | Localized anomaly peak           | Particle contamination, scratch, measurement error |
| Global Shift     | Whole-wafer level offset         | Recipe change, process drift                       |
| Normal           | Normal pattern                   | —                                                 |

> Requires `scikit-learn`: `pip install scikit-learn`

---

## 9. Workflow Examples

### Example A: First Launch — Instant Start with Manual Input

```
1. Run: streamlit run app.py
2. Wafer Map tab → Single Analysis → select ✏️ Manual Input
3. Type coordinates and values in the empty table (or paste from Excel with Ctrl+V)
4. Once 3+ points are entered, Heatmap, Contour, and stats appear automatically
```

### Example B: Try Everything with Sample Data

```
1. Run: streamlit run app.py
2. Sidebar → click 🎯 Generate 5 Samples
3. Select wafer_01.csv from the file list
4. Single Analysis → review Heatmap, Contour, and statistics
5. Compare Analysis → add all 5 files via ➕ Add Dataset → compare side by side
6. ML tab → 🔄 Sync from App → run anomaly detection
```

### Example C: Multi-Parameter Comparison

```
1. Prepare an Excel file with multiple measurement columns (thickness, rs, stress, etc.)
2. Go to Compare Analysis sub-tab
3. ➕ Add Dataset: same file, different Data column → add 3 datasets
4. Check 🔒 Lock Color Scale → compare absolute values across parameters
```

### Example D: ALD GPC Analysis

```
1. Prepare CSV with columns: x, y, thickness_nm, n_cycles
2. Select the folder and file in the sidebar
3. Column mapping: X=x, Y=y, Data=thickness_nm
4. GPC tab → Thickness col: thickness_nm / Mode: Column / Cycle col: n_cycles
5. Review GPC Heatmap and Center/Mid/Edge zone statistics
6. Report tab → check GPC → click Generate → download report
```

### Example E: Correlate Defect Locations with Process Data

```
1. Prepare two files: thickness CSV + defect coordinates CSV
2. Single Analysis → load thickness file → review Heatmap
3. Defect Overlay tab → select defect file → choose classes (e.g. Particle, Scratch)
4. Visually compare thin regions on the map with defect cluster locations
```

### Example F: Screen a Lot for Anomalous Wafers

```
1. Prepare measurement CSVs for 10 wafers from the same process step
2. Compare Analysis → add all 10 files (also gives a visual side-by-side comparison)
3. ML tab → 🔄 Sync from App → import all 10 datasets
4. Contamination: 0.10, Resolution: 40 → click Run
5. PCA scatter: identify wafers that deviate from the main cluster
```

---

## 10. FAQ

| Question                                    | Answer                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| No data files to test with?                 | Click**🎯 Generate 5 Samples**in the sidebar, or use**✏️ Manual Input**to enter data directly |
| Can I use the app without files?            | Yes, since v3.0. Go to Wafer Map → Single Analysis → ✏️ Manual Input                              |
| Where did the Multi-Parameter tab go?       | Merged into the Compare sub-tab in v3.0. Add the same file with different Data columns                |
| Where did the Compare Mode toggle go?       | Go to Wafer Map tab → 🔀 Compare Analysis sub-tab                                                    |
| A tab shows ⚠️ after its name?            | The module file is missing or a required package is not installed                                     |
| 🤖 ML tab is disabled?                      | Run `pip install scikit-learn`and restart the app                                                   |
| No chart images in the Excel report?        | Run `pip install kaleido`and restart the app                                                        |
| Folder picker won't open on macOS?          | Run `brew install python-tk`and restart the app                                                     |
| Defect coordinates are misaligned?          | Use the**Coordinate Unit**selector in the Defect Overlay tab (mm / cm / m / inch)               |
| Interpolation warning appears?              | Occurs when data points are too few or all collinear. Falls back to nearest-neighbor automatically    |
| I want to edit measurement values directly? | Edit cells in the Raw Data table at the bottom of Single Analysis                                     |
| I want to save manually entered data?       | Click the 📥 Download CSV button below the charts                                                     |

---

## 11. requirements.txt

Save the following as `requirements.txt` in the project root.

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
