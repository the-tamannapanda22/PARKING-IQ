# 🅿️ ParkingIQ — Parking-Induced Congestion Intelligence System

> **Flipkart Gridlock Hackathon 2.0 — Round 2 — Theme 1**
> Smart Congestion Intelligence & Spatial Enforcement Registry for Bengaluru Traffic Police

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parking-iq-372b5ovjarxvsu5bphhxp8.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚦 Overview

**ParkingIQ** is an AI-powered traffic intelligence dashboard built to detect illegal parking hotspots, quantify their congestion impact, and enable targeted enforcement across Bengaluru. It processes historical violation records spanning **November 2023 – April 2024**, clusters them into spatial H3 hexagonal grid cells, and scores every zone using a composite **Zone Risk Index (ZRI)**.

The system gives traffic police a single portal to monitor live hotspots, forecast peak-hour violations, identify habitual offenders, simulate infrastructure interventions, and dispatch SMS enforcement alerts — all from one interface.

---

## 🌐 Live Demo

**[https://parking-iq-372b5ovjarxvsu5bphhxp8.streamlit.app/](https://parking-iq-372b5ovjarxvsu5bphhxp8.streamlit.app/)**

---

## ✨ Features

| Module | Description |
|--------|-------------|
| 📊 **Executive Overview** | KPI dashboard — total violations, hotspot clusters, watchlist vehicles, heavy vehicle share, junction proximity |
| 🗺️ **Spatial Hotspot Map** | Interactive Folium/Plotly map with H3 grid markers, heatmap layer, and burst-alert pins |
| 📐 **Zone Risk Analysis** | Scatter plots, histograms, and police-station comparisons using ZRI, Spillover Index, and Persistence Index |
| 🤖 **Peak Hour Forecast** | RandomForest ML model predicts per-hour violation volumes; 24×7 heatmap; patrol officer deployment simulator |
| ⛽ **Petrol Pump Simulator** | Traffic engineering model estimating how designated roadside facilities reduce illegal parking by 10–15% |
| 🚔 **Habitual Offender Engine** | Watchlist lookup, escalating challan generation, and bulk summons export for repeat violators |
| 📡 **Live Feed Replay** | Simulated real-time intercept ticker with watchlist plate alerts at the simulated hour |
| 📝 **Citizen Complaints Portal** | Field report submission with photo evidence upload and session-level complaint tracking |
| 📈 **Zone Improvement Tracker** | Log interventions (bollards, signage, towing drives) and model before-vs-after ZRI reduction |
| 📲 **SMS Alert Center** | Draft and simulate SMS dispatches to individual or bulk-tier offenders |
| 🅿️ **Safe Parking Finder** | Haversine-distance search to find low-ZRI zones near a destination; highlights areas to avoid |

---

## 🧠 Methodology

### Zone Risk Index (ZRI) — 0 to 10

A composite score computed per H3 hexagon (resolution 9, ~100m cell size):

| Component | Weight | Description |
|-----------|--------|-------------|
| Violation Density | 30% | Log-normalised count within the hexagon |
| Repeat Offender Ratio | 20% | Share of plates with 3+ tickets |
| Heavy Vehicle Ratio | 15% | Trucks, buses, tankers, LGVs |
| Spillover Index | 15% | Spatial dispersion of violations from centroid |
| Persistence Index | 10% | Active violation days ÷ total dataset days |
| Junction Proximity | 10% | Violations within 150m of a junction |

### Zone Classification

| Type | Behaviour | Recommended Action |
|------|-----------|-------------------|
| 🟣 CHRONIC | High persistence, low variance | Permanent infrastructure (CCTV, bollards, barriers) |
| 🟡 SEMI-CHRONIC | Moderate persistence | Scheduled patrol windows |
| 🔵 SPORADIC | Low persistence, bursty | Tactical towing drives |

### ML Forecast Model

- **Algorithm:** `RandomForestRegressor(n_estimators=50)`
- **Features:** `zone_id`, `dayofweek`, `hour`, `spillover_index`, `persistence_index`, `latitude`, `longitude`
- **Output:** Predicted per-hour violation volume per police station corridor

---

## 🗂️ Project Structure

```
parking-iq-main/
├── app.py                    # Main Streamlit application
├── preprocess_data.py        # Data preprocessing & feature engineering script
├── requirements.txt          # Python dependencies
├── data/
│   ├── cluster_stats.csv     # H3 zone-level features & indices
│   ├── station_summary.csv   # Aggregated stats per police station
│   ├── habitual_offenders.csv# Repeat offender vehicle registry
│   ├── hourly_by_station.csv # Historical hourly violation counts
│   ├── summary.json          # Dataset-level KPIs
│   ├── predictor_model.pkl   # Trained RandomForest model
│   └── predictor_metadata.json # Model feature schema
└── data_backup/              # Backup copies of all CSVs
```

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/PARKING-IQ.git
cd PARKING-IQ/parking-iq-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess data (if raw source data is available)

```bash
python preprocess_data.py
```

> Skip this step if the `data/` folder already contains the CSV files.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repository to GitHub (ensure the `data/` folder is **not** in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set the branch, and set **Main file path** to `parking-iq-main/app.py`.
4. Click **Deploy**.

> **Note:** `app.py` uses `os.path.dirname(os.path.abspath(__file__))` to resolve data paths — this ensures all CSV files load correctly regardless of the Streamlit Cloud working directory.

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.32.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| plotly | ≥ 5.18.0 |
| scikit-learn | ≥ 1.3.0 |
| folium | ≥ 0.15.0 |
| streamlit-folium | ≥ 0.18.0 |
| h3 | ≥ 4.0.0 |

---

## 📊 Dataset

- **Source:** Bengaluru Traffic Police violation records
- **Window:** November 2023 – April 2024
- **Spatial indexing:** Uber H3 hexagonal grid at resolution 9 (~100m cells)
- **Key fields:** vehicle registration, violation type, timestamp, GPS coordinates, police station jurisdiction

---

## 🏆 Hackathon Context

Built for **Flipkart Gridlock Hackathon 2.0 — Round 2 — Theme 1: Parking-Induced Congestion**.

The challenge required designing a data-driven system to:
- Identify illegal parking hotspots causing traffic congestion
- Quantify their impact using spatial and temporal indices
- Enable smarter enforcement decisions using AI/ML

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ for smarter cities · Powered by Streamlit & Plotly
</div>
