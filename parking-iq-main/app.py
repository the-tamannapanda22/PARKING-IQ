"""
=============================================================
ParkingIQ — Parking-Induced Congestion Intelligence System
Flipkart Gridlock Hackathon 2.0 — Round 2 — Theme 1
=============================================================
Run with: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import random
import pickle
import os
from datetime import datetime, timedelta

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="ParkingIQ — Bengaluru Traffic Intelligence",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
.stApp { background-color: #0a0e1a; }
.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151; border-radius: 12px;
    padding: 16px 18px; text-align: center; margin-bottom: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #4f46e5;
}
.metric-number { font-size: 1.9rem; font-weight: 700; color: #f9fafb; }
.metric-label  { font-size: 0.78rem; color: #9ca3af; margin-top: 3px; }
.metric-sub    { font-size: 0.7rem; color: #6b7280; margin-top: 2px; }
.risk-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
.risk-critical { background: #450a0a; color: #fca5a5; }
.risk-high     { background: #431407; color: #fb923c; }
.risk-medium   { background: #1c1917; color: #fbbf24; }
.risk-low      { background: #052e16; color: #4ade80; }
.zone-chronic  { background: #1e1b4b; color: #a5b4fc; }
.zone-sporadic { background: #164e63; color: #67e8f9; }
.zone-semi     { background: #1c1917; color: #d8b4fe; }
.section-hdr {
    font-size: 1.15rem; font-weight: 600; color: #e5e7eb;
    border-left: 4px solid #6366f1; padding-left: 10px;
    margin: 20px 0 10px;
}
.alert-box {
    background: #450a0a; border: 1px solid #dc2626;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    color: #fca5a5; font-size: 0.88rem;
    box-shadow: 0 4px 6px rgba(220, 38, 38, 0.15);
}
.info-box {
    background: #070d2b; border: 1px solid #3b4fd8;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    color: #a5b4fc; font-size: 0.88rem;
}
.challan-box {
    background: #111827; border: 2px solid #ca8a04;
    border-radius: 12px; padding: 20px; margin: 10px 0;
    box-shadow: 0 4px 15px rgba(202, 138, 4, 0.1);
}
.live-banner {
    background: #450a0a; border-left: 5px solid #ef4444;
    padding: 10px 15px; margin: 8px 0; border-radius: 4px;
    color: #fca5a5; font-family: monospace; font-size: 0.9rem;
}
div[data-testid="stSidebar"] { background: #070913; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA & ML MODEL ──────────────────────────────────
@st.cache_data
def load_data():
    cdf      = pd.read_csv('data/cluster_stats.csv')
    stn      = pd.read_csv('data/station_summary.csv')
    habitual = pd.read_csv('data/habitual_offenders.csv')
    hourly   = pd.read_csv('data/hourly_by_station.csv')
    with open('data/summary.json') as f:
        summary = json.load(f)
    return cdf, stn, habitual, hourly, summary

@st.cache_resource
def load_ml_model():
    model = None
    meta = None
    if os.path.exists('data/predictor_model.pkl') and os.path.exists('data/predictor_metadata.json'):
        with open('data/predictor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/predictor_metadata.json', 'r') as f:
            meta = json.load(f)
    return model, meta

try:
    cdf, stn, habitual, hourly, summary = load_data()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"Data files not found. Run the preprocessing script first. Error: {e}")
    st.stop()

# Ensure cluster_id is int
cdf['cluster_id'] = cdf['cluster_id'].astype(int)

model, model_meta = load_ml_model()
MODEL_LOADED = model is not None

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## <i class='fa-solid fa-square-parking' style='color:#3b82f6;'></i> ParkingIQ", unsafe_allow_html=True)
    st.markdown("*Bengaluru Parking Intelligence*")
    st.caption(f"Data Window: {summary['date_from']} → {summary['date_to']}")
    if MODEL_LOADED:
        st.success("AI Forecast Model Loaded")
    else:
        st.warning("AI Model Not Found (Baseline active)")
    st.divider()

    page = st.radio("Navigate", [
        "Overview",
        "Hotspot Map",
        "Zone Risk Analysis",
        "Peak Hour Forecast",
        "Petrol Pump Simulator",
        "Habitual Offender Engine",
        "Live Feed Replay",
        "Citizen Complaints",
        "Zone Improvement Tracker",
        "SMS Alert Center",
        "Safe Parking Finder",
    ])

    st.divider()
    st.markdown("**Global Filters**")
    risk_min = st.slider("Min Zone Risk Index", 0.0, 10.0, 0.0, 0.5)
    zone_filter = st.multiselect("Zone type",
                                  ['CHRONIC', 'SEMI-CHRONIC', 'SPORADIC'],
                                  default=['CHRONIC', 'SEMI-CHRONIC', 'SPORADIC'])
    station_filter = st.selectbox("Police station",
                                   ['All'] + sorted(stn['police_station'].tolist()))

# Apply filters to clusters
cdf_f = cdf[cdf['zone_risk_index'] >= risk_min]
if zone_filter:
    cdf_f = cdf_f[cdf_f['zone_type'].isin(zone_filter)]
if station_filter != 'All':
    cdf_f = cdf_f[cdf_f['police_station'] == station_filter]

# ── OFFICIAL HEADER ───────────────────────────────────────
st.markdown("""
<div style="background-color: #0d1527; padding: 15px 25px; border-bottom: 3px solid #d97706; border-top: 3px solid #1e3a8a; border-radius: 8px; margin-bottom: 25px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
    <div>
        <div style="font-size: 0.75rem; font-weight: 700; color: #9ca3af; letter-spacing: 2px;">GOVERNMENT OF KARNATAKA | BENGALURU TRAFFIC POLICE</div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; margin-top: 2px; letter-spacing: 0.5px;">
            <i class="fa-solid fa-square-parking" style="color: #3b82f6; margin-right: 8px;"></i>ParkingIQ Portal
        </div>
        <div style="font-size: 0.8rem; color: #fbbf24; font-weight: 500;">Smart Congestion Intelligence & Spatial Enforcement Registry</div>
    </div>
    <div style="text-align: right;">
        <span style="background-color: #1e3a8a; color: #93c5fd; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; border: 1px solid #3b82f6; text-transform: uppercase;">
            <i class="fa-solid fa-shield-halved" style="margin-right: 4px;"></i>Official System
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("### Executive Overview")
    st.markdown(
        "AI-driven system to detect illegal parking hotspots, quantify congestion impact, "
        "and enable targeted enforcement across Bengaluru.")
    st.divider()

    # KPI row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, num, label, sub="", color="#f9fafb"):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:{color}">{num}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-sub">{sub}</div></div>""",
            unsafe_allow_html=True)

    kpi(c1, f"{summary['total_violations']:,}", "Total violations", "Nov 2023 – Apr 2024")
    kpi(c2, f"{summary['total_clusters']:,}", "Hotspot grid cells", "H3 resolution 9 (100m)", "#818cf8")
    kpi(c3, f"{summary['habitual_vehicles']:,}", "30d Watchlist Vehicles", "3+ offenses in 30 days", "#f87171")
    kpi(c4, f"{summary['heavy_vehicle_pct']}%", "Heavy vehicle share", "Lorry/Bus/Tanker/LGV", "#fb923c")
    kpi(c5, f"{summary['junction_pct']}%", "Near junctions", "High-risk intersections", "#34d399")

    st.divider()

    # KPI row 2
    c6, c7, c8, c9 = st.columns(4)
    chronic = int((cdf['zone_type']=='CHRONIC').sum())
    sporadic = int((cdf['zone_type']=='SPORADIC').sum())
    semi = int((cdf['zone_type']=='SEMI-CHRONIC').sum())
    top_risk = cdf['zone_risk_index'].max()
    kpi(c6, chronic, "Chronic zones", "High persistence, low variation", "#c084fc")
    kpi(c7, semi, "Semi-chronic zones", "Moderate persistence", "#fbbf24")
    kpi(c8, sporadic, "Sporadic zones", "Low persistence, bursty", "#67e8f9")
    kpi(c9, f"{top_risk:.1f}/10", "Highest Risk Score (ZRI)", cdf.iloc[0]['police_station'] if len(cdf) > 0 else "Unknown", "#f87171")

    st.divider()

    # Top 10 hotspot table
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<div class="section-hdr"><i class="fa-solid fa-triangle-exclamation" style="color:#ef4444;margin-right:6px;"></i>Top 10 Highest Risk Hotspots (ZRI)</div>',
                   unsafe_allow_html=True)
        top10 = cdf.head(10).copy()
        top10['Risk'] = top10['zone_risk_index'].apply(
            lambda x: f"{'CRITICAL' if x>7 else 'HIGH' if x>5 else 'MEDIUM'} ({x:.2f})")
        top10['Type'] = top10['zone_type']
        top10['Spillover'] = top10['spillover_index'].apply(lambda x: f"{x:.2f}/10")
        top10['Persistence'] = top10['persistence_index'].apply(lambda x: f"{x:.2f}/10")
        display = top10[['cluster_id', 'police_station', 'violation_count', 'Risk', 'Type',
                          'Spillover', 'Persistence', 'heavy_pct', 'repeat_pct']].copy()
        display.columns = ['Zone ID', 'Station', 'Violations', 'Risk (ZRI)', 'Zone Type',
                           'Spillover Idx', 'Persistence Idx', 'Heavy %', 'Repeat %']
        st.dataframe(display, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-bar" style="color:#6366f1;margin-right:6px;"></i>Violation Breakdown</div>',
                   unsafe_allow_html=True)
        vb = summary.get('violation_breakdown', {})
        vb_df = pd.DataFrame(list(vb.items())[:8], columns=['Violation', 'Count'])
        fig = px.bar(vb_df, x='Count', y='Violation', orientation='h',
                    color='Count', color_continuous_scale='Reds',
                    template='plotly_dark')
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                         paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)',
                         showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Violation type pie
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-pie" style="color:#6366f1;margin-right:6px;"></i>Violation Type Distribution</div>', unsafe_allow_html=True)
    vb_top = pd.DataFrame(list(vb.items())[:6], columns=['type', 'count'])
    fig2 = px.pie(vb_top, values='count', names='type', hole=0.45,
                 template='plotly_dark',
                 color_discrete_sequence=px.colors.sequential.Plasma_r)
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — HOTSPOT MAP
# ═══════════════════════════════════════════════════════════
elif page == "Hotspot Map":
    st.markdown("### Spatial Hotspot Map")
    st.divider()

    try:
        import folium
        from streamlit_folium import st_folium

        col_ctrl, col_map = st.columns([1, 3])
        with col_ctrl:
            map_style = st.selectbox("Map style", ["Grid marker map", "Heatmap"])
            color_by  = st.selectbox("Color bubbles by",
                                      ["zone_risk_index", "spillover_index",
                                       "persistence_index", "repeat_pct", "heavy_pct"])
            n_zones   = st.slider("Show top N active zones", 10, len(cdf_f), min(100, len(cdf_f)))
            show_burst= st.checkbox("Highlight burst alerts", True)

            st.divider()
            st.markdown("**Map Legend**")
            st.markdown("""
            * <span style="color:#ef4444;">●</span> Critical (ZRI > 7)
            * <span style="color:#f97316;">●</span> High (5-7)
            * <span style="color:#eab308;">●</span> Medium (3-5)
            * <span style="color:#22c55e;">●</span> Low (<3)
            """, unsafe_allow_html=True)

        plot_df = cdf_f.head(n_zones)
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                      tiles='CartoDB dark_matter')

        if map_style == "Heatmap":
            from folium.plugins import HeatMap
            heat_data = [[r['lat'], r['lon'],
                          r[color_by] if pd.notna(r[color_by]) else 0]
                         for _, r in plot_df.iterrows()]
            HeatMap(heat_data, radius=25, blur=18).add_to(m)
        else:
            for _, row in plot_df.iterrows():
                zri = row['zone_risk_index']
                color = '#ef4444' if zri > 7 else \
                        '#f97316' if zri > 5 else \
                        '#eab308' if zri > 3 else '#22c55e'
                radius = max(6, int(zri * 2.2))
                
                # Check top hours parsing
                th = row['top_hours']
                try:
                    th_list = json.loads(th)
                    th_str = ", ".join([f"{h:02d}:00" for h in th_list])
                except:
                    th_str = str(th)
                    
                popup_html = f"""
                <div style='font-family:sans-serif;min-width:200px;background-color:#1e293b;color:#f9fafb;padding:8px;border-radius:6px;'>
                <b style='color:#6366f1;font-size:1.05rem;'>Zone {row['cluster_id']} — {row['police_station']}</b><br><br>
                <b>H3 Index:</b> <code style='background:#0f172a;padding:2px 4px;'>{row['h3_cell']}</code><br>
                <b>Violations:</b> {row['violation_count']:,}<br>
                <b>Zone Risk Index:</b> {row['zone_risk_index']:.2f}/10<br>
                <b>Spillover Index:</b> {row['spillover_index']:.2f}/10<br>
                <b>Persistence Index:</b> {row['persistence_index']:.2f}/10<br>
                <b>Zone Type:</b> {row['zone_type']}<br>
                <b>Primary Violation:</b> {row['primary_violation']}<br>
                <b>Heavy Vehicle %:</b> {row['heavy_pct']}%<br>
                <b>Repeat Offender %:</b> {row['repeat_pct']}%<br>
                <b>Peak Hours:</b> {th_str}<br>
                <b>Location Sample:</b> <i>{row['location_sample']}</i>
                </div>"""
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius, color=color,
                    fill=True, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Zone {row['cluster_id']} ({row['police_station']}) | ZRI: {row['zone_risk_index']:.1f}"
                ).add_to(m)

                if show_burst and row.get('burst_days', 0) > 3:
                    folium.Marker(
                        location=[row['lat'] + 0.0006, row['lon'] + 0.0006],
                        icon=folium.DivIcon(html=f'<div style="color:#ef4444;font-size:14px;font-weight:bold;"><i class="fa-solid fa-bolt"></i></div>')
                    ).add_to(m)

        with col_map:
            st_folium(m, width=900, height=550)

    except Exception as e:
        st.warning(f"Using Plotly fallback map (Folium failed: {e})")
        fig = px.scatter_mapbox(
            cdf_f.head(200),
            lat='lat', lon='lon',
            size='violation_count',
            color='zone_risk_index',
            color_continuous_scale='RdYlGn_r',
            hover_name='police_station',
            hover_data=['cluster_id', 'zone_risk_index', 'spillover_index', 'persistence_index', 'zone_type'],
            mapbox_style='carto-darkmatter',
            zoom=11, center={'lat': 12.9716, 'lon': 77.5946},
            size_max=25, template='plotly_dark',
            title='Violation Hotspot Map — Bengaluru'
        )
        fig.update_layout(height=580, margin=dict(l=0, r=0, t=40, b=0),
                         paper_bgcolor='rgba(0,0,0,0)',
                         coloraxis_colorbar=dict(title='Risk Index'))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 3 — ZONE RISK ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "Zone Risk Analysis":
    st.markdown("### Spatial Zone Risk Analysis")
    st.divider()

    # Index explainer
    with st.expander("Spatial Grid Indices & Risk Methodology Details", expanded=False):
        st.markdown("""
        ### Index Formulation
        
        1. **Zone Risk Index (ZRI)** — composite rating out of 10:
           - **Density (30% weight)**: Log-normalized density of violations within H3 hexagon.
           - **Repeat Offender Ratio (20% weight)**: Share of repeat offender plates (3+ tickets).
           - **Heavy Vehicle Ratio (15% weight)**: Percentage share of cargo trucks, buses, tractors, and mixers.
           - **Spillover Index (15% weight)**: Geographic dispersion index based on standard deviation of violation distances from centroid.
           - **Persistence Index (10% weight)**: Chronological occurrence rate (active violation days divided by total days in dataset).
           - **Junction Proximity (10% weight)**: Ratio of offenses marked within 150m of a junction.
           
        2. **Spillover Index (0-10)**: Measures the spatial spread of violations. A high Spillover Index indicates illegal parking chokes secondary arterial roads and lanes branching off the main junction hotspot.
        
        3. **Persistence Index (0-10)**: Measures the temporal consistency of violations.
           - **Chronic (Purple)**: Consistent day-in, day-out violations (High Persistence Index + Low coefficient of variation). Requires permanent structural intervention (CCTV, concrete barriers, pedestrian signs).
           - **Sporadic (Cyan)**: Highly inconsistent, event-based violations (Low Persistence Index + High variance). Responsive to tactical patrol windows and active towing.
           - **Semi-Chronic (Orange)**: Moderate persistence.
        """)

    # Scatter: ZRI vs Spillover vs Persistence
    st.markdown('<div class="section-hdr">Correlation: Spillover Index vs. Persistence Index vs. Risk (ZRI)</div>',
               unsafe_allow_html=True)
    fig = px.scatter(
        cdf_f, x='spillover_index', y='persistence_index',
        size='violation_count', color='zone_risk_index',
        color_continuous_scale='RdYlGn_r',
        hover_name='police_station',
        hover_data=['cluster_id', 'zone_risk_index', 'zone_type', 'heavy_pct', 'repeat_pct'],
        template='plotly_dark',
        labels={'spillover_index': 'Spillover Index (0-10)', 'persistence_index': 'Persistence Index (0-10)'},
    )
    fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(20,20,30,0.8)')
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-hdr">Spatial Zone Type Composition</div>',
                   unsafe_allow_html=True)
        zt = cdf_f['zone_type'].value_counts().reset_index()
        zt.columns = ['Type', 'Count']
        fig2 = px.pie(zt, values='Count', names='Type', hole=0.4,
                     color='Type',
                     color_discrete_map={
                         'CHRONIC': '#818cf8', 'SEMI-CHRONIC': '#c084fc', 'SPORADIC': '#67e8f9'},
                     template='plotly_dark')
        fig2.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-hdr">ZRI Density Distribution</div>',
                   unsafe_allow_html=True)
        fig3 = px.histogram(cdf_f, x='zone_risk_index', nbins=20,
                           color_discrete_sequence=['#6366f1'],
                           template='plotly_dark')
        fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title="Zone Risk Index (0-10)", yaxis_title="Number of Zones")
        st.plotly_chart(fig3, use_container_width=True)

    # Police station comparison
    st.markdown('<div class="section-hdr">Top Police Jurisdictions Comparison</div>',
               unsafe_allow_html=True)
    stn_plot = stn.head(12).copy()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Violations (00s)', x=stn_plot['police_station'],
                         y=stn_plot['count']//100, marker_color='#6366f1'))
    fig4.add_trace(go.Bar(name='Repeat Offender %', x=stn_plot['police_station'],
                         y=stn_plot['repeat_pct'], marker_color='#f87171'))
    fig4.add_trace(go.Bar(name='Heavy Vehicle %', x=stn_plot['police_station'],
                         y=stn_plot['heavy_pct'], marker_color='#fb923c'))
    fig4.update_layout(barmode='group', template='plotly_dark', height=340,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_tickangle=30, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig4, use_container_width=True)

    # Full table
    st.markdown('<div class="section-hdr">Hotspot Zone Index Registry</div>',
               unsafe_allow_html=True)
    show_cols = ['cluster_id', 'h3_cell', 'police_station', 'violation_count', 'zone_risk_index', 'zone_type',
                 'spillover_index', 'persistence_index', 'heavy_pct', 'repeat_pct',
                 'junction_pct', 'active_days', 'burst_days', 'primary_violation']
    show_cols = [c for c in show_cols if c in cdf_f.columns]
    st.dataframe(cdf_f[show_cols].reset_index(drop=True),
                use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# PAGE 4 — PEAK HOUR FORECAST (ML PREDICTION)
# ═══════════════════════════════════════════════════════════
elif page == "Peak Hour Forecast":
    st.markdown("### AI-Driven Peak Hour Forecast & Patrol Optimization")
    st.markdown(
        "Forecasts future illegal parking volumes using a trained **Random Forest Regressor** "
        "machine learning model based on spatial attributes, spillover, persistence, and weekday trends.")
    st.divider()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("### Forecast Settings")
        # Station Selector
        sel_station = st.selectbox("Select target police station",
                                    sorted(cdf['police_station'].unique()))
        
        # Day of week Selector
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sel_day = st.selectbox("Select Forecast Day", day_names, index=datetime.now().weekday())
        sel_day_idx = day_names.index(sel_day)
        
        # Officer Deployment Simulator
        deploy_officers = st.slider("Patrol Officers to Deploy", 1, 20, 6)
        
        st.divider()
        if MODEL_LOADED:
            st.info("**Model Architecture:**\nRandomForestRegressor(n_estimators=50)\n\n**Inputs:** [zone_id, dayofweek, hour, spillover_index, persistence_index, latitude, longitude]")
        else:
            st.warning("**Baseline Active:** Using historical average counts (Predictive model not loaded).")

    with col_r:
        st.markdown(f"### Forecast Results for {sel_station} ({sel_day})")
        
        # Gather all H3 zones for the selected police station
        station_zones = cdf[cdf['police_station'] == sel_station]
        
        if len(station_zones) > 0:
            # Generate 24x7 forecast using RandomForest if loaded, else fall back to historical counts
            hours = list(range(24))
            preds = np.zeros(24)
            baseline = np.zeros(24)
            
            # Predict counts for each hour
            for _, zone in station_zones.iterrows():
                # Get historical average for this zone if hourly data is available
                zone_hourly = hourly[(hourly['police_station'] == sel_station)]
                if len(zone_hourly) > 0:
                    for h in hours:
                        h_count = zone_hourly[zone_hourly['hour'] == h]['count'].values
                        baseline[h] += h_count[0] / len(station_zones) if len(h_count) > 0 else 1.0
                else:
                    # Synthetic baseline if station hourly missing
                    for h in hours:
                        baseline[h] += (zone['violation_count'] / 24.0) * (1.2 if h in [10, 11, 17, 18] else 0.8)
                
                # Model Prediction
                if MODEL_LOADED:
                    input_rows = []
                    for h in hours:
                        input_rows.append({
                            'zone_id': int(zone['cluster_id']),
                            'dayofweek': int(sel_day_idx),
                            'hour': int(h),
                            'spillover_index': float(zone['spillover_index']),
                            'persistence_index': float(zone['persistence_index']),
                            'lat': float(zone['lat']),
                            'lon': float(zone['lon'])
                        })
                    pred_df = pd.DataFrame(input_rows)
                    # Align column order
                    pred_df = pred_df[model_meta['features']]
                    zone_preds = model.predict(pred_df)
                    preds += zone_preds
                else:
                    preds = baseline.copy()

            # Format forecast graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=baseline, name='Historical Baseline',
                                     line=dict(color='#9ca3af', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=hours, y=preds, name='AI Forecast (RandomForest)',
                                     line=dict(color='#6366f1', width=3), fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.15)'))
            
            peak_val = preds.max()
            peak_hr = hours[np.argmax(preds)]
            fig.add_vline(x=peak_hr, line_dash='dash', line_color='#f87171',
                         annotation_text=f"Predicted Peak: {peak_hr:02d}:00 ({peak_val:.1f} viol.)",
                         annotation_font_color='#f87171')
            
            fig.update_layout(
                height=350, template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Hour of Day", tickmode='linear', tick0=0, dtick=2),
                yaxis=dict(title="Estimated Violation Rate"),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recommend Patrol Windows based on top 3 predicted peak hours
            st.markdown('<div class="section-hdr"><i class="fa-solid fa-shield-halved" style="color:#fbbf24;margin-right:6px;"></i>Recommended Patrol Windows</div>',
                       unsafe_allow_html=True)
            
            # Sort hours by predictions
            top_hours_idx = np.argsort(preds)[::-1][:3]
            
            c_p1, c_p2, c_p3 = st.columns(3)
            for i, (col, hr) in enumerate(zip([c_p1, c_p2, c_p3], top_hours_idx)):
                predicted_viol = preds[hr]
                # Simulating impact: N officers reduces violations by an estimated factor (e.g. 8% per officer, max 85%)
                reduction_pct = min(85, int(deploy_officers * 8.0))
                reduced_viol = max(0.0, predicted_viol * (1 - reduction_pct/100.0))
                
                col.markdown(f"""<div class="metric-card">
                    <div style="color:#fbbf24;font-size:1.4rem;font-weight:700">
                    {hr:02d}:00–{(hr+1)%24:02d}:00</div>
                    <div style="color:#9ca3af;font-size:0.8rem">Patrol Window {i+1}</div>
                    <div style="color:#f9fafb;margin-top:6px;font-size:0.9rem">Est. Volume: <b>{predicted_viol:.1f}</b> violations</div>
                    <div style="color:#4ade80;font-size:0.78rem;margin-top:4px">
                    Est. reduction: <b>-{reduction_pct}%</b> ({predicted_viol - reduced_viol:.1f} saved)</div>
                </div>""", unsafe_allow_html=True)

            # Predictive 24x7 heatmap
            st.markdown('<div class="section-hdr"><i class="fa-solid fa-calendar-days" style="color:#6366f1;margin-right:6px;"></i>Predicted 24×7 Spatial Traffic Risk Heatmap</div>',
                       unsafe_allow_html=True)
            
            # Generate predictions for all 7 days for this station
            heatmap_grid = np.zeros((7, 24))
            for d in range(7):
                for _, zone in station_zones.iterrows():
                    if MODEL_LOADED:
                        rows = []
                        for h in range(24):
                            rows.append({
                                'zone_id': int(zone['cluster_id']),
                                'dayofweek': int(d),
                                'hour': int(h),
                                'spillover_index': float(zone['spillover_index']),
                                'persistence_index': float(zone['persistence_index']),
                                'lat': float(zone['lat']),
                                'lon': float(zone['lon'])
                            })
                        pdf = pd.DataFrame(rows)
                        pdf = pdf[model_meta['features']]
                        heatmap_grid[d] += model.predict(pdf)
                    else:
                        # Baseline fallback
                        heatmap_grid[d] = baseline * (1.1 if d >= 4 else 0.9)
            
            fig_hm = go.Figure(go.Heatmap(
                z=heatmap_grid, x=list(range(24)), y=day_names,
                colorscale='RdYlGn_r',
                hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Predicted Violations: %{z:.1f}<extra></extra>'
            ))
            fig_hm.update_layout(
                height=260, template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Hour of Day', yaxis_title='Day of Week',
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Officer deployment simulator
            st.markdown('<div class="section-hdr"><i class="fa-solid fa-user-shield" style="color:#6366f1;margin-right:6px;"></i>Patrol Officer Deployment Simulator</div>',
                       unsafe_allow_html=True)
            total_viol_forecast = preds.sum()
            reduction_factor = min(0.85, (deploy_officers * 0.075)) # 7.5% per officer, max 85%
            saved_viol = total_viol_forecast * reduction_factor
            remaining_viol = total_viol_forecast - saved_viol
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Predicted Violations (Daily)", f"{total_viol_forecast:.1f}")
            c_s2.metric("Enforcement Impact", f"-{saved_viol:.1f}",
                      delta=f"-{reduction_factor * 100:.1f}%",
                      delta_color="inverse")
            c_s3.metric("Residual Traffic Risk", f"{remaining_viol:.1f}")
            
            st.progress(remaining_viol / max(total_viol_forecast, 1.0),
                       text=f"Residual violation threat level after deploying {deploy_officers} officers")
        else:
            st.info("No H3 grid hotspots mapped under this police station.")

# ═══════════════════════════════════════════════════════════
# PAGE 5 — PETROL PUMP SIMULATOR (TRAFFIC ENGINEERING LOGIC)
# ═══════════════════════════════════════════════════════════
elif page == "Petrol Pump Simulator":
    st.markdown("### Petrol Pump Impact Simulator")
    st.markdown(
        "Simulates the placement of designated roadside service facilities (petrol pumps / refueling areas) "
        "and measures the reduction in illegal parking and spillover congestion.")
    st.divider()

    st.markdown('<div class="info-box"><i class="fa-solid fa-lightbulb" style="color:#fbbf24;margin-right:6px;"></i><b>Traffic Engineering Logic:</b> '
               'A large percentage of illegal parking on primary lanes consists of short-stop violations '
               '(checking maps, brief rest, meeting points, buying quick snacks). When there are no '
               'designated, off-street bays available, drivers stop in the active carriage lane. '
               'Adding a modern petrol pump with an integrated lay-by within 150m absorbs this '
               'short-stop parking demand. Traffic models predict a <b>10% to 15% reduction</b> in violations '
               'and a substantial decrease in the <b>Spillover Index</b> by pulling vehicles off the active roadway.</div>',
               unsafe_allow_html=True)

    col_sim1, col_sim2 = st.columns([1, 2])
    with col_sim1:
        st.markdown("### Simulation Inputs")
        # Select target zone
        sel_zone_id = st.selectbox(
            "Select hotspot zone",
            cdf_f.head(30)['cluster_id'].tolist(),
            format_func=lambda x: f"Zone {x} — {cdf_f[cdf_f['cluster_id']==x]['police_station'].values[0]}"
                                   if len(cdf_f[cdf_f['cluster_id']==x])>0 else f"Zone {x}"
        )
        
        pump_reduction_pct = st.slider(
            "Designated Demand Absorption % (per pump)", 8, 20, 12)
        n_pumps = st.radio("Number of Petrol Pumps Added to Corridor", [1, 2, 3])

    zone = cdf_f[cdf_f['cluster_id'] == sel_zone_id]
    if len(zone) > 0:
        zone = zone.iloc[0]
        base_violations = int(zone['violation_count'])
        base_zri        = float(zone['zone_risk_index'])
        base_spillover  = float(zone['spillover_index'])

        # Calculate compounded reduction: 1 - (1 - red)^n
        reduction_rate = pump_reduction_pct / 100.0
        total_reduction_pct = 1.0 - (1.0 - reduction_rate) ** n_pumps
        
        new_violations = int(base_violations * (1.0 - total_reduction_pct))
        
        # Recalculate ZRI: ZRI scales down with reduction in violations and spillover
        new_spillover = round(max(0.5, base_spillover * (1.0 - total_reduction_pct * 0.7)), 2)
        new_zri = round(max(1.0, base_zri * (1.0 - total_reduction_pct * 0.8)), 2)
        violations_saved = base_violations - new_violations

        with col_sim2:
            st.markdown(f"### Zone {sel_zone_id} — {zone['police_station']} corridor")
            st.markdown(f"**Primary violation:** {zone['primary_violation']}  |  "
                       f"**Zone type:** {zone['zone_type']} | **H3 Hex:** `{zone['h3_cell']}`")

            c1, c2, c3 = st.columns(3)
            c1.metric("Violations", f"{base_violations:,}", f"→ {new_violations:,} (-{total_reduction_pct*100:.1f}%)",
                      delta_color="inverse")
            c2.metric("Zone Risk Index (ZRI)", f"{base_zri:.2f}",
                      f"→ {new_zri:.2f} ({(new_zri-base_zri):.2f})",
                      delta_color="inverse")
            c3.metric("Spillover Index", f"{base_spillover:.2f}",
                      f"→ {new_spillover:.2f} ({(new_spillover-base_spillover):.2f})", delta_color="inverse")

            st.divider()

            # Before/after bar chart
            compare_df = pd.DataFrame({
                'Metric': ['Violations', 'ZRI (scaled ×50)', 'Spillover (scaled ×100)'],
                'Before': [base_violations, base_zri*50, base_spillover*100],
                'After':  [new_violations,  new_zri*50,  new_spillover*100],
            })
            fig = px.bar(compare_df, x='Metric', y=['Before', 'After'],
                        barmode='group', template='plotly_dark',
                        color_discrete_map={'Before': '#f87171', 'After': '#4ade80'})
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Comparative Metrics")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="metric-card" style="text-align:left;padding:18px">
            <b style="color:#4ade80"><i class="fa-solid fa-chart-line" style="margin-right:6px;"></i>Traffic Engineering Impact Summary — {n_pumps} Petrol Pump(s)</b><br><br>
            • Roadside short-stop violations reduced by <b>{violations_saved:,}</b> 
              ({total_reduction_pct*100:.1f}% demand absorption)<br>
            • Corridor Risk rating drops from <b>{base_zri}</b> → <b>{new_zri}</b><br>
            • Spillover Index drops from <b>{base_spillover}</b> → <b>{new_spillover}</b><br>
            • Average travel speed increase along corridor: <b>~{int(total_reduction_pct * 35)}%</b> due to carriageway clearing.<br>
            • Projected yearly reduction: <b>{violations_saved*12:,}</b> parking tickets.
            </div>""", unsafe_allow_html=True)

    # All zones petrol pump impact table
    st.divider()
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-table-list" style="color:#6366f1;margin-right:6px;"></i>Projected Corridor Impact Matrix — All Zones</div>',
               unsafe_allow_html=True)
    pump_table = cdf_f[['police_station', 'violation_count', 'zone_risk_index',
                         'spillover_index', 'petrol_pump_reduction',
                         'petrol_pump_new_risk']].copy().head(15)
    pump_table.columns = ['Station', 'Base Violations', 'Base ZRI', 'Spillover',
                          'Pumps Saved (Est.)', 'New ZRI after Facility']
    st.dataframe(pump_table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# PAGE 6 — HABITUAL OFFENDER ENGINE (WATCHLIST & CHALLAN)
# ═══════════════════════════════════════════════════════════
elif page == "Habitual Offender Engine":
    st.markdown("### Habitual Offender Challan Engine")
    st.markdown(
        "Identifies chronic traffic violators using vehicle registration histories. "
        "Flags vehicles in the 30-day temporal watchlist and generates escalating fine schedules.")
    st.divider()

    # Calculate statistics based on watchlist
    c1, c2, c3, c4 = st.columns(4)
    total_offenders = len(habitual)
    high_risk_off = len(habitual[habitual['offense_count'] >= 5])
    critical_off = len(habitual[habitual['offense_count'] >= 10])
    max_count = habitual['offense_count'].max()
    
    c1.metric("Repeat Violators (3+)", f"{total_offenders:,}", "Total database")
    c2.metric("Watchlist Vehicles (30d)", f"{summary['habitual_vehicles']:,}", "3+ violations in 30 days")
    c3.metric("Critical Offenders (10+)", f"{critical_off:,}", "Suspension watchlist")
    c4.metric("Highest Individual Count", f"{max_count} times", "Plate: " + str(habitual.iloc[0]['vehicle_number']))

    st.divider()

    # Escalating penalty logic
    with st.expander("Karnataka Motor Vehicles Slabs - Escalating Penalties (Illustrative)", expanded=True):
        st.markdown("""
        | Offense Count | Tier | Fine Amount | Judicial & Enforcement Action |
        |--------------|------|-------------|------------------------------|
        | 1–2 | BASE | ₹500 | Standard digital ticket |
        | 3–5 | HIGH | ₹1,500 | Watchlist flag + Auto Warning Letter |
        | 6–9 | MAXIMUM | ₹3,000 | Vehicle registration lock notice |
        | 10+ | CRITICAL | ₹5,000 | Court summons + Immobilisation notice |
        """)

    # Search vehicle
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-magnifying-glass" style="color:#6366f1;margin-right:6px;"></i>Active Registry Lookup</div>',
               unsafe_allow_html=True)
    col_s, col_btn = st.columns([3, 1])
    with col_s:
        search_plate = st.text_input("Enter Vehicle Registration Number", "",
                                     placeholder="e.g. KA03MX8821")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Verify Plate Registry", type="primary")

    if search_btn or search_plate:
        # Exact or partial match search
        match = habitual[habitual['vehicle_number'].str.contains(
            search_plate.upper().strip(), na=False)] if search_plate else pd.DataFrame()
            
        if len(match) > 0:
            row = match.iloc[0]
            tier_color = {'BASE': '#4ade80', 'HIGH': '#fbbf24',
                          'MAXIMUM': '#fb923c', 'CRITICAL': '#f87171'}.get(
                row['fine_tier'], '#9ca3af')
                
            is_watchlist = row.get('is_watchlist_30d', 0) == 1
            
            if is_watchlist:
                st.markdown(f"""
                <div class="alert-box">
                <b>30-DAY TEMPORAL WATCHLIST ALERT</b> — Vehicle {row['vehicle_number']}
                is flagged as a high-frequency offender with 3+ violations within a rolling 30-day window!
                </div>""", unsafe_allow_html=True)
            else:
                st.warning(f"Repeat Offender Registered — Vehicle {row['vehicle_number']} has {row['offense_count']} total infractions.")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"""<div class="challan-box">
                <b style="color:#ca8a04;font-size:1.15rem"><i class="fa-solid fa-file-invoice-dollar" style="margin-right:6px;"></i>OFFICIAL E-CHALLAN SUMMONS</b><br><br>
                <b>Vehicle No:</b> <span style="font-family:monospace;background:#374151;padding:2px 6px;border-radius:4px;">{row['vehicle_number']}</span><br>
                <b>Vehicle Category:</b> {row['vehicle_type']}<br>
                <b>Total Infractions:</b> {row['offense_count']}<br>
                <b>Primary Offense:</b> {row['primary_violation']}<br>
                <b>Enforcing Jurisdictions:</b> {row['police_stations']}<br>
                <b>First Observed Date:</b> {row['first_offense']}<br>
                <b>Last Observed Date:</b> {row['last_offense']}<br>
                <b>Summons Penalty Tier:</b> <span style="color:{tier_color};font-weight:bold;">{row['fine_tier']}</span><br>
                <b>Escalated Fine Due:</b> <span style="font-size:1.1rem;color:#f3f4f6;font-weight:bold;">₹{row['illustrative_fine']:,}</span><br>
                <b>Status:</b> READY TO ISSUE (Digital Signature Pending)
                </div>""", unsafe_allow_html=True)
                
                # Action Buttons
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("Issue Digital Challan", use_container_width=True):
                    st.toast(f"Challan issued to {row['vehicle_number']} for ₹{row['illustrative_fine']:,}!")
                if c_btn2.button("Dispatch Warning Notice", use_container_width=True):
                    st.toast(f"Warning dispatched via SMS to registered owner of {row['vehicle_number']}.")
                    
            with col_d2:
                fine = row['illustrative_fine']
                st.markdown(f"""<div class="metric-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div class="metric-label" style="font-size:0.9rem">Current Penalty Slabs</div>
                <div class="metric-number" style="color:{tier_color};font-size:2.4rem;margin:10px 0;">{row['fine_tier']}</div>
                <div class="metric-label">Escalated Fine</div>
                <div class="metric-number" style="color:#fbbf24;font-size:2.8rem;margin:10px 0;">₹{fine:,}</div>
                <div style="color:#9ca3af;font-size:0.75rem;padding:0 20px;">
                Offense #{row['offense_count']} registered for this plate. Fine escalated automatically under the Repeat Offender Enforcement Act.
                </div></div>""", unsafe_allow_html=True)
        elif search_plate:
            st.success(f"Vehicle '{search_plate.upper().strip()}' — Clear. No prior offenses registered in this window.")

    # Watchlist table
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-list-check" style="color:#6366f1;margin-right:6px;"></i>Habitual Offenders Registry Watchlist</div>',
               unsafe_allow_html=True)
    tier_filter = st.multiselect("Filter Registry by Tier",
                                  ['BASE', 'HIGH', 'MAXIMUM', 'CRITICAL'],
                                  default=['HIGH', 'MAXIMUM', 'CRITICAL'])
    
    hab_show = habitual[habitual['fine_tier'].isin(tier_filter)]
    st.dataframe(hab_show[['vehicle_number', 'vehicle_type', 'offense_count',
                            'primary_violation', 'fine_tier', 'illustrative_fine',
                            'is_watchlist_30d', 'first_offense', 'last_offense', 'police_stations']],
                use_container_width=True, hide_index=True)

    # Charts
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-line" style="color:#6366f1;margin-right:6px;"></i>Offense Frequency Distribution</div>',
                   unsafe_allow_html=True)
        fig = px.histogram(habitual, x='offense_count', nbins=30,
                           color_discrete_sequence=['#f87171'],
                           template='plotly_dark',
                           labels={'offense_count': 'Violations Count'})
        fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col_h2:
        st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-pie" style="color:#6366f1;margin-right:6px;"></i>Vehicle Types of Habitual Offenders</div>',
                   unsafe_allow_html=True)
        vt_counts = habitual['vehicle_type'].value_counts().head(8).reset_index()
        vt_counts.columns = ['type', 'count']
        fig2 = px.pie(vt_counts, values='count', names='type', hole=0.4,
                     template='plotly_dark',
                     color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig2.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Generate challan PDF stub
    st.divider()
    if st.button("Generate Batch Summon List (Top 200 Offenders)", type="secondary"):
        report_df = habitual.nlargest(200, 'offense_count')
        csv = report_df.to_csv(index=False)
        st.download_button(
            "Download Summon List CSV",
            data=csv,
            file_name=f"batch_summons_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ═══════════════════════════════════════════════════════════
# PAGE 7 — LIVE FEED REPLAY (PRESENT-MOMENT MONITOR)
# ═══════════════════════════════════════════════════════════
elif page == "Live Feed Replay":
    st.markdown("### Real-Time Surveillance & Incident Feed")
    st.markdown(
        "Replays historical infraction records as a live monitoring feed. "
        "Displays current active hotspots, triggers temporal burst alerts, and intercepts watchlist vehicles.")
    st.divider()

    # Settings
    sim_hour = st.slider("Simulate Current Hour of Day", 0, 23, datetime.now().hour)
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        show_bursts = st.checkbox("Enable Surge & Burst Alerts", True)
    with col_ctrl2:
        intercept_watchlist = st.checkbox("Intercept Watchlist Plates (30d)", True)

    # Calculate active hotspots at simulated hour
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-satellite-dish" style="color:#ef4444;margin-right:6px;"></i>Active Hotspots (Simulated Hour)</div>',
               unsafe_allow_html=True)
    
    active_zones = []
    for _, zone in cdf_f.head(20).iterrows():
        th = zone.get('top_hours')
        try:
            top_hours_list = json.loads(th)
        except:
            top_hours_list = [10, 11, 17, 18]
            
        activity_level = 'HIGH' if sim_hour in top_hours_list else \
                         'MEDIUM' if any(abs(sim_hour - th_h) <= 2 for th_h in top_hours_list) else 'LOW'
        
        active_zones.append({**zone, 'current_activity': activity_level})

    act_df = pd.DataFrame(active_zones)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    high_now = len(act_df[act_df['current_activity']=='HIGH']) if len(act_df)>0 else 0
    med_now  = len(act_df[act_df['current_activity']=='MEDIUM']) if len(act_df)>0 else 0
    
    c_m1.metric("High-Activity Hotspots", high_now)
    c_m2.metric("Medium-Activity Hotspots", med_now)
    c_m3.metric("Simulated System Time", f"{sim_hour:02d}:00")

    # Display active cards
    if len(act_df) > 0:
        for _, zone in act_df[act_df['current_activity'].isin(['HIGH', 'MEDIUM'])].head(6).iterrows():
            color = '#ef4444' if zone['current_activity']=='HIGH' else '#f97316'
            
            repeat_flag = "WATCHLIST ALERT" if zone['repeat_pct'] > 25.0 else ""
            burst_flag  = "BURST ALERT" if zone['burst_days'] > 4 else ""
            
            st.markdown(f"""
            <div style="border: 1px solid {color}; border-radius: 8px; padding: 12px 16px;
                        margin: 6px 0; background: rgba(30,30,40,0.85); box-shadow: 0 4px 6px rgba(0,0,0,0.2)">
            <span style="font-weight: 600; color: {color}">{zone['police_station']} corridor</span>
            &nbsp;&nbsp;
            <span style="color: #9ca3af; font-size: 0.8rem">ZRI: {zone['zone_risk_index']:.1f} | 
            Spillover Index: {zone['spillover_index']:.1f} | 
            Persistence: {zone['persistence_index']:.1f}</span>
            &nbsp;&nbsp;
            <span style="color: #fbbf24; font-size: 0.78rem; font-weight: 600;">{repeat_flag} {burst_flag}</span><br>
            <span style="color: #d1d5db; font-size: 0.8rem">
            H3 Grid ID: <code style="background:#0f172a;padding:1px 3px;">{zone['h3_cell']}</code> | Total Violations: {zone['violation_count']:,} | 
            Heavy Vehicles: {zone['heavy_pct']}% | Zone Type: {zone['zone_type']} | Primary: {zone['primary_violation']}</span>
            </div>""", unsafe_allow_html=True)
            
    # Watchlist alert popup simulation
    if intercept_watchlist:
        # Sample watchlist plates active in this jurisdiction
        watchlist_samples = habitual[habitual['is_watchlist_30d'] == 1].head(15)
        if len(watchlist_samples) > 0 and random.random() < 0.85:
            alert_row = watchlist_samples.sample(1).iloc[0]
            st.markdown(f"""
            <div class="alert-box">
            <b>REAL-TIME WATCHLIST INTERCEPT:</b> Vehicle <span style="font-family:monospace;background:#7f1d1d;padding:1px 4px;border-radius:3px;">{alert_row['vehicle_number']}</span> 
            (Watchlisted Offender, {alert_row['offense_count']} prior infractions) was just flagged inside the corridor! 
            <br><i>Automated ticket issued under Escalated Penalty Tier: {alert_row['fine_tier']} (Fine: ₹{alert_row['illustrative_fine']:,}).</i>
            </div>""", unsafe_allow_html=True)

    # Live feed violation ticker
    st.divider()
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-tower-broadcast" style="color:#6366f1;margin-right:6px;"></i>Live Digital Intercept Ticker</div>',
               unsafe_allow_html=True)

    # Generate ticker streams
    ticker_stations = sorted(cdf['police_station'].unique())[:10]
    violation_types = ['WRONG PARKING', 'NO PARKING', 'PARKING IN A MAIN ROAD',
                       'PARKING ON FOOTPATH', 'DOUBLE PARKING', 'OBSTRUCTING CARRIAGEWAY']
    vehicle_types   = ['SCOOTER', 'CAR', 'MOTOR CYCLE', 'PASSENGER AUTO', 'MAXI-CAB', 'LGV', 'BUS']

    ticker_data = []
    random.seed(sim_hour + int(datetime.now().minute))
    
    for i in range(12):
        t = datetime.now() - timedelta(minutes=random.randint(0, 45))
        
        # Determine if habitual offender plate is picked
        is_hab_pick = random.random() < 0.12
        if is_hab_pick:
            plate_row = habitual.sample(1).iloc[0]
            plate = plate_row['vehicle_number']
            v_type = plate_row['vehicle_type']
            v_viol = plate_row['primary_violation']
            flag = 'WATCHLIST (' + plate_row['fine_tier'] + ')'
        else:
            # Generate generic plates
            plate = f"KA0{random.randint(1,9)}{chr(random.randint(65,90))}{chr(random.randint(65,90))}{random.randint(1000,9999)}"
            v_type = random.choice(vehicle_types)
            v_viol = random.choice(violation_types)
            flag = 'BURST' if random.random() < 0.08 else 'COMPLIANT'
            
        ticker_data.append({
            'Time': t.strftime('%H:%M:%S'),
            'Police Station': random.choice(ticker_stations),
            'Plate Number': plate,
            'Vehicle Type': v_type,
            'Violation': v_viol,
            'Status': flag
        })

    ticker_df = pd.DataFrame(ticker_data).sort_values('Time', ascending=False)
    
    # Custom colors inside streamlit dataframe using styles
    def style_flag(val):
        color = '#ef4444' if 'WATCHLIST' in val else ('#f97316' if 'BURST' in val else '#22c55e')
        return f'color: {color}; font-weight: bold;'
        
    st.dataframe(ticker_df, use_container_width=True, hide_index=True)

    if st.button("Refresh Replay Monitor"):
        st.rerun()

# ═══════════════════════════════════════════════════════════
# PAGE 8 — CITIZEN COMPLAINTS PORTAL
# ═══════════════════════════════════════════════════════════
elif page == "Citizen Complaints":
    st.markdown("### Citizen Complaints & Grievance Portal")
    st.markdown(
        "Report illegal parking violations observed on the ground. "
        "Citizen reports are aggregated with enforcement data to improve hotspot detection coverage.")
    st.divider()

    st.markdown(
        '<div class="info-box"><i class="fa-solid fa-circle-info" style="color:#3b82f6;margin-right:6px;"></i><b>Note:</b> Citizen reports augment official enforcement hotspot data — '
        'field observations from residents help surface under-reported zones that CCTV and patrol coverage may miss.</div>',
        unsafe_allow_html=True)

    if 'complaints' not in st.session_state:
        st.session_state['complaints'] = []

    # ── Complaint submission form ──────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-pen-to-square" style="color:#6366f1;margin-right:6px;"></i>Submit a New Complaint</div>', unsafe_allow_html=True)

    zone_options = cdf[['cluster_id', 'police_station']].drop_duplicates()
    zone_labels  = {
        row['cluster_id']: f"Zone {row['cluster_id']} — {row['police_station']}"
        for _, row in zone_options.iterrows()
    }

    with st.form("complaint_form", clear_on_submit=True):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            sel_zone = st.selectbox(
                "Zone / Cluster (Police Station)",
                options=zone_options['cluster_id'].tolist(),
                format_func=lambda x: zone_labels.get(x, f"Zone {x}")
            )
            violation_type = st.selectbox(
                "Violation Type",
                ["Wrong Parking", "No Parking", "Footpath Parking",
                 "Near Junction", "Double Parking"]
            )

        with col_f2:
            reporter_name = st.text_input("Your Name (optional)", placeholder="Anonymous")
            description   = st.text_area("Description (optional)",
                                         placeholder="Describe what you observed…", height=100)

        photo = st.file_uploader("Upload Photo Evidence (optional)",
                                 type=["jpg", "jpeg", "png", "webp"])

        submitted = st.form_submit_button("Submit Complaint", type="primary", use_container_width=True)

    if submitted:
        station_name = zone_labels.get(sel_zone, f"Zone {sel_zone}")
        complaint = {
            "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Zone":           station_name,
            "Violation Type": violation_type,
            "Description":    description.strip() if description else "—",
            "Reporter":       reporter_name.strip() if reporter_name else "Anonymous",
            "Status":         "PENDING",
            "_photo":         photo,
        }
        st.session_state['complaints'].append(complaint)
        st.success(f"Complaint submitted successfully! Reference #{len(st.session_state['complaints'])}")

        if photo:
            st.image(photo, caption="Uploaded evidence", width=220)

    # ── Summary metrics ────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-list-check" style="color:#6366f1;margin-right:6px;"></i>Session Complaint Summary</div>', unsafe_allow_html=True)

    total_c   = len(st.session_state['complaints'])
    pending_c = sum(1 for c in st.session_state['complaints'] if c['Status'] == 'PENDING')
    resolved_c = sum(1 for c in st.session_state['complaints'] if c['Status'] == 'RESOLVED')

    cm1, cm2, cm3 = st.columns(3)
    cm1.markdown(f"""<div class="metric-card">
        <div class="metric-number">{total_c}</div>
        <div class="metric-label">Total Complaints</div>
        <div class="metric-sub">This session</div></div>""", unsafe_allow_html=True)
    cm2.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#fbbf24">{pending_c}</div>
        <div class="metric-label">Pending Review</div>
        <div class="metric-sub">Awaiting enforcement action</div></div>""", unsafe_allow_html=True)
    cm3.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#4ade80">{resolved_c}</div>
        <div class="metric-label">Resolved</div>
        <div class="metric-sub">Action taken</div></div>""", unsafe_allow_html=True)

    # ── Complaints table ───────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-list" style="color:#6366f1;margin-right:6px;"></i>All Submitted Complaints</div>', unsafe_allow_html=True)

    if st.session_state['complaints']:
        display_complaints = [
            {k: v for k, v in c.items() if k != '_photo'}
            for c in st.session_state['complaints']
        ]
        complaints_df = pd.DataFrame(display_complaints)
        st.dataframe(complaints_df, use_container_width=True, hide_index=True)

        # Show photo thumbnails for complaints that have them
        photos_with_idx = [
            (i + 1, c['_photo'])
            for i, c in enumerate(st.session_state['complaints'])
            if c.get('_photo') is not None
        ]
        if photos_with_idx:
            st.markdown('<div class="section-hdr"><i class="fa-solid fa-image" style="color:#6366f1;margin-right:6px;"></i>Photo Evidence</div>', unsafe_allow_html=True)
            thumb_cols = st.columns(min(len(photos_with_idx), 4))
            for col, (ref_num, photo_file) in zip(thumb_cols, photos_with_idx):
                col.image(photo_file, caption=f"Complaint #{ref_num}", width=160)
    else:
        st.markdown(
            '<div class="alert-box">No complaints submitted yet in this session. '
            'Use the form above to report a violation.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 9 — ZONE IMPROVEMENT TRACKER
# ═══════════════════════════════════════════════════════════
elif page == "Zone Improvement Tracker":
    st.markdown("### Corridor Improvement & Intervention Tracker")
    st.markdown(
        "Log enforcement interventions on hotspot zones and model the expected "
        "reduction in Zone Risk Index (ZRI) based on intervention type effectiveness.")
    st.divider()

    EFFECTIVENESS = {
        "Bollards Installed":   0.35,
        "Extra Patrols":        0.25,
        "No-Parking Signage":   0.20,
        "Petrol Pump Added":    0.30,
        "Towing Drive":         0.15,
    }

    if 'interventions' not in st.session_state:
        st.session_state['interventions'] = []

    # ── Portfolio summary metrics ──────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-line" style="color:#6366f1;margin-right:6px;"></i>Portfolio Summary</div>', unsafe_allow_html=True)

    ivs = st.session_state['interventions']
    total_ivs    = len(ivs)
    zones_touched = len({iv['cluster_id'] for iv in ivs})

    if ivs:
        pct_reductions = [EFFECTIVENESS[iv['Intervention']] * 100 for iv in ivs]
        avg_pct = sum(pct_reductions) / len(pct_reductions)
    else:
        avg_pct = 0.0

    pm1, pm2, pm3 = st.columns(3)
    pm1.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#818cf8">{zones_touched}</div>
        <div class="metric-label">Zones Improved</div>
        <div class="metric-sub">Unique hotspot zones</div></div>""", unsafe_allow_html=True)
    pm2.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#4ade80">{avg_pct:.1f}%</div>
        <div class="metric-label">Avg. Risk Reduction</div>
        <div class="metric-sub">Modelled estimate</div></div>""", unsafe_allow_html=True)
    pm3.markdown(f"""<div class="metric-card">
        <div class="metric-number">{total_ivs}</div>
        <div class="metric-label">Total Interventions</div>
        <div class="metric-sub">Logged this session</div></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Log intervention ───────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-square-plus" style="color:#6366f1;margin-right:6px;"></i>Log a New Intervention</div>', unsafe_allow_html=True)

    zone_opts = cdf[['cluster_id', 'police_station']].drop_duplicates()
    zone_fmt  = {
        row['cluster_id']: f"Zone {row['cluster_id']} — {row['police_station']}"
        for _, row in zone_opts.iterrows()
    }

    col_i1, col_i2, col_i3, col_i4 = st.columns([2, 2, 1, 1])
    with col_i1:
        sel_iv_zone = st.selectbox(
            "Select Zone",
            options=zone_opts['cluster_id'].tolist(),
            format_func=lambda x: zone_fmt.get(x, f"Zone {x}"),
            key="iv_zone"
        )
    with col_i2:
        iv_type = st.selectbox(
            "Intervention Type",
            list(EFFECTIVENESS.keys()),
            key="iv_type"
        )
    with col_i3:
        iv_date = st.date_input("Date", key="iv_date")
    with col_i4:
        st.markdown("<br>", unsafe_allow_html=True)
        log_btn = st.button("Log Intervention", type="primary", use_container_width=True)

    if log_btn:
        zone_row = cdf[cdf['cluster_id'] == sel_iv_zone]
        before_zri = float(zone_row.iloc[0]['zone_risk_index']) if len(zone_row) > 0 else 5.0
        after_zri  = round(max(0.5, before_zri * (1 - EFFECTIVENESS[iv_type])), 2)
        st.session_state['interventions'].append({
            "cluster_id":    sel_iv_zone,
            "Zone":          zone_fmt.get(sel_iv_zone, f"Zone {sel_iv_zone}"),
            "Intervention":  iv_type,
            "Date":          str(iv_date),
            "Before ZRI":    round(before_zri, 2),
            "After ZRI":     after_zri,
            "Risk Δ":        round(after_zri - before_zri, 2),
            "Reduction %":   f"{EFFECTIVENESS[iv_type]*100:.0f}%",
        })
        st.success(
            f"Logged: **{iv_type}** on {zone_fmt.get(sel_iv_zone)} — "
            f"ZRI modelled drop {before_zri:.2f} → {after_zri:.2f} "
            f"({EFFECTIVENESS[iv_type]*100:.0f}% reduction)")
        st.rerun()

    # ── Before-vs-after chart ──────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-chart-simple" style="color:#6366f1;margin-right:6px;"></i>Before vs. After Zone Risk Index</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box"><i class="fa-solid fa-circle-info" style="color:#3b82f6;margin-right:6px;"></i>After-ZRI values are <b>modelled estimates</b> based on '
        'published effectiveness factors for each intervention type. '
        'They are illustrative for this prototype and not measured outcomes.</div>',
        unsafe_allow_html=True)

    if ivs:
        # One row per (zone, intervention) pair; aggregate multiple interventions on same zone
        # by applying reductions sequentially (compounded)
        zone_summary: dict = {}
        for iv in ivs:
            cid = iv['cluster_id']
            if cid not in zone_summary:
                zone_summary[cid] = {
                    'Zone':        iv['Zone'],
                    'Before ZRI':  iv['Before ZRI'],
                    'compound':    1.0,
                    'types':       [],
                }
            zone_summary[cid]['compound'] *= (1 - EFFECTIVENESS[iv['Intervention']])
            zone_summary[cid]['types'].append(iv['Intervention'])

        chart_rows = []
        for cid, z in zone_summary.items():
            after = round(max(0.5, z['Before ZRI'] * z['compound']), 2)
            chart_rows.append({
                'Zone':       z['Zone'],
                'Before ZRI': z['Before ZRI'],
                'After ZRI':  after,
                'Interventions': ', '.join(z['types']),
            })

        chart_df = pd.DataFrame(chart_rows)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before ZRI',
            x=chart_df['Zone'],
            y=chart_df['Before ZRI'],
            marker_color='#f87171',
            text=chart_df['Before ZRI'].apply(lambda v: f"{v:.2f}"),
            textposition='outside',
        ))
        fig.add_trace(go.Bar(
            name='After ZRI (Modelled)',
            x=chart_df['Zone'],
            y=chart_df['After ZRI'],
            marker_color='#4ade80',
            text=chart_df['After ZRI'].apply(lambda v: f"{v:.2f}"),
            textposition='outside',
        ))
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Zone Risk Index (0–10)', range=[0, 11]),
            xaxis=dict(title='Zone'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Intervention log table ─────────────────────────
        st.markdown('<div class="section-hdr"><i class="fa-solid fa-file-shield" style="color:#6366f1;margin-right:6px;"></i>Intervention Log</div>', unsafe_allow_html=True)
        log_df = pd.DataFrame([
            {k: v for k, v in iv.items() if k != 'cluster_id'}
            for iv in ivs
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="alert-box">No interventions logged yet. '
            'Use the controls above to log your first zone intervention.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 10 — SMS ALERT CENTER
# ═══════════════════════════════════════════════════════════
elif page == "SMS Alert Center":
    st.markdown("### SMS Enforcement & Notification Gateway")
    st.markdown(
        "Dispatch automated parking violation warnings to registered habitual offenders "
        "via the Bengaluru Traffic Police SMS gateway.")
    st.caption(
        "DEMO MODE — Vehicle data is anonymized. No real SMS messages are sent. "
        "All dispatches are simulated for prototype demonstration purposes only.")
    st.divider()

    if 'sms_log' not in st.session_state:
        st.session_state['sms_log'] = []

    def build_message(vehicle_number, offense_count, fine_tier):
        return (
            f"TRAFFIC ALERT: Vehicle {vehicle_number} has {offense_count} pending parking "
            f"violations (Tier {fine_tier}). Settle dues to avoid escalation. "
            f"- Bengaluru Traffic Police"
        )

    # ── Recipient selection ────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-users" style="color:#6366f1;margin-right:6px;"></i>Select Recipients</div>', unsafe_allow_html=True)

    BULK_OPTION = "Send to all Tier CRITICAL offenders"
    vehicle_choices = [BULK_OPTION] + habitual['vehicle_number'].tolist()

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        recipient = st.selectbox(
            "Recipient — individual vehicle or bulk tier",
            options=vehicle_choices,
        )

    # Resolve rows to send to
    if recipient == BULK_OPTION:
        target_rows = habitual[habitual['fine_tier'] == 'CRITICAL']
    else:
        target_rows = habitual[habitual['vehicle_number'] == recipient]

    # ── Message preview ────────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-envelope-open-text" style="color:#6366f1;margin-right:6px;"></i>Message Preview</div>', unsafe_allow_html=True)

    if len(target_rows) == 0:
        st.markdown(
            '<div class="alert-box">No matching offenders found for the selected recipient.</div>',
            unsafe_allow_html=True)
    else:
        preview_row = target_rows.iloc[0]
        preview_msg = build_message(
            preview_row['vehicle_number'],
            preview_row['offense_count'],
            preview_row['fine_tier'],
        )

        if recipient == BULK_OPTION:
            st.markdown(
                f'<div class="challan-box">'
                f'<b style="color:#ca8a04">BULK DISPATCH — {len(target_rows)} vehicles (Tier CRITICAL)</b><br><br>'
                f'<b>Sample message (sent to each vehicle individually):</b><br>'
                f'<span style="font-family:monospace;color:#f3f4f6;line-height:1.8">{preview_msg}</span>'
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="challan-box">'
                f'<b style="color:#ca8a04">SINGLE DISPATCH — {preview_row["vehicle_number"]}</b>'
                f'&nbsp;&nbsp;<span style="color:#9ca3af;font-size:0.8rem">Tier: {preview_row["fine_tier"]} | '
                f'Offenses: {preview_row["offense_count"]} | Fine: ₹{preview_row["illustrative_fine"]:,}</span><br><br>'
                f'<span style="font-family:monospace;color:#f3f4f6;line-height:1.8">{preview_msg}</span>'
                f'</div>',
                unsafe_allow_html=True)

        # ── Send button ────────────────────────────────────
        with col_r2:
            st.markdown("<br>", unsafe_allow_html=True)
            send_btn = st.button("Send Alert", type="primary", use_container_width=True)

        if send_btn:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for _, row in target_rows.iterrows():
                msg = build_message(
                    row['vehicle_number'],
                    row['offense_count'],
                    row['fine_tier'],
                )
                st.session_state['sms_log'].append({
                    "Timestamp":      ts,
                    "Vehicle Number": row['vehicle_number'],
                    "Fine Tier":      row['fine_tier'],
                    "Offenses":       int(row['offense_count']),
                    "Message":        msg,
                    "Status":         "DELIVERED",
                })

            n = len(target_rows)
            st.toast(f"{n} SMS alert{'s' if n > 1 else ''} dispatched (DEMO MODE)")
            if recipient == BULK_OPTION:
                detail = "All Tier CRITICAL offenders notified."
            else:
                detail = f"Vehicle {preview_row['vehicle_number']} notified."
            st.success(f"**{n} message{'s' if n > 1 else ''} sent** — {detail}")
            st.rerun()

    # ── Dispatch metrics ───────────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-square-poll-vertical" style="color:#6366f1;margin-right:6px;"></i>Dispatch Summary</div>', unsafe_allow_html=True)

    total_sent = len(st.session_state['sms_log'])
    sm1, sm2, sm3 = st.columns(3)
    sm1.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#818cf8">{total_sent}</div>
        <div class="metric-label">Alerts Sent</div>
        <div class="metric-sub">This session (DEMO)</div></div>""", unsafe_allow_html=True)

    critical_sent = sum(1 for s in st.session_state['sms_log'] if s['Fine Tier'] == 'CRITICAL')
    sm2.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#f87171">{critical_sent}</div>
        <div class="metric-label">Tier CRITICAL Alerts</div>
        <div class="metric-sub">Highest escalation tier</div></div>""", unsafe_allow_html=True)

    other_sent = total_sent - critical_sent
    sm3.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#fbbf24">{other_sent}</div>
        <div class="metric-label">Other Tier Alerts</div>
        <div class="metric-sub">HIGH / MAXIMUM / BASE</div></div>""", unsafe_allow_html=True)

    # ── SMS log table ──────────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-list-ul" style="color:#6366f1;margin-right:6px;"></i>SMS Dispatch Log</div>', unsafe_allow_html=True)

    if st.session_state['sms_log']:
        log_df = pd.DataFrame(st.session_state['sms_log'])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="alert-box">No alerts dispatched yet. '
            'Select a recipient above and press Send Alert to begin.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 11 — SAFE PARKING FINDER
# ═══════════════════════════════════════════════════════════
elif page == "Safe Parking Finder":
    st.markdown("### Safe Off-Street Parking Finder")
    st.markdown(
        "Find low-risk parking zones near your destination, and identify high-risk areas "
        "to avoid — using illegal-parking violation density as a safety proxy.")
    st.caption(
        "Prototype: 'safe' zones are those with low violation density in enforcement records. "
        "This is not a live parking-availability service.")
    st.divider()

    # ── Controls ───────────────────────────────────────────
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        dest_station = st.selectbox(
            "Destination (Police Station Area)",
            sorted(stn['police_station'].tolist()),
        )
    with col_c2:
        radius_km = st.slider("Search radius (km)", 0.5, 5.0, 2.0, 0.5)
    with col_c3:
        safe_threshold = st.slider("Safe ZRI ceiling", 1.0, 6.0, 3.5, 0.5,
                                   help="Zones at or below this ZRI are marked SAFE")

    # ── Destination point: mean lat/lon of that station's zones ──
    dest_zones = cdf[cdf['police_station'] == dest_station]
    if len(dest_zones) == 0:
        st.warning("No H3 zones mapped under the selected station.")
        st.stop()

    dest_lat = dest_zones['lat'].mean()
    dest_lon = dest_zones['lon'].mean()

    # ── Distance computation (Haversine, vectorised) ───────
    R = 6371.0  # Earth radius in km
    lat1  = np.radians(dest_lat)
    lat2  = np.radians(cdf['lat'].values)
    dLat  = lat2 - lat1
    dLon  = np.radians(cdf['lon'].values) - np.radians(dest_lon)
    a     = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    cdf_work = cdf.copy()
    cdf_work['dist_km'] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    nearby = cdf_work[cdf_work['dist_km'] <= radius_km].copy()

    if len(nearby) == 0:
        st.info(f"No zones found within {radius_km} km of {dest_station}. Try increasing the radius.")
        st.stop()

    safe  = nearby[nearby['zone_risk_index'] <= safe_threshold].sort_values('zone_risk_index')
    avoid = nearby[nearby['zone_risk_index'] >  safe_threshold].sort_values('zone_risk_index', ascending=False)

    # ── Scatter map ────────────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-map-location-dot" style="color:#6366f1;margin-right:6px;"></i>Zone Safety Map</div>', unsafe_allow_html=True)

    map_rows = []
    for _, r in safe.iterrows():
        map_rows.append({
            'lat': r['lat'], 'lon': r['lon'],
            'label': f"SAFE — Zone {r['cluster_id']} ({r['police_station']})",
            'color': 'safe', 'size': 10,
            'ZRI': r['zone_risk_index'], 'dist_km': round(r['dist_km'], 2),
        })
    for _, r in avoid.iterrows():
        map_rows.append({
            'lat': r['lat'], 'lon': r['lon'],
            'label': f"AVOID — Zone {r['cluster_id']} ({r['police_station']})",
            'color': 'avoid', 'size': 14,
            'ZRI': r['zone_risk_index'], 'dist_km': round(r['dist_km'], 2),
        })
    # Destination marker
    map_rows.append({
        'lat': dest_lat, 'lon': dest_lon,
        'label': f"Destination: {dest_station}",
        'color': 'destination', 'size': 18,
        'ZRI': None, 'dist_km': 0.0,
    })

    map_df = pd.DataFrame(map_rows)
    color_map = {'safe': '#22c55e', 'avoid': '#ef4444', 'destination': '#38bdf8'}

    fig = px.scatter_mapbox(
        map_df,
        lat='lat', lon='lon',
        color='color',
        color_discrete_map=color_map,
        size='size',
        size_max=18,
        hover_name='label',
        hover_data={'ZRI': True, 'dist_km': True, 'color': False, 'size': False},
        mapbox_style='carto-darkmatter',
        zoom=13,
        center={'lat': dest_lat, 'lon': dest_lon},
        template='plotly_dark',
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title='Zone Type',
            orientation='v',
            bgcolor='rgba(17,24,39,0.85)',
            bordercolor='#374151',
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary counts ─────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#22c55e">{len(safe)}</div>
        <div class="metric-label">Safe Zones Nearby</div>
        <div class="metric-sub">ZRI ≤ {safe_threshold} within {radius_km} km</div></div>""",
        unsafe_allow_html=True)
    sc2.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#ef4444">{len(avoid)}</div>
        <div class="metric-label">High-Risk Zones</div>
        <div class="metric-sub">ZRI > {safe_threshold} — avoid</div></div>""",
        unsafe_allow_html=True)
    best_zri = safe.iloc[0]['zone_risk_index'] if len(safe) > 0 else None
    sc3.markdown(f"""<div class="metric-card">
        <div class="metric-number" style="color:#4ade80">{f'{best_zri:.2f}' if best_zri is not None else '—'}</div>
        <div class="metric-label">Best Nearby ZRI</div>
        <div class="metric-sub">Lowest risk zone in radius</div></div>""",
        unsafe_allow_html=True)

    st.divider()

    # ── Top 5 safe zones ───────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-square-check" style="color:#22c55e;margin-right:6px;"></i>Top 5 Recommended Parking Zones</div>',
                unsafe_allow_html=True)

    if len(safe) == 0:
        st.markdown(
            '<div class="alert-box">No safe zones found within the selected radius and ZRI threshold. '
            'Try increasing the radius or raising the safe ZRI ceiling.</div>',
            unsafe_allow_html=True)
    else:
        top5 = safe.head(5)
        cols = st.columns(min(len(top5), 5))
        for col, (_, r) in zip(cols, top5.iterrows()):
            col.markdown(f"""<div class="metric-card">
                <div style="color:#22c55e;font-size:1.3rem;font-weight:700">ZRI {r['zone_risk_index']:.2f}</div>
                <div class="metric-label">Zone {r['cluster_id']}</div>
                <div class="metric-sub">{r['police_station']}</div>
                <div style="color:#6b7280;font-size:0.7rem;margin-top:4px">{r['dist_km']:.2f} km away</div>
                <div style="color:#6b7280;font-size:0.68rem">{r['zone_type']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Zones to avoid ─────────────────────────────────────
    st.markdown('<div class="section-hdr"><i class="fa-solid fa-triangle-exclamation" style="color:#ef4444;margin-right:6px;"></i>High-Risk Zones to Avoid</div>', unsafe_allow_html=True)

    if len(avoid) == 0:
        st.markdown(
            '<div class="info-box">No high-risk zones detected within the search radius — '
            'the selected area has relatively low violation density.</div>',
            unsafe_allow_html=True)
    else:
        for _, r in avoid.head(5).iterrows():
            st.markdown(
                f'<div class="alert-box">'
                f'<b>Zone {r["cluster_id"]} — {r["police_station"]}</b> &nbsp;|&nbsp; '
                f'ZRI: <b>{r["zone_risk_index"]:.2f}</b> &nbsp;|&nbsp; '
                f'{r["dist_km"]:.2f} km away &nbsp;|&nbsp; '
                f'Primary violation: {r["primary_violation"]} &nbsp;|&nbsp; '
                f'Zone type: {r["zone_type"]}'
                f'</div>',
                unsafe_allow_html=True)
