"""
=============================================================
ParkingIQ Data Preprocessing & Model Training Pipeline
=============================================================
Run: python preprocess_data.py
=============================================================
"""
import pandas as pd
import numpy as np
import ast
import json
import os
import warnings
import h3
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

print("=" * 60)
print("ParkingIQ - Advanced Preprocessing & Model Training")
print("=" * 60)

# 1. Load dataset
csv_path = 'c:/Users/taman/Downloads/jan to may police violation_anonymized791b166.csv'
if not os.path.exists(csv_path):
    csv_path = 'jan_to_may_police_violation_anonymized791b166.csv'

if not os.path.exists(csv_path):
    print(f"❌ Error: Dataset file not found at {csv_path}!")
    print("Please place the CSV file in your Downloads directory or project folder.")
    exit(1)

print(f"\n[1/6] Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path,
                 usecols=['id', 'latitude', 'longitude', 'location', 'vehicle_number',
                          'vehicle_type', 'violation_type', 'created_datetime',
                          'police_station', 'junction_name'])

print(f"   Successfully loaded {len(df):,} records.")

# Parse datetimes
df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed', utc=True)
df['hour'] = df['created_datetime'].dt.hour
df['dayofweek'] = df['created_datetime'].dt.dayofweek
df['date'] = df['created_datetime'].dt.date
df['week'] = df['created_datetime'].dt.isocalendar().week.astype(int)
df['month'] = df['created_datetime'].dt.month

# 2. Parse violations
print("\n[2/6] Parsing violation types...")
def parse_first(s):
    try:
        v = ast.literal_eval(s)
        return v[0] if v else 'UNKNOWN'
    except:
        return 'UNKNOWN'

def parse_all(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

df['primary_violation'] = df['violation_type'].apply(parse_first)
df['all_violations'] = df['violation_type'].apply(parse_all)

# Build violation counts dictionary
vtypes = {}
for v in df['violation_type'].dropna():
    try:
        for t in ast.literal_eval(v):
            vtypes[t] = vtypes.get(t, 0) + 1
    except:
        pass

# 3. Categorize heavy vehicles & junctions
print("\n[3/6] Flagging vehicle categories & junction proximity...")
HEAVY = ['LORRY', 'TANKER', 'TIPPER', 'BUS', 'MAXI-CAB', 'TRUCK',
         'PRIVATE BUS', 'BUS (BMTC/KSRTC)', 'LGV']
df['is_heavy'] = df['vehicle_type'].str.upper().isin([h.upper() for h in HEAVY])
df['at_junction'] = (df['junction_name'] != 'No Junction') & (df['junction_name'].notna())

# 4. Habitual Offenders (Watchlist: 3+ violations in 30 days)
print("\n[4/6] Computing habitual offender watchlist (3+ violations in 30 days)...")
vc = df['vehicle_number'].value_counts()
repeat_candidates = vc[vc >= 3].index

# Filter records to speed up 30-day window check
sub_df = df[df['vehicle_number'].isin(repeat_candidates)][['vehicle_number', 'created_datetime']]

def is_habitual_30days(group):
    if len(group) < 3:
        return False
    sorted_dates = sorted(group)
    for i in range(len(sorted_dates) - 2):
        if (sorted_dates[i+2] - sorted_dates[i]).days <= 30:
            return True
    return False

# Run check
print("   Analyzing temporal spacing of violations per vehicle...")
habitual_status = sub_df.groupby('vehicle_number')['created_datetime'].apply(is_habitual_30days)
habitual_watchlist = set(habitual_status[habitual_status].index)
print(f"   Found {len(habitual_watchlist):,} vehicles matching the 30-day watchlist rule.")

df['is_repeat'] = df['vehicle_number'].map(vc) >= 3
df['is_watchlist'] = df['vehicle_number'].isin(habitual_watchlist)

# Generate detailed habitual offenders records for output (limit to 12,000 for size/demo)
print("   Compiling habitual offender profiles...")
repeat_records_df = df[df['vehicle_number'].isin(repeat_candidates)]
grouped_hab = repeat_records_df.groupby('vehicle_number')

habitual_data = []
# Aggregating efficiently using mode and aggregates
for vnum, group in grouped_hab:
    cnt = len(group)
    vtype = group['vehicle_type'].mode().iloc[0] if not group['vehicle_type'].empty else 'UNKNOWN'
    p_violation = group['primary_violation'].mode().iloc[0] if not group['primary_violation'].empty else 'UNKNOWN'
    stations = ', '.join(group['police_station'].dropna().unique()[:3])
    first_off = str(group['created_datetime'].min().date())
    last_off = str(group['created_datetime'].max().date())
    
    # Escalating penalty logic
    if cnt < 3:
        tier = 'BASE'
        fine = 500
    elif cnt < 6:
        tier = 'HIGH'
        fine = 1500
    elif cnt < 10:
        tier = 'MAXIMUM'
        fine = 3000
    else:
        tier = 'CRITICAL'
        fine = 5000
        
    habitual_data.append({
        'vehicle_number': vnum,
        'vehicle_type': vtype,
        'offense_count': cnt,
        'primary_violation': p_violation,
        'police_stations': stations if stations else 'Unknown',
        'first_offense': first_off,
        'last_offense': last_off,
        'fine_tier': tier,
        'illustrative_fine': fine,
        'is_watchlist_30d': 1 if vnum in habitual_watchlist else 0
    })

habitual_output_df = pd.DataFrame(habitual_data).sort_values('offense_count', ascending=False)
habitual_output_df.to_csv('data/habitual_offenders.csv', index=False)
print(f"   Saved {len(habitual_output_df):,} profiles to data/habitual_offenders.csv")

# 5. Spatial Indexing & Clustering (H3 Grid)
print("\n[5/6] Creating spatial hotspots grid using H3 (Resolution 9)...")
df['h3_cell'] = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(df['latitude'], df['longitude'])]

cell_counts = df['h3_cell'].value_counts()
# Select active hotspot cells (at least 15 violations)
active_cells = cell_counts[cell_counts >= 15].index
print(f"   Detected {len(active_cells)} dense hotspot cells out of {len(cell_counts)} total active H3 cells.")

total_days = max((df['created_datetime'].max() - df['created_datetime'].min()).days, 1)

cluster_rows = []
h3_to_zone_id = {}

for idx, cell in enumerate(active_cells):
    h3_to_zone_id[cell] = int(idx)
    sub = df[df['h3_cell'] == cell]
    n = len(sub)
    lat_c = sub['latitude'].mean()
    lon_c = sub['longitude'].mean()
    
    # Spillover Index (spread of coordinates in km, scaled to 0-10)
    dists = np.sqrt((sub['latitude'] - lat_c)**2 + (sub['longitude'] - lon_c)**2) * 111.0
    spillover_index = min(round(float(dists.std() * 60.0), 2), 10.0) if n > 1 else 0.0
    
    # Persistence Index (active days fraction scaled to 0-10)
    active_days = sub['date'].nunique()
    persistence_index = min(round(active_days / total_days * 10.0, 2), 10.0)
    
    heavy_pct = sub['is_heavy'].mean()
    repeat_pct = sub['is_repeat'].mean()
    junction_pct = sub['at_junction'].mean()
    
    # Zone Risk Index (ZRI) Composite Formula
    density_score = min(np.log1p(n) / np.log1p(500) * 3.0, 3.0)
    heavy_score = heavy_pct * 1.5
    repeat_score = repeat_pct * 2.0
    spillover_score = (spillover_index / 10.0) * 1.5
    junction_score = junction_pct * 1.0
    persistence_score = (persistence_index / 10.0) * 1.0
    
    zone_risk = round(min(density_score + heavy_score + repeat_score + spillover_score + junction_score + persistence_score, 10.0), 2)
    
    # Classification: Chronic, Semi-Chronic, Sporadic
    weekly = sub.groupby('week').size()
    cv = weekly.std() / (weekly.mean() + 0.001) if len(weekly) > 1 else 1.5
    
    if persistence_index >= 5.0 and cv < 0.6:
        zone_type = 'CHRONIC'
    elif persistence_index < 3.0 or cv > 1.1:
        zone_type = 'SPORADIC'
    else:
        zone_type = 'SEMI-CHRONIC'
        
    # Burst Days (surges)
    daily = sub.groupby('date').size()
    burst_threshold = daily.mean() + 2 * daily.std()
    burst_days = int((daily > burst_threshold).sum()) if len(daily) > 2 else 0
    
    # Top Hours
    top_hours = sub.groupby('hour').size().nlargest(3).index.tolist()
    
    # Petrol Pump Impact
    # 1 pump: 12% reduction. 2 pumps: 22%. 3 pumps: 30%.
    petrol_pump_reduction = int(n * 0.12)
    
    cluster_rows.append({
        'cluster_id': int(idx),
        'h3_cell': cell,
        'lat': round(lat_c, 6),
        'lon': round(lon_c, 6),
        'violation_count': n,
        'police_station': sub['police_station'].mode().iloc[0] if sub['police_station'].notna().any() else 'Unknown',
        'primary_violation': sub['primary_violation'].mode().iloc[0] if sub['primary_violation'].notna().any() else 'Unknown',
        'heavy_pct': round(heavy_pct * 100, 1),
        'repeat_pct': round(repeat_pct * 100, 1),
        'junction_pct': round(junction_pct * 100, 1),
        'top_hours': json.dumps(top_hours),
        'spillover_index': spillover_index,
        'persistence_index': persistence_index,
        'zone_risk_index': zone_risk,
        'zone_type': zone_type,
        'active_days': int(active_days),
        'burst_days': burst_days,
        'petrol_pump_reduction': petrol_pump_reduction,
        'petrol_pump_new_risk': round(zone_risk * 0.88, 2),
        'location_sample': sub['location'].dropna().iloc[0][:80] if sub['location'].notna().any() else ''
    })

cdf = pd.DataFrame(cluster_rows).sort_values('zone_risk_index', ascending=False)
cdf.to_csv('data/cluster_stats.csv', index=False)
print(f"   Hotspot stats generated and saved to data/cluster_stats.csv")

# Save station summary
stn = df.groupby('police_station').agg(
    count=('id', 'count'),
    heavy_pct=('is_heavy', 'mean'),
    repeat_pct=('is_repeat', 'mean'),
    junction_pct=('at_junction', 'mean'),
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean')
).reset_index()
for col in ['heavy_pct', 'repeat_pct', 'junction_pct']:
    stn[col] = (stn[col] * 100).round(1)
stn = stn.sort_values('count', ascending=False).head(20)
stn.to_csv('data/station_summary.csv', index=False)

# Hourly by station
top5 = df['police_station'].value_counts().head(5).index.tolist()
hourly = df[df['police_station'].isin(top5)].groupby(['police_station', 'hour']).size().reset_index(name='count')
hourly.to_csv('data/hourly_by_station.csv', index=False)

# 6. Train Predictive ML Model (Random Forest)
print("\n[6/6] Formatting hourly data and training predictive model...")

# Assign zone IDs to df
df['zone_id'] = df['h3_cell'].map(h3_to_zone_id)
# Drop records not in active hotspot cells
df_active = df[df['zone_id'].notna()]

# Group by zone_id, dayofweek, hour to get training samples
hourly_grouped = df_active.groupby(['zone_id', 'dayofweek', 'hour']).size().reset_index(name='violation_count')

# Join zone-level static features
zone_features = cdf[['cluster_id', 'spillover_index', 'persistence_index', 'lat', 'lon']]
hourly_grouped = hourly_grouped.merge(zone_features, left_on='zone_id', right_on='cluster_id').drop(columns=['cluster_id'])

# Prepare feature matrix
features = ['zone_id', 'dayofweek', 'hour', 'spillover_index', 'persistence_index', 'lat', 'lon']
X = hourly_grouped[features]
y = hourly_grouped['violation_count']

# Train Random Forest Regressor
print("   Fitting RandomForestRegressor model...")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X, y)

# Save model and H3 mapping data
with open('data/predictor_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("   Saved model to data/predictor_model.pkl")

# Save summary and metadata
summary_dict = {
    'total_violations': int(len(df)),
    'total_clusters': int(len(cdf)),
    'chronic_zones': int((cdf['zone_type'] == 'CHRONIC').sum()),
    'semi_chronic_zones': int((cdf['zone_type'] == 'SEMI-CHRONIC').sum()),
    'sporadic_zones': int((cdf['zone_type'] == 'SPORADIC').sum()),
    'habitual_vehicles': len(habitual_watchlist),
    'heavy_vehicle_pct': round(df['is_heavy'].mean() * 100, 1),
    'junction_pct': round(df['at_junction'].mean() * 100, 1),
    'date_from': str(df['created_datetime'].min().date()),
    'date_to': str(df['created_datetime'].max().date()),
    'top_station': df['police_station'].value_counts().index[0],
    'violation_breakdown': {k: int(v) for k, v in sorted(vtypes.items(), key=lambda x: -x[1])}
}

with open('data/summary.json', 'w') as f:
    json.dump(summary_dict, f, indent=2)

# Save metadata for H3 cells and station lookups
metadata = {
    'h3_to_zone_id': {k: int(v) for k, v in h3_to_zone_id.items()},
    'zone_id_to_h3': {int(v): k for k, v in h3_to_zone_id.items()},
    'features': features
}
with open('data/predictor_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("Success: Preprocessing & model training successfully complete!")
print(f"   Total Violations Processed: {summary_dict['total_violations']:,}")
print(f"   Hotspot Hexagons Grid:      {summary_dict['total_clusters']}")
print(f"   Habitual watchlist:        {summary_dict['habitual_vehicles']:,}")
print(f"   Date Range:                 {summary_dict['date_from']} to {summary_dict['date_to']}")
print("=" * 60)
