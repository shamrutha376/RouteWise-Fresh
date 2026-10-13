import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Synthetic Accident Dataset Generator
# For Dynamic Traffic AI - Tambaram prototype
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

INPUT_FILE = DATA_DIR / "X_features.csv"
EVENT_FILE = DATA_DIR / "accident_events.csv"
FEATURE_FILE = DATA_DIR / "accident_features.csv"
MERGED_FILE = DATA_DIR / "X_features_with_accidents.csv"

RANDOM_SEED = 42
NUMBER_OF_ACCIDENT_EVENTS = 1500

np.random.seed(RANDOM_SEED)

print("Loading existing road features...")
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "u", "v", "key", "length_m", "road_type", "lanes",
    "maxspeed", "u_degree", "v_degree",
    "road_connectivity", "road_centrality",
    "school_count", "hospital_count",
    "place_of_worship_count", "bus_stop_count",
    "office_count", "mall_count", "commercial_count",
    "retail_count", "college_count", "total_poi_count",
    "traffic_density"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# X_features contains four time-period records for each road segment.
# We only need one static record per road segment for accident generation.
segments = (
    df.sort_values(["u", "v", "key"])
      .drop_duplicates(["u", "v", "key"])
      .copy()
      .reset_index(drop=True)
)

print(f"Unique road segments found: {len(segments)}")


def percentile_score(series):
    """Convert a numeric feature to a 0-1 percentile score."""
    return series.rank(pct=True).fillna(0.0)


# ------------------------------------------------------------
# 1. Build an accident-proneness score
# ------------------------------------------------------------
# This does NOT use the existing risk_score or historical_accident_count.
# Therefore the synthetic accident data are not circularly generated
# from the target we are later going to add.

traffic_score = percentile_score(segments["traffic_density"])

centrality_score = percentile_score(segments["road_centrality"])

connectivity_score = percentile_score(segments["road_connectivity"])

poi_score = percentile_score(segments["total_poi_count"])

intersection_value = (
    segments["u_degree"].fillna(0) +
    segments["v_degree"].fillna(0)
) / 2

intersection_score = percentile_score(intersection_value)

road_type_weight = {
    "motorway": 1.00,
    "trunk": 0.95,
    "primary": 0.90,
    "secondary": 0.75,
    "tertiary": 0.60,
    "trunk_link": 0.85,
    "primary_link": 0.75,
    "motorway_link": 0.90,
    "residential": 0.45,
    "living_street": 0.30,
    "unclassified": 0.35,
    "service": 0.25,
    "busway": 0.55
}

road_score = (
    segments["road_type"]
    .map(road_type_weight)
    .fillna(0.40)
)

# Infrastructure activity is based on the POI categories already
# present in X_features.csv.
activity_score = percentile_score(
    segments[
        [
            "school_count",
            "hospital_count",
            "place_of_worship_count",
            "bus_stop_count",
            "office_count",
            "mall_count",
            "commercial_count",
            "retail_count",
            "college_count"
        ]
    ].sum(axis=1)
)

propensity = (
    0.28 * traffic_score +
    0.18 * intersection_score +
    0.15 * centrality_score +
    0.12 * connectivity_score +
    0.12 * poi_score +
    0.10 * activity_score +
    0.05 * road_score
)

propensity = propensity.clip(lower=0.01)

# Probability of selecting a road for an accident event.
event_probability = propensity / propensity.sum()


# ------------------------------------------------------------
# 2. Select road segments for synthetic accident events
# ------------------------------------------------------------
selected_indices = np.random.choice(
    segments.index,
    size=NUMBER_OF_ACCIDENT_EVENTS,
    replace=True,
    p=event_probability
)

events = segments.loc[selected_indices].copy().reset_index(drop=True)

events.insert(
    0,
    "accident_id",
    [f"ACC_{i:05d}" for i in range(1, len(events) + 1)]
)

# ------------------------------------------------------------
# 3. Generate synthetic accident circumstances
# ------------------------------------------------------------

# Higher activity roads are more likely to produce peak-period events.
event_activity = percentile_score(
    events["total_poi_count"] +
    events["commercial_count"] +
    events["office_count"] +
    events["retail_count"]
).to_numpy()

peak_probability = np.clip(0.30 + 0.30 * event_activity, 0.30, 0.65)

is_peak = np.random.random(len(events)) < peak_probability

time_period = []

for peak in is_peak:
    if peak:
        time_period.append(
            np.random.choice(["Morning", "Evening"], p=[0.42, 0.58])
        )
    else:
        time_period.append(
            np.random.choice(["Afternoon", "Night"], p=[0.65, 0.35])
        )

events["time_period"] = time_period

hour_map = {
    "Morning": [7, 8, 9],
    "Afternoon": [12, 13, 14, 15],
    "Evening": [17, 18, 19, 20],
    "Night": [21, 22, 23, 0, 1, 2, 3, 4, 5]
}

events["hour"] = [
    np.random.choice(hour_map[p])
    for p in events["time_period"]
]

events["peak_hour"] = events["time_period"].isin(
    ["Morning", "Evening"]
).astype(int)

events["traffic_level"] = pd.cut(
    events["traffic_density"],
    bins=[-np.inf, 50, 75, 90, np.inf],
    labels=["Low", "Moderate", "High", "Very High"]
).astype(str)

# Intersection-related accident probability.
intersection_strength = (
    events["u_degree"].fillna(0) +
    events["v_degree"].fillna(0)
) / 2

intersection_probability = np.clip(
    0.15 + 0.12 * intersection_strength,
    0.15,
    0.70
)

events["intersection_related"] = (
    np.random.random(len(events)) < intersection_probability
).astype(int)

# Road-departure probability is higher on lower-connectivity roads.
departure_probability = np.clip(
    0.30 - 0.05 * events["road_connectivity"].fillna(2.5),
    0.10,
    0.30
)

events["road_departure"] = (
    np.random.random(len(events)) < departure_probability
).astype(int)

events["lighting_condition"] = np.where(
    events["time_period"].eq("Night"),
    np.random.choice(
        ["Dark", "Artificial_Light"],
        size=len(events),
        p=[0.35, 0.65]
    ),
    "Daylight"
)

events["vehicles_involved"] = np.random.choice(
    [1, 2, 3],
    size=len(events),
    p=[0.45, 0.45, 0.10]
)

# ------------------------------------------------------------
# 4. Generate accident severity
# ------------------------------------------------------------

risk_strength = (
    0.45 * traffic_score.loc[events.index].to_numpy()
    if False else np.zeros(len(events))
)

# Recalculate local risk factors for the sampled events.
local_traffic = percentile_score(events["traffic_density"]).to_numpy()
local_centrality = percentile_score(events["road_centrality"]).to_numpy()
local_intersection = percentile_score(
    (events["u_degree"].fillna(0) + events["v_degree"].fillna(0)) / 2
).to_numpy()

severity_strength = (
    0.45 * local_traffic +
    0.30 * local_centrality +
    0.25 * local_intersection
)

severity = []

for value in severity_strength:
    # Base distribution: Minor > Serious > Fatal.
    # Higher-risk road/event conditions increase serious/fatal probability.
    fatal_probability = np.clip(0.03 + 0.08 * value, 0.03, 0.11)
    serious_probability = np.clip(0.20 + 0.15 * value, 0.20, 0.35)
    minor_probability = 1.0 - fatal_probability - serious_probability

    severity.append(
        np.random.choice(
            ["Minor", "Serious", "Fatal"],
            p=[
                minor_probability,
                serious_probability,
                fatal_probability
            ]
        )
    )

events["accident_severity"] = severity

# Synthetic date range for historical prototype scenarios.
start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2025-12-31")

days = (end_date - start_date).days

events["date"] = [
    start_date + pd.Timedelta(days=int(np.random.randint(0, days + 1)))
    for _ in range(len(events))
]

events["date"] = events["date"].dt.strftime("%Y-%m-%d")

# Keep the event-level dataset focused on accident information.
event_columns = [
    "accident_id",
    "u",
    "v",
    "key",
    "road_type",
    "length_m",
    "lanes",
    "maxspeed",
    "time_period",
    "hour",
    "peak_hour",
    "traffic_level",
    "traffic_density",
    "intersection_related",
    "road_departure",
    "lighting_condition",
    "vehicles_involved",
    "accident_severity",
    "date"
]

events = events[event_columns]

events.to_csv(EVENT_FILE, index=False)


# ------------------------------------------------------------
# 5. Aggregate events to road-segment accident features
# ------------------------------------------------------------

group_columns = ["u", "v", "key"]

accident_features = (
    segments[group_columns]
    .copy()
)

counts = (
    events.groupby(group_columns)
    .size()
    .rename("historical_accident_count")
    .reset_index()
)

accident_features = accident_features.merge(
    counts,
    on=group_columns,
    how="left"
)

accident_features["historical_accident_count"] = (
    accident_features["historical_accident_count"]
    .fillna(0)
    .astype(int)
)

severity_table = pd.crosstab(
    [events["u"], events["v"], events["key"]],
    events["accident_severity"]
).reset_index()

for column in ["Minor", "Serious", "Fatal"]:
    if column not in severity_table.columns:
        severity_table[column] = 0

severity_table = severity_table.rename(
    columns={
        "Minor": "minor_accident_count",
        "Serious": "serious_accident_count",
        "Fatal": "fatal_accident_count"
    }
)

accident_features = accident_features.merge(
    severity_table[
        group_columns +
        [
            "minor_accident_count",
            "serious_accident_count",
            "fatal_accident_count"
        ]
    ],
    on=group_columns,
    how="left"
)

for column in [
    "minor_accident_count",
    "serious_accident_count",
    "fatal_accident_count"
]:
    accident_features[column] = (
        accident_features[column]
        .fillna(0)
        .astype(int)
    )

peak_counts = (
    events[events["peak_hour"] == 1]
    .groupby(group_columns)
    .size()
    .rename("peak_hour_accident_count")
    .reset_index()
)

night_counts = (
    events[events["time_period"] == "Night"]
    .groupby(group_columns)
    .size()
    .rename("night_accident_count")
    .reset_index()
)

intersection_counts = (
    events[events["intersection_related"] == 1]
    .groupby(group_columns)
    .size()
    .rename("intersection_accident_count")
    .reset_index()
)

departure_counts = (
    events[events["road_departure"] == 1]
    .groupby(group_columns)
    .size()
    .rename("road_departure_accident_count")
    .reset_index()
)

for table in [
    peak_counts,
    night_counts,
    intersection_counts,
    departure_counts
]:
    accident_features = accident_features.merge(
        table,
        on=group_columns,
        how="left"
    )

for column in [
    "peak_hour_accident_count",
    "night_accident_count",
    "intersection_accident_count",
    "road_departure_accident_count"
]:
    accident_features[column] = (
        accident_features[column]
        .fillna(0)
        .astype(int)
    )

# Weighted severity score:
# Minor = 1, Serious = 2, Fatal = 3.
accident_features["severity_weighted_score"] = (
    accident_features["minor_accident_count"] * 1 +
    accident_features["serious_accident_count"] * 2 +
    accident_features["fatal_accident_count"] * 3
)

# Normalize to a 0-100 accident-risk scale.
max_score = accident_features["severity_weighted_score"].max()

if max_score > 0:
    accident_features["accident_risk_score"] = (
        accident_features["severity_weighted_score"] /
        max_score * 100
    ).round(2)
else:
    accident_features["accident_risk_score"] = 0.0

# Accident rate per kilometre for the synthetic period.
accident_features["accident_rate_per_km"] = (
    accident_features["historical_accident_count"] /
    (segments["length_m"].replace(0, np.nan) / 1000)
).replace([np.inf, -np.inf], 0).fillna(0).round(3)

accident_features.to_csv(FEATURE_FILE, index=False)


# ------------------------------------------------------------
# 6. Create a separate merged feature file for inspection
# ------------------------------------------------------------

# Rename the generated accident count before merging so that it
# does not collide with the placeholder column already in X_features.
accident_for_merge = accident_features.rename(
    columns={
        "historical_accident_count": "generated_accident_count"
    }
)

merged = df.drop(
    columns=["historical_accident_count"],
    errors="ignore"
).merge(
    accident_for_merge,
    on=group_columns,
    how="left"
)

merged = merged.rename(
    columns={
        "generated_accident_count": "historical_accident_count"
    }
)

merged.to_csv(MERGED_FILE, index=False)


# ------------------------------------------------------------
# 7. Print validation information
# ------------------------------------------------------------

print()
print("Synthetic accident data generation completed.")
print(f"Accident events: {len(events)}")
print(f"Road segments: {len(accident_features)}")
print(
    f"Road segments with at least one accident: "
    f"{(accident_features['historical_accident_count'] > 0).sum()}"
)
print()
print("Severity distribution:")
print(events["accident_severity"].value_counts())
print()
print("Time-period distribution:")
print(events["time_period"].value_counts())
print()
print("Files created:")
print(EVENT_FILE)
print(FEATURE_FILE)
print(MERGED_FILE)
