
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"

features = pd.read_csv(DATA / "X_features_with_accidents.csv")
events = pd.read_csv(DATA / "accident_events.csv")

def normalize_id(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(-1)
        .astype("int64")
        .astype(str)
    )

for frame in (features, events):
    for col in ("u", "v", "key"):
        frame[col] = normalize_id(frame[col])

feature_keys = set(
    map(tuple, features[["u", "v", "key"]].drop_duplicates().values)
)
event_keys = set(
    map(tuple, events[["u", "v", "key"]].drop_duplicates().values)
)

matched = feature_keys & event_keys

print("Feature rows:", len(features))
print("Unique feature road segments:", len(feature_keys))
print("Accident records:", len(events))
print("Unique accident road segments:", len(event_keys))
print("Matching road segments:", len(matched))
print("Unmatched accident road segments:", len(event_keys - feature_keys))
