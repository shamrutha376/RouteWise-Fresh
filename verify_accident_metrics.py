from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
FEATURES = DATA / "X_features_with_accidents.csv"
EVENTS = DATA / "accident_events.csv"

def normalize_id(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return value

def add_pair(frame):
    frame = frame.copy()
    frame["_u"] = frame["u"].map(normalize_id)
    frame["_v"] = frame["v"].map(normalize_id)
    frame["_pair_u"] = frame["_u"].where(frame["_u"] <= frame["_v"], frame["_v"])
    frame["_pair_v"] = frame["_v"].where(frame["_u"] <= frame["_v"], frame["_u"])
    return frame

def main():
    for path in (FEATURES, EVENTS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run this script from your project root and confirm both CSVs are inside Data/."
            )

    print("Loading feature roads and accident events...")
    features = pd.read_csv(FEATURES, usecols=["u", "v", "historical_accident_count"])
    events = pd.read_csv(EVENTS)

    required = {"u", "v", "accident_severity"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"accident_events.csv is missing required columns: {sorted(missing)}")

    features = add_pair(features)
    events = add_pair(events)

    # Count accident event records per direction-independent road pair.
    event_counts = events.groupby(["_pair_u", "_pair_v"]).size().rename("recomputed_count")
    feature_pairs = features[["_pair_u", "_pair_v"]].drop_duplicates()
    matched = feature_pairs.merge(
        event_counts.reset_index(),
        on=["_pair_u", "_pair_v"],
        how="left"
    )
    matched["recomputed_count"] = matched["recomputed_count"].fillna(0).astype(int)

    stored = pd.to_numeric(features["historical_accident_count"], errors="coerce").fillna(0)
    severity = events["accident_severity"].astype(str).str.lower()
    print("\n=== Accident metrics verification ===")
    print(f"Accident event rows: {len(events):,}")
    print(f"Unique accident road pairs: {len(event_counts):,}")
    print(f"Unique road pairs in feature file: {len(feature_pairs):,}")
    print(f"Feature road pairs with >=1 event: {(matched['recomputed_count'] > 0).sum():,}")
    print(f"Feature rows with stored historical count > 0: {(stored > 0).sum():,} / {len(features):,}")
    print(f"Feature rows with recomputed event count > 0: {matched['recomputed_count'].gt(0).sum():,} unique pairs")
    print(f"Minor events: {severity.str.contains('minor|low', regex=True).sum():,}")
    print(f"Serious events: {severity.str.contains('serious|major|moderate', regex=True).sum():,}")
    print(f"Fatal events: {severity.str.contains('fatal|death', regex=True).sum():,}")

    for col in ("intersection_related", "road_departure"):
        if col in events.columns:
            values = pd.to_numeric(events[col], errors="coerce").fillna(0)
            print(f"{col} column: present; summed flags = {values.sum():,.0f}")
        else:
            print(f"{col} column: MISSING (the app will report this metric as zero)")

    print("\nInterpretation:")
    print("- Stored historical counts can be stale; the fixed app recomputes metrics from accident_events.csv.")
    print("- Roads with no matching event records should still show 0; that is expected.")
    print("- In Streamlit, inspect a road with a positive recomputed count and confirm the popup/route details show it.")

if __name__ == "__main__":
    main()
