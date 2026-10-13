"""Small SQLite storage layer for Dynamic Traffic AI."""

from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DB_FILE = DATA_DIR / "traffic_ai.db"


def _connect():
    DATA_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_FILE)


def build_database(modelling_df, road_df, poi=None, accident_file=None):
    """Create/update the local SQLite database after data preparation."""
    with _connect() as conn:
        modelling_df.to_sql("model_features", conn, if_exists="replace", index=False)
        road_df.to_sql("road_features", conn, if_exists="replace", index=False)

        if poi is not None and not poi.empty:
            poi_table = poi.copy()
            poi_table["geometry_wkt"] = poi_table.geometry.to_wkt()
            poi_table = pd.DataFrame(poi_table.drop(columns="geometry", errors="ignore"))
            poi_table.to_sql("infrastructure", conn, if_exists="replace", index=False)
        else:
            pd.DataFrame(columns=["poi_category", "geometry_wkt"]).to_sql(
                "infrastructure", conn, if_exists="replace", index=False
            )

        if accident_file is not None and Path(accident_file).exists():
            accidents = pd.read_csv(accident_file)
        else:
            accidents = pd.DataFrame(columns=["latitude", "longitude"])
        accidents.to_sql("accidents", conn, if_exists="replace", index=False)

        metadata = pd.DataFrame([
            ["study_area", "Tambaram, Chennai, Tamil Nadu, India"],
            ["database_version", "1.0"],
            ["traffic_targets", "Prototype/simulated until real time-stamped traffic observations are supplied"],
        ], columns=["key", "value"])
        metadata.to_sql("metadata", conn, if_exists="replace", index=False)


def read_table(table_name, columns="*"):
    with _connect() as conn:
        return pd.read_sql_query(f"SELECT {columns} FROM {table_name}", conn)
