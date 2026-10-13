from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import re
import time
import requests

from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
import geopandas as gpd
import plotly.express as px
import joblib
from shapely import wkt
from shapely.geometry import Point

from streamlit_folium import st_folium
from database import read_table
import streamlit.components.v1 as components


def render_chart(fig):
    """Apply the dashboard's light almond-pink chart styling consistently."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFAF7",
        plot_bgcolor="#FFFAF7",
        font=dict(color="#29212C", family="Arial, sans-serif", size=12),
        title_font=dict(color="#6843B8", size=17),
        legend=dict(font=dict(color="#29212C"), bgcolor="rgba(255,250,247,0.85)"),
        margin=dict(l=18, r=18, t=58, b=18),
        hoverlabel=dict(bgcolor="#29212C", font_color="#FFFFFF", bordercolor="#805AD5"),
    )
    fig.update_xaxes(
        title_font=dict(color="#29212C"),
        tickfont=dict(color="#514653"),
        gridcolor="#EADBD7",
        zerolinecolor="#D9C9C5",
        linecolor="#D9C9C5",
    )
    fig.update_yaxes(
        title_font=dict(color="#29212C"),
        tickfont=dict(color="#514653"),
        gridcolor="#EADBD7",
        zerolinecolor="#D9C9C5",
        linecolor="#D9C9C5",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


st.set_page_config(
    page_title="Dynamic Traffic AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
  --emerald: #20B486;
  --violet: #805AD5;
  --lavender: #D8C7FF;
  --mint: #E8F5EC;
  --almond: #FFF1EB;
  --almond-card: #FFFAF7;
  --ink: #29212C;
  --ink-muted: #6F626C;
  --border: #EADBD7;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--almond) !important;
  color: var(--ink) !important;
}
[data-testid="stHeader"] { background: rgba(255,241,235,.92) !important; }
[data-testid="stSidebar"] {
  background: #F8E5DF !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #6843B8 !important; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp [data-testid="stMarkdownContainer"] {
  color: var(--ink);
}
.stCaption, [data-testid="stCaptionContainer"] { color: var(--ink-muted) !important; }
.stButton > button, [data-testid="stFormSubmitButton"] button {
  background: var(--violet); color: #FFFFFF; border: 1px solid var(--violet);
  border-radius: 10px; font-weight: 600; transition: all .15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--emerald); border-color: var(--emerald); color: #172A25;
}
[data-testid="stMetric"] {
  background: var(--almond-card); padding: 16px 18px;
  border: 1px solid var(--border); border-top: 3px solid var(--emerald);
  border-radius: 14px; box-shadow: 0 3px 12px rgba(41,33,44,.045);
}
[data-testid="stMetricLabel"] { color: var(--ink-muted) !important; }
[data-testid="stMetricValue"] { color: #6843B8 !important; }
[data-testid="stMetricDelta"] { color: #168A67 !important; }
/* Ensure dataframe/table headers remain readable in the light theme. */
[data-testid="stDataFrame"] thead th,
[data-testid="stDataFrame"] th,
[data-testid="stTable"] thead th,
[data-testid="stTable"] th {
  background-color: #805AD5 !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
}
[data-testid="stRadio"] > label { display: none; }
div[role="radiogroup"] { gap: 10px; flex-wrap: wrap; }
div[role="radiogroup"] label {
  background: var(--almond-card); border: 1px solid #DCCDEB; padding: 9px 18px;
  border-radius: 10px; color: var(--ink) !important; transition: all .15s ease;
}
div[role="radiogroup"] label:has(input:checked) {
  background: var(--violet); border-color: var(--violet); color: #FFFFFF !important;
  box-shadow: 0 3px 10px rgba(128,90,213,.18);
}
div[role="radiogroup"] label:has(input:checked) * { color: #FFFFFF !important; }
div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: var(--almond-card); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stExpander"] {
  background: var(--almond-card); border: 1px solid var(--border); border-radius: 12px;
}
input, textarea, [data-baseweb="select"] > div {
  background-color: var(--almond-card) !important; color: var(--ink) !important;
  border-color: #DCCDEB !important;
}
hr { border-color: var(--border); }
a { color: #6843B8 !important; }
h1, h2, h3 { letter-spacing: -.02em; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
MODEL_DIR = ROOT / "models"
DB_FILE = DATA / "traffic_ai.db"
GRAPH_FILE = DATA / "road_network.graphml"
POI_FILE = DATA / "infrastructure_pois.geojson"
X_FEATURES_FILE = DATA / "X_features_with_accidents.csv"

TRAFFIC_MODEL_FILE = MODEL_DIR / "traffic_prediction_model.pkl"
RISK_MODEL_FILE = MODEL_DIR / "risk_prediction_model.pkl"
RISK_CLASSIFIER_FILE = MODEL_DIR / "risk_level_classifier.pkl"

CENTER_LAT = 12.9249
CENTER_LON = 80.1000
STUDY_AREA = "Tambaram, Chennai"


@st.cache_data
def load_features():
    data = pd.read_csv(X_FEATURES_FILE)
    events_file = DATA / "accident_events.csv"

    required_keys = ["u", "v"]
    if not all(c in data.columns for c in required_keys):
        raise ValueError(
            "Road feature data must contain u and v columns."
        )

    if not events_file.exists():
        raise FileNotFoundError(
            f"Accident event file not found: {events_file}"
        )

    events = pd.read_csv(events_file)

    if not all(c in events.columns for c in required_keys):
        raise ValueError(
            "accident_events.csv must contain u and v columns."
        )

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

    def add_road_pair(frame):
        frame = frame.copy()
        frame["_u"] = frame["u"].map(normalize_id)
        frame["_v"] = frame["v"].map(normalize_id)

        frame["_pair_u"] = frame.apply(
            lambda r: min(r["_u"], r["_v"]), axis=1
        )
        frame["_pair_v"] = frame.apply(
            lambda r: max(r["_u"], r["_v"]), axis=1
        )
        return frame

    data = add_road_pair(data)
    events = add_road_pair(events)

    severity = events.get(
        "accident_severity",
        pd.Series("Minor", index=events.index)
    ).astype(str).str.lower()

    events["_minor"] = severity.str.contains(
        "minor|low", regex=True
    ).astype(int)
    events["_serious"] = severity.str.contains(
        "serious|major|moderate", regex=True
    ).astype(int)
    events["_fatal"] = severity.str.contains(
        "fatal|death", regex=True
    ).astype(int)

    events["_weighted"] = (
        events["_minor"]
        + 2 * events["_serious"]
        + 3 * events["_fatal"]
    )

    peak = events.get(
        "peak_hour", pd.Series(0, index=events.index)
    )
    events["_peak"] = pd.to_numeric(
        peak, errors="coerce"
    ).fillna(0)

    period = events.get(
        "time_period", pd.Series("", index=events.index)
    ).astype(str).str.lower()
    events["_night"] = period.eq("night").astype(int)

    # Support the column names actually present in accident_events.csv,
    # while retaining compatibility with alternate prepared schemas.
    for sources, target in [
        (("intersection_related", "intersection_accident"), "_intersection"),
        (("road_departure", "road_departure_accident"), "_departure"),
    ]:
        source = next((name for name in sources if name in events.columns), None)
        values = (
            events[source] if source is not None
            else pd.Series(0, index=events.index)
        )
        events[target] = pd.to_numeric(
            values, errors="coerce"
        ).fillna(0)

    metrics = events.groupby(
        ["_pair_u", "_pair_v"], as_index=False
    ).agg(
        historical_accident_count=("_u", "size"),
        minor_accident_count=("_minor", "sum"),
        serious_accident_count=("_serious", "sum"),
        fatal_accident_count=("_fatal", "sum"),
        peak_hour_accident_count=("_peak", "sum"),
        night_accident_count=("_night", "sum"),
        intersection_accident_count=("_intersection", "sum"),
        road_departure_accident_count=("_departure", "sum"),
        severity_weighted_score=("_weighted", "sum"),
    )

    metrics["accident_risk_score"] = 0.0
    max_score = metrics["severity_weighted_score"].max()

    if max_score > 0:
        metrics["accident_risk_score"] = (
            metrics["severity_weighted_score"] / max_score * 100
        )

    # Preserve the historical count already stored in the feature dataset.
    # The event file may not contain every previously prepared count, so do
    # not discard the source count before joining the event-derived metrics.
    data["_source_historical_accident_count"] = pd.to_numeric(
        data.get("historical_accident_count", pd.Series(0, index=data.index)),
        errors="coerce"
    ).fillna(0)

    data = data.drop(
        columns=[
            c for c in data.columns
            if c in metrics.columns
        ],
        errors="ignore"
    )
    
    # Use the same ID normalization for roads and accident events.
    _u = data["u"].map(normalize_id)
    _v = data["v"].map(normalize_id)

    data["_pair_u"] = _u.where(_u <= _v, _v)
    data["_pair_v"] = _v.where(_u <= _v, _u)


    # Ensure the metrics table uses the same key format
    metrics["_pair_u"] = metrics["_pair_u"].astype(str)
    metrics["_pair_v"] = metrics["_pair_v"].astype(str)

    data = data.merge(
        metrics,
        on=["_pair_u", "_pair_v"],
        how="left",
        validate="many_to_one"
    )


    metric_columns = [
        "historical_accident_count",
        "minor_accident_count",
        "serious_accident_count",
        "fatal_accident_count",
        "peak_hour_accident_count",
        "night_accident_count",
        "intersection_accident_count",
        "road_departure_accident_count",
        "severity_weighted_score",
        "accident_risk_score",
    ]

    for column in metric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(0)

    # Use the larger of the prepared feature count and event-derived count.
    # This keeps valid historical counts while still incorporating events.
    data["historical_accident_count"] = np.maximum(
        data["historical_accident_count"],
        data["_source_historical_accident_count"]
    )
    data.drop(columns=["_source_historical_accident_count"], inplace=True)

    length_km = pd.to_numeric(
        data["length_m"], errors="coerce"
    ).fillna(0) / 1000

    data["accident_rate_per_km"] = np.where(
        length_km > 0,
        data["historical_accident_count"] / length_km,
        0
    )

    
# Normalize road node IDs to match the string IDs in road_network.graphml.
# This prevents IDs such as 12345.0 from failing to match node "12345".
    data["u"] = data["u"].map(normalize_id)
    data["v"] = data["v"].map(normalize_id)

    data.drop(
        columns=["_u", "_v", "_pair_u", "_pair_v"],
        inplace=True,
        errors="ignore"
    )

    return data


@st.cache_data
def load_pois():
    if not POI_FILE.exists():
        return None
    return gpd.read_file(POI_FILE)


@st.cache_resource
def load_graph():
    return nx.read_graphml(GRAPH_FILE)


@st.cache_resource
def load_models():
    return (
        joblib.load(TRAFFIC_MODEL_FILE),
        joblib.load(RISK_MODEL_FILE),
        joblib.load(RISK_CLASSIFIER_FILE)
    )


required = [
    GRAPH_FILE,
    TRAFFIC_MODEL_FILE,
    RISK_MODEL_FILE,
    RISK_CLASSIFIER_FILE,
]

missing = [str(path) for path in required if not path.exists()]

if missing:
    st.error("The project has not been prepared and trained yet.")
    st.write("Run `python prepare_data.py` and then `python train_models.py`.")
    for path in missing:
        st.write(path)
    st.stop()


df = load_features()
graph = load_graph()
traffic_model, risk_model, risk_classifier = load_models()
pois = load_pois()


# ------------------------------------------------------------
# PREDICTION HELPERS
# ------------------------------------------------------------

def get_time_period(hour):
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    return "Night"


def prepare_prediction_data(data):
    columns = [
        "length_m", "road_type", "lanes", "maxspeed",
        "u_degree", "v_degree", "u_betweenness", "v_betweenness",
        "road_connectivity", "road_centrality",
        "school_count", "hospital_count", "place_of_worship_count",
        "bus_stop_count", "office_count", "mall_count",
        "commercial_count", "retail_count", "college_count",
        "total_poi_count",
        "historical_accident_count", "minor_accident_count",
        "serious_accident_count", "fatal_accident_count",
        "peak_hour_accident_count", "night_accident_count",
        "intersection_accident_count", "road_departure_accident_count",
        "severity_weighted_score", "accident_risk_score",
        "accident_rate_per_km",
        "time_period", "hour", "peak_hour"
    ]

    data = data.copy()
    for column in columns:
        if column not in data.columns:
            if column in ["road_type", "time_period"]:
                data[column] = "unknown"
            else:
                data[column] = 0
    return data[columns]




def calculate_predictions(data):
    prediction_data = prepare_prediction_data(data)

    result = data.copy()
    result["predicted_traffic"] = traffic_model.predict(
        prediction_data
    )
    result["predicted_risk"] = np.clip(
        risk_model.predict(prediction_data), 0, 100
    )

    result["predicted_risk_level"] = pd.cut(
        result["predicted_risk"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return result

def unique_edges(data):
    keys = [column for column in ["u", "v", "key"] if column in data.columns]
    if keys:
        return data.drop_duplicates(subset=keys).copy()
    return data.copy()


def risk_color(level):
    level = str(level).lower()
    if level == "high":
        return "#d62728"
    if level == "medium":
        return "#ff9800"
    return "#2ca02c"


def road_point(graph_obj, node):
    data = graph_obj.nodes[node]
    return float(data["y"]), float(data["x"])


def nearest_node(graph_obj, latitude, longitude):
    best_node = None
    best_distance = float("inf")

    for node, data in graph_obj.nodes(data=True):
        try:
            lat = float(data["y"])
            lon = float(data["x"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = (lat - latitude) ** 2 + (lon - longitude) ** 2
        if distance < best_distance:
            best_distance = distance
            best_node = node

    return best_node


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def nearest_node_distance_km(graph_obj, node, latitude, longitude):
    node_lat, node_lon = road_point(graph_obj, node)
    return haversine_km(latitude, longitude, node_lat, node_lon)


def snap_location_to_road(graph_obj, location, max_distance_km=3.0):
    """
    Snap a geocoded location to the nearest actual road segment.

    The previous implementation measured distance only to the nearest
    graph NODE. That can reject valid locations that are close to a road
    but happen to be between two graph nodes. The graph edges already
    contain their full OSM LINESTRING geometry, so use that geometry.
    """
    point = Point(float(location["lon"]), float(location["lat"]))

    best = None
    best_distance_km = float("inf")

    for u, v, key, data in graph_obj.edges(keys=True, data=True):
        geometry_text = data.get("geometry")

        try:
            if geometry_text:
                road_geometry = wkt.loads(str(geometry_text))
                nearest_point = road_geometry.interpolate(
                    road_geometry.project(point)
                )
            else:
                # Fallback for an edge without geometry.
                start = graph_obj.nodes[u]
                end = graph_obj.nodes[v]
                from shapely.geometry import LineString
                road_geometry = LineString([
                    (float(start["x"]), float(start["y"])),
                    (float(end["x"]), float(end["y"]))
                ])
                nearest_point = road_geometry.interpolate(
                    road_geometry.project(point)
                )

            distance_km = haversine_km(
                float(location["lat"]),
                float(location["lon"]),
                float(nearest_point.y),
                float(nearest_point.x)
            )

            if distance_km < best_distance_km:
                best_distance_km = distance_km
                best = {
                    "u": str(u),
                    "v": str(v),
                    "key": key,
                    "lat": float(nearest_point.y),
                    "lon": float(nearest_point.x),
                    "distance_km": distance_km,
                    "geometry": road_geometry,
                }

        except Exception:
            continue

    if best is None or best_distance_km > max_distance_km:
        return None

    # Route from the endpoint nearest to the actual snapped point.
    # The previous code compared endpoints to the original place coordinate,
    # which could select the wrong end of a long road segment.
    u_lat, u_lon = road_point(graph_obj, best["u"])
    v_lat, v_lon = road_point(graph_obj, best["v"])

    u_distance = haversine_km(
        best["lat"], best["lon"], u_lat, u_lon
    )
    v_distance = haversine_km(
        best["lat"], best["lon"], v_lat, v_lon
    )

    best["node"] = best["u"] if u_distance <= v_distance else best["v"]
    return best


def create_route_graph(hour):
    """
    Build a directed routing graph with separate cost profiles.
    All profiles keep physical distance as the base cost, while traffic,
    predicted risk and accident risk influence the route differently.
    Values are normalised across the selected time slice so that metres
    do not overwhelm the prediction features.
    """
    period = get_time_period(hour)
    selected = df[(df["hour"] == hour) & (df["time_period"] == period)].copy()

    if selected.empty:
        selected = df[df["hour"] == hour].copy()
    if selected.empty:
        selected = df.copy()

    predictions = calculate_predictions(selected)
    route_graph = nx.DiGraph()

    for _, row in predictions.iterrows():
        try:
            u = str(row["u"]).strip()
            v = str(row["v"]).strip()

            if u.endswith(".0"):
                u = u[:-2]
            if v.endswith(".0"):
                v = v[:-2]
            length = max(float(row["length_m"]), 1.0)
            traffic = float(row["predicted_traffic"])
            risk = float(row["predicted_risk"])
            accident_risk = float(row.get("accident_risk_score", 0))

            if u not in graph.nodes or v not in graph.nodes:
                continue

            geometry = None
            if graph.has_edge(u, v):
                edge_candidates = graph.get_edge_data(u, v)
                if edge_candidates:
                    candidate = min(
                        edge_candidates.values(),
                        key=lambda item: float(item.get("length", length))
                    )
                    geometry = candidate.get("geometry")

            edge_data = {
                "weight": length,
                "length": length,
                "traffic": traffic,
                "risk": risk,
                "risk_level": row["predicted_risk_level"],
                "accident_risk": accident_risk,
                "historical_accidents": float(row.get("historical_accident_count", 0)),
                "severity_weighted_score": float(row.get("severity_weighted_score", 0)),
                "infrastructure_count": float(row.get("total_poi_count", 0)),
                "geometry": geometry,
            }

            
            if route_graph.has_edge(u, v):
                existing = route_graph[u][v]

    # Preserve the shortest road length, but don't lose accident
    # information when another record represents the same directed edge.
                existing["historical_accidents"] = max(
                    float(existing.get("historical_accidents", 0)),
                    edge_data["historical_accidents"]
                )
                existing["severity_weighted_score"] = max(
                    float(existing.get("severity_weighted_score", 0)),
                    edge_data["severity_weighted_score"]
                )
                existing["accident_risk"] = max(
                    float(existing.get("accident_risk", 0)),
                    edge_data["accident_risk"]
                )

                if length < existing["length"]:
                    existing.update(edge_data)
            else:
                route_graph.add_edge(u, v, **edge_data)

        except Exception:
            continue

    if route_graph.number_of_edges() == 0:
        return route_graph

    # Min-max normalisation makes each predicted factor comparable.
    edge_values = list(route_graph.edges(data=True))

    def normalise(attribute):
        values = [max(float(data.get(attribute, 0)), 0.0) for _, _, data in edge_values]
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        span = high - low
        for _, _, data in edge_values:
            value = max(float(data.get(attribute, 0)), 0.0)
            data[f"{attribute}_norm"] = (value - low) / span if span > 1e-9 else 0.0

    normalise("traffic")
    normalise("risk")
    normalise("accident_risk")
    normalise("infrastructure_count")

    # Prototype relative-risk estimate on a 0–100 scale.
    # This is not a calibrated accident probability.
    for _, _, data in edge_values:
        data["infrastructure_risk"] = 100 * (
            0.50 * data["risk_norm"]
            + 0.30 * data["traffic_norm"]
            + 0.20 * data["infrastructure_count_norm"]
        )
        data["infrastructure_risk_norm"] = (
            data["infrastructure_risk"] / 100
        )

    for _, _, data in edge_values:
        length = data["length"]
        traffic_n = data["traffic_norm"]
        risk_n = data["risk_norm"]
        accident_n = data["accident_risk_norm"]

        # Relative costs: faster favours low traffic; safest favours low
        # predicted risk and accident exposure; balanced considers all.
        data["fastest_weight"] = length * (1.0 + 0.90 * traffic_n)
        data["safest_weight"] = length * (
            1.0
            + 0.60 * risk_n
            + 0.70 * data["infrastructure_risk_norm"]
            + 0.35 * accident_n
        )

        data["balanced_weight"] = length * (
            1.0
            + 0.35 * traffic_n
            + 0.30 * risk_n
            + 0.40 * data["infrastructure_risk_norm"]
            + 0.25 * accident_n
        )
        data["weight"] = data["balanced_weight"]

    return route_graph


def find_diverse_routes(route_graph, source_node, destination_node, limit=3):
    """
    Find routes using fastest, safest and balanced costs.
    Prefer alternatives that share less road distance with already
    selected routes, while avoiding excessively long detours.
    """
    source_node = str(source_node)
    destination_node = str(destination_node)

    profiles = [
        ("Fastest", "fastest_weight"),
        ("Safest", "safest_weight"),
        ("Balanced", "balanced_weight"),
    ]

    if source_node not in route_graph or destination_node not in route_graph:
        return [], []

    if source_node == destination_node:
        return [], []

    chosen = []
    chosen_names = []
    used_edges = []

    def path_edges(path):
        return set(zip(path[:-1], path[1:]))

    def path_length(path):
        return sum(
            max(0.0, float(route_graph[u][v].get("length", 0)))
            for u, v in zip(path[:-1], path[1:])
        )

    def path_cost(path, weight_key):
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = route_graph[u][v]
            cost = edge.get(weight_key, edge.get("length", 1))
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                cost = float(edge.get("length", 1))

            if not cost >= 0:
                cost = float(edge.get("length", 1))

            total += cost

        return total

    def overlap_ratio(candidate, existing):
        candidate_edges = path_edges(candidate)
        existing_edges = path_edges(existing)

        candidate_length = path_length(candidate)
        if candidate_length <= 0:
            return 1.0

        shared_length = sum(
            max(0.0, float(route_graph[u][v].get("length", 0)))
            for u, v in candidate_edges.intersection(existing_edges)
        )

        return min(1.0, shared_length / candidate_length)

    def select_candidate(candidates, weight_key):
        valid = []

        for path in candidates:
            length = path_length(path)
            if length <= 0:
                continue

            # Avoid paths that are effectively duplicates of existing routes.
            overlaps = [
                overlap_ratio(path, existing)
                for existing in chosen
            ]

            # A path with very high overlap is acceptable only when
            # alternatives are unavailable or the route limit is nearly met.
            if overlaps and max(overlaps) >= 0.95:
                continue

            cost = path_cost(path, weight_key)

            # Penalize shared roads, but keep the route cost meaningful.
            penalty = 1.0 + (
                0.8 * sum(overlaps) / max(1, len(overlaps))
            )
            valid.append((cost * penalty, path))

        if not valid:
            return None

        valid.sort(key=lambda item: item[0])
        return valid[0][1]

    for label, weight_key in profiles:
        if len(chosen) >= limit:
            break

        try:
            generator = nx.shortest_simple_paths(
                route_graph,
                source_node,
                destination_node,
                weight=weight_key
            )

            # Examine more alternatives before deciding that profiles overlap.
            candidates = []
            for _, candidate in zip(range(40), generator):
                candidates.append(candidate)

        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            continue

        best_path = select_candidate(candidates, weight_key)

        if best_path is not None and best_path not in chosen:
            chosen.append(best_path)
            chosen_names.append(label)

    # Fill any remaining slots with balanced alternatives.
    if len(chosen) < limit:
        try:
            generator = nx.shortest_simple_paths(
                route_graph,
                source_node,
                destination_node,
                weight="balanced_weight"
            )

            for candidate in generator:
                if candidate in chosen:
                    continue

                length = path_length(candidate)
                if length <= 0:
                    continue

                overlaps = [
                    overlap_ratio(candidate, existing)
                    for existing in chosen
                ]

                # Skip near-duplicates when possible.
                if overlaps and max(overlaps) >= 0.95:
                    continue

                chosen.append(candidate)
                chosen_names.append("Alternative")

                if len(chosen) >= limit:
                    break

        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            pass

    return chosen[:limit], chosen_names[:limit]


def route_statistics(route, route_graph):
    total_distance = 0.0
    traffic_values = []
    risk_values = []
    risk_levels = []
    accident_risk_values = []
    accident_counts = []
    severity_values = []
    infrastructure_risk_values = []
    segments_with_accidents = 0
    total_route_segments = 0

    for u, v in zip(route[:-1], route[1:]):
        edge = route_graph.get_edge_data(u, v)
        if not edge:
            continue
        
        total_route_segments += 1
        if float(edge.get("historical_accidents", 0)) > 0:
            segments_with_accidents += 1
        infrastructure_risk_values.append(
            float(edge.get("infrastructure_risk", 0))
        )

        total_distance += float(edge.get("length", 0))
        traffic_values.append(float(edge.get("traffic", 0)))
        risk_values.append(float(edge.get("risk", 0)))
        risk_levels.append(str(edge.get("risk_level", "Low")))
        accident_risk_values.append(float(edge.get("accident_risk", 0)))
        accident_counts.append(float(edge.get("historical_accidents", 0)))
        severity_values.append(float(edge.get("severity_weighted_score", 0)))

    average_traffic = float(np.mean(traffic_values)) if traffic_values else 0.0
    average_risk = float(np.mean(risk_values)) if risk_values else 0.0
    average_accident_risk = float(np.mean(accident_risk_values)) if accident_risk_values else 0.0
    total_accidents = float(np.sum(accident_counts)) if accident_counts else 0.0
    severity_score = float(np.sum(severity_values)) if severity_values else 0.0
    average_infrastructure_risk = (
        float(np.mean(infrastructure_risk_values))
        if infrastructure_risk_values else 0.0
    )

    route_score = (
        0.40 * total_distance
        + 0.25 * average_traffic * 10
        + 0.20 * average_risk * 10
        + 0.15 * average_accident_risk * 10
    )

    if "High" in risk_levels:
        highest = "High"
    elif "Medium" in risk_levels:
        highest = "Medium"
    else:
        highest = "Low"

    return {
        "distance_km": total_distance / 1000,
        "traffic": average_traffic,
        "risk": average_risk,
        "risk_level": highest,
        "accident_risk": average_accident_risk,
        "historical_accidents": total_accidents,
        "severity_weighted_score": severity_score,
        "score": route_score,
        "segments_with_accidents": segments_with_accidents,
        "infrastructure_risk": average_infrastructure_risk,
        "total_route_segments": total_route_segments,
    }


def route_coordinates(route, route_graph):
    """
    Return the actual OSM road geometry for the route.

    Falling back to node coordinates is retained for safety, but when
    geometry is available the map follows the real curved road instead
    of drawing straight lines between graph nodes.
    """
    coordinates = []

    for u, v in zip(route[:-1], route[1:]):
        try:
            edge = route_graph.get_edge_data(u, v)
            geometry_text = edge.get("geometry") if edge else None

            if geometry_text:
                geom = (
                    geometry_text
                    if hasattr(geometry_text, "coords")
                    else wkt.loads(str(geometry_text))
                )

                segment = [(float(lat), float(lon)) for lon, lat in geom.coords]

                # Orient the segment to match u -> v.
                u_lat, u_lon = road_point(graph, u)
                if segment:
                    first_error = haversine_km(
                        u_lat, u_lon, segment[0][0], segment[0][1]
                    )
                    last_error = haversine_km(
                        u_lat, u_lon, segment[-1][0], segment[-1][1]
                    )
                    if last_error < first_error:
                        segment.reverse()

                if coordinates and segment:
                    # Avoid duplicating the join point.
                    if haversine_km(
                        coordinates[-1][0], coordinates[-1][1],
                        segment[0][0], segment[0][1]
                    ) < 0.001:
                        coordinates.extend(segment[1:])
                    else:
                        coordinates.extend(segment)
                else:
                    coordinates.extend(segment)

            else:
                if not coordinates:
                    coordinates.append(road_point(graph, u))
                coordinates.append(road_point(graph, v))

        except Exception:
            if not coordinates:
                try:
                    coordinates.append(road_point(graph, u))
                except Exception:
                    pass
            try:
                coordinates.append(road_point(graph, v))
            except Exception:
                pass

    return coordinates


def geocode_location(query):
    """
    Resolve a place name to real coordinates. Do not discard geocoder results
    just because they fall outside a guessed rectangle; the road-network
    snapping step below is responsible for checking local road coverage.
    """
    query = str(query).strip()
    if not query:
        return None

    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2
    }

    search_queries = [
        f"{query}, Tambaram, Chennai, Tamil Nadu, India",
        f"{query}, Chennai, Tamil Nadu, India",
        f"{query}, Tamil Nadu, India",
    ]
    candidates = []
    seen = set()
    errors = []

    # Try Nominatim first, then Photon as a fallback if it returns no results.
    providers = [
        (
            "https://nominatim.openstreetmap.org/search",
            {"format": "jsonv2", "limit": 8, "countrycodes": "in",
             "addressdetails": 1},
            "nominatim",
        ),
        (
            "https://photon.komoot.io/api/",
            {"limit": 8, "lang": "en"},
            "photon",
        ),
    ]

    for endpoint, base_params, provider in providers:
        provider_found = False
        for search_query in search_queries:
            params = dict(base_params)
            if provider == "nominatim":
                params["q"] = search_query
            else:
                params["q"] = search_query

            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    headers={
                        "User-Agent": "Dynamic-Traffic-AI-student-project/1.0",
                        "Accept": "application/json",
                    },
                    timeout=12,
                )
                response.raise_for_status()
                payload = response.json()

                if provider == "nominatim":
                    results = payload if isinstance(payload, list) else []
                    for item in results:
                        try:
                            lat, lon = float(item["lat"]), float(item["lon"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        display = item.get("display_name", query)
                        address_text = display.lower()
                        score = sum(token in address_text for token in query_tokens)
                        key = (round(lat, 6), round(lon, 6))
                        if key not in seen:
                            seen.add(key)
                            candidates.append({
                                "lat": lat, "lon": lon,
                                "display": display, "_match_score": score,
                            })
                            provider_found = True
                else:
                    features = payload.get("features", []) if isinstance(payload, dict) else []
                    for item in features:
                        try:
                            lon, lat = item["geometry"]["coordinates"][:2]
                            lat, lon = float(lat), float(lon)
                        except (KeyError, TypeError, ValueError, IndexError):
                            continue
                        props = item.get("properties", {})
                        display = ", ".join(
                            str(props[key]) for key in
                            ("name", "street", "district", "city", "state", "country")
                            if props.get(key)
                        ) or query
                        address_text = display.lower()
                        score = sum(token in address_text for token in query_tokens)
                        key = (round(lat, 6), round(lon, 6))
                        if key not in seen:
                            seen.add(key)
                            candidates.append({
                                "lat": lat, "lon": lon,
                                "display": display, "_match_score": score,
                            })
                            provider_found = True

            except (requests.RequestException, ValueError, TypeError) as error:
                errors.append(f"{provider}: {error}")

            time.sleep(1.0)

        if provider_found:
            break

    if not candidates:
        if errors:
            st.warning(
                "Location search failed for both place names. Check your internet "
                "connection. Geocoding details: " + errors[-1]
            )
        return None

    # Rank name matches first, then prefer places close to this loaded road graph.
    # A candidate is not accepted for routing until it successfully snaps to a road.
    ranked = []
    for candidate in candidates:
        snap = snap_location_to_road(graph, candidate, max_distance_km=3.0)
        distance = snap["distance_km"] if snap is not None else 999.0
        ranked.append((candidate["_match_score"], -distance, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = ranked[0][2]
    best.pop("_match_score", None)
    return best


def add_poi_layer(map_object):
    if pois is None or pois.empty:
        return

    poi_copy = pois
    if poi_copy.crs and str(poi_copy.crs) != "EPSG:4326":
        poi_copy = poi_copy.to_crs("EPSG:4326")

    for _, row in poi_copy.head(300).iterrows():
        try:
            geometry = row.geometry
            if geometry is None:
                continue
            if geometry.geom_type == "Point":
                point = geometry
            else:
                point = geometry.centroid

            category = str(row.get("poi_category", "Infrastructure")).replace("_", " ").title()
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=3,
                weight=1,
                fill=True,
                fill_opacity=0.75,
                popup=category,
            ).add_to(map_object)
        except Exception:
            continue


def build_live_map(predictions, hour):
    map_object = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    for _, row in predictions.head(1800).iterrows():
        try:
            u = str(row["u"])
            v = str(row["v"])
            if u not in graph.nodes or v not in graph.nodes:
                continue

            start = road_point(graph, u)
            end = road_point(graph, v)
            level = row["predicted_risk_level"]

            popup = (
                f"Road: {row['road_type']}<br>"
                f"Predicted traffic: {row['predicted_traffic']:.2f}<br>"
                f"Predicted risk: {row['predicted_risk']:.2f}<br>"
                f"Risk level: {level}<br>"
                f"Infrastructure count: {row.get('total_poi_count', 0):.0f}<br>"
                f"Historical accidents: {row.get('historical_accident_count', 0):.0f}"
            )

            folium.PolyLine(
                [start, end],
                color=risk_color(level),
                weight=4,
                opacity=0.75,
                popup=popup,
            ).add_to(map_object)
        except Exception:
            continue

    add_poi_layer(map_object)
    return map_object


# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("Dynamic Traffic AI")
st.sidebar.caption("Tambaram traffic and route intelligence")

st.markdown("### Dynamic Traffic AI")
st.caption("Infrastructure-aware traffic and road-risk intelligence · Tambaram")
page = st.radio(
    "Navigation",
    ["Live Traffic", "Analyse Route", "Infrastructure Analysis"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_page_navigation",
)
st.markdown("---")


# ------------------------------------------------------------
# LIVE TRAFFIC
# ------------------------------------------------------------

if page == "Live Traffic":
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    default_hour = now.hour

    st.title("Live Traffic")
    st.write("Model-predicted traffic conditions for the Tambaram study area.")

    hour = st.slider(
        "Traffic time",
        min_value=0,
        max_value=23,
        value=default_hour,
        format="%d:00",
    )

    period = get_time_period(hour)

# Use one base record for each unique road segment.
# This prevents the four temporal copies of each road
# from being displayed repeatedly.
    selected = unique_edges(df).copy()

# Apply the currently selected time to every road segment.
# The trained models use hour, time_period and peak_hour
# as temporal prediction features.
    selected["hour"] = hour
    selected["time_period"] = period
    selected["peak_hour"] = int(period in ["Morning", "Evening"])

    predictions = calculate_predictions(selected)

    # Presentation fail-safe for the demo: a model can return exact zero for
    # some hours. Keep the raw output for transparency, and use a small
    # data-derived floor only for map/display so roads do not appear as
    # missing traffic. This is a display fallback, not a measured live value.
    predictions["predicted_traffic_raw"] = pd.to_numeric(
        predictions["predicted_traffic"], errors="coerce"
    ).fillna(0)
    zero_traffic_mask = predictions["predicted_traffic_raw"] <= 0
    positive_traffic = predictions.loc[
        predictions["predicted_traffic_raw"] > 0, "predicted_traffic_raw"
    ]
    traffic_floor = (
        float(positive_traffic.quantile(0.05))
        if not positive_traffic.empty else 1.0
    )
    traffic_floor = max(traffic_floor, 1.0)
    predictions["traffic_display_fallback"] = zero_traffic_mask
    predictions.loc[zero_traffic_mask, "predicted_traffic"] = traffic_floor

    c1, c2, c3 = st.columns(3)
    c1.metric("Study area", "Tambaram")
    c2.metric("Prediction period", f"{period} {hour:02d}:00")
    c3.metric("Road segments", f"{len(unique_edges(predictions)):,}")

    st.markdown("---")

    avg_traffic = predictions["predicted_traffic"].mean()
    avg_risk = predictions["predicted_risk"].mean()
    high_risk = (predictions["predicted_risk_level"].astype(str).str.lower() == "high").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average predicted traffic", f"{avg_traffic:.2f}")
    c2.metric("Average predicted risk", f"{avg_risk:.2f}")
    c3.metric("High-risk road segments", f"{high_risk:,}")
    fallback_count = int(predictions["traffic_display_fallback"].sum())
    if fallback_count:
        st.caption(
            f"Demo display fallback applied to {fallback_count:,} road segments "
            f"with zero model output (display floor: {traffic_floor:.2f}). "
            "These are estimated display values, not live sensor readings."
        )

    st.subheader("Tambaram Traffic Map")
    st.caption("Green, orange and red roads represent model-predicted low, medium and high risk levels.")

    traffic_map = build_live_map(predictions, hour)
    st_folium(traffic_map, width=None, height=650, key="live_traffic_map")


# ------------------------------------------------------------
# ANALYSE ROUTE
# ------------------------------------------------------------

elif page == "Analyse Route":
    st.title("Analyse Route")
    st.write("Enter two locations inside the Tambaram study area to compare alternative routes.")

    col1, col2 = st.columns(2)

    with col1:
        source_text = st.text_input(
            "From",
            placeholder="Example: Tambaram Railway Station"
        )

    with col2:
        destination_text = st.text_input(
            "To",
            placeholder="Example: Tambaram Bus Stand"
        )

    hour = st.slider(
        "Departure time",
        0,
        23,
        datetime.now(ZoneInfo("Asia/Kolkata")).hour,
        format="%d:00",
    )

    if st.button("Analyse Route", type="primary"):
        # Never keep an old successful route visible after a new failed attempt.
        st.session_state.pop("route_analysis", None)
        if not source_text or not destination_text:
            st.warning("Enter both From and To locations.")
        else:
            with st.spinner("Finding locations and analysing routes..."):
                source = geocode_location(source_text)
                time.sleep(1.0)
                destination = geocode_location(destination_text)

                if source is None or destination is None:
                    st.error(
                        "The location service could not find one or both place names. "
                        "Check the network warning above, or try a more specific name "
                        "such as a landmark plus Tambaram, Chennai."
                    )
                    st.write("Start location:", source_text, "→", source)
                    st.write("Destination:", destination_text, "→", destination)
                else:
                    source_snap = snap_location_to_road(
                        graph, source, max_distance_km=0.75
                    )
                    destination_snap = snap_location_to_road(graph, destination, max_distance_km=0.75)

                    if source_snap is None or destination_snap is None:
                        st.error(
                            "The selected location could not be matched to a road in the current "
                            "Tambaram road network. Try the full place name or a nearby road name."
                        )
                    else:
                        source_node = source_snap["node"]
                        destination_node = destination_snap["node"]

                        source_distance = source_snap["distance_km"]
                        destination_distance = destination_snap["distance_km"]

                        st.caption(
                            f"Road snapping: From {source_distance:.2f} km, "
                            f"To {destination_distance:.2f} km from the selected road network."
                        )

                        route_graph = create_route_graph(hour)

                        try:
                            routes, route_names = find_diverse_routes(
                                route_graph,
                                str(source_node),
                                str(destination_node),
                                limit=3,
                            )
                        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
                            routes, route_names = [], []

                        if not routes:
                            st.error(
                                "No route could be found between these locations "
                                "in the current road network."
                        )
                        else:
                            results = []

                            for index, route in enumerate(routes, start=1):
                                stats = route_statistics(route, route_graph)
                                stats["Route"] = route_names[index - 1] if index - 1 < len(route_names) else f"Alternative {index}"
                                results.append(stats)

                            st.session_state.route_analysis = {
                                "source": source,
                                "destination": destination,
                                "source_snap": source_snap,
                                "destination_snap": destination_snap,
                                "source_text": source_text,
                                "destination_text": destination_text,
                                "hour": hour,
                                "routes": routes,
                                "route_graph": route_graph,
                                "results": results,
        }
    if "route_analysis" in st.session_state:
        analysis = st.session_state.route_analysis
        results = analysis["results"]

        
        result_df = pd.DataFrame(results)
        result_df = result_df[
            [
                "Route", "distance_km", "traffic", "risk",
                "accident_risk", "infrastructure_risk",
                "historical_accidents", "severity_weighted_score",
                "risk_level", "score"
            ]
        ].copy()

        # A comparative exposure proxy prevents an all-zero accident column
        # from making the prototype look broken, without fabricating historical
        # accident counts. It is derived from model risk, traffic and the
        # infrastructure-based score, and is explicitly labelled as an estimate.
        traffic_component = pd.to_numeric(result_df["traffic"], errors="coerce").fillna(0)
        traffic_max = max(float(traffic_component.max()), 1.0)
        traffic_norm = (traffic_component / traffic_max) * 100
        risk_proxy = (
            0.50 * pd.to_numeric(result_df["infrastructure_risk"], errors="coerce").fillna(0)
            + 0.30 * pd.to_numeric(result_df["risk"], errors="coerce").fillna(0)
            + 0.20 * traffic_norm
        ).clip(0, 100)
        result_df["accident_exposure_proxy"] = risk_proxy.round(2)

        result_df.columns = [
            "Route", "Distance (km)", "Predicted Traffic",
            "Predicted Risk", "Historical Accident Risk",
            "Infrastructure-Based Risk (0-100)",
            "Historical Accidents", "Severity-Weighted Accidents",
            "Risk Level", "Route Score", "Estimated Accident Exposure (0-100)"
        ]

        best_index = int(result_df["Route Score"].idxmin())
        best_route_name = result_df.loc[best_index, "Route"]

        st.success(f"Route analysis completed. The current prototype score selects {best_route_name}.")

        # Fail-safe presentation: preserve positive measured values, but do not
        # present an unmatched zero as proof that a road has no accidents.
        # The internal numeric values remain unchanged for calculations.
        measured_columns = [
            "Historical Accident Risk",
            "Historical Accidents",
            "Severity-Weighted Accidents",
        ]
        display_df = result_df.copy()
        for column in measured_columns:
            numeric_values = pd.to_numeric(display_df[column], errors="coerce").fillna(0)
            if column == "Historical Accident Risk":
                display_df[column] = [
                    f"{value:.2f}" if value > 0 else "—"
                    for value in numeric_values
                ]
            else:
                display_df[column] = [
                    f"{value:.0f}" if value > 0 else "—"
                    for value in numeric_values
                ]

        
        st.subheader("Route Comparison")

        table_html = display_df.to_html(
            index=False,
            escape=True,
            classes="route-comparison-table",
            border=0,
        )

        styled_table_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    padding: 4px;
                    background-color: #FFF1EB;
                    font-family: Arial, sans-serif;
                }}

                .table-wrapper {{
                    width: 100%;
                    overflow-x: auto;
                    border: 1px solid #EADBD7;
                    border-radius: 10px;
                    background-color: #FFFAF7;
                }}

                .route-comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #FFFAF7;
                    color: #29212C;
                    font-size: 13px;
                }}

                .route-comparison-table thead th {{
                    background-color: #805AD5;
                    color: #FFFFFF;
                    padding: 12px 10px;
                    text-align: left;
                    font-weight: 700;
                    border: 1px solid #D8C7FF;
                    white-space: nowrap;
                }}

                .route-comparison-table tbody td {{
                    background-color: #FFFAF7;
                    color: #29212C;
                    padding: 10px;
                    border: 1px solid #EADBD7;
                    white-space: nowrap;
                }}

                .route-comparison-table tbody tr:nth-child(even) td {{
                    background-color: #F5E6F1;
                }}

                .route-comparison-table tbody tr:hover td {{
                    background-color: #E8F5EC;
                }}
            </style>
        </head>
        <body>
            <div class="table-wrapper">
                {table_html}
            </div>
        </body>
        </html>
        """

        # Render the HTML in a component instead of displaying the HTML
        # source as text inside the Streamlit page.
        components.html(
            styled_table_html,
            height=260,
            scrolling=True,
        )
        st.caption(
            "Estimated Accident Exposure (0–100) is a model-based comparison proxy "
            "using infrastructure, predicted risk and traffic; it is not a count or "
            "probability of accidents. A dash in historical columns means no matching "
            "historical records were available for those route segments."
        )

        matched_total = sum(
            int(stats.get("segments_with_accidents", 0)) for stats in results
        )
        segment_total = sum(
            int(stats.get("total_route_segments", 0)) for stats in results
        )
        if matched_total:
            st.caption(
                f"Historical event matches: {matched_total} route-segment matches "
                f"across {segment_total} route segments."
            )
        else:
            st.caption(
                "Historical events did not match the road segments on these routes; "
                "the estimated exposure proxy is shown for comparison instead."
            )

        best = results[best_index]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Selected route", best_route_name)
        c2.metric("Distance", f"{best['distance_km']:.2f} km")
        c3.metric("Predicted traffic", f"{best['traffic']:.2f}")
        c4.metric("Predicted risk", f"{best['risk']:.2f}")
        if float(best.get("accident_risk", 0)) > 0:
            c5.metric("Historical Accident Risk", f"{best['accident_risk']:.2f}")
        else:
            c5.metric(
                "Estimated Accident Exposure",
                f"{float(best.get('infrastructure_risk', 0)):.2f}",
                help="Comparative prototype estimate, not historical accident data."
            )

        st.caption(
            "Infrastructure-Based Risk and Estimated Accident Exposure are comparative prototype scores, not calibrated accident probabilities. "
            "Historical accident values are shown only when records match the route segments; a dash means records did not match, not that accidents never occurred. "
            "Routes use different priorities: fastest (traffic-aware), safest (predicted-risk/available-accident-data-aware), and balanced. "
            "The route score is a prototype comparison metric, not a verified travel-time estimate."
        )

        route_map = folium.Map(
            location=[analysis["source_snap"]["lat"], analysis["source_snap"]["lon"]],
            zoom_start=13,
            tiles="OpenStreetMap",
        )

        folium.Marker(
            [analysis["source"]["lat"], analysis["source"]["lon"]],
            tooltip=f"From: {analysis['source_text']}",
        ).add_to(route_map)

        folium.Marker(
            [analysis["destination"]["lat"], analysis["destination"]["lon"]],
            tooltip=f"To: {analysis['destination_text']}",
        ).add_to(route_map)

        # Show the exact road-snapped points used by the routing graph.
        source_snap = analysis.get("source_snap")
        destination_snap = analysis.get("destination_snap")

        if source_snap:
            folium.CircleMarker(
                [source_snap["lat"], source_snap["lon"]],
                radius=6,
                color="#1f77b4",
                fill=True,
                fill_opacity=1,
                tooltip=f"Road snap: {source_snap['distance_km']:.3f} km",
            ).add_to(route_map)

        if destination_snap:
            folium.CircleMarker(
                [destination_snap["lat"], destination_snap["lon"]],
                radius=6,
                color="#d62728",
                fill=True,
                fill_opacity=1,
                tooltip=f"Road snap: {destination_snap['distance_km']:.3f} km",
            ).add_to(route_map)

        route_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        route_names = [item.get("Route", f"Alternative {i + 1}") for i, item in enumerate(results)]

        st.subheader("Route Map")

        # Clear route legend
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            background: white;
            padding: 12px 16px;
            border: 2px solid #555;
            border-radius: 8px;
            font-size: 14px;
            color: #222;
        ">
            <b>Route Legend</b><br>
            <span style="color:#1f77b4;">━━</span> {route_names[0] if len(route_names) > 0 else "Route 1"}<br>
            <span style="color:#ff7f0e;">━━</span> {route_names[1] if len(route_names) > 1 else "Route 2"}<br>
            <span style="color:#2ca02c;">━━</span> {route_names[2] if len(route_names) > 2 else "Route 3"}
        </div>
        """

        route_map.get_root().html.add_child(
            folium.Element(legend_html)
        )

        for index, route in enumerate(analysis["routes"]):
            coordinates = route_coordinates(route, analysis["route_graph"])

            # Extend the visible route from the actual snapped road point
            # to the first/last graph node used by the routing algorithm.
            # This prevents the route from visually stopping before the
            # requested location.
            if coordinates:
                source_snap = analysis.get("source_snap")
                destination_snap = analysis.get("destination_snap")

                if source_snap:
                    coordinates.insert(
                        0,
                        (source_snap["lat"], source_snap["lon"])
                    )

                if destination_snap:
                    coordinates.append(
                        (destination_snap["lat"], destination_snap["lon"])
                    )

            if len(coordinates) >= 2:
                route_color = route_colors[index % len(route_colors)]

                folium.PolyLine(
                    coordinates,
                    color=route_color,
                    weight=7 if index == best_index else 5,
                    opacity=0.9 if index == best_index else 0.6,
                    tooltip=route_names[index] if index < len(route_names) else f"Alternative {index + 1}",
                ).add_to(route_map)

                # Permanent Route 1 / Route 2 / Route 3 label
                middle = coordinates[len(coordinates) // 2]

                label_html = f"""
                <div style="
                    background-color: white;
                    border: 2px solid {route_color};
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 13px;
                    font-weight: bold;
                    color: {route_color};
                    white-space: nowrap;
                ">
                    {route_names[index] if index < len(route_names) else f"Alternative {index + 1}"}
                </div>
                """

                folium.Marker(
                    middle,
                    icon=folium.DivIcon(html=label_html)
                ).add_to(route_map)

        st_folium(
            route_map,
            width=None,
            height=650,
            key="route_analysis_map"
        )



# ------------------------------------------------------------
# INFRASTRUCTURE ANALYSIS
# ------------------------------------------------------------

else:
    st.title("Infrastructure Analysis")
    st.write("Explore how nearby infrastructure, road-network structure, traffic predictions, and risk patterns vary across the Tambaram study area.")

    base = unique_edges(df)
    infrastructure_columns = [
        "school_count", "hospital_count", "place_of_worship_count",
        "bus_stop_count", "office_count", "mall_count",
        "commercial_count", "retail_count", "college_count"
    ]
    available = [column for column in infrastructure_columns if column in base.columns]
    summary = pd.DataFrame({
        "Infrastructure": [column.replace("_count", "").replace("_", " ").title() for column in available],
        "Feature Count": [pd.to_numeric(base[column], errors="coerce").fillna(0).sum() for column in available],
        "Road Segments Affected": [(pd.to_numeric(base[column], errors="coerce").fillna(0) > 0).sum() for column in available],
    })
    palette = ["#20B486", "#805AD5", "#D8C7FF", "#E8F5EC", "#45C4A0", "#A78BFA", "#5EEAD4", "#B9A1F5", "#6EE7B7"]
    chart_template = "plotly_white"

    # Top-level indicators
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Road segments", f"{len(base):,}")
    c2.metric("Infrastructure categories", str(len(available)))
    c3.metric("Segments near infrastructure", f"{int((pd.to_numeric(base.get('total_poi_count', pd.Series(0, index=base.index)), errors='coerce').fillna(0) > 0).sum()):,}")
    c4.metric("Maximum POIs on a segment", f"{pd.to_numeric(base['total_poi_count'], errors='coerce').fillna(0).max():.0f}" if "total_poi_count" in base.columns else "N/A")

    st.subheader("Infrastructure footprint")
    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.bar(summary.sort_values("Feature Count", ascending=True), x="Feature Count", y="Infrastructure", orientation="h",
                     color="Infrastructure", color_discrete_sequence=palette, title="Infrastructure features by category",
                     labels={"Feature Count": "Feature count", "Infrastructure": "Category"}, template=chart_template)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="#FFFAF7", plot_bgcolor="#FFFAF7", font_color="#29212C")
        render_chart(fig)
        if not summary.empty:
            top_row = summary.loc[summary["Feature Count"].idxmax()]
            st.caption(f"Finding: {top_row['Infrastructure']} has the largest total feature count in the prepared dataset.")
    with right:
        pie_summary = summary[summary["Feature Count"] > 0]
        if not pie_summary.empty:
            fig = px.pie(pie_summary, names="Infrastructure", values="Feature Count", hole=0.55,
                         color_discrete_sequence=palette, title="Share of infrastructure features", template=chart_template)
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Category")
            render_chart(fig)
            st.caption("Finding: the donut chart compares each infrastructure category's share of all recorded POI features.")
        else:
            st.info("No positive infrastructure category counts are available to plot.")

    st.subheader("Traffic and risk explorer")
    hour = st.slider("Focus hour", 0, 23, 18, format="%d:00", key="infra_focus_hour")
    period = get_time_period(hour)
    selected = df[(df["hour"] == hour) & (df["time_period"] == period)].copy()
    if selected.empty:
        selected = df[df["hour"] == hour].copy()
    predictions = calculate_predictions(selected)
    sample = predictions.sample(min(2500, len(predictions)), random_state=42) if len(predictions) else predictions

    left, right = st.columns(2)
    with left:
        fig = px.scatter(sample, x="total_poi_count", y="predicted_traffic", color="predicted_risk_level",
                         color_discrete_map={"Low": "#20B486", "Medium": "#D8C7FF", "High": "#F0445B"},
                         hover_data=[c for c in ["road_type", "historical_accident_count"] if c in sample.columns],
                         title=f"Infrastructure density vs predicted traffic · {hour:02d}:00",
                         labels={"total_poi_count": "Nearby infrastructure count", "predicted_traffic": "Predicted traffic", "predicted_risk_level": "Predicted risk"}, template=chart_template)
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="#FFFAF7", plot_bgcolor="#FFFAF7", font_color="#29212C")
        render_chart(fig)
        st.caption("Finding: each point represents a road segment; color separates the model's predicted risk classes.")
    with right:
        risk_counts = predictions["predicted_risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk Level").reset_index(name="Road Segments")
        fig = px.bar(risk_counts, x="Risk Level", y="Road Segments", color="Risk Level",
                     color_discrete_map={"Low": "#20B486", "Medium": "#D8C7FF", "High": "#F0445B"},
                     title=f"Predicted risk distribution · {hour:02d}:00", template=chart_template)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="#FFFAF7", plot_bgcolor="#FFFAF7", font_color="#29212C")
        render_chart(fig)
        dominant = risk_counts.loc[risk_counts["Road Segments"].idxmax(), "Risk Level"] if len(risk_counts) else "Unavailable"
        st.caption(f"Finding: {dominant} is the most common predicted risk class at the selected hour.")

    # Hour-by-hour model predictions for a consistent sample; used for line and animated views.
    st.subheader("Daily traffic pattern")
    hourly_sample = base.sample(min(350, len(base)), random_state=21).copy() if len(base) else base.copy()
    hourly_frames = []
    for h in range(24):
        frame = hourly_sample.copy()
        frame["hour"] = h
        frame["time_period"] = get_time_period(h)
        if "peak_hour" in frame.columns:
            frame["peak_hour"] = int(h in [7, 8, 9, 17, 18, 19])
        if len(frame):
            hourly_pred = calculate_predictions(frame)
            hourly_frames.append(pd.DataFrame({
                "Hour": [f"{h:02d}:00"] * len(hourly_pred),
                "hour_num": [h] * len(hourly_pred),
                "Predicted traffic": hourly_pred["predicted_traffic"].to_numpy(),
                "Predicted risk": hourly_pred["predicted_risk"].to_numpy(),
                "Risk level": hourly_pred["predicted_risk_level"].to_numpy(),
                "Infrastructure count": pd.to_numeric(hourly_pred.get("total_poi_count", 0), errors="coerce").fillna(0).to_numpy() if isinstance(hourly_pred.get("total_poi_count", 0), pd.Series) else np.zeros(len(hourly_pred)),
                "Road type": hourly_pred["road_type"].astype(str).to_numpy() if "road_type" in hourly_pred.columns else ["Unknown"] * len(hourly_pred),
                "Segment sample": np.arange(len(hourly_pred)),
            }))
    if hourly_frames:
        hourly_data = pd.concat(hourly_frames, ignore_index=True)
        hourly_line = hourly_data.groupby("hour_num", as_index=False).agg({"Predicted traffic": "mean", "Predicted risk": "mean"})
        hourly_line["Time"] = hourly_line["hour_num"].map(lambda h: f"{h:02d}:00")
        line_long = hourly_line.melt(id_vars=["hour_num", "Time"], value_vars=["Predicted traffic", "Predicted risk"], var_name="Metric", value_name="Average score")
        fig = px.line(line_long, x="Time", y="Average score", color="Metric", markers=True,
                      color_discrete_map={"Predicted traffic": "#20B486", "Predicted risk": "#805AD5"},
                      title="Average predicted traffic and risk across 24 hours", template=chart_template)
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
        render_chart(fig)
        peak_row = hourly_line.loc[hourly_line["Predicted traffic"].idxmax()]
        st.caption(f"Finding: the sampled road segments reach their highest average predicted traffic around {int(peak_row['hour_num']):02d}:00.")

        st.subheader("Motion view · how sampled road segments change by hour")
        motion = hourly_data.copy()
        motion["Hour"] = motion["hour_num"].map(lambda h: f"{h:02d}:00")
        fig = px.scatter(motion, x="Infrastructure count", y="Predicted traffic", animation_frame="Hour",
                         animation_group="Segment sample", color="Risk level", size="Predicted risk",
                         hover_name="Road type", range_x=[0, max(1, float(motion["Infrastructure count"].max()))],
                         range_y=[0, max(100, float(motion["Predicted traffic"].max()) * 1.05)],
                         color_discrete_map={"Low": "#20B486", "Medium": "#D8C7FF", "High": "#F0445B"},
                         title="Animated traffic prediction by hour", template=chart_template)
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), transition_duration=350)
        render_chart(fig)
        st.caption("Finding: use the play control to compare model predictions across the day; this is a sampled prototype simulation, not live sensor playback.")

    st.subheader("Road network structure")
    left, right = st.columns(2)
    with left:
        if "road_centrality" in base.columns:
            fig = px.histogram(base, x="road_centrality", nbins=30, color_discrete_sequence=["#805AD5"], title="Road centrality distribution", template=chart_template)
            render_chart(fig)
            st.caption("Finding: centrality highlights segments that may play a more connected role in the network.")
        if "road_connectivity" in base.columns:
            fig = px.histogram(base, x="road_connectivity", nbins=30, color_discrete_sequence=["#20B486"], title="Road connectivity distribution", template=chart_template)
            render_chart(fig)
            st.caption("Finding: connectivity shows how the road segments vary in local network links.")
    with right:
        if "road_type" in base.columns:
            road_counts = base["road_type"].fillna("Unknown").value_counts().head(12).rename_axis("Road Type").reset_index(name="Segments")
            fig = px.bar(road_counts, x="Road Type", y="Segments", color="Road Type", color_discrete_sequence=palette, title="Most represented road types", template=chart_template)
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="#FFFAF7", plot_bgcolor="#FFFAF7", font_color="#29212C")
            render_chart(fig)
            st.caption(f"Finding: {road_counts.iloc[0]['Road Type']} is the most represented road type in the plotted top categories." if len(road_counts) else "Finding: road type data is unavailable.")

        accident_total = float(pd.to_numeric(base.get("historical_accident_count", pd.Series(0, index=base.index)), errors="coerce").fillna(0).sum())
        if accident_total > 0:
            fig = px.histogram(base, x="historical_accident_count", nbins=25, color_discrete_sequence=["#F0445B"], title="Prepared accident-feature distribution", template=chart_template)
            render_chart(fig)
            st.caption("Finding: this distribution reflects the accident features present in the prepared prototype dataset.")
        else:
            st.info("Historical accident observations are not available for this analysis; no accident counts are inferred here.")

    st.caption("Data note: traffic and risk are model predictions from the prepared prototype features. Animated and hourly views re-run the model on a fixed sample; they are not live sensor measurements.")

