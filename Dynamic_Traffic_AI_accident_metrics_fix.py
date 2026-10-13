from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import time
import requests

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
import geopandas as gpd
import plotly.express as px
import joblib

from streamlit_folium import st_folium
from database import read_table


st.set_page_config(
    page_title="Dynamic Traffic AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Load road features and attach event-derived accident metrics by OSM edge ID."""
    data = pd.read_csv(X_FEATURES_FILE)
    events_file = DATA / "accident_events.csv"
    keys = ["u", "v", "key"]
    if not events_file.exists() or not all(k in data.columns for k in keys):
        return data

    events = pd.read_csv(events_file)
    if not all(k in events.columns for k in keys):
        return data

    # Normalize GraphML/CSV IDs: values like 123.0 and 123 must match.
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

    for frame in (data, events):
        for key in keys:
            frame[key] = frame[key].map(normalize_id)

    severity = events.get("accident_severity", pd.Series("Minor", index=events.index)).astype(str).str.lower()
    events["_minor"] = severity.str.contains("minor|low").astype(int)
    events["_serious"] = severity.str.contains("serious|major|moderate").astype(int)
    events["_fatal"] = severity.str.contains("fatal|death").astype(int)
    events["_weighted"] = events["_minor"] + 2 * events["_serious"] + 3 * events["_fatal"]

    hour_series = pd.to_numeric(events.get("hour", pd.Series(np.nan, index=events.index)), errors="coerce")
    if hour_series.isna().all() and "date" in events.columns:
        hour_series = pd.to_datetime(events["date"], errors="coerce").dt.hour
    events["_peak"] = hour_series.between(7, 10) | hour_series.between(16, 20)
    events["_night"] = hour_series.ge(21) | hour_series.lt(6)

    agg = events.groupby(keys, dropna=False).agg(
        historical_accident_count=(keys[0], "size"),
        minor_accident_count=("_minor", "sum"),
        serious_accident_count=("_serious", "sum"),
        fatal_accident_count=("_fatal", "sum"),
        peak_hour_accident_count=("_peak", "sum"),
        night_accident_count=("_night", "sum"),
        severity_weighted_score=("_weighted", "sum"),
    ).reset_index()

    # Drop stale versions first so event data replaces all-zero old columns.
    metric_cols = [c for c in agg.columns if c not in keys]
    data = data.drop(columns=[c for c in metric_cols if c in data.columns], errors="ignore")
    data = data.merge(agg, on=keys, how="left", validate="many_to_one")
    for c in metric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

    lengths_km = pd.to_numeric(data.get("length_m", 0), errors="coerce").fillna(0) / 1000
    data["accident_rate_per_km"] = np.where(lengths_km > 0, data["historical_accident_count"] / lengths_km, 0)
    max_weight = float(data["severity_weighted_score"].max())
    data["accident_risk_score"] = data["severity_weighted_score"] / max_weight if max_weight > 0 else 0.0
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
    "accident_rate_per_km",

    "time_period",
    "hour",
    "peak_hour"
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
    result["predicted_traffic"] = traffic_model.predict(prediction_data)
    result["predicted_risk"] = risk_model.predict(prediction_data)
    result["predicted_risk_level"] = risk_classifier.predict(prediction_data)
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


def create_route_graph(hour):
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
            u = str(row["u"])
            v = str(row["v"])
            length = float(row["length_m"])
            traffic = float(row["predicted_traffic"])
            risk = float(row["predicted_risk"])
            accidents = float(row.get("historical_accident_count", 0) or 0)
            accident_risk = float(row.get("accident_risk_score", 0) or 0)
            severity_score = float(row.get("severity_weighted_score", 0) or 0)

            if u not in graph.nodes or v not in graph.nodes:
                continue

            # Prototype composite route cost.
            weight = (
                0.45 * length
                + 0.30 * traffic * 10
                + 0.25 * risk * 10
            )

            if route_graph.has_edge(u, v):
                if weight < route_graph[u][v]["weight"]:
                    route_graph[u][v].update({
                        "weight": weight,
                        "length": length,
                        "traffic": traffic,
                        "risk": risk,
                        "risk_level": row["predicted_risk_level"],
                        "accidents": accidents,
                        "accident_risk": accident_risk,
                        "severity_score": severity_score,
                    })
            else:
                route_graph.add_edge(
                    u, v,
                    weight=weight,
                    length=length,
                    traffic=traffic,
                    risk=risk,
                    risk_level=row["predicted_risk_level"],
                    accidents=accidents,
                    accident_risk=accident_risk,
                    severity_score=severity_score,
                )
        except Exception:
            continue

    return route_graph


def route_statistics(route, route_graph):
    total_distance = 0.0
    traffic_values = []
    risk_values = []
    risk_levels = []
    accident_risks = []
    total_accidents = 0.0
    total_severity_score = 0.0

    for u, v in zip(route[:-1], route[1:]):
        edge = route_graph.get_edge_data(u, v)
        if not edge:
            continue
        total_distance += float(edge.get("length", 0))
        traffic_values.append(float(edge.get("traffic", 0)))
        risk_values.append(float(edge.get("risk", 0)))
        risk_levels.append(str(edge.get("risk_level", "Low")))
        accident_risks.append(float(edge.get("accident_risk", 0)))
        total_accidents += float(edge.get("accidents", 0))
        total_severity_score += float(edge.get("severity_score", 0))

    average_accident_risk = float(np.mean(accident_risks)) if accident_risks else 0.0
    average_traffic = float(np.mean(traffic_values)) if traffic_values else 0.0
    average_risk = float(np.mean(risk_values)) if risk_values else 0.0
    route_score = (0.40 * total_distance + 0.25 * average_traffic * 10
                   + 0.20 * average_risk * 10 + 0.15 * average_accident_risk * 10)

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
        "severity_weighted_score": total_severity_score,
        "score": route_score,
    }


def route_coordinates(route):
    coordinates = []
    for node in route:
        try:
            coordinates.append(road_point(graph, node))
        except Exception:
            continue
    return coordinates


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(query):
    query = query.strip()
    if not query:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "Dynamic-Traffic-AI-student-project/1.0"
    }

    # Try Tambaram-specific search first
    queries = [
        f"{query}, Tambaram, Chennai, Tamil Nadu, India",
        f"{query}, Chennai, Tamil Nadu, India",
        f"{query}, Tamil Nadu, India",
    ]

    for search_query in queries:
        params = {
            "q": search_query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in",
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            results = response.json()

            if results:
                return {
                    "lat": float(results[0]["lat"]),
                    "lon": float(results[0]["lon"]),
                    "display": results[0].get(
                        "display_name",
                        query
                    ),
                }

        except Exception:
            continue

    return None

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

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Traffic",
        "Analyse Route",
        "Infrastructure Analysis",
    ]
)


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
    # Use one record per road, then set the selected hour explicitly.
    # This makes the slider change the temporal inputs for every segment.
    selected = unique_edges(df).copy()
    selected["hour"] = hour
    selected["time_period"] = period
    selected["peak_hour"] = int(period in ["Morning", "Evening"])
    predictions = calculate_predictions(selected)

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
        if not source_text or not destination_text:
            st.warning("Enter both From and To locations.")
        else:
            with st.spinner("Finding locations and analysing routes..."):
                source = geocode_location(source_text)
                time.sleep(1.0)
                destination = geocode_location(destination_text)

                if source is None or destination is None:
                    st.error("One or both locations could not be found. Try a more specific Tambaram location name.")
                else:
                    source_node = nearest_node(graph, source["lat"], source["lon"])
                    destination_node = nearest_node(graph, destination["lat"], destination["lon"])

                    source_distance = nearest_node_distance_km(graph, source_node, source["lat"], source["lon"])
                    destination_distance = nearest_node_distance_km(graph, destination_node, destination["lat"], destination["lon"])

                    if source_distance > 2.5 or destination_distance > 2.5:
                        st.error("The selected location is outside the current Tambaram road-network coverage.")
                    else:
                        route_graph = create_route_graph(hour)

                        try:
                            generator = nx.shortest_simple_paths(
                                route_graph,
                                str(source_node),
                                str(destination_node),
                                weight="weight",
                            )
                            routes = []
                            for route in generator:
                                routes.append(route)
                                if len(routes) == 3:
                                    break
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            routes = []

                        if not routes:
                            st.error("No route could be found between these locations in the current road network.")
                        else:
                            results = []
                            for index, route in enumerate(routes, start=1):
                                stats = route_statistics(route, route_graph)
                                stats["Route"] = f"Route {index}"
                                results.append(stats)

                            st.session_state.route_analysis = {
                                "source": source,
                                "destination": destination,
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
            ["Route", "distance_km", "traffic", "risk", "accident_risk", "historical_accidents", "severity_weighted_score", "risk_level", "score"]
        ]
        result_df.columns = [
            "Route", "Distance (km)", "Predicted Traffic",
            "Predicted Risk", "Accident Risk", "Historical Accidents", "Severity-Weighted Accidents", "Risk Level", "Route Score"
        ]

        best_index = int(result_df["Route Score"].idxmin())
        best_route_name = result_df.loc[best_index, "Route"]

        st.success(f"Route analysis completed. The current prototype score selects {best_route_name}.")

        st.subheader("Route Comparison")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        best = results[best_index]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Selected route", best_route_name)
        c2.metric("Distance", f"{best['distance_km']:.2f} km")
        c3.metric("Predicted traffic", f"{best['traffic']:.2f}")
        c4.metric("Predicted risk", f"{best['risk']:.2f}")
        c5.metric("Accident risk", f"{best['accident_risk']:.2f}")

        st.caption(
            "Accident metrics are aggregated from Data/accident_events.csv by matching OSM edge IDs. "
            "They represent the supplied event dataset, not independently verified official accident records."
        )

        route_map = folium.Map(
            location=[analysis["source"]["lat"], analysis["source"]["lon"]],
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

        route_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        st.subheader("Route Map")

        # Clear route legend
        legend_html = """
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
            <span style="color:#1f77b4;">━━</span> Route 1<br>
            <span style="color:#ff7f0e;">━━</span> Route 2<br>
            <span style="color:#2ca02c;">━━</span> Route 3
        </div>
        """

        route_map.get_root().html.add_child(
            folium.Element(legend_html)
        )

        for index, route in enumerate(analysis["routes"]):
            coordinates = route_coordinates(route)

            if len(coordinates) >= 2:
                route_color = route_colors[index % len(route_colors)]

                folium.PolyLine(
                    coordinates,
                    color=route_color,
                    weight=7 if index == best_index else 5,
                    opacity=0.9 if index == best_index else 0.6,
                    tooltip=f"Route {index + 1}",
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
                    Route {index + 1}
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
    st.write("Infrastructure and road-network features used by the traffic and risk prediction pipeline.")

    base = unique_edges(df)

    infrastructure_columns = [
        "school_count", "hospital_count", "place_of_worship_count",
        "bus_stop_count", "office_count", "mall_count",
        "commercial_count", "retail_count", "college_count"
    ]

    available = [column for column in infrastructure_columns if column in base.columns]
    summary = pd.DataFrame({
        "Infrastructure": [column.replace("_count", "").replace("_", " ").title() for column in available],
        "Feature Count": [base[column].sum() for column in available],
        "Road Segments Affected": [(base[column] > 0).sum() for column in available],
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Road segments", f"{len(base):,}")
    c2.metric("Infrastructure categories", str(len(available)))
    c3.metric("Maximum POI feature count", f"{base['total_poi_count'].max():.0f}" if "total_poi_count" in base.columns else "0")

    st.subheader("Infrastructure Feature Distribution")
    fig = px.bar(
        summary,
        x="Infrastructure",
        y="Feature Count",
        title="Infrastructure Features Represented Across Road Segments",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Infrastructure and Predicted Traffic")
    hour = st.slider("Analysis time", 0, 23, 18, format="%d:00")
    period = get_time_period(hour)
    selected = unique_edges(df).copy()
    selected["hour"] = hour
    selected["time_period"] = period
    selected["peak_hour"] = int(period in ["Morning", "Evening"])
    predictions = calculate_predictions(selected)

    sample = predictions.sample(min(3000, len(predictions)), random_state=42)
    fig = px.scatter(
        sample,
        x="total_poi_count",
        y="predicted_traffic",
        color="predicted_risk_level",
        hover_data=["road_type", "historical_accident_count"],
        title=f"Infrastructure Density vs Predicted Traffic — {period} {hour:02d}:00",
        labels={"total_poi_count": "Nearby Infrastructure Feature Count", "predicted_traffic": "Predicted Traffic"},
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Road Network Characteristics")
        if "road_centrality" in base.columns:
            fig = px.histogram(base, x="road_centrality", nbins=30, title="Road Centrality")
            st.plotly_chart(fig, use_container_width=True)
        if "road_connectivity" in base.columns:
            fig = px.histogram(base, x="road_connectivity", nbins=30, title="Road Connectivity")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Historical Accident Feature")
        accident_total = float(base.get("historical_accident_count", pd.Series(dtype=float)).sum())

        if accident_total > 0:
            fig = px.histogram(
                base,
                x="historical_accident_count",
                nbins=25,
                title="Historical Accident Counts by Road Segment",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No non-zero Tambaram historical accident records are currently loaded. "
                "Add Data/accidents_tambaram.csv with latitude and longitude columns and rerun data preparation."
            )

        if "road_type" in base.columns:
            road_counts = base["road_type"].value_counts().head(15).reset_index()
            road_counts.columns = ["Road Type", "Segments"]
            fig = px.bar(road_counts, x="Road Type", y="Segments", title="Road Types")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted Risk Distribution")
    risk_counts = predictions["predicted_risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Road Segments"]
    fig = px.bar(risk_counts, x="Risk Level", y="Road Segments", title="Predicted Risk Levels")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The current traffic and risk targets are prototype targets generated during data preparation. "
        "Connecting real time-stamped traffic observations is the next step for a production-grade live prediction system."
    )
