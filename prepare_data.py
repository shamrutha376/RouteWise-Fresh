"""
Dynamic Traffic AI - Data Preparation
Study area: Tambaram, Chennai, Tamil Nadu

This script:
1. Downloads the OSM driving road network.
2. Downloads infrastructure POIs in small, separate queries.
3. Calculates road-network metrics.
4. Calculates nearby infrastructure counts/distances.
5. Supports a local Tambaram accident CSV when available.
6. Creates a modelling dataset with temporal traffic/risk prototype labels.

Run:
    python prepare_data.py
"""

from pathlib import Path
import warnings
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from database import build_database

warnings.filterwarnings("ignore")

# ============================================================
# STUDY AREA
# ============================================================

STUDY_AREA = "Tambaram, Chennai, Tamil Nadu, India"

# Approximate central Tambaram point.
# Change ONLY these values if you later want another locality.
CENTER = (12.9249, 80.1000)

# 2 km is deliberately used for the first run so Overpass queries
# remain manageable. Increase after the pipeline works.
ROAD_RADIUS_M = 2000
POI_RADIUS_M = 2000

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

ROAD_GRAPH = DATA_DIR / "road_network.graphml"
ROAD_GRAPH_PROJECTED = DATA_DIR / "road_network_projected.graphml"
POI_FILE = DATA_DIR / "infrastructure_pois.geojson"
ROAD_FEATURE_FILE = DATA_DIR / "osm_road_features.csv"
NODE_METRIC_FILE = DATA_DIR / "node_network_metrics.csv"
ACCESSIBILITY_FILE = DATA_DIR / "accessibility_features.csv"
FINAL_FEATURE_FILE = DATA_DIR / "X_features.csv"
ADJACENCY_FILE = DATA_DIR / "adjacency_matrix.npy"
ACCIDENT_FILE = DATA_DIR / "accidents_tambaram.csv"

# ============================================================
# OSM SETTINGS
# ============================================================

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.requests_timeout = 300
ox.settings.overpass_url = "https://overpass.kumi.systems/api"

# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def download_pois():
    """
    Read infrastructure POIs from the local BBBike OpenStreetMap
    GeoPackage instead of querying Overpass.

    The rest of the data-preparation pipeline is intentionally kept
    unchanged. The GeoPackage contains OSM points and multipolygons,
    so both layers are checked for infrastructure categories.
    """

    print("\nReading infrastructure POIs from local GeoPackage...")

    GPKG_FILE = DATA_DIR / (
        "planet_80.0851,12.8911_80.184,12.9519-geopackage"
        "/planet_80.0851,12.8911_80.184,12.9519.gpkg"
    )

    if not GPKG_FILE.exists():
        print(f"GeoPackage not found: {GPKG_FILE}")
        print("No POIs were loaded.")
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    # These are the OSM categories used by the existing pipeline.
    poi_categories = {
        "school": ("amenity", "school"),
        "hospital": ("amenity", "hospital"),
        "place_of_worship": ("amenity", "place_of_worship"),
        "bus_stop": ("highway", "bus_stop"),
        "college": ("amenity", "college"),
        "office": ("office", None),
        "mall": ("shop", "mall"),
        "commercial": ("landuse", "commercial"),
        "retail": ("landuse", "retail"),
    }

    layers = ["points", "multipolygons"]
    frames = []

    for layer in layers:
        try:
            layer_gdf = gpd.read_file(
                GPKG_FILE,
                layer=layer
            )
        except Exception as exc:
            print(f"Could not read GeoPackage layer '{layer}': {exc}")
            continue

        if layer_gdf.empty:
            continue

        if layer_gdf.crs is None:
            layer_gdf = layer_gdf.set_crs("EPSG:4326")
        elif layer_gdf.crs.to_string() != "EPSG:4326":
            layer_gdf = layer_gdf.to_crs("EPSG:4326")

        for category, (key, value) in poi_categories.items():
            if key not in layer_gdf.columns:
                continue

            column = layer_gdf[key]

            if value is None:
                mask = column.notna() & (
                    column.astype(str).str.strip() != ""
                )
            else:
                mask = column.astype(str).str.lower() == value.lower()

            selected = layer_gdf.loc[mask].copy()

            if selected.empty:
                continue

            selected["poi_category"] = category
            selected["osm_type"] = layer

            # GeoPackage uses osm_id for the original OSM identifier.
            if "osm_id" not in selected.columns:
                selected["osm_id"] = selected.index.astype(str)

            frames.append(selected)

    # Some useful OSM tags can be stored inside other_tags rather than
    # dedicated columns. Parse them for bus-stop/public-transport cases.
    try:
        points = gpd.read_file(GPKG_FILE, layer="points")
        if not points.empty:
            if points.crs is None:
                points = points.set_crs("EPSG:4326")

            other_tags = points.get(
                "other_tags",
                pd.Series(index=points.index, dtype="object")
            ).fillna("").astype(str).str.lower()

            bus_mask = (
                points.get(
                    "highway",
                    pd.Series(index=points.index, dtype="object")
                ).astype(str).str.lower().eq("bus_stop")
                | other_tags.str.contains(
                    '"public_transport"=>"platform"',
                    regex=False
                )
                | other_tags.str.contains(
                    '"public_transport"=>"stop_position"',
                    regex=False
                )
            )

            bus_points = points.loc[bus_mask].copy()

            if not bus_points.empty:
                bus_points["poi_category"] = "bus_stop"
                bus_points["osm_type"] = "points"

                if "osm_id" not in bus_points.columns:
                    bus_points["osm_id"] = bus_points.index.astype(str)

                frames.append(bus_points)
    except Exception as exc:
        print(f"Could not inspect additional bus-stop tags: {exc}")

    if not frames:
        print("\nNo infrastructure POIs were found in the GeoPackage.")
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    poi_all = gpd.GeoDataFrame(
        pd.concat(
            frames,
            ignore_index=True
        ),
        geometry="geometry",
        crs="EPSG:4326"
    )

    # Keep only the categories expected by the existing accessibility
    # and modelling functions.
    poi_all = poi_all[
        poi_all["poi_category"].isin(poi_categories.keys())
    ].copy()

    # Remove duplicate OSM objects that may have been captured by both
    # the dedicated tag and the additional bus-stop tag logic.
    poi_all = poi_all.drop_duplicates(
        subset=[
            "poi_category",
            "osm_type",
            "osm_id"
        ]
    ).reset_index(drop=True)

    poi_all.to_file(
        POI_FILE,
        driver="GeoJSON"
    )

    print(
        f"\nSaved infrastructure POIs: {POI_FILE}"
    )

    print(
        f"Total POIs: {len(poi_all)}"
    )

    print("\nInfrastructure summary:")

    print(
        poi_all["poi_category"].value_counts()
    )

    return poi_all

def graph_matches_study_area(G):
    """Check whether an existing GraphML file belongs to Tambaram."""
    if not G or len(G.nodes) == 0:
        return False

    latitudes = []
    longitudes = []

    for _, data in G.nodes(data=True):
        try:
            latitudes.append(float(data["y"]))
            longitudes.append(float(data["x"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not latitudes:
        return False

    median_lat = float(np.median(latitudes))
    median_lon = float(np.median(longitudes))

    # About 5 km tolerance around the selected Tambaram centre.
    distance_km = (
        ((median_lat - CENTER[0]) * 111.0) ** 2
        + ((median_lon - CENTER[1]) * 108.5) ** 2
    ) ** 0.5

    return distance_km <= 5.0

def build_road_network():
    if ROAD_GRAPH.exists():
        print("\nChecking existing road network...")
        G = ox.load_graphml(ROAD_GRAPH)

        if graph_matches_study_area(G):
            print("Existing graph matches the Tambaram study area.")
            print(f"Loaded {len(G.nodes):,} nodes and {len(G.edges):,} edges.")
            return G

        print("Existing graph belongs to a different study area.")
        print("Removing stale road graph and downloading the Tambaram network.")
        ROAD_GRAPH.unlink(missing_ok=True)

    print("\nDownloading OpenStreetMap road network for Tambaram...")

    G = ox.graph_from_point(
        CENTER,
        dist=ROAD_RADIUS_M,
        network_type="drive",
        simplify=True
    )

    # Current OSMnx versions already provide edge length for the
    # simplified drive graph. We intentionally do not call the
    # removed ox.add_edge_lengths() function.
    ox.save_graphml(G, ROAD_GRAPH)

    print(f"Saved road graph: {ROAD_GRAPH}")
    print(f"Nodes: {len(G.nodes):,}")
    print(f"Edges: {len(G.edges):,}")

    return G


def calculate_network_metrics(G):
    print("\nCalculating road-network metrics...")

    # Undirected graph for structural centrality.
    Gu = ox.convert.to_undirected(G)

    degree_dict = dict(Gu.degree())

    # Betweenness can be expensive on a large network.
    # k limits the number of sampled nodes while still producing
    # a useful prototype feature.
    sample_k = min(300, len(Gu))

    if len(Gu) > 2:
        try:
            betweenness = nx.betweenness_centrality(
                Gu,
                k=sample_k,
                normalized=True,
                seed=42
            )
        except Exception:
            betweenness = {node: 0.0 for node in Gu.nodes}
    else:
        betweenness = {node: 0.0 for node in Gu.nodes}

    metrics = pd.DataFrame({
        "node_id": list(Gu.nodes),
        "degree": [degree_dict.get(n, 0) for n in Gu.nodes],
        "betweenness_centrality": [
            betweenness.get(n, 0.0) for n in Gu.nodes
        ]
    })

    metrics.to_csv(NODE_METRIC_FILE, index=False)

    print(f"Saved node metrics: {NODE_METRIC_FILE}")

    return metrics


def create_edge_features(G, node_metrics):
    print("\nCreating road-segment features...")

    rows = []

    node_metric_map = node_metrics.set_index("node_id").to_dict("index")

    for u, v, key, data in G.edges(keys=True, data=True):

        length = float(data.get("length", 0) or 0)

        highway = data.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unknown"

        lanes = data.get("lanes", np.nan)
        if isinstance(lanes, list):
            lanes = lanes[0] if lanes else np.nan

        maxspeed = data.get("maxspeed", np.nan)
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0] if maxspeed else np.nan

        rows.append({
            "u": u,
            "v": v,
            "key": key,
            "length_m": length,
            "road_type": str(highway),
            "lanes": pd.to_numeric(lanes, errors="coerce"),
            "maxspeed": pd.to_numeric(
                str(maxspeed).split()[0],
                errors="coerce"
            ),
            "u_degree": node_metric_map.get(u, {}).get("degree", 0),
            "v_degree": node_metric_map.get(v, {}).get("degree", 0),
            "u_betweenness": node_metric_map.get(
                u, {}
            ).get("betweenness_centrality", 0),
            "v_betweenness": node_metric_map.get(
                v, {}
            ).get("betweenness_centrality", 0),
        })

    road_df = pd.DataFrame(rows)

    road_df["road_connectivity"] = (
        road_df["u_degree"] + road_df["v_degree"]
    ) / 2

    road_df["road_centrality"] = (
        road_df["u_betweenness"] + road_df["v_betweenness"]
    ) / 2

    road_df["lanes"] = road_df["lanes"].fillna(
        road_df["lanes"].median()
        if road_df["lanes"].notna().any()
        else 1
    )

    road_df["maxspeed"] = road_df["maxspeed"].fillna(
        road_df["maxspeed"].median()
        if road_df["maxspeed"].notna().any()
        else 40
    )

    road_df.to_csv(ROAD_FEATURE_FILE, index=False)

    print(f"Saved road features: {ROAD_FEATURE_FILE}")

    return road_df


def create_adjacency_matrix(G):
    print("\nCreating adjacency matrix...")

    nodes = list(G.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    adjacency = np.zeros(
        (len(nodes), len(nodes)),
        dtype=np.uint8
    )

    for u, v in G.edges():
        if u in node_index and v in node_index:
            i = node_index[u]
            j = node_index[v]
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    np.save(ADJACENCY_FILE, adjacency)

    print(f"Saved adjacency matrix: {ADJACENCY_FILE}")

    return adjacency


def create_accessibility_features(G, poi):
    print("\nCreating infrastructure accessibility features...")

    if poi.empty:
        print("No POIs available; creating zero-valued accessibility features.")
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(
            G,
            nodes=True,
            edges=True
        )

        result = pd.DataFrame({
            "node_id": nodes_gdf.index,
            "school_count": 0,
            "hospital_count": 0,
            "place_of_worship_count": 0,
            "bus_stop_count": 0,
            "office_count": 0,
            "mall_count": 0,
            "commercial_count": 0,
            "retail_count": 0,
            "college_count": 0,
            "total_poi_count": 0,
        })

        result.to_csv(ACCESSIBILITY_FILE, index=False)
        return result

    nodes_gdf, _ = ox.graph_to_gdfs(
        G,
        nodes=True,
        edges=True
    )

    # Work in a projected CRS for spatial distances.
    nodes_projected = nodes_gdf.to_crs("EPSG:32644")
    poi_projected = poi.to_crs("EPSG:32644")

    # 250 m is used as the local infrastructure influence radius.
    influence_radius = 250

    counts = []

    for node_id, node_row in nodes_projected.iterrows():

        buffer = node_row.geometry.buffer(influence_radius)

        nearby = poi_projected[
            poi_projected.geometry.intersects(buffer)
        ]

        category_counts = nearby["poi_category"].value_counts()

        counts.append({
            "node_id": node_id,
            "school_count": int(category_counts.get("school", 0)),
            "hospital_count": int(category_counts.get("hospital", 0)),
            "place_of_worship_count": int(
                category_counts.get("place_of_worship", 0)
            ),
            "bus_stop_count": int(category_counts.get("bus_stop", 0)),
            "office_count": int(category_counts.get("office", 0)),
            "mall_count": int(category_counts.get("mall", 0)),
            "commercial_count": int(
                category_counts.get("commercial", 0)
            ),
            "retail_count": int(category_counts.get("retail", 0)),
            "college_count": int(category_counts.get("college", 0)),
        })

    result = pd.DataFrame(counts)

    count_columns = [
        "school_count",
        "hospital_count",
        "place_of_worship_count",
        "bus_stop_count",
        "office_count",
        "mall_count",
        "commercial_count",
        "retail_count",
        "college_count",
    ]

    result["total_poi_count"] = result[count_columns].sum(axis=1)

    result.to_csv(ACCESSIBILITY_FILE, index=False)

    print(f"Saved accessibility features: {ACCESSIBILITY_FILE}")

    return result


def attach_node_features_to_edges(G, road_df, accessibility):
    print("\nCombining node and road features...")

    access_map = accessibility.set_index("node_id").to_dict("index")

    rows = []

    for _, row in road_df.iterrows():
        u_features = access_map.get(row["u"], {})
        v_features = access_map.get(row["v"], {})

        record = row.to_dict()

        for col in [
            "school_count",
            "hospital_count",
            "place_of_worship_count",
            "bus_stop_count",
            "office_count",
            "mall_count",
            "commercial_count",
            "retail_count",
            "college_count",
            "total_poi_count",
        ]:
            record[col] = (
                float(u_features.get(col, 0))
                + float(v_features.get(col, 0))
            ) / 2

        rows.append(record)

    final_df = pd.DataFrame(rows)

    return final_df


def attach_accidents(final_df):
    """
    If Data/accidents_tambaram.csv exists, it should contain:
        latitude, longitude

    Optional:
        accident_count

    We keep this separate so Kanyakumari/other-city accident data is
    never silently treated as Tambaram data.
    """

    final_df["historical_accident_count"] = 0.0

    if not ACCIDENT_FILE.exists():
        print(
            "\nNo Tambaram accident file found."
            "\nHistorical accident feature currently defaults to 0."
            "\nPlace a compatible Data/accidents_tambaram.csv file here"
            "\nwhen genuine local accident coordinates are available."
        )
        return final_df

    accidents = pd.read_csv(ACCIDENT_FILE)

    lat_col = first_existing_column(
        accidents,
        ["latitude", "Latitude", "lat", "LATITUDE"]
    )
    lon_col = first_existing_column(
        accidents,
        ["longitude", "Longitude", "lon", "LONGITUDE"]
    )

    if lat_col is None or lon_col is None:
        print(
            "\nAccident file found, but latitude/longitude columns "
            "were not detected. Accident feature left at 0."
        )
        return final_df

    accidents[lat_col] = safe_numeric(accidents[lat_col])
    accidents[lon_col] = safe_numeric(accidents[lon_col])

    accidents = accidents.dropna(
        subset=[lat_col, lon_col]
    ).copy()

    if accidents.empty:
        return final_df

    accident_gdf = gpd.GeoDataFrame(
        accidents,
        geometry=gpd.points_from_xy(
            accidents[lon_col],
            accidents[lat_col]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:32644")

    nodes_gdf, _ = ox.graph_to_gdfs(
        ox.load_graphml(ROAD_GRAPH),
        nodes=True,
        edges=True
    )

    nodes_projected = nodes_gdf.to_crs("EPSG:32644")

    # Match each accident to the nearest road-network node.
    accident_points = list(
        zip(
            accident_gdf.geometry.x,
            accident_gdf.geometry.y
        )
    )

    node_xy = np.array([
        [geom.x, geom.y]
        for geom in nodes_projected.geometry
    ])

    node_ids = list(nodes_projected.index)

    accident_node_counts = {}

    for x, y in accident_points:
        distances = np.sqrt(
            (node_xy[:, 0] - x) ** 2
            + (node_xy[:, 1] - y) ** 2
        )

        nearest_index = int(np.argmin(distances))
        nearest_node = node_ids[nearest_index]

        if distances[nearest_index] <= 500:
            accident_node_counts[nearest_node] = (
                accident_node_counts.get(nearest_node, 0) + 1
            )

    final_df["u_accidents"] = final_df["u"].map(
        accident_node_counts
    ).fillna(0)

    final_df["v_accidents"] = final_df["v"].map(
        accident_node_counts
    ).fillna(0)

    final_df["historical_accident_count"] = (
        final_df["u_accidents"]
        + final_df["v_accidents"]
    )

    final_df.drop(
        columns=["u_accidents", "v_accidents"],
        inplace=True,
        errors="ignore"
    )

    print(
        "Historical accident feature successfully attached "
        "from Tambaram accident coordinates."
    )

    return final_df


def create_temporal_prototype_features(final_df):
    """
    Creates a prototype modelling dataset when live/time-stamped traffic
    observations are unavailable.

    The resulting target is explicitly a prototype/simulated traffic
    signal, not a measured live-traffic value.
    """

    print("\nCreating temporal prototype features...")

    rng = np.random.default_rng(42)

    # Multiple time periods allow the model to learn temporal variation.
    time_periods = [
        ("Morning", 8, 1.20),
        ("Afternoon", 14, 0.90),
        ("Evening", 18, 1.30),
        ("Night", 22, 0.55),
    ]

    repeated = []

    base = final_df.copy()

    for period, hour, multiplier in time_periods:

        temp = base.copy()

        temp["time_period"] = period
        temp["hour"] = hour
        temp["peak_hour"] = int(
            period in ["Morning", "Evening"]
        )

        road_factor = (
            temp["road_connectivity"].rank(pct=True).fillna(0.5)
        )

        infrastructure_factor = (
            temp["total_poi_count"].rank(pct=True).fillna(0.5)
        )

        centrality_factor = (
            temp["road_centrality"].rank(pct=True).fillna(0.5)
        )

        accident_factor = (
            temp["historical_accident_count"].rank(
                pct=True
            ).fillna(0)
        )

        base_traffic = (
            20
            + 45 * road_factor
            + 20 * infrastructure_factor
            + 15 * centrality_factor
            + 10 * accident_factor
        )

        traffic = (
            base_traffic
            * multiplier
            + rng.normal(0, 5, len(temp))
        )

        temp["traffic_density"] = np.clip(
            traffic,
            0,
            100
        )

        risk_score = (
            0.35 * temp["traffic_density"]
            + 15 * infrastructure_factor
            + 15 * accident_factor
            + 10 * centrality_factor
        )

        temp["risk_score"] = np.clip(
            risk_score,
            0,
            100
        )

        temp["risk_level"] = pd.cut(
            temp["risk_score"],
            bins=[-np.inf, 40, 70, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        repeated.append(temp)

    modelling_df = pd.concat(
        repeated,
        ignore_index=True
    )

    modelling_df.to_csv(
        FINAL_FEATURE_FILE,
        index=False
    )

    print(f"Saved modelling dataset: {FINAL_FEATURE_FILE}")
    print(
        f"Rows: {len(modelling_df):,} | "
        f"Columns: {len(modelling_df.columns):,}"
    )

    print(
        "\nIMPORTANT: traffic_density and risk_score are "
        "prototype/simulated targets until real time-stamped traffic "
        "observations are connected."
    )

    return modelling_df


def main():
    print("=" * 60)
    print("DYNAMIC TRAFFIC AI - TAMBARAM DATA PREPARATION")
    print("=" * 60)
    print(f"Study area: {STUDY_AREA}")
    print(f"Center: {CENTER}")
    print(f"Road radius: {ROAD_RADIUS_M} m")
    print(f"POI radius: {POI_RADIUS_M} m")

    G = build_road_network()

    node_metrics = calculate_network_metrics(G)

    road_df = create_edge_features(
        G,
        node_metrics
    )

    create_adjacency_matrix(G)

    poi = download_pois()

    accessibility = create_accessibility_features(
        G,
        poi
    )

    final_df = attach_node_features_to_edges(
        G,
        road_df,
        accessibility
    )

    final_df = attach_accidents(final_df)

    modelling_df = create_temporal_prototype_features(
        final_df
    )

    build_database(
        modelling_df=modelling_df,
        road_df=road_df,
        poi=poi,
        accident_file=ACCIDENT_FILE
    )

    # Save projected graph for mapping and spatial analysis.
    try:
        G_projected = ox.project_graph(G)
        ox.save_graphml(
            G_projected,
            ROAD_GRAPH_PROJECTED
        )
        print(
            f"\nSaved projected road graph: "
            f"{ROAD_GRAPH_PROJECTED}"
        )
    except Exception as exc:
        print(
            f"\nCould not save projected graph: {exc}"
        )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")
    for path in [
        ROAD_GRAPH,
        ROAD_GRAPH_PROJECTED,
        POI_FILE,
        ROAD_FEATURE_FILE,
        NODE_METRIC_FILE,
        ACCESSIBILITY_FILE,
        ADJACENCY_FILE,
        FINAL_FEATURE_FILE,
        DATA_DIR / "traffic_ai.db",
    ]:
        if path.exists():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
