import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

import joblib


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "Data/X_features_with_accidents.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("DYNAMIC TRAFFIC AI - MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading accident-enhanced modelling dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. CHECK ACCIDENT FEATURES
# ============================================================

accident_features = [
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
    "accident_rate_per_km"
]

print("\nChecking accident features...")

for column in accident_features:
    if column in df.columns:
        print(f"  {column}: included")
    else:
        print(f"  {column}: MISSING")


# ============================================================
# 3. REMOVE TARGET / ID COLUMNS FROM MODEL INPUT
# ============================================================

drop_columns = [
    "u",
    "v",
    "key",
    "traffic_density",
    "risk_score",
    "risk_level"
]

feature_columns = [
    column
    for column in df.columns
    if column not in drop_columns
]

X = df[feature_columns].copy()

y_traffic = df["traffic_density"].copy()
y_risk = df["risk_score"].copy()
y_risk_level = df["risk_level"].copy()


print("\nFeatures used for modelling:")

for column in feature_columns:
    print(f"  {column}")


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "road_type",
    "time_period"
]

numeric_features = [
    column
    for column in feature_columns
    if column not in categorical_features
]


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 6. TRAFFIC PREDICTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING TRAFFIC PREDICTION MODEL")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_traffic,
    test_size=0.20,
    random_state=42
)

traffic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=80,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

traffic_model.fit(
    X_train,
    y_train
)

traffic_predictions = traffic_model.predict(X_test)

traffic_mae = mean_absolute_error(
    y_test,
    traffic_predictions
)

traffic_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        traffic_predictions
    )
)

traffic_r2 = r2_score(
    y_test,
    traffic_predictions
)

print("\nTraffic model results:")
print(f"MAE  : {traffic_mae:.4f}")
print(f"RMSE : {traffic_rmse:.4f}")
print(f"R²   : {traffic_r2:.4f}")


# ============================================================
# 7. ROAD RISK REGRESSION MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ROAD RISK MODEL")
print("=" * 60)

X_train_risk, X_test_risk, y_train_risk, y_test_risk = (
    train_test_split(
        X,
        y_risk,
        test_size=0.20,
        random_state=42
    )
)

risk_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=80,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

risk_model.fit(
    X_train_risk,
    y_train_risk
)

risk_predictions = risk_model.predict(
    X_test_risk
)

risk_mae = mean_absolute_error(
    y_test_risk,
    risk_predictions
)

risk_rmse = np.sqrt(
    mean_squared_error(
        y_test_risk,
        risk_predictions
    )
)

risk_r2 = r2_score(
    y_test_risk,
    risk_predictions
)

print("\nRisk model results:")
print(f"MAE  : {risk_mae:.4f}")
print(f"RMSE : {risk_rmse:.4f}")
print(f"R²   : {risk_r2:.4f}")


# ============================================================
# 8. RISK LEVEL CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RISK LEVEL CLASSIFIER")
print("=" * 60)

X_train_class, X_test_class, y_train_class, y_test_class = (
    train_test_split(
        X,
        y_risk_level,
        test_size=0.20,
        random_state=42,
        stratify=y_risk_level
    )
)

risk_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=80,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

risk_classifier.fit(
    X_train_class,
    y_train_class
)

risk_level_predictions = risk_classifier.predict(
    X_test_class
)

classification_accuracy = accuracy_score(
    y_test_class,
    risk_level_predictions
)

print("\nRisk classification accuracy:")
print(f"{classification_accuracy:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_test_class,
        risk_level_predictions
    )
)


# ============================================================
# 9. SAVE MODELS
# ============================================================

traffic_model_file = (
    "models/traffic_prediction_model.pkl"
)

risk_model_file = (
    "models/risk_prediction_model.pkl"
)

risk_classifier_file = (
    "models/risk_level_classifier.pkl"
)


joblib.dump(
    traffic_model,
    traffic_model_file,
    compress=3
)

joblib.dump(
    risk_model,
    risk_model_file,
    compress=3
)

joblib.dump(
    risk_classifier,
    risk_classifier_file,
    compress=3
)


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("MODELS SAVED")
print("=" * 60)

print(
    f"Traffic model     : {traffic_model_file}"
)

print(
    f"Risk model        : {risk_model_file}"
)

print(
    f"Risk classifier   : {risk_classifier_file}"
)

print("\nModel training complete.")