# Dynamic Traffic AI
## AI-Based Infrastructure-Aware Road Risk Prediction System

### 1. Project Overview

Dynamic Traffic AI is a Python-based web application designed to analyse road traffic conditions and road-related risk in the Tambaram study area. It combines road-network information, infrastructure data, traffic predictions, risk predictions, and accident-related metrics to support route analysis and infrastructure-aware road assessment.

The application is developed using Streamlit and provides an interactive interface for exploring traffic information, comparing routes, and analysing infrastructure-related patterns.

### 2. Main Features

- **Live Traffic:** View traffic-related information and predictions for the study area.
- **Analyse Route:** Explore route alternatives using road distance, predicted traffic, predicted risk, and accident-risk information.
- **Infrastructure Analysis:** Explore relationships between nearby infrastructure and traffic or risk indicators.
- **Accident Metrics:** Calculate accident-related measures from the prepared accident-event dataset.
- **Interactive Visualisation:** Display maps, charts, and analytical summaries through a web interface.

### 3. Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn and Joblib
- NetworkX and OSMnx
- GeoPandas and Shapely
- Folium and Streamlit-Folium
- Plotly
- SQLite, where used by the project database module
- OpenStreetMap-based road and infrastructure data

### 4. Project Structure

```text
final_project/
├── app.py
├── database.py
├── accident_data.py
├── prepare_data.py
├── train_models.py
├── verify_accident_metrics.py
├── requirements.txt
├── README.md
├── Data/
├── models/
├── cache/
├── reports/
├── documentation/
├── screenshots/
└── backup_versions/
```

Additional development and testing scripts may remain in the project directory.

### 5. Installation

Use Python 3.13 or another compatible Python version with the required libraries.

Open a terminal in the project directory and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 6. Running the Application

From the project directory, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a web browser to access the application.

### 7. Data and Model Requirements

The application depends on prepared datasets and trained model files stored in the `Data/` and `models/` directories. Keep these directories in their existing locations unless the corresponding file paths in the code are updated.

The accident-analysis functionality uses the prepared accident-event data. The road network, infrastructure records, feature dataset, database, and model files should be retained for the application to work as intended.

### 8. Scope

The current implementation focuses on the Tambaram study area. Its results depend on the prepared datasets, trained models, and available road-network coverage. Predictions and risk scores should be interpreted as analytical estimates rather than guaranteed real-time road conditions.

### 9. Future Enhancements

- Integrate a continuously updated, validated live-traffic data source.
- Expand the geographic coverage of the road network.
- Improve model evaluation using independently validated data.
- Enhance route validation and alternative-route comparisons.
- Add further infrastructure and accident-pattern analysis.

### 10. Conclusion

Dynamic Traffic AI demonstrates how road-network data, machine-learning predictions, accident metrics, and infrastructure information can be combined in an interactive application to support traffic analysis and route-risk assessment.
