# Institute Rank Prediction

A machine learning project that predicts institute rankings based on NIRF scores and location data. Built with Python, Flask, and Streamlit.

## Project Structure

```
institute-rank-predictor/
├── data/
│   └── Institute_Rankings.csv    # Dataset of 100 NIRF-ranked institutes
├── models/                        # Trained model artifacts (generated)
│   ├── model.pkl                 # Best trained model
│   ├── scaler.pkl                # Feature scaler
│   ├── state_encoder.pkl         # State label encoder
│   ├── city_encoder.pkl          # City label encoder
│   └── metadata.json             # Model metrics and metadata
├── reports/
│   └── images/                   # Generated charts and screenshots
├── train_model.py                # ML model training script
├── app.py                        # Flask backend API (port 5000)
├── ui.py                         # Streamlit frontend UI (port 8501)
├── generate_report.py            # Report generation script
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Train the model
python train_model.py

# Step 2: Start the Flask backend API
python app.py
# API runs on http://localhost:5000

# Step 3: Start the Streamlit frontend UI
streamlit run ui.py --server.port 8501
# UI runs on http://localhost:8501
```

## API Endpoints

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| GET    | /api/health        | Health check                       |
| GET    | /api/model-info     | Model metadata and metrics         |
| GET    | /api/dataset        | Full institute dataset             |
| GET    | /api/states         | States and their cities            |
| POST   | /api/predict        | Predict rank from score/state/city|

### Prediction Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"score": 70, "state": "Delhi", "city": "New Delhi"}'
```

## Models Trained

| Model               | R2 Score | MAE   | RMSE  |
|---------------------|----------|-------|-------|
| Linear Regression   | 0.723    | 11.88 | 14.90 |
| Random Forest       | 0.998    | 1.03  | 1.24  |
| K-Nearest Neighbors | 0.868    | 8.92  | 10.29 |

**Best model: Random Forest** (R2 = 0.998)

## Features

- NIRF Score (numeric, 0-100)
- State (categorical, label-encoded)
- City (categorical, label-encoded)

## Dataset

100 institutes from the NIRF India Rankings, including fields:
- Institute ID, Name, City, State, Score (0-100), Rank (1-100)
