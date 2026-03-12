# Customer Analytics Dashboard

An end-to-end retail customer analytics platform built with Streamlit, covering five core data science workflows — from exploratory analysis to deep learning forecasting.

---

## Features

| Module | Description |
|---|---|
| **Exploratory Data Analysis** | Customer demographics, transaction patterns, RFM distributions, geographic analysis, YoY revenue |
| **Customer Segmentation** | RFM scoring, K-Means clustering (elbow + silhouette), cluster profiling with radar charts |
| **Churn Prediction** | Logistic Regression & Random Forest classifiers, ROC-AUC, feature importance |
| **Sales Forecasting** | Bidirectional LSTM model, 6-month future forecast, RMSE/MAE/MAPE evaluation |
| **Product Recommendations** | Popularity-based, user-based & item-based collaborative filtering (cosine similarity) |

---

## Project Structure

```
├── app.py                          # Streamlit dashboard entry point
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Segmentation.ipynb
│   ├── 03_Churn_Prediction.ipynb
│   ├── 04_LSTM_Sales_Forecasting.ipynb
│   └── 05_Product_Recommendation.ipynb
├── models/                         # Saved models (pickle / keras)
├── data/                           # Raw datasets
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks (in order)

Execute notebooks `01` through `05` to generate the processed data and saved models before launching the dashboard.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

---

##  Models & Artifacts

Trained models are saved to the `models/` directory after running the notebooks:

- `kmeans_model.pkl` — Customer segmentation
- `scaler.pkl` — RFM feature scaler
- `random_forest.pkl` — Churn classifier
- `logistic_regression.pkl` — Churn baseline
- `lstm_model.keras` — Sales forecasting model
- `user_similarity.pkl` / `item_similarity.pkl` — Recommendation matrices

---

##  Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Key libraries:

- `streamlit` — Dashboard UI
- `tensorflow` — LSTM forecasting
- `scikit-learn` — ML models & preprocessing
- `pandas` / `numpy` — Data manipulation
- `matplotlib` / `seaborn` — Visualisation

---
