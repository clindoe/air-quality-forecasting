# Time-Series Forecasting of Carbon Monoxide and Nitrogen Dioxide Levels

> **Live Demo →** [air-quality-forecasting-ukvol.streamlit.app](https://air-quality-forecasting-ukvol.streamlit.app/)

---

## Abstract

Air pollution remains one of the most critical environmental and public health challenges of the modern era. Among the key pollutants, **Carbon Monoxide (CO)** and **Nitrogen Dioxide (NO₂)** are directly linked to respiratory diseases, cardiovascular conditions, and long-term environmental degradation. Accurate forecasting of these concentrations enables governments, urban planners, and health authorities to take proactive measures before dangerous thresholds are reached.

This project develops and evaluates a complete end-to-end **time-series forecasting pipeline** for daily CO(GT) and NO₂(GT) concentrations using the UCI Air Quality dataset collected in an Italian city between March 2004 and April 2005. Three classes of forecasting models are implemented and compared: a classical statistical model **(ARIMA)**, a gradient-boosted machine learning model **(XGBoost)**, and a deep learning sequence model **(LSTM)**. The best-performing model is then used to generate **30-day ahead forecasts** with uncertainty estimates and actionable public health recommendations.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Forecasting Models](#6-forecasting-models)
7. [Evaluation](#7-evaluation)
8. [Future Forecasting](#8-future-forecasting)
9. [Insights and Recommendations](#9-insights-and-recommendations)
10. [Project Structure](#10-project-structure)
11. [How to Run](#11-how-to-run)
12. [References](#12-references)

---

## 1. Introduction

Urban air quality monitoring has become a priority in the context of increasing industrialization and traffic density. Sensor-based air quality stations continuously record the concentrations of various pollutants, but raw sensor readings alone are insufficient for proactive decision-making — **predictive models are needed** to forecast future pollution levels.

This project addresses the following core problem:

> *Given historical hourly readings of CO, NO₂, and related environmental variables (temperature, humidity, sensor proxies), can we accurately forecast daily CO(GT) and NO₂(GT) concentrations for the next 30 days?*

The pipeline covers the full machine learning lifecycle:
- Data ingestion and cleaning
- Exploratory data analysis
- Feature engineering
- Multi-model training with temporal cross-validation
- Model evaluation and comparison
- Interactive deployment via Streamlit

---

## 2. Dataset

| Attribute         | Detail                                         |
|-------------------|------------------------------------------------|
| **Source**        | UCI Machine Learning Repository               |
| **Location**      | Italian city (road-level monitoring station)  |
| **Period**        | March 2004 – April 2005                       |
| **Frequency**     | Hourly (resampled to daily averages)           |
| **Total Records** | ~9,357 hourly rows → ~391 daily rows          |
| **Targets**       | `CO(GT)`, `NO2(GT)`                           |

### Features

| Column           | Description                                        |
|------------------|----------------------------------------------------|
| `CO(GT)`         | True hourly CO concentration (mg/m³) — **Target** |
| `NO2(GT)`        | True hourly NO₂ concentration (μg/m³) — **Target**|
| `PT08.S1(CO)`    | Tin oxide sensor (CO proxy)                        |
| `C6H6(GT)`       | Benzene concentration (μg/m³)                     |
| `PT08.S2(NMHC)`  | Titania sensor (NMHC proxy)                        |
| `NOx(GT)`        | Nitrogen oxides concentration (ppb)               |
| `PT08.S3(NOx)`   | Tungsten oxide sensor (NOx proxy)                  |
| `PT08.S4(NO2)`   | Tungsten oxide sensor (NO₂ proxy)                  |
| `PT08.S5(O3)`    | Indium oxide sensor (O₃ proxy)                     |
| `T`              | Temperature (°C)                                   |
| `RH`             | Relative Humidity (%)                              |
| `AH`             | Absolute Humidity                                  |

> **Note:** The dataset uses `-200` as a sentinel value for missing readings, replaced with `NaN` during preprocessing.

---

## 3. Methodology

The pipeline follows a structured six-stage workflow:

```
Raw Data
   │
   ▼
1. Data Preprocessing
   │  - Datetime parsing & indexing
   │  - Daily resampling (mean)
   │  - Linear interpolation + forward/back fill
   │  - IQR-based outlier capping (k = 3.0)
   ▼
2. Exploratory Data Analysis
   │  - Trend visualization with 7-day moving average
   │  - Monthly & day-of-week seasonality
   │  - Pearson correlation heatmap
   ▼
3. Feature Engineering
   │  - Lag features: t-1, t-2, t-3, t-7, t-14
   │  - Rolling statistics: 3-day, 7-day (mean & std)
   │  - Cyclical time encoding: sin/cos of month & day-of-week
   ▼
4. Model Training (Temporal Split: 70 / 15 / 15)
   │  - ARIMA(2,1,2)
   │  - XGBoost Regressor (early stopping on validation MAE)
   │  - LSTM (2-layer: 64 → 32 units, dropout=0.2, early stopping)
   ▼
5. Evaluation
   │  - MAE, RMSE, MAPE on held-out test set
   │  - Actual vs Predicted plots
   ▼
6. 30-Day Ahead Forecast
      - Iterative rolling forecast with XGBoost (best ML model)
      - ±12% uncertainty band
      - High-risk day identification
```

---

## 4. Exploratory Data Analysis

### 4.1 Trend Analysis

Both CO(GT) and NO₂(GT) exhibit clear **seasonal variation** across the dataset period. Concentrations peak during autumn–winter months (October–January) and dip during summer, consistent with:
- Reduced atmospheric dispersion in colder months
- Higher vehicular and heating-related emissions in winter

The **7-day moving average** smooths short-term sensor noise and reveals the underlying medium-term trend clearly.

### 4.2 Seasonality

| Pattern         | CO(GT)                              | NO₂(GT)                             |
|----------------|--------------------------------------|--------------------------------------|
| Monthly         | Peak: Nov–Jan, Trough: Jun–Aug       | Peak: Oct–Jan, Trough: Jun–Aug       |
| Day of Week     | Slightly elevated on weekdays        | Noticeably higher Mon–Fri vs weekend |

The weekday effect in NO₂ is particularly pronounced, strongly implicating **traffic emissions** as the dominant source.

### 4.3 Correlation

CO(GT) and NO₂(GT) show a **strong positive Pearson correlation (r ≈ 0.72)**, indicating shared emission sources (primarily combustion). Both are also significantly correlated with:
- `C6H6(GT)` (benzene — traffic exhaust marker)
- `NOx(GT)` (direct combustion product)
- `PT08.S4(NO₂)` (dedicated NO₂ sensor proxy)

Temperature (`T`) shows a **negative correlation** with both targets, consistent with the seasonal winter peaks.

---

## 5. Feature Engineering

Raw timestamps and target values alone are insufficient for ML-based forecasting. The following features are constructed:

### 5.1 Lag Features
Historical values of both targets at offsets of **1, 2, 3, 7, and 14 days** are included as features. These capture short-term autocorrelation and weekly periodicity. All lags use `.shift(n)` to ensure **zero data leakage**.

### 5.2 Rolling Statistics
**3-day and 7-day rolling mean and standard deviation** of each target (shifted by 1 day) capture the local level and volatility of the series without leaking future information.

### 5.3 Cyclical Time Encoding
Rather than raw integers (month = 1–12), sine and cosine transformations are used:

```
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

This ensures December and January are numerically adjacent — a property that linear integer encoding fails to capture.

### 5.4 Additional Time Features
`dayofweek`, `dayofyear`, and `quarter` provide further temporal context for the models.

**Total features after engineering: 33** (excluding the 2 target columns).

---

## 6. Forecasting Models

### 6.1 ARIMA(2,1,2) — Statistical Baseline

**ARIMA** (AutoRegressive Integrated Moving Average) is the classical univariate time-series model. The order `(p=2, d=1, q=2)` was selected based on:
- **ADF test** on first-differenced series confirming stationarity at d=1
- ACF/PACF pattern inspection suggesting AR(2) and MA(2) terms

ARIMA is trained separately for CO(GT) and NO₂(GT) using only the training portion of each target series. It serves as the **statistical baseline**.

**Limitation:** ARIMA cannot incorporate external features (temperature, humidity, sensor readings), limiting its accuracy on multivariate datasets.

### 6.2 XGBoost Regressor — Machine Learning Model

**XGBoost** (Extreme Gradient Boosting) is trained on the full engineered feature matrix. Key hyperparameters:

| Parameter           | Value |
|--------------------|-------|
| `n_estimators`      | 600   |
| `learning_rate`     | 0.04  |
| `max_depth`         | 5     |
| `subsample`         | 0.8   |
| `colsample_bytree`  | 0.8   |
| `early_stopping`    | 30 rounds on validation MAE |

Early stopping on the validation set prevents overfitting while maximizing model capacity. XGBoost consistently outperforms ARIMA by leveraging **multi-feature context**.

**Top predictive features (both targets):** lag-1, lag-2, lag-7, 7-day rolling mean — confirming the dominant role of **recent history and weekly periodicity**.

### 6.3 LSTM — Deep Learning Model

A two-layer **LSTM** (Long Short-Term Memory) network processes 14-day sliding windows of the scaled feature matrix:

```
Input (14 timesteps × 33 features)
    → LSTM(64, return_sequences=True)
    → Dropout(0.2)
    → LSTM(32)
    → Dropout(0.2)
    → Dense(16, ReLU)
    → Dense(1)
```

Training uses:
- **Adam optimizer**, MAE loss
- **EarlyStopping** (patience=10, restore best weights)
- **MinMaxScaler** on both features and target (inverse-transformed for evaluation)
- Up to 100 epochs, batch size 16

LSTM is particularly well-suited for capturing **non-linear long-range dependencies** in sequential data that ARIMA and even XGBoost may miss.

---

## 7. Evaluation

Models are evaluated on the **held-out test set (last 15% of data)** using three metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|actual - pred|)` | Average absolute error in original units |
| **RMSE** | `sqrt(mean((actual - pred)²))` | Penalizes large errors more heavily |
| **MAPE** | `mean(|actual - pred| / actual) × 100` | Scale-independent percentage error |

### Results Summary

> *Exact values vary per run; representative results below.*

| Model    | CO(GT) MAE | CO(GT) RMSE | NO₂(GT) MAE | NO₂(GT) RMSE |
|----------|-----------|------------|------------|-------------|
| ARIMA    | ~0.35     | ~0.48      | ~12.5      | ~16.8       |
| XGBoost  | ~0.21     | ~0.29      | ~8.4       | ~11.2       |
| LSTM     | ~0.24     | ~0.33      | ~9.1       | ~12.0       |

**XGBoost achieves the lowest MAE and RMSE for both targets**, combining the power of multi-feature input with regularized gradient boosting. LSTM is competitive but requires substantially more compute and data.

---

## 8. Future Forecasting

The trained XGBoost model generates a **30-day ahead forecast** using an iterative rolling strategy:

1. Predict the next day using the most recent window of lag and rolling features
2. Append the prediction to the history
3. Recompute lag and rolling features for the following step
4. Repeat for all forecast steps

Environmental features (temperature, humidity, sensor readings) are carried forward from the last known observation.

A **±12% uncertainty band** is overlaid on the forecast to communicate prediction confidence to non-technical stakeholders.

---

## 9. Insights and Recommendations

Based on the forecasts and seasonal patterns identified during EDA:

1. **Issue public health alerts** on days where the forecast exceeds WHO limits:
   - CO: 4 mg/m³ (8-hour mean)
   - NO₂: 25 μg/m³ (24-hour mean)

2. **Enforce stricter traffic and industrial emission controls** during predicted peak periods (autumn–winter months, weekday mornings).

3. **Cross-validate forecasts** with real-time meteorological data (wind speed, precipitation) for improved accuracy.

4. **Retrain models monthly** to account for seasonal drift, sensor degradation, and changing emission profiles.

5. **Apply SHAP (SHapley Additive exPlanations)** to the XGBoost model to quantify and communicate the contribution of each feature to individual predictions.

6. **Extend to hourly forecasting** for intraday emission management and real-time alert systems.

---

## 10. Project Structure

```
air-quality-forecasting/
├── app.py                                        # Streamlit web application
├── requirements.txt                              # Python dependencies
├── Time-Series_Forecasting_CO_NO2.ipynb          # Full analysis notebook
└── README.md                                     # This document
```

---

## 11. How to Run

### Run the Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
jupyter notebook "Time-Series_Forecasting_CO_NO2.ipynb"
```

### Run the Streamlit App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Community Cloud (Free)
1. Fork or clone this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select this repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

---

## 12. References

1. De Vito, S. et al. (2008). *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.* Sensors and Actuators B: Chemical.
2. UCI Machine Learning Repository — [Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
3. Box, G. E. P., Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control.* Holden-Day.
4. Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM KDD.
5. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
6. World Health Organization (2021). *WHO Global Air Quality Guidelines.*

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built with Python · Streamlit · XGBoost · Statsmodels · TensorFlow/Keras*
