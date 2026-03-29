import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(42)
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

np.random.seed(42)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGETS  = ['CO(GT)', 'NO2(GT)']
DATA_URL = 'https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/airquality.csv'

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and preprocessing data…")
def load_data():
    df_raw = pd.read_csv(DATA_URL)
    df_raw['Datetime'] = pd.to_datetime(
        df_raw['Date'].astype(str) + ' ' + df_raw['Time'].astype(str)
    )
    df_raw.set_index('Datetime', inplace=True)
    df_raw.drop(['Date', 'Time'], axis=1, inplace=True)
    df_raw.replace(-200, np.nan, inplace=True)

    df = df_raw.resample('D').mean(numeric_only=True)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.ffill().bfill()

    for col in TARGETS:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        iqr = Q3 - Q1
        df[col] = df[col].clip(Q1 - 3 * iqr, Q3 + 3 * iqr)

    return df


@st.cache_data(show_spinner="Engineering features…")
def build_features(_df):
    feat = _df.copy()
    for col in TARGETS:
        for lag in [1, 2, 3, 7, 14]:
            feat[f'{col}_lag{lag}'] = feat[col].shift(lag)
        for w in [3, 7]:
            base = feat[col].shift(1)
            feat[f'{col}_roll{w}_mean'] = base.rolling(w).mean()
            feat[f'{col}_roll{w}_std']  = base.rolling(w).std()

    feat['month']     = feat.index.month
    feat['dayofweek'] = feat.index.dayofweek
    feat['dayofyear'] = feat.index.dayofyear
    feat['quarter']   = feat.index.quarter
    feat['month_sin'] = np.sin(2 * np.pi * feat['month']     / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month']     / 12)
    feat['dow_sin']   = np.sin(2 * np.pi * feat['dayofweek'] / 7)
    feat['dow_cos']   = np.cos(2 * np.pi * feat['dayofweek'] / 7)
    feat.dropna(inplace=True)
    return feat


def split_data(df_feat):
    n      = len(df_feat)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)
    train  = df_feat.iloc[:i_val]
    val    = df_feat.iloc[i_val:i_test]
    test   = df_feat.iloc[i_test:]
    fcols  = [c for c in df_feat.columns if c not in TARGETS]
    return train, val, test, fcols


def compute_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / np.where(actual == 0, 1e-8, actual))) * 100
    return mae, rmse, mape


# ── Model training (cached as resources – trained once per session) ────────────
@st.cache_resource(show_spinner="Training ARIMA models…")
def train_arima(_df, cutoff_str, test_index_list):
    results = {}
    for col in TARGETS:
        train_s = _df.loc[_df.index <= cutoff_str, col]
        test_s  = _df.loc[test_index_list, col]
        fitted  = ARIMA(train_s, order=(2, 1, 2)).fit()
        fc      = np.clip(fitted.forecast(steps=len(test_s)).values, 0, None)
        results[col] = {'actual': test_s.values, 'forecast': fc, 'aic': fitted.aic}
    return results


@st.cache_resource(show_spinner="Training XGBoost models…")
def train_xgboost(_X_train, _y_train, _X_val, _y_val, _X_test, _y_test, _feature_cols):
    models, results = {}, {}
    for col in TARGETS:
        m = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, random_state=42, n_jobs=-1,
            early_stopping_rounds=30, eval_metric='mae',
        )
        m.fit(_X_train, _y_train[col],
              eval_set=[(_X_val, _y_val[col])], verbose=False)
        preds = np.clip(m.predict(_X_test), 0, None)
        models[col]  = m
        results[col] = {
            'actual':   _y_test[col].values,
            'forecast': preds,
            'fi':       pd.Series(m.feature_importances_, index=_feature_cols).nlargest(10),
        }
    return models, results


@st.cache_resource(show_spinner="Training LSTM models…")
def train_lstm(_X_train, _y_train, _X_val, _y_val, _X_test, _y_test,
               _df_feat, _feature_cols, val_start_str, test_start_str):
    if not KERAS_AVAILABLE:
        return {}

    LOOKBACK = 14
    scaler_X = MinMaxScaler().fit(_X_train.values)
    X_all_s  = scaler_X.transform(_df_feat[_feature_cols].values)
    seq_dates = _df_feat.index[LOOKBACK:]

    results = {}
    for col in TARGETS:
        scaler_y = MinMaxScaler().fit(_y_train[[col]].values)
        y_all_s  = scaler_y.transform(_df_feat[col].values.reshape(-1, 1)).ravel()

        Xs, ys = [], []
        for i in range(LOOKBACK, len(X_all_s)):
            Xs.append(X_all_s[i - LOOKBACK:i])
            ys.append(y_all_s[i])
        X_seq, y_seq = np.array(Xs), np.array(ys)

        is_tr = seq_dates < val_start_str
        is_va = (seq_dates >= val_start_str) & (seq_dates < test_start_str)
        is_te = seq_dates >= test_start_str

        X_tr, y_tr = X_seq[is_tr], y_seq[is_tr]
        X_va, y_va = X_seq[is_va], y_seq[is_va]
        X_te, y_te = X_seq[is_te], y_seq[is_te]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, X_tr.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mae')
        es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=100, batch_size=16, callbacks=[es], verbose=0)

        preds_s = model.predict(X_te, verbose=0).ravel()
        preds   = np.clip(scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel(), 0, None)
        actual  = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
        results[col] = {'actual': actual, 'forecast': preds}

    return results


def rolling_forecast_xgb(model, df_feat, feature_cols, target_col, df_orig, steps):
    history   = df_feat.copy()
    hist_orig = df_orig[[target_col]].copy()
    preds     = []

    for _ in range(steps):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        row = {}

        for lag in [1, 2, 3, 7, 14]:
            key = f'{target_col}_lag{lag}'
            row[key] = hist_orig[target_col].iloc[-lag] if lag <= len(hist_orig) else hist_orig[target_col].iloc[0]
        for w in [3, 7]:
            vals = hist_orig[target_col].iloc[-w:]
            row[f'{target_col}_roll{w}_mean'] = vals.mean()
            row[f'{target_col}_roll{w}_std']  = vals.std()
        for c in feature_cols:
            if c not in row:
                row[c] = history[c].iloc[-1]
        row.update({
            'month':     next_date.month,
            'dayofweek': next_date.dayofweek,
            'dayofyear': next_date.dayofyear,
            'quarter':   next_date.quarter,
            'month_sin': np.sin(2 * np.pi * next_date.month     / 12),
            'month_cos': np.cos(2 * np.pi * next_date.month     / 12),
            'dow_sin':   np.sin(2 * np.pi * next_date.dayofweek / 7),
            'dow_cos':   np.cos(2 * np.pi * next_date.dayofweek / 7),
        })

        X_next = pd.DataFrame([row], index=[next_date])[feature_cols]
        pred   = float(np.clip(model.predict(X_next)[0], 0, None))
        preds.append(pred)

        new_row = X_next.copy()
        new_row[target_col] = pred
        for t in TARGETS:
            if t != target_col and t not in new_row.columns:
                new_row[t] = history[t].iloc[-1]
        history   = pd.concat([history, new_row])
        hist_orig = pd.concat([
            hist_orig,
            pd.DataFrame({target_col: [pred]}, index=[next_date])
        ])

    future_idx = pd.date_range(df_feat.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(preds, index=future_idx)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ Air Quality\nForecasting")
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)

    st.markdown("---")
    st.subheader("🤖 Models")
    run_arima   = st.checkbox("ARIMA",   value=True)
    run_xgboost = st.checkbox("XGBoost", value=True)
    run_lstm    = st.checkbox(
        "LSTM (TensorFlow)", value=False,
        disabled=not KERAS_AVAILABLE,
        help="Requires TensorFlow. Disabled if not installed.",
    )

    st.markdown("---")
    st.caption("Data: UCI Air Quality Dataset\n2004–2005, Italy")

# ── Load data & build features ────────────────────────────────────────────────
df      = load_data()
df_feat = build_features(df)
train, val, test, feature_cols = split_data(df_feat)

X_train, y_train = train[feature_cols], train[TARGETS]
X_val,   y_val   = val[feature_cols],   val[TARGETS]
X_test,  y_test  = test[feature_cols],  test[TARGETS]

# ── Train models ──────────────────────────────────────────────────────────────
arima_results = (
    train_arima(df, str(train.index.max()), list(test.index))
    if run_arima else {}
)
xgb_models, xgb_results = (
    train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    if run_xgboost else ({}, {})
)
lstm_results = (
    train_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
               df_feat, feature_cols, str(val.index[0]), str(test.index[0]))
    if (run_lstm and KERAS_AVAILABLE) else {}
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Models", "🔮 Forecast"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Days",    len(df))
    m2.metric("Features",      len(df.columns))
    m3.metric("CO(GT) Mean",   f"{df['CO(GT)'].mean():.2f} mg/m³")
    m4.metric("NO₂(GT) Mean",  f"{df['NO2(GT)'].mean():.2f} μg/m³")

    st.subheader("Time Series")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("CO(GT)", "NO₂(GT)"),
                        vertical_spacing=0.08)
    palette = [("steelblue", "rgba(70,130,180,0.12)"),
               ("darkorange", "rgba(255,140,0,0.12)")]
    for i, (col, (line_c, fill_c)) in enumerate(zip(TARGETS, palette), 1):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=line_c, width=1.4),
            fill='tozeroy', fillcolor=fill_c,
        ), row=i, col=1)
        ma7 = df[col].rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma7, name=f"7-day MA",
            line=dict(color='black', width=2, dash='dash'),
        ), row=i, col=1)
    fig.update_layout(height=480, hovermode='x unified',
                      title_text="Daily Concentrations with 7-Day Moving Average")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Summary Statistics"):
        st.dataframe(df[TARGETS].describe().round(3), use_container_width=True)

    with st.expander("🔢 Train / Val / Test Split"):
        split_info = pd.DataFrame({
            "Split":  ["Train (70%)", "Val (15%)", "Test (15%)"],
            "From":   [train.index.min().date(), val.index.min().date(), test.index.min().date()],
            "To":     [train.index.max().date(), val.index.max().date(), test.index.max().date()],
            "Days":   [len(train), len(val), len(test)],
        })
        st.dataframe(split_info, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Exploratory Data Analysis")

    sub1, sub2 = st.tabs(["Seasonality", "Correlation"])

    with sub1:
        sel_t = st.selectbox("Target variable", TARGETS, key="eda_target")
        df_eda = df[[sel_t]].copy()
        df_eda['Month']     = df_eda.index.month
        df_eda['DayOfWeek'] = df_eda.index.dayofweek

        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        dow_map   = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

        ca, cb = st.columns(2)
        with ca:
            monthly = (df_eda.groupby('Month')[sel_t]
                       .agg(['mean', 'std']).reset_index())
            monthly['Label'] = monthly['Month'].map(month_map)
            fig_m = go.Figure(go.Bar(
                x=monthly['Label'], y=monthly['mean'],
                error_y=dict(type='data', array=monthly['std']),
                marker_color='steelblue',
            ))
            fig_m.update_layout(title=f"{sel_t} — Monthly", height=340,
                                 yaxis_title=sel_t)
            st.plotly_chart(fig_m, use_container_width=True)

        with cb:
            dow_agg = (df_eda.groupby('DayOfWeek')[sel_t]
                       .agg(['mean', 'std']).reset_index())
            dow_agg['Label'] = dow_agg['DayOfWeek'].map(dow_map)
            fig_d = go.Figure(go.Bar(
                x=dow_agg['Label'], y=dow_agg['mean'],
                error_y=dict(type='data', array=dow_agg['std']),
                marker_color='darkorange',
            ))
            fig_d.update_layout(title=f"{sel_t} — Day of Week", height=340,
                                  yaxis_title=sel_t)
            st.plotly_chart(fig_d, use_container_width=True)

    with sub2:
        corr_cols = TARGETS + ['T', 'RH', 'AH']
        corr = df[corr_cols].corr()
        fig_h = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1, title='Feature Correlation Heatmap')
        fig_h.update_layout(height=420)
        st.plotly_chart(fig_h, use_container_width=True)

        r = df['CO(GT)'].corr(df['NO2(GT)'])
        fig_sc = px.scatter(df, x='CO(GT)', y='NO2(GT)', opacity=0.5,
                            title=f'CO(GT) vs NO₂(GT)  (r = {r:.2f})',
                            labels={'CO(GT)': 'CO(GT)', 'NO2(GT)': 'NO₂(GT)'})
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Model Evaluation")

    model_res_map = {}
    if arima_results:   model_res_map['ARIMA']   = arima_results
    if xgb_results:     model_res_map['XGBoost'] = xgb_results
    if lstm_results:    model_res_map['LSTM']     = lstm_results

    if not model_res_map:
        st.info("Enable at least one model in the sidebar.")
    else:
        # Metrics table
        rows = []
        for mname, mres in model_res_map.items():
            for col in TARGETS:
                mae, rmse, mape = compute_metrics(mres[col]['actual'], mres[col]['forecast'])
                rows.append({'Model': mname, 'Target': col,
                             'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
                             'MAPE(%)': round(mape, 2)})
        metrics_df = pd.DataFrame(rows)

        st.subheader("Metrics (Test Set)")
        st.dataframe(
            metrics_df.style.highlight_min(
                subset=['MAE', 'RMSE', 'MAPE(%)'], color='#d4f1d4'
            ),
            use_container_width=True,
        )

        # MAE bar chart
        fig_bar = px.bar(
            metrics_df, x='Model', y='MAE', color='Target', barmode='group',
            title='MAE Comparison Across Models',
            color_discrete_sequence=['steelblue', 'darkorange'],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Actual vs Predicted")
        c1, c2 = st.columns(2)
        sel_target = c1.selectbox("Target",  TARGETS,                      key="avp_t")
        sel_model  = c2.selectbox("Model",   list(model_res_map.keys()),   key="avp_m")

        if sel_target in model_res_map.get(sel_model, {}):
            res = model_res_map[sel_model][sel_target]
            act = np.array(res['actual'])
            prd = np.array(res['forecast'])
            n   = min(len(act), len(prd))
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(y=act[:n], name='Actual',
                                        line=dict(color='black', width=2)))
            fig_ap.add_trace(go.Scatter(y=prd[:n], name=sel_model,
                                        line=dict(color='crimson', width=2, dash='dash')))
            fig_ap.update_layout(
                title=f'{sel_target} — {sel_model}',
                xaxis_title='Test Day', yaxis_title=sel_target, height=380,
            )
            st.plotly_chart(fig_ap, use_container_width=True)

        # XGBoost feature importance
        if xgb_results:
            st.subheader("XGBoost — Top 10 Feature Importances")
            fi_t = st.selectbox("Target", TARGETS, key="fi_t")
            fi   = xgb_results[fi_t]['fi']
            fig_fi = px.bar(x=fi.values, y=fi.index, orientation='h',
                            title=f'Feature Importance — {fi_t}',
                            labels={'x': 'Importance', 'y': 'Feature'})
            fig_fi.update_layout(height=360, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header(f"{forecast_days}-Day Ahead Forecast")

    if not xgb_models:
        st.warning("Enable **XGBoost** in the sidebar to generate forecasts.")
    else:
        with st.spinner("Generating forecasts…"):
            future_preds = {
                col: rolling_forecast_xgb(
                    xgb_models[col], df_feat, feature_cols, col, df, forecast_days
                )
                for col in TARGETS
            }

        for col in TARGETS:
            hist = df[col].iloc[-90:]
            fp   = future_preds[col]
            color = 'steelblue' if col == 'CO(GT)' else 'darkorange'

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=hist.index, y=hist.values, name='Historical',
                line=dict(color=color, width=2),
            ))
            # Uncertainty band
            fig_f.add_trace(go.Scatter(
                x=fp.index, y=fp.values * 1.12,
                line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
            fig_f.add_trace(go.Scatter(
                x=fp.index, y=fp.values * 0.88, name='±12% band',
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(220,20,60,0.15)', hoverinfo='skip',
            ))
            fig_f.add_trace(go.Scatter(
                x=fp.index, y=fp.values, name='Forecast',
                line=dict(color='crimson', width=2.5, dash='dash'),
            ))
            fig_f.add_shape(
                type='line',
                x0=str(df.index[-1].date()), x1=str(df.index[-1].date()),
                y0=0, y1=1, yref='paper',
                line=dict(dash='dot', color='gray', width=1.5),
            )
            fig_f.add_annotation(
                x=str(df.index[-1].date()), y=1, yref='paper',
                text='Forecast start', showarrow=False,
                xanchor='left', yanchor='top',
                font=dict(size=10, color='gray'),
            )
            fig_f.update_layout(
                title=f'{col}: {forecast_days}-Day Forecast (XGBoost)',
                xaxis_title='Date', yaxis_title=col,
                height=400, hovermode='x unified',
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # Insight metrics
        st.subheader("Forecast Insights")
        cols_ui = st.columns(len(TARGETS))
        for col_ui, col in zip(cols_ui, TARGETS):
            fp        = future_preds[col]
            hist_mean = df[col].mean()
            hist_std  = df[col].std()
            fore_mean = fp.mean()
            fore_max  = fp.max()
            fore_peak = fp.idxmax().date()
            high_days = int((fp > hist_mean + hist_std).sum())
            delta_pct = (fore_mean - hist_mean) / hist_mean * 100
            trend     = "Rising ↑" if fp.iloc[-7:].mean() > fp.iloc[:7].mean() else "Falling ↓"

            with col_ui:
                st.markdown(f"#### {col}")
                st.metric("Forecast Mean",    f"{fore_mean:.2f}",
                          f"{delta_pct:+.1f}% vs historical")
                st.metric("Forecast Peak",    f"{fore_max:.2f}",
                          f"on {fore_peak}")
                st.metric("High-risk days",   f"{high_days}/{forecast_days}",
                          trend)

        with st.expander("📋 Action Recommendations"):
            st.markdown("""
| # | Recommendation |
|---|---------------|
| 1 | Issue **public health alerts** on forecasted high-concentration days |
| 2 | Enforce **stricter emission controls** during predicted peaks |
| 3 | Cross-validate with **meteorological data** (wind, precipitation) |
| 4 | **Retrain models monthly** to capture seasonal drift |
| 5 | Apply **SHAP analysis** to identify top causal drivers |
| 6 | Set **automated threshold alerts** — WHO limits: CO 4 mg/m³ (8h), NO₂ 25 μg/m³ (24h) |
| 7 | Extend to **hourly forecasting** for intraday emission management |
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · UCI Air Quality Dataset (2004–2005, Italy)")
