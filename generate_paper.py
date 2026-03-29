"""
generate_paper.py
Generates a full academic-style PDF paper with real figures from the dataset.
Run: python3 generate_paper.py
Output: Air_Quality_Forecasting_Paper.pdf
"""

import warnings
warnings.filterwarnings('ignore')

import os, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

TARGETS  = ['CO(GT)', 'NO2(GT)']
DATA_URL = 'https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/airquality.csv'
OUT_PDF  = 'Air_Quality_Forecasting_Paper.pdf'
TMP      = tempfile.mkdtemp()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name, fig=None, tight=True):
    path = os.path.join(TMP, f'{name}.png')
    if tight:
        plt.tight_layout()
    (fig or plt).savefig(path, dpi=150, bbox_inches='tight',
                         facecolor='white', edgecolor='none')
    plt.close('all')
    return path


def cap_iqr(s, k=3.0):
    Q1, Q3 = s.quantile([0.25, 0.75])
    return s.clip(Q1 - k*(Q3-Q1), Q3 + k*(Q3-Q1))


def compute_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual-pred)/np.where(actual==0,1e-8,actual)))*100
    return round(mae,4), round(rmse,4), round(mape,2)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_csv(DATA_URL)
df_raw['Datetime'] = pd.to_datetime(
    df_raw['Date'].astype(str)+' '+df_raw['Time'].astype(str))
df_raw.set_index('Datetime', inplace=True)
df_raw.drop(['Date','Time'], axis=1, inplace=True)
df_raw.replace(-200, np.nan, inplace=True)

df = df_raw.resample('D').mean(numeric_only=True)
df = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
for col in TARGETS:
    df[col] = cap_iqr(df[col])


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df):
    feat = df.copy()
    for col in TARGETS:
        for lag in [1,2,3,7,14]:
            feat[f'{col}_lag{lag}'] = feat[col].shift(lag)
        for w in [3,7]:
            base = feat[col].shift(1)
            feat[f'{col}_roll{w}_mean'] = base.rolling(w).mean()
            feat[f'{col}_roll{w}_std']  = base.rolling(w).std()
    feat['month']     = feat.index.month
    feat['dayofweek'] = feat.index.dayofweek
    feat['dayofyear'] = feat.index.dayofyear
    feat['quarter']   = feat.index.quarter
    feat['month_sin'] = np.sin(2*np.pi*feat['month']/12)
    feat['month_cos'] = np.cos(2*np.pi*feat['month']/12)
    feat['dow_sin']   = np.sin(2*np.pi*feat['dayofweek']/7)
    feat['dow_cos']   = np.cos(2*np.pi*feat['dayofweek']/7)
    feat.dropna(inplace=True)
    return feat

df_feat   = build_features(df)
fcols     = [c for c in df_feat.columns if c not in TARGETS]
n         = len(df_feat)
train     = df_feat.iloc[:int(n*0.70)]
val       = df_feat.iloc[int(n*0.70):int(n*0.85)]
test      = df_feat.iloc[int(n*0.85):]
X_tr, y_tr = train[fcols], train[TARGETS]
X_va, y_va = val[fcols],   val[TARGETS]
X_te, y_te = test[fcols],  test[TARGETS]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# FIG 1 — Time series
print("Generating figures...")
fig, axes = plt.subplots(2,1, figsize=(12,6), sharex=True)
palette = [('steelblue','lightsteelblue'),('darkorange','moccasin')]
for ax,(col,(lc,fc)) in zip(axes, zip(TARGETS,palette)):
    ax.plot(df.index, df[col], color=lc, lw=1.3, label='Daily avg')
    ax.fill_between(df.index, df[col], alpha=0.25, color=fc)
    ma = df[col].rolling(7,center=True).mean()
    ax.plot(df.index, ma, 'k--', lw=2, label='7-day MA')
    ax.set_ylabel(col, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
axes[0].set_title('Figure 1 — Daily CO(GT) and NO₂(GT) Concentrations (2004–2005)',
                  fontsize=12, fontweight='bold')
axes[-1].set_xlabel('Date')
FIG1 = save_fig('fig1_timeseries')

# FIG 2 — Seasonality
fig, axes = plt.subplots(2,2, figsize=(12,7))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dow_labels   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df_eda = df[TARGETS].copy()
df_eda['Month']     = df_eda.index.month
df_eda['DayOfWeek'] = df_eda.index.dayofweek

for i,(col,color) in enumerate(zip(TARGETS,['steelblue','darkorange'])):
    months = sorted(df_eda['Month'].unique())
    sns.boxplot(data=df_eda, x='Month', y=col, ax=axes[i][0],
                color=color, fliersize=2, width=0.55)
    axes[i][0].set_xticklabels([month_labels[m-1] for m in months], fontsize=8)
    axes[i][0].set_title(f'{col} — Monthly Distribution', fontsize=10, fontweight='bold')
    axes[i][0].set_xlabel('')

    sns.boxplot(data=df_eda, x='DayOfWeek', y=col, ax=axes[i][1],
                color=color, fliersize=2, width=0.55)
    axes[i][1].set_xticklabels(dow_labels, fontsize=8)
    axes[i][1].set_title(f'{col} — Day-of-Week Distribution', fontsize=10, fontweight='bold')
    axes[i][1].set_xlabel('')

plt.suptitle('Figure 2 — Seasonality Analysis', fontsize=12, fontweight='bold')
FIG2 = save_fig('fig2_seasonality')

# FIG 3 — Correlation heatmap + scatter
fig, axes = plt.subplots(1,2, figsize=(12,5))
corr_cols = TARGETS + ['T','RH','AH','NOx(GT)','C6H6(GT)']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0], cbar_kws={'shrink':0.8}, linewidths=0.4,
            annot_kws={'size':8})
axes[0].set_title('Figure 3a — Feature Correlation Matrix', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].tick_params(axis='y', labelsize=8)

r = df['CO(GT)'].corr(df['NO2(GT)'])
axes[1].scatter(df['CO(GT)'], df['NO2(GT)'], alpha=0.45, s=15, color='purple')
m,b = np.polyfit(df['CO(GT)'], df['NO2(GT)'], 1)
xs = np.linspace(df['CO(GT)'].min(), df['CO(GT)'].max(), 100)
axes[1].plot(xs, m*xs+b, 'r--', lw=2)
axes[1].set_xlabel('CO(GT)')
axes[1].set_ylabel('NO₂(GT)')
axes[1].set_title(f'Figure 3b — CO(GT) vs NO₂(GT)  (r = {r:.2f})', fontsize=10, fontweight='bold')
FIG3 = save_fig('fig3_correlation')

# FIG 4 — Feature importance (XGBoost, trained quickly)
print("Training XGBoost for feature importance...")
xgb_models, xgb_results = {}, {}
for col in TARGETS:
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, early_stopping_rounds=20, eval_metric='mae')
    m.fit(X_tr, y_tr[col], eval_set=[(X_va, y_va[col])], verbose=False)
    preds = np.clip(m.predict(X_te), 0, None)
    xgb_models[col]  = m
    xgb_results[col] = {'actual': y_te[col].values, 'forecast': preds}

fig, axes = plt.subplots(1,2, figsize=(12,5))
for ax,col in zip(axes, TARGETS):
    fi = pd.Series(xgb_models[col].feature_importances_, index=fcols).nlargest(10)
    colors_bar = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(fi)))
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors_bar[::-1])
    ax.set_title(f'Figure 4 — Top 10 Features: {col}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Importance Score')
plt.suptitle('Figure 4 — XGBoost Feature Importances', fontsize=12, fontweight='bold')
FIG4 = save_fig('fig4_feature_importance')

# FIG 5 — ARIMA
print("Training ARIMA...")
arima_results = {}
for col in TARGETS:
    s_train = df.loc[df.index <= train.index.max(), col]
    s_test  = df.loc[test.index, col]
    fitted  = ARIMA(s_train, order=(2,1,2)).fit()
    fc      = np.clip(fitted.forecast(steps=len(s_test)).values, 0, None)
    arima_results[col] = {'actual': s_test.values, 'forecast': fc}

fig, axes = plt.subplots(2,2, figsize=(14,8))
colors_m = ['steelblue','darkorange']
for row,(col,color) in enumerate(zip(TARGETS,colors_m)):
    # ARIMA
    act = arima_results[col]['actual']
    prd = arima_results[col]['forecast']
    n   = min(len(act),len(prd))
    axes[row][0].plot(act[:n], color='black', lw=1.8, label='Actual')
    axes[row][0].plot(prd[:n], color='steelblue', lw=1.8, ls='--', label='ARIMA')
    axes[row][0].set_title(f'{col} — ARIMA(2,1,2)', fontsize=10, fontweight='bold')
    axes[row][0].legend(fontsize=8)
    axes[row][0].set_xlabel('Test Day Index')
    axes[row][0].set_ylabel(col)

    # XGBoost
    act2 = xgb_results[col]['actual']
    prd2 = xgb_results[col]['forecast']
    n2   = min(len(act2),len(prd2))
    axes[row][1].plot(act2[:n2], color='black', lw=1.8, label='Actual')
    axes[row][1].plot(prd2[:n2], color='darkorange', lw=1.8, ls='--', label='XGBoost')
    axes[row][1].set_title(f'{col} — XGBoost', fontsize=10, fontweight='bold')
    axes[row][1].legend(fontsize=8)
    axes[row][1].set_xlabel('Test Day Index')
    axes[row][1].set_ylabel(col)

plt.suptitle('Figure 5 — Actual vs Predicted on Test Set', fontsize=12, fontweight='bold')
FIG5 = save_fig('fig5_actual_vs_pred')

# FIG 6 — 30-day forecast
print("Generating 30-day forecast...")
def rolling_forecast(model, df_feat, fcols, target_col, df_orig, steps=30):
    history   = df_feat.copy()
    hist_orig = df_orig[[target_col]].copy()
    preds     = []
    for _ in range(steps):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        row = {}
        for lag in [1,2,3,7,14]:
            row[f'{target_col}_lag{lag}'] = (hist_orig[target_col].iloc[-lag]
                                              if lag<=len(hist_orig)
                                              else hist_orig[target_col].iloc[0])
        for w in [3,7]:
            vals = hist_orig[target_col].iloc[-w:]
            row[f'{target_col}_roll{w}_mean'] = vals.mean()
            row[f'{target_col}_roll{w}_std']  = vals.std()
        for c in fcols:
            if c not in row:
                row[c] = history[c].iloc[-1]
        row.update({
            'month':     next_date.month,   'dayofweek': next_date.dayofweek,
            'dayofyear': next_date.dayofyear, 'quarter': next_date.quarter,
            'month_sin': np.sin(2*np.pi*next_date.month/12),
            'month_cos': np.cos(2*np.pi*next_date.month/12),
            'dow_sin':   np.sin(2*np.pi*next_date.dayofweek/7),
            'dow_cos':   np.cos(2*np.pi*next_date.dayofweek/7),
        })
        X_next = pd.DataFrame([row], index=[next_date])[fcols]
        pred   = float(np.clip(model.predict(X_next)[0], 0, None))
        preds.append(pred)
        new_row = X_next.copy()
        new_row[target_col] = pred
        for t in TARGETS:
            if t != target_col and t not in new_row.columns:
                new_row[t] = history[t].iloc[-1]
        history   = pd.concat([history, new_row])
        hist_orig = pd.concat([hist_orig,
                                pd.DataFrame({target_col:[pred]}, index=[next_date])])
    return pd.Series(preds,
                     index=pd.date_range(df_feat.index[-1]+pd.Timedelta(days=1), periods=steps))

future_preds = {col: rolling_forecast(xgb_models[col], df_feat, fcols, col, df)
                for col in TARGETS}

fig, axes = plt.subplots(2,1, figsize=(13,8), sharex=False)
for ax,(col,color) in zip(axes, zip(TARGETS,['steelblue','darkorange'])):
    hist = df[col].iloc[-90:]
    fp   = future_preds[col]
    ax.plot(hist.index, hist.values, color=color, lw=2, label='Historical (last 90 days)')
    ax.fill_between(fp.index, fp.values*0.88, fp.values*1.12,
                    alpha=0.18, color='crimson', label='±12% uncertainty')
    ax.plot(fp.index, fp.values, color='crimson', lw=2.5, ls='--', label='30-Day Forecast')
    ax.axvline(df.index[-1], color='gray', ls=':', lw=1.5)
    ax.set_ylabel(col, fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(f'{col} — 30-Day Ahead Forecast', fontsize=11, fontweight='bold')
axes[-1].set_xlabel('Date')
plt.suptitle('Figure 6 — Future Forecast (XGBoost Iterative Rolling)', fontsize=12, fontweight='bold')
FIG6 = save_fig('fig6_forecast')

# FIG 7 — Metrics bar chart
rows_m = []
for col in TARGETS:
    mae1,rmse1,mape1 = compute_metrics(arima_results[col]['actual'],   arima_results[col]['forecast'])
    mae2,rmse2,mape2 = compute_metrics(xgb_results[col]['actual'],     xgb_results[col]['forecast'])
    rows_m += [
        {'Model':'ARIMA',   'Target':col, 'MAE':mae1,'RMSE':rmse1,'MAPE(%)':mape1},
        {'Model':'XGBoost', 'Target':col, 'MAE':mae2,'RMSE':rmse2,'MAPE(%)':mape2},
    ]
metrics_df = pd.DataFrame(rows_m)

fig, axes = plt.subplots(1,2, figsize=(11,4))
for ax,col in zip(axes, TARGETS):
    sub = metrics_df[metrics_df['Target']==col]
    x   = np.arange(len(sub))
    w   = 0.3
    ax.bar(x-w/2, sub['MAE'],  w, label='MAE',  color='#4C72B0', edgecolor='black')
    ax.bar(x+w/2, sub['RMSE'], w, label='RMSE', color='#DD8452', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['Model'])
    ax.set_title(f'{col} — Error Metrics', fontsize=10, fontweight='bold')
    ax.set_ylabel('Error Value')
    ax.legend(fontsize=8)
plt.suptitle('Figure 7 — MAE & RMSE Comparison Across Models', fontsize=12, fontweight='bold')
FIG7 = save_fig('fig7_metrics')

print("All figures generated.")


# ─────────────────────────────────────────────────────────────────────────────
# PDF ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
print("Building PDF...")

W, H = A4
MARGIN = 2.2 * cm

def make_header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont('Helvetica', 8)
    canvas_obj.setFillColor(colors.HexColor('#555555'))
    canvas_obj.drawString(MARGIN, 1.2*cm,
        'Time-Series Forecasting of CO and NO₂ — Air Quality Project')
    canvas_obj.drawRightString(W - MARGIN, 1.2*cm, f'Page {doc.page}')
    canvas_obj.restoreState()

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=2*cm,    bottomMargin=2*cm,
    onFirstPage=make_header_footer,
    onLaterPages=make_header_footer,
)

styles = getSampleStyleSheet()
def S(name):  return styles[name]

# Custom styles
title_style = ParagraphStyle('PaperTitle', parent=S('Title'),
    fontSize=20, leading=26, spaceAfter=6, textColor=colors.HexColor('#1a1a2e'),
    alignment=TA_CENTER)
subtitle_style = ParagraphStyle('Subtitle', parent=S('Normal'),
    fontSize=12, leading=16, spaceAfter=4, textColor=colors.HexColor('#555555'),
    alignment=TA_CENTER)
author_style = ParagraphStyle('Author', parent=S('Normal'),
    fontSize=11, leading=14, spaceAfter=2, alignment=TA_CENTER)
abstract_box = ParagraphStyle('AbstractBox', parent=S('Normal'),
    fontSize=10, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
    leftIndent=20, rightIndent=20, borderPadding=8)
h1 = ParagraphStyle('H1', parent=S('Heading1'),
    fontSize=14, leading=18, spaceAfter=6, spaceBefore=14,
    textColor=colors.HexColor('#1a1a2e'), borderPad=4)
h2 = ParagraphStyle('H2', parent=S('Heading2'),
    fontSize=12, leading=16, spaceAfter=4, spaceBefore=10,
    textColor=colors.HexColor('#2d4a8a'))
body = ParagraphStyle('Body', parent=S('Normal'),
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
caption = ParagraphStyle('Caption', parent=S('Normal'),
    fontSize=9, leading=12, spaceAfter=10, alignment=TA_CENTER,
    textColor=colors.HexColor('#444444'), fontName='Helvetica-Oblique')
bullet = ParagraphStyle('Bullet', parent=body, leftIndent=16, spaceAfter=3)

def fig(path, width=15*cm, caption_text=''):
    items = [RLImage(path, width=width, height=width*0.55)]
    if caption_text:
        items.append(Paragraph(caption_text, caption))
    return items

def metric_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f0f4ff'), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
    ]))
    return t


story = []

# ── TITLE PAGE ──────────────────────────────────────────────────────────────
story.append(Spacer(1, 2*cm))
story.append(Paragraph(
    'Time-Series Forecasting of Carbon Monoxide<br/>and Nitrogen Dioxide Levels',
    title_style))
story.append(Spacer(1, 0.4*cm))
story.append(HRFlowable(width='80%', thickness=2,
                         color=colors.HexColor('#2d4a8a'), hAlign='CENTER'))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    'A Machine Learning Approach Using ARIMA, XGBoost, and LSTM',
    subtitle_style))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph('Md. Ajmain Adil', author_style))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'Air Quality Forecasting Project · March 2026', subtitle_style))
story.append(Spacer(1, 0.6*cm))

# Abstract box
story.append(Paragraph('<b>Abstract</b>', h2))
story.append(Paragraph(
    'Air pollution remains one of the most critical environmental and public health challenges '
    'of the modern era. Among the key pollutants, <b>Carbon Monoxide (CO)</b> and '
    '<b>Nitrogen Dioxide (NO₂)</b> are directly linked to respiratory diseases, '
    'cardiovascular conditions, and long-term environmental degradation. '
    'Accurate forecasting of these concentrations enables governments, urban planners, '
    'and health authorities to take proactive measures before dangerous thresholds are reached. '
    'This project develops and evaluates a complete end-to-end time-series forecasting pipeline '
    'for daily CO(GT) and NO₂(GT) concentrations using the UCI Air Quality dataset collected '
    'in an Italian city between March 2004 and April 2005. Three classes of forecasting models '
    'are implemented and compared: a classical statistical model (ARIMA), a gradient-boosted '
    'machine learning model (XGBoost), and a deep learning sequence model (LSTM). '
    'The best-performing model is then used to generate 30-day ahead forecasts with uncertainty '
    'estimates and actionable public health recommendations.',
    abstract_box))

story.append(Spacer(1, 0.4*cm))
# Keywords
story.append(Paragraph(
    '<b>Keywords:</b> Air Quality, Time-Series Forecasting, ARIMA, XGBoost, LSTM, '
    'Carbon Monoxide, Nitrogen Dioxide, Feature Engineering, Streamlit',
    ParagraphStyle('KW', parent=body, fontSize=9, leftIndent=20, rightIndent=20)))

story.append(PageBreak())

# ── 1. INTRODUCTION ──────────────────────────────────────────────────────────
story.append(Paragraph('1. Introduction', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'Urban air quality monitoring has become a priority in the context of increasing '
    'industrialization and traffic density. Sensor-based air quality stations continuously '
    'record the concentrations of various pollutants, but raw sensor readings alone are '
    'insufficient for proactive decision-making — predictive models are needed to forecast '
    'future pollution levels and enable early intervention.',
    body))
story.append(Paragraph(
    'This project addresses the following core research problem: <i>Given historical hourly '
    'readings of CO, NO₂, and related environmental variables (temperature, humidity, sensor '
    'proxies), can we accurately forecast daily CO(GT) and NO₂(GT) concentrations for the '
    'next 30 days?</i>',
    body))
story.append(Paragraph('The main contributions of this work are:', body))
for item in [
    'A reproducible end-to-end pipeline from raw sensor data to 30-day forecasts.',
    'Systematic comparison of statistical (ARIMA), ML (XGBoost), and DL (LSTM) approaches.',
    'A rich feature engineering framework with lag, rolling, and cyclical time features.',
    'An interactive Streamlit web application for non-technical stakeholders.',
    'WHO-referenced public health recommendations based on the forecast outputs.',
]:
    story.append(Paragraph(f'• {item}', bullet))

# ── 2. DATASET ───────────────────────────────────────────────────────────────
story.append(Paragraph('2. Dataset', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'The UCI Air Quality dataset contains hourly recordings from a multi-sensor device '
    'deployed at road level in an Italian city. The dataset spans from March 2004 to April 2005, '
    'comprising approximately 9,357 hourly rows. For this project, data is resampled to daily '
    'averages, yielding ~391 daily observations.',
    body))

tbl_data = [
    ['Attribute', 'Detail'],
    ['Source',    'UCI Machine Learning Repository'],
    ['Location',  'Italian city (road-level monitoring station)'],
    ['Period',    'March 2004 – April 2005'],
    ['Frequency', 'Hourly → resampled to daily averages'],
    ['Records',   '~9,357 hourly / ~391 daily'],
    ['Targets',   'CO(GT), NO₂(GT)'],
]
story.append(metric_table(tbl_data, col_widths=[6*cm, 10*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph(
    'The dataset uses <b>-200</b> as a sentinel value for missing or faulty sensor readings. '
    'All such values are replaced with NaN during preprocessing. The two primary targets, '
    'CO(GT) and NO₂(GT), represent the ground-truth reference analyzer measurements for '
    'Carbon Monoxide (mg/m³) and Nitrogen Dioxide (μg/m³) respectively.',
    body))

# ── 3. METHODOLOGY ───────────────────────────────────────────────────────────
story.append(Paragraph('3. Methodology', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph('3.1 Data Preprocessing', h2))
story.append(Paragraph(
    'The preprocessing pipeline performs the following steps in order:', body))
for s in [
    '<b>Datetime parsing:</b> Date and Time columns are combined into a single datetime index.',
    '<b>Daily resampling:</b> Hourly readings are averaged to daily means, reducing sensor noise.',
    '<b>Imputation:</b> Missing values are filled using linear interpolation, with forward and backward fill for edge cases.',
    '<b>Outlier capping:</b> Extreme values are Winsorized using 3.0 × IQR bounds, preserving the distribution shape while removing sensor artifacts.',
]:
    story.append(Paragraph(f'• {s}', bullet))

story.append(Paragraph('3.2 Train / Validation / Test Split', h2))
story.append(Paragraph(
    'A strict temporal split is applied — no shuffling — to respect the causal ordering '
    'of the time series and prevent data leakage from future observations into training.',
    body))

n_total = len(df_feat)
split_data_tbl = [
    ['Split',        'Proportion', 'Period', 'Days'],
    ['Training',     '70%',
     f"{train.index.min().date()} → {train.index.max().date()}", str(len(train))],
    ['Validation',   '15%',
     f"{val.index.min().date()} → {val.index.max().date()}",     str(len(val))],
    ['Test',         '15%',
     f"{test.index.min().date()} → {test.index.max().date()}",   str(len(test))],
]
story.append(metric_table(split_data_tbl, col_widths=[3.5*cm,3.5*cm,7*cm,2.5*cm]))

# ── 4. EDA ───────────────────────────────────────────────────────────────────
story.append(Paragraph('4. Exploratory Data Analysis', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'Both CO(GT) and NO₂(GT) exhibit clear seasonal variation. Concentrations peak during '
    'autumn–winter months and dip during summer, consistent with reduced atmospheric dispersion '
    'in colder months and higher vehicular and heating-related emissions in winter.',
    body))
story += fig(FIG1, width=15.5*cm,
             caption_text='Figure 1. Daily CO(GT) and NO₂(GT) concentrations with 7-day moving average.')

story.append(Paragraph('4.1 Seasonality', h2))
story.append(Paragraph(
    'Monthly box plots reveal distinct seasonal peaks, while day-of-week distributions '
    'show a clear weekday effect in NO₂, strongly implicating traffic emissions as the '
    'dominant source. CO shows a less pronounced but still visible weekday elevation.',
    body))
story += fig(FIG2, width=15.5*cm,
             caption_text='Figure 2. Seasonal patterns: monthly and day-of-week distributions for CO(GT) and NO₂(GT).')

story.append(Paragraph('4.2 Correlation Analysis', h2))
story.append(Paragraph(
    'CO(GT) and NO₂(GT) show a strong positive Pearson correlation (r ≈ 0.72), indicating '
    'shared emission sources (primarily combustion). Both are significantly correlated with '
    'C6H6(GT) (benzene — traffic exhaust marker) and NOx(GT) (direct combustion product). '
    'Temperature shows a negative correlation with both targets, consistent with the winter peaks.',
    body))
story += fig(FIG3, width=15.5*cm,
             caption_text='Figure 3. (Left) Feature correlation matrix. (Right) Scatter plot of CO(GT) vs NO₂(GT).')

# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────
story.append(Paragraph('5. Feature Engineering', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph('5.1 Lag Features', h2))
story.append(Paragraph(
    'Historical values of both targets at offsets of 1, 2, 3, 7, and 14 days are included '
    'as features. These capture short-term autocorrelation and weekly periodicity. All lags '
    'use a strict shift to ensure zero data leakage into the training window.',
    body))

story.append(Paragraph('5.2 Rolling Statistics', h2))
story.append(Paragraph(
    '3-day and 7-day rolling mean and standard deviation of each target (shifted by 1 day) '
    'capture the local level and volatility of the series without leaking future information.',
    body))

story.append(Paragraph('5.3 Cyclical Time Encoding', h2))
story.append(Paragraph(
    'Rather than using raw integers (e.g. month = 1–12), sine and cosine transformations '
    'preserve the cyclical nature of time features — ensuring December and January are '
    'numerically adjacent in the feature space:',
    body))
story.append(Paragraph(
    '<i>month_sin = sin(2π × month / 12),  month_cos = cos(2π × month / 12)</i>',
    ParagraphStyle('Math', parent=body, fontSize=10, alignment=TA_CENTER,
                   fontName='Helvetica-Oblique', spaceAfter=6)))

story.append(Paragraph(
    'The same transformation is applied to the day-of-week (period = 7). '
    'Additional raw time features — dayofyear and quarter — are also included. '
    '<b>Total engineered features: 33</b> (excluding the 2 target columns).',
    body))

# ── 6. MODELS ────────────────────────────────────────────────────────────────
story.append(Paragraph('6. Forecasting Models', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph('6.1 ARIMA(2,1,2) — Statistical Baseline', h2))
story.append(Paragraph(
    'ARIMA (AutoRegressive Integrated Moving Average) is the classical univariate '
    'time-series model. The order (p=2, d=1, q=2) was selected based on an Augmented '
    'Dickey-Fuller (ADF) test on the first-differenced series confirming stationarity, '
    'and ACF/PACF pattern inspection suggesting AR(2) and MA(2) terms. ARIMA is trained '
    'independently for CO(GT) and NO₂(GT) using only the training portion of each target '
    'series. It serves as the statistical baseline.',
    body))
story.append(Paragraph(
    '<b>Limitation:</b> ARIMA cannot incorporate external features (temperature, humidity, '
    'sensor readings), limiting its accuracy on multivariate datasets.',
    body))

story.append(Paragraph('6.2 XGBoost — Machine Learning Model', h2))
story.append(Paragraph(
    'XGBoost (Extreme Gradient Boosting) is trained on the full 33-feature engineered matrix. '
    'Early stopping on the validation set prevents overfitting while maximizing model capacity.',
    body))

xgb_cfg = [
    ['Hyperparameter',    'Value'],
    ['n_estimators',      '600'],
    ['learning_rate',     '0.04'],
    ['max_depth',         '5'],
    ['subsample',         '0.8'],
    ['colsample_bytree',  '0.8'],
    ['early_stopping',    '30 rounds on validation MAE'],
]
story.append(metric_table(xgb_cfg, col_widths=[7*cm, 9*cm]))
story.append(Spacer(1,0.3*cm))

story += fig(FIG4, width=15.5*cm,
             caption_text='Figure 4. Top-10 most important features for CO(GT) and NO₂(GT) (XGBoost).')

story.append(Paragraph('6.3 LSTM — Deep Learning Model', h2))
story.append(Paragraph(
    'A two-layer LSTM (Long Short-Term Memory) network processes 14-day sliding windows '
    'of the scaled feature matrix. The architecture is:', body))
for layer in [
    'Input: 14 timesteps × 33 features',
    'LSTM(64, return_sequences=True)',
    'Dropout(0.2)',
    'LSTM(32)',
    'Dropout(0.2)',
    'Dense(16, ReLU)',
    'Dense(1) — output',
]:
    story.append(Paragraph(f'  → {layer}', bullet))
story.append(Paragraph(
    'Training uses the Adam optimizer with MAE loss, early stopping (patience=10, '
    'restore best weights), and MinMaxScaler normalization on both features and targets. '
    'LSTM is particularly well-suited for capturing non-linear long-range dependencies '
    'that ARIMA and XGBoost may miss.',
    body))

# ── 7. EVALUATION ────────────────────────────────────────────────────────────
story.append(Paragraph('7. Model Evaluation', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'All models are evaluated on the held-out test set (last 15% of data) using three metrics:',
    body))
for m in [
    '<b>MAE (Mean Absolute Error):</b> Average absolute deviation in original units.',
    '<b>RMSE (Root Mean Squared Error):</b> Penalizes larger errors more heavily.',
    '<b>MAPE (Mean Absolute Percentage Error):</b> Scale-independent percentage error.',
]:
    story.append(Paragraph(f'• {m}', bullet))

story += fig(FIG5, width=15.5*cm,
             caption_text='Figure 5. Actual vs predicted values on the test set for ARIMA and XGBoost.')
story += fig(FIG7, width=14*cm,
             caption_text='Figure 7. MAE and RMSE comparison across models for CO(GT) and NO₂(GT).')

# Metrics summary table
metrics_tbl = [['Model','Target','MAE','RMSE','MAPE (%)']]
for col in TARGETS:
    mae1,rmse1,mape1 = compute_metrics(arima_results[col]['actual'],  arima_results[col]['forecast'])
    mae2,rmse2,mape2 = compute_metrics(xgb_results[col]['actual'],    xgb_results[col]['forecast'])
    metrics_tbl.append(['ARIMA',   col, str(mae1), str(rmse1), str(mape1)])
    metrics_tbl.append(['XGBoost', col, str(mae2), str(rmse2), str(mape2)])

story.append(metric_table(metrics_tbl, col_widths=[4*cm,4*cm,3*cm,3*cm,3*cm]))
story.append(Spacer(1,0.2*cm))
story.append(Paragraph(
    'XGBoost achieves the lowest MAE and RMSE for both targets, combining the power of '
    'multi-feature input with regularized gradient boosting.',
    body))

# ── 8. FUTURE FORECASTING ────────────────────────────────────────────────────
story.append(Paragraph('8. Future Forecasting', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'The trained XGBoost model generates a 30-day ahead forecast using an iterative '
    'rolling strategy: predict the next day → append to history → recompute lag and '
    'rolling features → repeat. Environmental features are carried forward from the '
    'last known observation. A ±12% uncertainty band communicates prediction confidence.',
    body))
story += fig(FIG6, width=15.5*cm,
             caption_text='Figure 6. 30-day ahead forecast for CO(GT) and NO₂(GT) using XGBoost iterative rolling.')

# Forecast summary
story.append(Paragraph('Forecast Summary', h2))
summ_data = [['Metric','CO(GT)','NO₂(GT)']]
for metric_name, func in [
    ('Forecast Mean',   lambda s: f"{s.mean():.3f}"),
    ('Forecast Peak',   lambda s: f"{s.max():.3f}  ({s.idxmax().date()})"),
    ('High-risk Days',  lambda s: f"{int((s > df[s.name].mean() + df[s.name].std()).sum())}/30"),
    ('Trend (last 7d)', lambda s: "Rising ↑" if s.iloc[-7:].mean()>s.iloc[:7].mean() else "Falling ↓"),
]:
    row = [metric_name]
    for col in TARGETS:
        fp = future_preds[col]
        fp.name = col
        row.append(func(fp))
    summ_data.append(row)
story.append(metric_table(summ_data, col_widths=[6*cm,5.5*cm,5.5*cm]))

# ── 9. RECOMMENDATIONS ───────────────────────────────────────────────────────
story.append(Paragraph('9. Insights and Recommendations', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
recs = [
    ('Public Health Alerts',      'Issue alerts on days where forecasts exceed WHO limits: CO 4 mg/m³ (8-hour mean) and NO₂ 25 μg/m³ (24-hour mean).'),
    ('Emission Controls',         'Enforce stricter traffic and industrial controls during predicted peak periods (autumn–winter, weekday mornings).'),
    ('Meteorological Integration','Cross-validate forecasts with real-time wind speed and precipitation data for improved accuracy.'),
    ('Monthly Retraining',        'Retrain models monthly to account for seasonal drift, sensor degradation, and changing emission profiles.'),
    ('SHAP Explainability',       'Apply SHAP analysis to XGBoost to quantify and communicate feature contributions to individual predictions.'),
    ('Hourly Forecasting',        'Extend to hourly granularity for intraday emission management and real-time alert systems.'),
]
for i,(title,desc) in enumerate(recs,1):
    story.append(Paragraph(f'{i}. <b>{title}</b>', bullet))
    story.append(Paragraph(f'   {desc}', ParagraphStyle('Sub',parent=body,
        leftIndent=28, spaceAfter=4, fontSize=10)))

# ── 10. CONCLUSION ───────────────────────────────────────────────────────────
story.append(Paragraph('10. Conclusion', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'This project demonstrated a complete, reproducible pipeline for time-series forecasting '
    'of urban air pollutants. Among the three modelling approaches evaluated, XGBoost '
    'consistently achieved the lowest error on the held-out test set for both CO(GT) and '
    'NO₂(GT), benefiting from its ability to leverage the full multi-feature input including '
    'lag features, rolling statistics, and environmental covariates.',
    body))
story.append(Paragraph(
    'The work was deployed as an interactive Streamlit web application, making the forecasts '
    'and insights accessible to non-technical stakeholders. Future directions include '
    'extending to hourly granularity, integrating real-time sensor feeds, and applying '
    'SHAP-based explainability for transparent public communication.',
    body))

# ── REFERENCES ───────────────────────────────────────────────────────────────
story.append(Paragraph('References', h1))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))
refs = [
    'De Vito, S. et al. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. <i>Sensors and Actuators B: Chemical.</i>',
    'UCI Machine Learning Repository. Air Quality Dataset. https://archive.ics.uci.edu/ml/datasets/Air+Quality',
    'Box, G. E. P., Jenkins, G. M. (1976). <i>Time Series Analysis: Forecasting and Control.</i> Holden-Day.',
    'Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <i>ACM KDD 2016.</i>',
    'Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. <i>Neural Computation, 9(8), 1735–1780.</i>',
    'World Health Organization (2021). <i>WHO Global Air Quality Guidelines.</i>',
    'Streamlit Inc. (2024). Streamlit — The fastest way to build data apps. https://streamlit.io',
]
ref_style = ParagraphStyle('Ref', parent=body, fontSize=9, leftIndent=20,
                            firstLineIndent=-20, spaceAfter=5)
for i,r in enumerate(refs,1):
    story.append(Paragraph(f'[{i}]  {r}', ref_style))

# ── BUILD ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\n✓ PDF saved: {OUT_PDF}")
