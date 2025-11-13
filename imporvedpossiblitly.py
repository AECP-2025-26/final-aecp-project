# aecp_improved.py
# Streamlit app: AECP improved forecasting pipeline with feature engineering and model comparison
# Requirements: streamlit, pandas, numpy, matplotlib, scikit-learn, statsmodels
# Optional (for LSTM): tensorflow

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io
import warnings
import math
warnings.filterwarnings("ignore")

# Try to import tensorflow (LSTM) — gracefully handle absence
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="AECP Improved", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper functions
# ---------------------------
def safe_min_nonzero(series, floor=1e-6):
    """Return min(series) but ensure > 0 to avoid division by zero."""
    m = series.min()
    return m if m > 0 else floor

def mape(y_true, y_pred):
    # Avoid divide by zero by using safe denominator
    denom = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def nrmse(y_true, y_pred, kind='mean'):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    if kind == 'range':
        denom = (np.max(y_true) - np.min(y_true))
    else:
        denom = np.mean(y_true)
    denom = denom if denom != 0 else 1e-6
    return rmse_val / denom

def train_test_split_time(df, test_size=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def add_time_features(df):
    # df indexed by year or has 'year' column
    if 'year' in df.columns:
        df = df.set_index(pd.to_datetime(df['year'], format='%Y'))
        df.drop(columns=['year'], inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # try to coerce index to datetime if it's numeric year
        try:
            df.index = pd.to_datetime(df.index.year, format='%Y')
        except Exception:
            pass

    # Add lag features and rolling means for population
    df['pop_lag1'] = df['population'].shift(1)
    df['pop_lag2'] = df['population'].shift(2)
    df['pop_diff1'] = df['population'].diff(1)
    df['pop_roll_mean_3'] = df['population'].rolling(window=3, min_periods=1).mean()
    df['pop_roll_mean_5'] = df['population'].rolling(window=5, min_periods=1).mean()

    # Fill NaNs from shift/diff with reasonable values (forward/backfill)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def evaluate_predictions(y_true, y_pred):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse_mean = nrmse(y_true, y_pred, kind='mean')
    nrmse_range = nrmse(y_true, y_pred, kind='range')
    return {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'MAPE (%)': mape_val,
        'R2': r2,
        'NRMSE_mean': nrmse_mean,
        'NRMSE_range': nrmse_range
    }

def plot_series(df, forecast_years=None, forecast_vals=None, title='Population Forecast'):
    plt.figure(figsize=(12,6))
    plt.plot(df.index.year, df['population'], label='Historical', linewidth=2)
    if forecast_years is not None and forecast_vals is not None:
        plt.plot(forecast_years, forecast_vals, linestyle='--', marker='o', label='Forecast', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------
# UI: Sidebar - Data input
# ---------------------------
st.sidebar.title("AECP: Improved Forecasting")
st.sidebar.markdown("Upload CSV or generate sample Datasets. Required columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`")

data_source = st.sidebar.radio("Data source", ["Upload CSV", "Use Sample Dodo Dataset", "Generate Synthetic (quick)"])

uploaded_df = None
if data_source == "Upload CSV":
    f = st.sidebar.file_uploader("Upload CSV file (year,population,temperature,rainfall,habitat_index)", type=['csv'])
    if f is not None:
        uploaded_df = pd.read_csv(f)
elif data_source == "Use Sample Dodo Dataset":
    # Provide the dodo dataset we created earlier (1450-1650)
    # We'll synthesize a smooth realistic dodo dataset again here to ensure app independence
    years = np.arange(1450, 1650)
    n = len(years)
    rng = np.random.default_rng(42)
    population = np.round(np.linspace(20000, 300, n) * (1 + rng.normal(0, 0.03, n))).astype(int)
    population = np.maximum(population, 100)
    temperature = np.round(np.linspace(25.0, 25.8, n) + rng.normal(0, 0.05, n), 2)
    rainfall = np.round(np.linspace(1800, 1600, n) + rng.normal(0, 12, n), 1)
    habitat_index = np.round(np.clip(np.linspace(0.95, 0.25, n) + rng.normal(0, 0.02, n), 0.05, 1.0), 3)
    uploaded_df = pd.DataFrame({
        'year': years,
        'population': population,
        'temperature': temperature,
        'rainfall': rainfall,
        'habitat_index': habitat_index
    })
    st.sidebar.success("Using internal Dodo sample dataset (1450-1649).")
elif data_source == "Generate Synthetic (quick)":
    # Let user choose quick synthetic parameters
    start_year = st.sidebar.number_input("Start Year", value=1800, step=1)
    end_year = st.sidebar.number_input("End Year", value=2020, step=1)
    initial_pop = st.sidebar.number_input("Initial population", value=100000, step=100)
    final_pop = st.sidebar.number_input("Approx final population", value=10000, step=100)
    temp_start = st.sidebar.number_input("Temp start (°C)", value=25.0)
    temp_end = st.sidebar.number_input("Temp end (°C)", value=27.0)
    rain_start = st.sidebar.number_input("Rain start (mm)", value=2000)
    rain_end = st.sidebar.number_input("Rain end (mm)", value=1800)
    habitat_start = st.sidebar.number_input("Habitat start (0-1)", value=0.95)
    habitat_end = st.sidebar.number_input("Habitat end (0-1)", value=0.40)
    if st.sidebar.button("Generate Synthetic"):
        years = np.arange(start_year, end_year + 1)
        n = len(years)
        rng = np.random.default_rng(123)
        pop_clean = np.linspace(initial_pop, final_pop, n) * (1 + rng.normal(0, 0.06, n))
        pop_clean = np.round(np.clip(pop_clean, 1, None)).astype(int)
        temp = np.round(np.linspace(temp_start, temp_end, n) + rng.normal(0, 0.08, n), 2)
        rain = np.round(np.linspace(rain_start, rain_end, n) + rng.normal(0, 6, n), 1)
        habitat = np.round(np.clip(np.linspace(habitat_start, habitat_end, n) + rng.normal(0, 0.02, n), 0.01, 1.0), 3)
        uploaded_df = pd.DataFrame({
            'year': years,
            'population': pop_clean,
            'temperature': temp,
            'rainfall': rain,
            'habitat_index': habitat
        })
        st.sidebar.success("Synthetic dataset generated.")

if uploaded_df is None:
    st.info("Please upload a CSV or choose/generate a dataset from the sidebar to proceed.")
    st.stop()

# ---------------------------
# Main: Data validation & preprocessing
# ---------------------------
st.title("AECP — Improved Forecasting & Evaluation")
st.markdown("This app preprocesses time-series population data, engineers features, and compares forecasting models. Target is `population`.")

# Validate required columns
required_cols = ['year', 'population', 'temperature', 'rainfall', 'habitat_index']
missing = [c for c in required_cols if c not in uploaded_df.columns]
if missing:
    st.error(f"Uploaded data is missing required columns: {missing}")
    st.stop()

# Ensure year is integer and continuous
df = uploaded_df.copy()
df['year'] = df['year'].astype(int)
df = df.sort_values('year').reset_index(drop=True)
if df['year'].duplicated().any():
    st.warning("Duplicate years detected; duplicates will be averaged.")
    df = df.groupby('year', as_index=False).mean()

# Ensure at least 150 rows (user asked earlier)
if len(df) < 150:
    st.warning(f"Dataset has {len(df)} rows. Recommended >= 150 rows for reliable modeling. You may continue, but results could be unstable.")

st.write(f"Data range: {df['year'].min()} — {df['year'].max()} ({len(df)} rows)")

# Convert to indexed dataframe for time features
df_proc = add_time_features(df.copy())

st.subheader("Data (first 10 rows)")
st.dataframe(df_proc.head(10))

# Provide quick visualization
st.subheader("Historical Population")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_proc.index.year, df_proc['population'], linewidth=2)
ax.set_xlabel('Year'); ax.set_ylabel('Population'); ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# Modeling options
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Modeling Options")

forecast_years = st.sidebar.slider("Forecast horizon (years)", min_value=10, max_value=200, value=50, step=10)
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

model_choices = st.sidebar.multiselect(
    "Select models to run",
    options=['ARIMA', 'SARIMAX', 'RandomForest', 'GradientBoosting'] + (['LSTM'] if TF_AVAILABLE else []),
    default=['RandomForest', 'ARIMA']
)

log_transform_target = st.sidebar.checkbox("Log-transform target (recommended)", value=True)
scale_features = st.sidebar.checkbox("Scale features (StandardScaler for tree models off by default)", value=False)

# Hyperparams for tree models
n_estimators = st.sidebar.slider("Tree models: n_estimators", 50, 1000, 200, step=50)
max_depth = st.sidebar.slider("Tree models: max_depth (0 = None)", 0, 10, 5)

# LSTM params
if TF_AVAILABLE and 'LSTM' in model_choices:
    lstm_epochs = st.sidebar.number_input("LSTM epochs", value=50, min_value=5, step=5)
    lstm_batch = st.sidebar.number_input("LSTM batch size", value=16, min_value=1, step=1)
    lstm_window = st.sidebar.slider("LSTM window size", 3, 30, 8)

# ---------------------------
# Prepare training and testing sets
# ---------------------------
train_df, test_df = train_test_split_time(df_proc, test_size=test_size)

# Choose features
feature_cols = ['population', 'pop_lag1', 'pop_lag2', 'pop_diff1', 'pop_roll_mean_3', 'pop_roll_mean_5', 
                'temperature', 'rainfall', 'habitat_index']
# For supervised models, we predict 'population' at t using t-1 features; we'll shift target to next year
supervised = df_proc.copy()
supervised['target'] = supervised['population'].shift(-1)  # predict next year's population
# Drop last row (no target)
supervised = supervised.dropna(subset=['target'])

# Split supervised dataset by time
supervised_train = supervised.iloc[:int(len(supervised)*(1-test_size))]
supervised_test = supervised.iloc[int(len(supervised)*(1-test_size)):]

# X/y
X_train = supervised_train[feature_cols].values
y_train = supervised_train['target'].values
X_test = supervised_test[feature_cols].values
y_test = supervised_test['target'].values

# Optional scaling
scaler_X = None
if scale_features:
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

# Optionally log-transform target
scaler_y = None
if log_transform_target:
    # use log1p to handle small values
    y_train_trans = np.log1p(y_train)
    y_test_trans = np.log1p(y_test)
else:
    y_train_trans = y_train.copy()
    y_test_trans = y_test.copy()

# ---------------------------
# Model training & evaluation
# ---------------------------
results = {}

# ---- Random Forest ----
if 'RandomForest' in model_choices:
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_trans)
    y_pred_tr = rf.predict(X_train)
    y_pred_te = rf.predict(X_test)
    # inverse transform if log
    if log_transform_target:
        y_pred_te_inv = np.expm1(y_pred_te)
        y_pred_tr_inv = np.expm1(y_pred_tr)
        y_test_inv = np.expm1(y_test_trans)
        y_train_inv = np.expm1(y_train_trans)
    else:
        y_pred_te_inv = y_pred_te
        y_pred_tr_inv = y_pred_tr
        y_test_inv = y_test
        y_train_inv = y_train
    results['RandomForest'] = {
        'model': rf,
        'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
        'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
        'y_pred_test': y_pred_te_inv
    }

# ---- Gradient Boosting ----
if 'GradientBoosting' in model_choices:
    gb = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)
    gb.fit(X_train, y_train_trans)
    y_pred_tr = gb.predict(X_train)
    y_pred_te = gb.predict(X_test)
    if log_transform_target:
        y_pred_te_inv = np.expm1(y_pred_te)
        y_pred_tr_inv = np.expm1(y_pred_tr)
        y_test_inv = np.expm1(y_test_trans)
        y_train_inv = np.expm1(y_train_trans)
    else:
        y_pred_te_inv = y_pred_te
        y_pred_tr_inv = y_pred_tr
        y_test_inv = y_test
        y_train_inv = y_train
    results['GradientBoosting'] = {
        'model': gb,
        'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
        'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
        'y_pred_test': y_pred_te_inv
    }

# ---- ARIMA / SARIMAX ----
# ARIMA will operate on univariate population series (predict next steps)
if 'ARIMA' in model_choices or 'SARIMAX' in model_choices:
    # Use last part of series for fitting (train on train_df population)
    series_train = train_df['population']
    series_all = df_proc['population']
    try:
        if 'ARIMA' in model_choices:
            # Simple ARIMA with fixed order — users could tune this
            arima = ARIMA(series_train, order=(3,1,1)).fit()
            # Predict test horizon (next len(test_df) steps)
            arima_pred_test = arima.forecast(steps=len(test_df))
            # Evaluate using test_df population (aligned)
            # We need to align indices: test_df index corresponds to years; arima_pred_test returned in order
            actual_test_pop = test_df['population'].values
            arima_pred_test_vals = np.array(arima_pred_test).astype(float)
            results['ARIMA'] = {
                'model': arima,
                'test_metrics': evaluate_predictions(actual_test_pop, arima_pred_test_vals),
                'y_pred_test': arima_pred_test_vals
            }
        if 'SARIMAX' in model_choices:
            # Seasonal order can be tuned; here we use seasonal_period=0 (no season) by default
            sarima = SARIMAX(series_train, order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
            sarima_pred_test = sarima.forecast(steps=len(test_df))
            results['SARIMAX'] = {
                'model': sarima,
                'test_metrics': evaluate_predictions(test_df['population'].values, np.array(sarima_pred_test).astype(float)),
                'y_pred_test': np.array(sarima_pred_test).astype(float)
            }
    except Exception as e:
        st.warning(f"ARIMA/SARIMAX training failed: {e}")

# ---- LSTM (optional) ----
if TF_AVAILABLE and 'LSTM' in model_choices:
    try:
        # Prepare scaler for LSTM on population only (sequence modeling)
        minmax = MinMaxScaler()
        pop_vals = df_proc['population'].values.reshape(-1,1)
        pop_scaled = minmax.fit_transform(pop_vals)
        # create sequences
        window = lstm_window if 'lstm_window' in locals() else 8
        def create_seqs(arr, w):
            X, y = [], []
            for i in range(len(arr)-w):
                X.append(arr[i:i+w])
                y.append(arr[i+w])
            return np.array(X), np.array(y)
        X_seq, y_seq = create_seqs(pop_scaled, window)
        # train/test split on sequences
        split_idx = int(X_seq.shape[0] * (1 - test_size))
        X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
        y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
        # build model
        model_lstm = Sequential([
            LSTM(64, activation='tanh', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        # fit
        model_lstm.fit(X_seq_train, y_seq_train, epochs=lstm_epochs, batch_size=lstm_batch, verbose=0)
        # predict
        y_seq_pred = model_lstm.predict(X_seq_test)
        y_seq_pred_inv = minmax.inverse_transform(y_seq_pred).flatten()
        y_seq_test_inv = minmax.inverse_transform(y_seq_test).flatten()
        results['LSTM'] = {
            'model': model_lstm,
            'test_metrics': evaluate_predictions(y_seq_test_inv, y_seq_pred_inv),
            'y_pred_test': y_seq_pred_inv
        }
    except Exception as e:
        st.warning(f"LSTM failed: {e}")

# ---------------------------
# Display model comparison table
# ---------------------------
st.subheader("Model Comparison (Test Set Metrics)")
summary_rows = []
for name, info in results.items():
    metrics = info.get('test_metrics', {})
    summary_rows.append({
        'Model': name,
        'RMSE': metrics.get('RMSE', np.nan),
        'MAE': metrics.get('MAE', np.nan),
        'MAPE (%)': metrics.get('MAPE (%)', np.nan),
        'R2': metrics.get('R2', np.nan),
        'NRMSE_mean': metrics.get('NRMSE_mean', np.nan)
    })

if len(summary_rows) == 0:
    st.warning("No models were trained. Select models in the sidebar and ensure dataset is valid.")
else:
    summary_df = pd.DataFrame(summary_rows).sort_values('RMSE')
    st.dataframe(summary_df.style.format({
        'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'MAPE (%)': '{:.2f}', 'R2': '{:.3f}', 'NRMSE_mean': '{:.3f}'
    }))

# ---------------------------
# Visualize best model predictions
# ---------------------------
st.subheader("Prediction Diagnostics & Forecasting")

if len(results) > 0:
    # Choose best model by RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_metrics']['RMSE'])
    st.info(f"Best model by RMSE on test set: **{best_model_name}**")
    best_info = results[best_model_name]
    y_pred_test = best_info['y_pred_test']
    # Plot actual vs predicted for test period
    # Identify years for test set: supervised_test corresponds to years from supervised_test.index
    test_years = supervised_test.index.year
    plt.figure(figsize=(12,5))
    plt.plot(supervised_test.index.year, y_test if not log_transform_target else np.expm1(y_test_trans), label='Actual (next year target)')
    plt.plot(supervised_test.index.year, y_pred_test, label=f'Predicted ({best_model_name})')
    plt.title(f'Actual vs Predicted — {best_model_name} (Test Set)')
    plt.xlabel('Year'); plt.ylabel('Population')
    plt.legend(); plt.grid(alpha=0.3)
    st.pyplot(plt.gcf()); plt.close()

    # Forecast into future using the best model
    st.markdown("### Multi-year Forecast (Iterative)")
    forecast_h = forecast_years

    if best_model_name in ['RandomForest', 'GradientBoosting']:
        # iterative forecasting: use last known features and predicted population to roll forward
        model = best_info['model']
        # start with most recent row of df_proc (which has features)
        last_row = df_proc.iloc[-1:].copy()
        forecasts = []
        current_row = last_row.copy()
        for i in range(forecast_h):
            # create feature vector
            feat = current_row[feature_cols].values.astype(float).reshape(1, -1)
            if scale_features and scaler_X is not None:
                feat_scaled = scaler_X.transform(feat)
            else:
                feat_scaled = feat
            pred_trans = model.predict(feat_scaled)
            if log_transform_target:
                pred = np.expm1(pred_trans)[0]
            else:
                pred = pred_trans[0]
            forecasts.append(pred)
            # shift current_row to next year by updating population & engineered features
            new_year = current_row.index.year[0] + 1
            new_row = current_row.copy()
            # update population to predicted value
            new_row['population'] = pred
            # update lags and rolls
            new_row['pop_lag2'] = new_row['pop_lag1']
            new_row['pop_lag1'] = pred
            new_row['pop_diff1'] = new_row['population'] - new_row['pop_lag1']
            # rolling means approximate by combining
            new_row['pop_roll_mean_3'] = (new_row['pop_roll_mean_3'] * 2 + pred) / 3
            new_row['pop_roll_mean_5'] = (new_row['pop_roll_mean_5'] * 4 + pred) / 5
            # environmental variables: assume trend continues linearly (extrapolate)
            # compute simple linear extrapolation based on last two real years
            for col in ['temperature', 'rainfall', 'habitat_index']:
                # estimate slope from last two historical points
                hist_vals = df_proc[col].values[-2:]
                slope = hist_vals[-1] - hist_vals[-2]
                new_row[col] = new_row[col] + slope
            # set new index
            new_row.index = pd.to_datetime([str(new_year)], format='%Y')
            current_row = new_row
        forecast_years_arr = np.arange(df_proc.index.year.max() + 1, df_proc.index.year.max() + 1 + forecast_h)
        # Plot
        plot_series(df_proc, forecast_years_arr, forecasts, title=f'Forecast using {best_model_name}')
        # Provide CSV download
        out_df = pd.DataFrame({'year': forecast_years_arr, 'predicted_population': np.round(forecasts).astype(int)})
        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        st.download_button("Download forecast CSV", csv_buffer.getvalue(), file_name=f'forecast_{best_model_name}.csv', mime='text/csv')
    elif best_model_name in ['ARIMA', 'SARIMAX']:
        # Use the fitted model to forecast forward
        model = best_info['model']
        try:
            fc = model.forecast(steps=forecast_h)
            forecast_years_arr = np.arange(df_proc.index.year.max() + 1, df_proc.index.year.max() + 1 + forecast_h)
            plot_series(df_proc, forecast_years_arr, np.array(fc).astype(float), title=f'Forecast using {best_model_name}')
            out_df = pd.DataFrame({'year': forecast_years_arr, 'predicted_population': np.round(np.array(fc).astype(float)).astype(int)})
            csv_buffer = io.StringIO()
            out_df.to_csv(csv_buffer, index=False)
            st.download_button("Download forecast CSV", csv_buffer.getvalue(), file_name=f'forecast_{best_model_name}.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Forecasting via {best_model_name} failed: {e}")
    elif best_model_name == 'LSTM':
        st.info("LSTM forecasting not available in the iterative pipeline — consider using ARIMA or tree models for multi-year forecasts.")
    else:
        st.info("Selected best model not supported for iterative forecasting in this app.")

# ---------------------------
# Offer tips & diagnostics
# ---------------------------
st.markdown("---")
st.subheader("Tips to further improve accuracy")
st.markdown("""
- **Try log-transforming the target** (enabled in sidebar). It stabilizes variance for exponential declines.
- **Feature engineering** (lags, diffs, rolling means) often yields big gains for time-series models.
- **Use cross-validation** (TimeSeriesSplit) to tune hyperparameters systematically.
- **Tune ARIMA/SARIMA orders** using auto_arima libraries (not included by default) or grid search.
- **For LSTM**, ensure long training data, proper windowing, and sufficient epochs. LSTMs generally need more data and tuning.
- **If your target includes very large scales (millions)** consider scaling the population down (e.g., divide by 1e3) before modeling to improve numerical stability.
""")

st.caption("AECP improved — model comparison, safe forecasting, and downloadable predictions. If you want, I can auto-tune hyperparameters or provide a version that runs scheduled backtesting and saves model artifacts.")
