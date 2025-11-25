import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
from sklearn.metrics import mean_squared_error
from itertools import product
import time

warnings.filterwarnings('ignore')

# --- Helper Functions (Cached for performance) ---

# Caching is crucial to avoid re-running expensive models when the UI updates
@st.cache_data(show_spinner=False)
def create_sequences(data, window):
    """Creates input/output sequences for LSTM training/testing."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

@st.cache_data(show_spinner=False)
def run_all_models(train_data, test_data, window_size=5, arima_order=(3, 1, 1), sarima_order=(1, 1, 1), sarima_seasonal_order=(0, 0, 0, 0), epochs=50, batch_size=8):
    """
    Trains, tests, and calculates RMSE for all three models (ARIMA, SARIMA, LSTM)
    using the train/test split.
    """
    results = {}

    # --- ARIMA Model ---
    try:
        arima_model = ARIMA(train_data['population'], order=arima_order).fit()
        arima_forecast_test = arima_model.forecast(steps=len(test_data))
        arima_mse = mean_squared_error(test_data['population'], arima_forecast_test)
        arima_rmse = np.sqrt(arima_mse)
        results['ARIMA'] = {'rmse': arima_rmse, 'order': arima_order}
    except Exception as e:
        results['ARIMA'] = {'rmse': 999999999, 'error': f"ARIMA failed: {e}"}

    # --- SARIMA Model ---
    try:
        sarima_model = SARIMAX(train_data['population'], order=sarima_order, seasonal_order=sarima_seasonal_order).fit(disp=False)
        sarima_forecast_test = sarima_model.forecast(steps=len(test_data))
        sarima_mse = mean_squared_error(test_data['population'], sarima_forecast_test)
        sarima_rmse = np.sqrt(sarima_mse)
        results['SARIMA'] = {'rmse': sarima_rmse, 'order': sarima_order, 'seasonal_order': sarima_seasonal_order}
    except Exception as e:
        results['SARIMA'] = {'rmse': 999999999, 'error': f"SARIMA failed: {e}"}

    # --- LSTM Model ---
    try:
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data[['population']])
        X_train, y_train = create_sequences(train_data_scaled, window_size)

        if len(X_train) == 0:
            results['LSTM'] = {'rmse': 999999999, 'error': "Insufficient data for LSTM sequence creation."}
        else:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            # Simple LSTM for pre-run performance evaluation
            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict on the test set
            test_data_scaled = scaler.transform(test_data[['population']])
            X_test, _ = create_sequences(test_data_scaled, window_size)
            
            if len(X_test) > 0:
                X_test_reshaped = np.array(X_test).reshape(len(X_test), window_size, 1)
                lstm_predictions_scaled = lstm_model.predict(X_test_reshaped, verbose=0)
                lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                
                actual_test_data = test_data['population'].iloc[window_size:]
                lstm_mse = mean_squared_error(actual_test_data, lstm_predictions)
                lstm_rmse = np.sqrt(lstm_mse)
                results['LSTM'] = {'rmse': lstm_rmse, 'window_size': window_size, 'epochs': epochs, 'batch_size': batch_size}
            else:
                 results['LSTM'] = {'rmse': 999999999, 'error': "Test data too small for LSTM window."}
                 
    except Exception as e:
        results['LSTM'] = {'rmse': 999999999, 'error': f"LSTM failed: {e}"}

    return results

@st.cache_data(show_spinner="Generating Full Forecast...")
def get_full_forecast_and_metrics(df, future_steps, model_choice, params):
    """
    Fits the chosen model on the FULL dataset and generates the final forecast.
    The RMSE from the initial train/test split is retrieved from session state.
    """
    population_data = df['population']
    
    # --- ARIMA/SARIMA ---
    if model_choice in ['ARIMA', 'SARIMA']:
        order = params['order']
        try:
            if model_choice == 'ARIMA':
                model_full = ARIMA(population_data, order=order).fit()
            else: # SARIMA
                seasonal_order = params['seasonal_order']
                model_full = SARIMAX(population_data, order=order, seasonal_order=seasonal_order).fit(disp=False)
                
            forecast_full = model_full.forecast(steps=future_steps).values
        except Exception as e:
            return None, None, f"Model fitting error: {e}"

    # --- LSTM ---
    elif model_choice == 'LSTM':
        window_size = params['window_size']
        epochs = params['epochs']
        batch_size = params['batch_size']
        
        try:
            scaler = MinMaxScaler()
            data_scaled_full = scaler.fit_transform(population_data.values.reshape(-1, 1))
            X_full, y_full = create_sequences(data_scaled_full, window_size)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
            
            # Configure TensorFlow to only log errors/warnings
            tf.get_logger().setLevel('ERROR')
            
            model_full = Sequential([
                LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
                Dense(1)
            ])
            model_full.compile(optimizer='adam', loss='mse')
            model_full.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=0)
            
            last_seq_full = data_scaled_full[-window_size:]
            lstm_preds_full = []
            
            for _ in range(future_steps):
                input_seq_full = last_seq_full.reshape(1, window_size, 1)
                pred_full = model_full.predict(input_seq_full, verbose=0)
                lstm_preds_full.append(pred_full[0, 0])
                last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
                
            forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1)).flatten()
            
        except Exception as e:
            return None, None, f"LSTM Model training error: {e}"

    # --- Extinction Check and Prediction DataFrame ---
    extinction_year = None
    years_future = np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps)
    
    for i, val in enumerate(forecast_full):
        if val <= 0:
            extinction_year = df.index.year.max() + i + 1
            break

    forecast_df = pd.DataFrame({'year': years_future, 'population': forecast_full.round(0).astype(int)})
    
    return forecast_df, extinction_year, None


# --- Page Configuration (MUST be the first st command) ---
st.set_page_config(
    layout="wide",
    page_title="AECP: Animal Extinction Calendar Predictor",
    page_icon="🐘"
)

# --- Custom CSS Styling (Revised for Vivid App Feel) ---
st.markdown("""
<style>
    /* Global Background and Content Container */
    .stApp {
        background-color: #f0f2f6; /* Light gray/blue background for app feel */
    }
    .main-header {
        font-size: 64px; /* Slightly smaller for better fit, but still large */
        font-weight: 900;
        color: #004d40; /* Deep Teal - Primary Color */
        text-align: center;
        padding: 20px 0 10px 0;
        border-bottom: 6px solid #8CE44C; /* Vivid Green accent */
        margin-bottom: 25px;
        letter-spacing: 2px;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 24px;
        font-weight: 400;
        color: #555555;
        text-align: center;
        margin-bottom: 40px;
        line-height: 1.5;
    }
    
    /* Custom button styling for the 'Start Analysis' button (More prominent) */
    .stButton>button {
        font-size: 28px;
        font-weight: 700;
        color: white;
        background-color: #2ecc71; /* Vibrant Green */
        border: none;
        border-radius: 15px;
        padding: 15px 70px;
        margin: 30px auto;
        display: block;
        box-shadow: 0 8px #1abc9c;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background-color: #1abc9c;
        box-shadow: 0 5px #16a085;
        transform: translateY(3px); /* Subtle press effect */
    }
    
    /* Card/Container Styles (for the large menus) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #ddd;
        border-radius: 15px; /* Rounded corners for app feel */
        padding: 30px; /* More padding for larger look */
        margin-bottom: 25px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1); /* Stronger shadow */
        background-color: #ffffff;
    }
    
    /* Metric Card Styling (Restored White Background with Vivid Colors) */
    [data-testid="stMetric"] {
        background-color: #ffffff; /* Explicitly white background */
        border: 2px solid #004d40;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #555555 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #004d40 !important; /* Deep Teal Value */
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        color: #e74c3c !important; /* Red Delta */
        font-weight: 600 !important;
    }

    /* Expander/Info Styling */
    .st-expander > summary {
        font-size: 1.2rem;
        font-weight: 700;
        color: #004d40;
    }
    .stAlert {
        border-radius: 10px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header & Initialization ---
st.markdown('<p class="main-header">AECP: Extinction Calendar Predictor 🐘</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A comprehensive, multi-model time-series analysis tool for forecasting population trends.</p>', unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = 'ARIMA'
if 'future_steps' not in st.session_state:
    st.session_state.future_steps = 50

# --- Synthetic Data Function (Unchanged) ---
@st.cache_data
def generate_custom_data(start_year, num_years, initial_pop, decline_rate):
    """Generates synthetic time-series data based on user-defined parameters."""
    end_year = start_year + num_years
    years = np.arange(start_year, end_year)
    
    population_base = initial_pop - decline_rate * (years - start_year)
    population_noise = np.random.normal(0, initial_pop * 0.05, len(years))
    population = np.maximum(50, population_base + population_noise)

    temperature = 10 + 0.1 * (years - start_year) + np.random.normal(0, 0.5, len(years))
    rainfall = 1000 - 5 * (years - start_year) + np.random.normal(0, 50, len(years))
    habitat_index_base = 1.0 - (decline_rate / initial_pop) * 5 * (years - start_year)
    habitat_index = habitat_index_base + np.random.normal(0, 0.05, len(years))
    habitat_index = np.clip(habitat_index, 0.1, 1.0) 

    df = pd.DataFrame({
        'year': years,
        'population': population.astype(int),
        'temperature': temperature.round(1),
        'rainfall': rainfall.astype(int),
        'habitat_index': habitat_index.round(2)
    })
    return df

# --- Introduction & Definitions (Restored) ---
with st.container(border=True):
    st.write("""
    Hi! This application uses time-series forecasting models (ARIMA, SARIMA, and LSTM) to predict future population based on historical data.
    
    **Workflow:**
    1. **Data Selection:** Choose a data source below.
    2. **Start Analysis:** Click the **🚀 Start Analysis** button. This automatically runs a benchmark test across all models.
    3. **Results:** View the best-performing model's forecast and metrics in the tabs that appear.
    """)
    
    with st.expander("📖 Essential Definitions & Model Overview", expanded=False):
        st.markdown("**ARIMA & SARIMA**")
        st.write("ARIMA (Autoregressive Integrated Moving Average) and SARIMA (Seasonal ARIMA). The **A**utoregressive (AR) part indicates that the variable is regressed on its own prior values. The **M**oving **A**verage (MA) part indicates that the regression error is a linear combination of past error terms. The **I**ntegrated (I) part indicates data values have been differenced to become stationary. SARIMA adds seasonal components for data with annual or recurring patterns.")
        
        st.markdown("**LSTM (Long Short-Term Memory)**")
        st.write("A specialized type of recurrent neural network (RNN). It is designed to capture long-term dependencies in sequential data, making it highly effective for complex, non-linear time series forecasting where traditional models may struggle.")
        
        st.markdown("**Root Mean Squared Error (RMSE)**")
        st.write("The square root of the Mean Squared Error (MSE). This is the standard evaluation metric as it's in the **same units as the original data** (i.e., population count), representing the typical magnitude of the model's error. **Lower RMSE is better.**")
        
        st.markdown("**Habitat Index**")
        st.write("Measures the proportion of suitable habitats for a country's species that remain intact, relative to a baseline year.")
    
# --- 1. Data Source Selection (Large Card Menu) ---
with st.container(border=True):
    st.subheader("1. Data Source Selection 💾")
    
    col_data_radio, col_data_detail = st.columns([1, 3])
    
    with col_data_radio:
        # 🚨 "Upload My Own CSV" remains first
        data_source_choice = st.radio(
            "Select Data Method:",
            options=["Upload My Own CSV", "Generate Custom Data"],
            index=0,
            key='data_source_radio',
            label_visibility='collapsed'
        )

    df_local = None
    with col_data_detail:
        if data_source_choice == "Upload My Own CSV":
            uploaded_file = st.file_uploader(
                "Upload your annual time-series CSV file", 
                type=['csv'],
                help="The CSV must contain columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`."
            )
            if uploaded_file is not None:
                try:
                    df_local = pd.read_csv(uploaded_file)
                    st.success("✅ Custom CSV file staged. Click 'Start Analysis' below.")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
                    df_local = None
            else:
                st.info("Please upload a CSV file to continue.")
                
        elif data_source_choice == "Generate Custom Data":
            st.markdown("##### Configure Synthetic Dataset Parameters")
            col_gen_1, col_gen_2 = st.columns(2)
            
            with col_gen_1:
                start_year = st.number_input(
                    "Start Year:",
                    min_value=1900, max_value=2023, value=1950, step=1, key='start_year'
                )
                initial_pop = st.number_input(
                    "Initial Population Size:",
                    min_value=1000, max_value=50000, value=10000, step=500, key='initial_pop',
                    help="The population count in the start year."
                )
            with col_gen_2:
                num_years = st.slider(
                    "Historical Data Length (Years):", 
                    min_value=50, max_value=200, value=75, step=10, key='num_years',
                    help="The number of historical data points to generate."
                )
                decline_rate = st.slider(
                    "Annual Population Decline Rate:", 
                    min_value=0.0, max_value=200.0, value=100.0, step=5.0, key='decline_rate',
                    help="The average number of individuals lost per year."
                )

            df_local = generate_custom_data(start_year, num_years, initial_pop, decline_rate)
            st.success("✅ Custom Synthetic Data generated. Click 'Start Analysis' below.")

st.session_state.df = df_local

# --- Start Analysis Button ---
if st.session_state.df is not None:
    # Use a placeholder to ensure the button is visually central and prominent
    placeholder = st.empty()
    if placeholder.button("🚀 Start Analysis", key='start_analysis_button'):
        st.session_state.analysis_started = True
        placeholder.empty() # Remove the button after press
        st.rerun()
else:
    st.markdown("<h4 style='text-align: center; color: #e74c3c;'>Waiting for Data Upload/Generation...</h4>", unsafe_allow_html=True)


# --- Main Application Logic (Runs only after the button is pressed) ---
if st.session_state.analysis_started and st.session_state.df is not None:
    try:
        df = st.session_state.df.copy()
        
        # --- 2. Data Preprocessing and Validation ---
        with st.spinner("Processing and validating data..."):
            required_cols = ['year', 'population', 'temperature', 'rainfall', 'habitat_index']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Data validation failed. Missing column(s): **{', '.join(missing_cols)}**. Please ensure your data includes all columns: {required_cols}")
                st.session_state.analysis_started = False
                st.stop()
                
            # Consistent data processing
            df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
            df.dropna(subset=['year'], inplace=True)
            df.set_index('year', inplace=True)
            df.sort_index(inplace=True)
            
            # 80/20 split for training and testing
            train_size = int(len(df) * 0.8)
            train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
            
            st.success(f"Data validated. Historical range: **{df.index.year.min()}** to **{df.index.year.max()}**. Total data points: **{len(df)}**.")

        # --- 3. Pre-run all models (Distraction Computation) ---
        with st.spinner("🧠 Training & Testing all models (ARIMA, SARIMA, LSTM) for best fit..."):
            # Use fixed, reasonable default hyperparameters for the pre-run
            pre_run_results = run_all_models(train_data, test_data)
            st.session_state.model_results = pre_run_results
            
            # Find the best model
            best_model_name = min(pre_run_results, key=lambda k: pre_run_results[k]['rmse'])
            st.session_state.best_model = best_model_name
            st.success(f"✅ Pre-analysis complete! **{best_model_name}** selected as the best fit (RMSE: {pre_run_results[best_model_name]['rmse']:.2f}).")


        # --- Define Tabs ---
        tab_config, tab_forecast, tab_environ, tab_data = st.tabs([
            "⚙️ Model Configuration", 
            "📈 Forecast & Results", 
            "🌳 Environmental Analysis", 
            "📊 View Data"
        ])

        # --- Model Selection & Configuration (Runs inside the main app logic) ---
        with tab_config:
            st.subheader("2. Select Model & Hyperparameters ⚙️")
            
            # Large, vivid selection menu
            with st.container(border=True):
                st.markdown("#### **Algorithm Selection & Forecast Horizon**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    algorithm_choice = st.selectbox(
                        "**Choose the Forecasting Algorithm:**", 
                        options=["ARIMA", "SARIMA", "LSTM"],
                        index=["ARIMA", "SARIMA", "LSTM"].index(st.session_state.best_model),
                        key='algorithm_choice',
                        help="The best model, based on pre-run RMSE, is selected by default."
                    )
                with col2:
                    future_steps = st.slider("**Forecast Horizon (Years):**", min_value=10, max_value=100, value=st.session_state.future_steps, step=10, key='future_steps_slider')
                    st.session_state.future_steps = future_steps
                
                current_rmse = st.session_state.model_results.get(algorithm_choice, {}).get('rmse', 'N/A')
                st.markdown(f"**Selected Model Pre-run RMSE:** <span style='color:#004d40; font-weight:bold;'>{current_rmse:.2f}</span>", unsafe_allow_html=True)
            
            # --- Interactive Hyperparameter Tuning (Large, prominent fields) ---
            with st.expander(f"🔬 Tune {algorithm_choice} Hyperparameters for Precision Forecast", expanded=True):
                st.markdown("Adjusting these parameters will re-run the final forecast using the **FULL** historical dataset.")
                
                if algorithm_choice == 'ARIMA':
                    default_order = st.session_state.model_results.get('ARIMA', {}).get('order', (3, 1, 1))
                    c1, c2, c3 = st.columns(3)
                    p = c1.number_input("p (AR order - Auto-regressive):", min_value=0, max_value=5, value=default_order[0], step=1, key='arima_p')
                    d = c2.number_input("d (Differencing - Integration):", min_value=0, max_value=3, value=default_order[1], step=1, key='arima_d')
                    q = c3.number_input("q (MA order - Moving Average):", min_value=0, max_value=5, value=default_order[2], step=1, key='arima_q')
                    model_params = {'order': (p, d, q)}
                
                elif algorithm_choice == 'SARIMA':
                    default_order = st.session_state.model_results.get('SARIMA', {}).get('order', (1, 1, 1))
                    default_s_order = st.session_state.model_results.get('SARIMA', {}).get('seasonal_order', (0, 0, 0, 0))
                    
                    st.markdown("##### Non-Seasonal Order (`p, d, q`)")
                    c1, c2, c3 = st.columns(3)
                    p = c1.number_input("p (AR order):", min_value=0, max_value=5, value=default_order[0], step=1, key='sarima_p')
                    d = c2.number_input("d (Differencing):", min_value=0, max_value=3, value=default_order[1], step=1, key='sarima_d')
                    q = c3.number_input("q (MA order):", min_value=0, max_value=5, value=default_order[2], step=1, key='sarima_q')
                    
                    st.markdown("##### Seasonal Order (`P, D, Q, s`)")
                    c4, c5, c6, c7 = st.columns(4)
                    P = c4.number_input("P (Seasonal AR):", min_value=0, max_value=5, value=default_s_order[0], step=1, key='sarima_P')
                    D = c5.number_input("D (Seasonal Diff):", min_value=0, max_value=3, value=default_s_order[1], step=1, key='sarima_D')
                    Q = c6.number_input("Q (Seasonal MA):", min_value=0, max_value=5, value=default_s_order[2], step=1, key='sarima_Q')
                    s = c7.number_input("s (Seasonality Period):", min_value=0, max_value=24, value=default_s_order[3], step=1, help="0 = no seasonality", key='sarima_s')
                    model_params = {'order': (p, d, q), 'seasonal_order': (P, D, Q, s)}
                
                elif algorithm_choice == 'LSTM':
                    default_window = st.session_state.model_results.get('LSTM', {}).get('window_size', 5)
                    default_epochs = st.session_state.model_results.get('LSTM', {}).get('epochs', 50)
                    default_batch = st.session_state.model_results.get('LSTM', {}).get('batch_size', 8)
                    
                    c1, c2, c3 = st.columns(3)
                    window_size = c1.slider("**Window Size** (Past Years Input):", min_value=2, max_value=15, value=default_window, step=1, key='lstm_window_size')
                    epochs = c2.slider("**Epochs** (Training Cycles):", min_value=10, max_value=100, value=default_epochs, step=5, key='lstm_epochs')
                    batch_size = c3.slider("**Batch Size** (Data per Update):", min_value=4, max_value=32, value=default_batch, step=4, key='lstm_batch_size')
                    model_params = {'window_size': window_size, 'epochs': epochs, 'batch_size': batch_size}

        # --- 4. Forecast & Results Tab ---
        with tab_forecast:
            st.subheader("3. Forecast Visualization & Results 📈")

            # Run the final full forecast using the selected model and parameters
            forecast_df, extinction_year, error = get_full_forecast_and_metrics(
                df, st.session_state.future_steps, algorithm_choice, model_params
            )
            
            if error:
                st.error(f"Failed to generate full forecast for {algorithm_choice}: {error}")
                st.stop()

            # --- Plotting ---
            st.markdown("##### Population Forecast")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            plt.style.use('seaborn-v0_8-notebook')
            
            # Historical Data
            ax.plot(df.index.year, df['population'], label='Historical Population (Train)', linewidth=2, color='#004d40')
            ax.plot(test_data.index.year, test_data['population'], label='Actual Test Data', linewidth=2, color='#27ae60') # Green
            
            # Forecast Data
            ax.plot(forecast_df['year'], forecast_df['population'], label=f'{algorithm_choice} Forecast', linestyle='--', linewidth=2.5, color='#e74c3c')
            
            # Add a vertical line for the start of the forecast
            ax.axvline(df.index.year.max(), color='grey', linestyle=':', linewidth=1.5, label=f'Forecast Start ({df.index.year.max()})')

            if extinction_year:
                ax.axvline(extinction_year, color='darkred', linestyle='-', linewidth=2, label=f'Predicted Extinction: {extinction_year}')
                ax.annotate(f'Extinction Year: {extinction_year}', 
                            (extinction_year, max(df['population']) * 0.9), 
                            textcoords="offset points", xytext=(-10,0), ha='right', color='darkred', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1, alpha=0.8))

            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Population Size', fontsize=12)
            ax.set_title(f'Population Time Series & {algorithm_choice} Forecast', fontsize=16, weight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

            # --- Results and Metrics (Vivid Cards) ---
            st.subheader("Model Performance and Outlook")
            col_metric_1, col_metric_2, col_alert = st.columns([1, 1, 2])
            
            with col_metric_1:
                st.metric(
                    label="Best Model Pre-run RMSE", 
                    value=f"{st.session_state.model_results[st.session_state.best_model]['rmse']:.2f}" if isinstance(st.session_state.model_results[st.session_state.best_model]['rmse'], float) else st.session_state.model_results[st.session_state.best_model]['rmse'],
                    delta=f"Based on {st.session_state.best_model}",
                    delta_color="off"
                )
            
            with col_metric_2:
                # RMSE for the currently selected model
                st.metric(
                    label=f"Current {algorithm_choice} RMSE (Test Set)", 
                    value=f"{current_rmse:.2f}" if isinstance(current_rmse, float) else current_rmse,
                    delta=f"{len(test_data)} points",
                    delta_color="off"
                )

            with col_alert:
                if extinction_year:
                    st.error(f"⚠️ **CRITICAL FORECAST:** The model predicts population extinction in the year **{extinction_year}**.", icon="🚨")
                else:
                    st.success(f"✅ **POSITIVE OUTLOOK:** The model did not predict extinction within the next {future_steps} years.", icon="🌳")
                
                st.info("The RMSE metric confirms the model's predictive accuracy on unseen data (the test set).")


        # --- 5. Environmental Analysis Tab (Unchanged Logic) ---
        with tab_environ:
            st.subheader("4. Environmental Analysis 🌳")

            # --- Thriving Conditions (Large Card Menu) ---
            with st.container(border=True):
                st.markdown("##### 🔍 Identify Thriving Conditions")
                
                min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
                
                col_input_1, col_input_2 = st.columns(2)
                with col_input_1:
                    low_thriving_default = max(min_pop, int(max_pop * 0.9)) if max_pop > 100 else min_pop 
                    low_thriving = st.number_input("**Lower Bound of 'Thriving' Population:**", min_value=min_pop, max_value=max_pop, value=low_thriving_default, step=100, key='low_thriving')
                with col_input_2:
                    high_thriving = st.number_input("**Upper Bound of 'Thriving' Population:**", min_value=low_thriving, max_value=max_pop, value=max_pop, step=100, key='high_thriving')

                thriving_years = df[(df['population'] >= low_thriving) & (df['population'] <= high_thriving)]

                st.markdown("##### Optimal Average Environmental Metrics:")
                
                if thriving_years.empty:
                    st.warning("No years found in the historical data matching that thriving population range. Try adjusting the bounds.")
                else:
                    environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
                    
                    col_env_1, col_env_2, col_env_3 = st.columns(3)
                    
                    with col_env_1:
                        st.metric(label="Optimal Temperature", value=f"{environmental_means['temperature']} °C")
                    with col_env_2:
                        st.metric(label="Optimal Rainfall", value=f"{environmental_means['rainfall']} mm")
                    with col_env_3:
                        st.metric(label="Optimal Habitat Index", value=f"{environmental_means['habitat_index']}")
                    
                    st.caption(f"Insight derived from **{len(thriving_years)}** historical data point(s).")

            # --- Environmental Variable Trends ---
            with st.container(border=True):
                st.markdown("##### Environmental Variable Trends Over Time")
                
                fig_env, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
                plt.style.use('seaborn-v0_8-notebook')

                variables = ['temperature', 'rainfall', 'habitat_index']
                titles = ['Annual Temperature Trend (°C)', 'Annual Rainfall Trend (mm)', 'Habitat Index Trend']
                colors = ['#e67e22', '#3498db', '#27ae60']

                for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
                    ax = axes[i]
                    ax.plot(df.index.year, df[var], label=var.replace('_', ' ').title(), linewidth=2.5, color=color)
                    ax.set_title(title, fontsize=14, weight='bold')
                    ax.set_ylabel(var.replace('_', ' ').title(), fontsize=12)
                    ax.legend(loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.6)

                axes[-1].set_xlabel('Year', fontsize=12)
                axes[-1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig_env)
                
        # --- 6. Data View Tab (Unchanged Logic) ---
        with tab_data:
            st.subheader("5. View Raw & Processed Data 📊")
            
            with st.container(border=True):
                st.markdown("#### Full Processed Dataset")
                st.dataframe(df)
            
            with st.container(border=True):
                st.markdown("#### Descriptive Statistics")
                st.dataframe(df.describe())
            

    except Exception as e:
        st.error(f"An unexpected error occurred during the main analysis. Please check your data or model configuration.")
        st.exception(e)
