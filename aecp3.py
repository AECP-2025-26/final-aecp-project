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

warnings.filterwarnings('ignore')

# --- Page Configuration (MUST be the first st command) ---
st.set_page_config(
    layout="wide",
    page_title="AECP: Animal Extinction Calendar Predictor",
    page_icon="🐘"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 48px; /* Reduced from 96px */
        font-weight: 700;
        color: #004d40; /* A deep, professional teal */
        text-align: center;
        padding: 20px 0 10px 0;
        border-bottom: 4px solid #00796b; /* A matching accent color */
        margin-bottom: 15px;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 24px;
        font-weight: 300;
        color: #333;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Make Streamlit's native containers look like cards */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        background-color: #ffffff; /* Explicitly set background */
    }
    
    /* Style the tabs */
    [data-testid="stTabs"] {
        margin-top: 20px;
    }
    [data-testid="stTab"] {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 15px;
    }
    
    /* Style metrics */
    [data-testid="stMetric"] {
        background-color: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        color: #333333; /* <-- FIX: Set a default dark text color */
    }

    /* FIX: Explicitly set colors for metric components */
    [data-testid="stMetricLabel"] {
        color: #555555 !important; /* Dark gray for label */
    }
    
    [data-testid="stMetricValue"] {
        color: #004d40 !important; /* Main theme color for the value */
    }
    
    [data-testid="stMetricDelta"] {
        color: #555555 !important; /* Dark gray for delta */
    }
    
    /* Custom expander header */
    .st-expander > summary {
        font-size: 1.1rem;
        font-weight: 600;
        color: #004d40;
    }

</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="main-header">AECP: Animal Extinction Calendar Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">An interactive tool for forecasting population trends and extinction timelines.</p>', unsafe_allow_html=True)

# --- Introduction & Definitions ---
with st.container(border=True):
    st.write("""
    Hi! This application uses an extensive, pre-processed dataset containing up to 250 years of data. 
    The libraries Scikit-learn, Statsmodels, and TensorFlow use this data to predict future population. 
    
    **Get started:**
    1.  Select a data source below (generate a synthetic dataset or upload your own).
    2.  Once data is loaded, new tabs will appear.
    3.  Go to the **⚙️ Model Configuration** tab to select a model and tune its hyperparameters.
    4.  View the results in the **📈 Forecast & Results** and **🌳 Environmental Analysis** tabs.
    """)
    
    with st.expander("Learn more about the key terms and models"):
        st.write("Here are the full forms and meanings of a few important terms:")
        st.markdown("**ARIMA & SARIMA**")
        st.write("ARIMA (Autoregressive Integrated Moving Average) and SARIMA (Seasonal ARIMA). The autoregressive (AR) part indicates that the variable is regressed on its own prior values. The moving average (MA) part indicates that the regression error is a linear combination of past error terms. The integrated (I) part indicates data values have been differenced to become stationary.")
        
        st.markdown("**LSTM (Long Short-Term Memory)**")
        st.write("A type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data, making it ideal for complex time series forecasting.")
        
        st.markdown("**Mean Squared Error (MSE)**")
        st.write("Calculated by taking the average of the squared prediction errors. It serves as the loss function during model training, heavily penalizing large errors.")
        
        st.markdown("**Root Mean Squared Error (RMSE)**")
        st.write("The square root of the MSE. This is the standard evaluation metric as it's in the same units as the original data (i.e., population count), representing the typical magnitude of the model's error.")
        
        st.markdown("**Habitat Index**")
        st.write("Measures the proportion of suitable habitats for a country's species that remain intact, relative to a baseline set in the year 2001.")

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

# --- 1. Data Source Selection ---
with st.container(border=True):
    st.subheader("1. Data Source Selection")
    data_source_choice = st.radio(
        "How would you like to obtain your data?",
        options=["Generate Custom Data", "Upload My Own CSV"],
        index=0,
        horizontal=True,
        help="Generate a synthetic dataset based on parameters, or upload your own time-series data."
    )

    df = None

    if data_source_choice == "Generate Custom Data":
        with st.expander("Configure Generated Dataset Parameters", expanded=True):
            col_gen_1, col_gen_2 = st.columns(2)
            
            with col_gen_1:
                start_year = st.number_input(
                    "Historical Data Start Year:",
                    min_value=1900, max_value=2023, value=1950, step=1
                )
                initial_pop = st.number_input(
                    "Initial Population Size:",
                    min_value=1000, max_value=50000, value=10000, step=500,
                    help="The population count in the start year."
                )
                
            with col_gen_2:
                num_years = st.slider(
                    "Historical Data Length (Years):", 
                    min_value=50, max_value=200, value=75, step=10,
                    help="The number of historical data points to generate."
                )
                decline_rate = st.slider(
                    "Annual Population Decline Rate:", 
                    min_value=0.0, max_value=200.0, value=100.0, step=5.0,
                    help="The average number of individuals lost per year (Higher = steeper decline, faster extinction)."
                )

        df = generate_custom_data(start_year, num_years, initial_pop, decline_rate)
        st.success("✅ Custom Synthetic Data generated successfully. See tabs below for analysis.")

    elif data_source_choice == "Upload My Own CSV":
        uploaded_file = st.file_uploader(
            "Upload your annual time-series CSV file", 
            type=['csv'],
            help="The CSV must contain columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`."
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Custom CSV file loaded successfully! See tabs below for analysis.")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                df = None

# --- Main Application Logic (Runs only if data is loaded) ---
if df is not None:
    try:
        # --- 2. Data Preprocessing and Validation ---
        # This step is crucial and must happen before tabs are used.
        with st.spinner("Processing and validating data..."):
            df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
            df.dropna(subset=['year'], inplace=True)
            df.set_index('year', inplace=True)
            
            required_cols = ['population', 'temperature', 'rainfall', 'habitat_index']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Data validation failed. Missing column(s): **{', '.join(missing_cols)}**. Please ensure your CSV includes all columns: {required_cols}")
                st.stop()
            
            df.sort_index(inplace=True)
            
            # 80/20 split for training and testing
            train_size = int(len(df) * 0.8)
            train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
            
            st.info(f"Historical range: **{df.index.year.min()}** to **{df.index.year.max()}**. Total data points: **{len(df)}**. (Training: {len(train_data)}, Test: {len(test_data)})")

        # --- Define Tabs ---
        tab_config, tab_forecast, tab_environ, tab_data = st.tabs([
            "⚙️ Model Configuration", 
            "📈 Forecast & Results", 
            "🌳 Environmental Analysis", 
            "📊 View Data"
        ])

        # --- 3. Model Configuration Tab ---
        with tab_config:
            st.subheader("3. Model Configuration & Hyperparameters")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                algorithm_choice = st.selectbox(
                    "Select a forecasting model:", 
                    options=["ARIMA", "SARIMA", "LSTM"],
                    index=0,
                    help="""
                    - **ARIMA**: Best for non-seasonal data with clear trends.
                    - **SARIMA**: An extension of ARIMA for data with seasonal patterns.
                    - **LSTM**: A neural network good at finding complex, non-linear patterns.
                    """
                )
            with col2:
                future_steps = st.slider("Forecast Years:", min_value=10, max_value=100, value=50, step=10, help="Number of years to forecast into the future.")

            st.info(f"Using **{algorithm_choice}** to forecast population over the next **{future_steps}** years.")
            
            # --- Interactive Hyperparameter Tuning ---
            with st.expander(f"Tune {algorithm_choice} Hyperparameters"):
                if algorithm_choice == 'ARIMA':
                    st.write("ARIMA `order=(p, d, q)`")
                    c1, c2, c3 = st.columns(3)
                    p = c1.number_input("p (AR order):", min_value=0, max_value=5, value=3, step=1)
                    d = c2.number_input("d (Differencing):", min_value=0, max_value=3, value=1, step=1)
                    q = c3.number_input("q (MA order):", min_value=0, max_value=5, value=1, step=1)
                    arima_order = (p, d, q)
                
                elif algorithm_choice == 'SARIMA':
                    st.write("SARIMA `order=(p, d, q)`")
                    c1, c2, c3 = st.columns(3)
                    p = c1.number_input("p (AR order):", min_value=0, max_value=5, value=1, step=1)
                    d = c2.number_input("d (Differencing):", min_value=0, max_value=3, value=1, step=1)
                    q = c3.number_input("q (MA order):", min_value=0, max_value=5, value=1, step=1)
                    sarima_order = (p, d, q)
                    
                    st.write("SARIMA `seasonal_order=(P, D, Q, s)`")
                    c4, c5, c6, c7 = st.columns(4)
                    P = c4.number_input("P (Seasonal AR):", min_value=0, max_value=5, value=0, step=1)
                    D = c5.number_input("D (Seasonal Diff):", min_value=0, max_value=3, value=0, step=1)
                    Q = c6.number_input("Q (Seasonal MA):", min_value=0, max_value=5, value=0, step=1)
                    s = c7.number_input("s (Seasonality):", min_value=0, max_value=24, value=0, step=1, help="0 = no seasonality")
                    sarima_seasonal_order = (P, D, Q, s)
                
                elif algorithm_choice == 'LSTM':
                    c1, c2, c3 = st.columns(3)
                    window_size = c1.slider("Window Size:", min_value=2, max_value=15, value=5, step=1, help="Number of past years to use for predicting the next year.")
                    epochs = c2.slider("Epochs:", min_value=10, max_value=100, value=50, step=5, help="Number of training cycles.")
                    batch_size = c3.slider("Batch Size:", min_value=4, max_value=32, value=8, step=4)


        # --- 4. Forecast & Results Tab ---
        with tab_forecast:
            st.subheader("4. Forecast Visualization & Results")

            forecast_label = f'{algorithm_choice} Forecast'
            rmse, mse = 0, 0
            
            try:
                if algorithm_choice == 'LSTM':
                    scaler = MinMaxScaler()
                    train_data_scaled = scaler.fit_transform(train_data[['population']])
                    test_data_scaled = scaler.transform(test_data[['population']])
                    
                    @st.cache_data
                    def create_sequences(data, window):
                        X, y = [], []
                        for i in range(len(data) - window):
                            X.append(data[i:i+window])
                            y.append(data[i+window])
                        return np.array(X), np.array(y)

                    X_train, y_train = create_sequences(train_data_scaled, window_size)
                    X_test, y_test = [], []
                    
                    if len(test_data_scaled) > window_size:
                        X_test, y_test = create_sequences(test_data_scaled, window_size)
                    else:
                        st.warning(f"Test data size ({len(test_data_scaled)}) is too small for a window size of {window_size}. Skipping test set evaluation.")
                    
                    if len(X_train) == 0:
                        st.error("Insufficient data to create LSTM sequences. Try increasing 'Historical Data Length' or decreasing 'Window Size'.")
                        st.stop()
                        
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

                    data_scaled_full = scaler.fit_transform(df[['population']])
                    X_full, y_full = create_sequences(data_scaled_full, window_size)
                    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                    
                    model_full = Sequential([
                        LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
                        Dense(1)
                    ])
                    model_full.compile(optimizer='adam', loss='mse')
                    
                    with st.spinner(f"Training {algorithm_choice} model for {epochs} epochs..."):
                        model_full.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    if len(X_test) > 0:
                        X_test_reshaped = np.array(X_test).reshape(len(X_test), window_size, 1)
                        lstm_predictions_scaled = model_full.predict(X_test_reshaped, verbose=0)
                        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                        
                        actual_test_data = test_data['population'].iloc[window_size:]
                        mse = mean_squared_error(actual_test_data, lstm_predictions)
                        rmse = np.sqrt(mse)

                    last_seq_full = data_scaled_full[-window_size:]
                    lstm_preds_full = []
                    
                    for _ in range(future_steps):
                        input_seq_full = last_seq_full.reshape(1, window_size, 1)
                        pred_full = model_full.predict(input_seq_full, verbose=0)
                        lstm_preds_full.append(pred_full[0, 0])
                        last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
                        
                    forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1)).flatten()

                elif algorithm_choice == 'ARIMA':
                    with st.spinner(f"Fitting {algorithm_choice}{arima_order} model..."):
                        arima_model = ARIMA(train_data['population'], order=arima_order).fit()
                        arima_forecast_test = arima_model.forecast(steps=len(test_data))
                        
                        mse = mean_squared_error(test_data['population'], arima_forecast_test)
                        rmse = np.sqrt(mse)
                        
                        arima_model_full = ARIMA(df['population'], order=arima_order).fit()
                        forecast_full = arima_model_full.forecast(steps=future_steps).values

                elif algorithm_choice == 'SARIMA':
                    with st.spinner(f"Fitting {algorithm_choice}{sarima_order}{sarima_seasonal_order} model..."):
                        sarima_model = SARIMAX(train_data['population'], order=sarima_order, seasonal_order=sarima_seasonal_order).fit(disp=False)
                        sarima_forecast_test = sarima_model.forecast(steps=len(test_data))
                        
                        mse = mean_squared_error(test_data['population'], sarima_forecast_test)
                        rmse = np.sqrt(mse)
                        
                        sarima_model_full = SARIMAX(df['population'], order=sarima_order, seasonal_order=sarima_seasonal_order).fit(disp=False)
                        forecast_full = sarima_model_full.forecast(steps=future_steps).values

                # --- Extinction Check ---
                extinction_year = None
                for i, val in enumerate(forecast_full):
                    if val <= 0:
                        extinction_year = df.index.year.max() + i + 1
                        break

                # --- Plotting ---
                st.subheader("Population Forecast Visualization")
                years_future = np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps)
                
                fig, ax = plt.subplots(figsize=(14, 7))
                plt.style.use('seaborn-v0_8-notebook')
                
                # Plot historical, test, and forecast data
                ax.plot(df.index.year, df['population'], label='Historical Population', linewidth=2, color='#004d40') # Deep teal
                ax.plot(test_data.index.year, test_data['population'], label='Actual Test Data', linewidth=2, color='#27ae60') # Green
                ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', linewidth=2.5, color='#e74c3c') # Red
                
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

                # --- Results and Metrics ---
                st.subheader("Model Performance and Outlook")
                min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
                col_metric_1, col_metric_2, col_alert = st.columns([1, 1, 2])
                
                acc = (mse / min_pop) if min_pop > 0 else 0
                
                with col_metric_1:
                    st.metric(
                        label="Scaled Loss Factor", 
                        value=f"{acc:.2f}",
                        help="An internal metric (MSE / min_pop) used to gauge error relative to the population's lowest point. Lower is better."
                    )
                
                with col_metric_2:
                    st.metric(
                        label="Root Mean Squared Error (RMSE)", 
                        value=f"{rmse:.2f}",
                        help="The average error of the model in population units. A value of 50 means the model is, on average, off by 50 individuals. Lower is better."
                    )

                with col_alert:
                    if extinction_year:
                        st.error(f"**CRITICAL FORECAST:** The model predicts population extinction in the year **{extinction_year}**.")
                    else:
                        st.success(f"**POSITIVE OUTLOOK:** The model did not predict extinction within the next {future_steps} years.")
                    
                    st.caption(f"Metrics were calculated by comparing the model's predictions against the {len(test_data)} data points in the test set.")

            except Exception as e:
                st.error(f"An unexpected error occurred during model processing. This can happen if the data is too small/sparse for the selected model or hyperparameters (especially LSTM).")
                st.exception(e)


        # --- 5. Environmental Analysis Tab ---
        with tab_environ:
            st.subheader("5. Environmental Analysis")

            # --- Thriving Conditions ---
            with st.container(border=True):
                st.subheader("Identify Thriving Conditions")
                
                min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
                
                col_input_1, col_input_2 = st.columns(2)
                with col_input_1:
                    low_thriving_default = max(min_pop, int(max_pop * 0.9)) if max_pop > 100 else min_pop 
                    low_thriving = st.number_input("Lower Bound of 'Thriving' Population:", min_value=min_pop, max_value=max_pop, value=low_thriving_default, step=100)
                with col_input_2:
                    high_thriving = st.number_input("Upper Bound of 'Thriving' Population:", min_value=low_thriving, max_value=max_pop, value=max_pop, step=100)

                thriving_years = df[(df['population'] >= low_thriving) & (df['population'] <= high_thriving)]

                st.markdown("##### Optimal Average Environmental Metrics:")
                
                if thriving_years.empty:
                    st.info("No years found in the historical data matching that thriving population range. Try adjusting the bounds.")
                else:
                    environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
                    
                    col_env_1, col_env_2, col_env_3 = st.columns(3)
                    
                    with col_env_1:
                        st.metric(
                            label="Optimal Temperature", 
                            value=f"{environmental_means['temperature']} °C",
                            delta="Avg. during thriving years"
                        )
                    with col_env_2:
                        st.metric(
                            label="Optimal Rainfall", 
                            value=f"{environmental_means['rainfall']} mm",
                            delta="Avg. during thriving years"
                        )
                    with col_env_3:
                        st.metric(
                            label="Optimal Habitat Index", 
                            value=f"{environmental_means['habitat_index']}",
                            delta="Avg. during thriving years"
                        )
                    
                    st.caption(f"Insight derived from **{len(thriving_years)}** historical data point(s) where population was between {low_thriving} and {high_thriving}.")

            # --- Environmental Variable Trends ---
            with st.container(border=True):
                st.subheader("6. Environmental Variable Trends")
                
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
        
        # --- 6. Data View Tab ---
        with tab_data:
            st.subheader("7. View Raw & Processed Data")
            
            st.markdown("#### Full Processed Dataset")
            st.dataframe(df)
            
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe())
            

    except Exception as e:
        # Catch-all for any other errors
        st.error(f"An unexpected error occurred. Please check your data or model configuration.")
        st.exception(e)
        
# --- Initial Message if no data is loaded yet ---
if df is None and data_source_choice == "Upload My Own CSV":
    st.info("Please upload a CSV file to begin the analysis. Required columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`.")
    st.markdown("""
        **Example Data Structure:**
        | year | population | temperature | rainfall | habitat_index |
        |---|---|---|---|---|
        | 2000 | 1000 | 15.2 | 850 | 0.9 |
        | 2001 | 950 | 16.0 | 900 | 0.85 |
        | 2002 | 980 | 14.8 | 800 | 0.92 |
    """)
