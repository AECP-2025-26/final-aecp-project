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

# --- Custom CSS for Aesthetics (Increased Header Size) ---
st.markdown("""
<style>
.main-header {
    font-size: 48px; /* Much Bigger */
    font-weight: 900; /* Bolder */
    color: #4B0082; /* Deep Purple */
    text-align: center;
    padding: 15px 0;
    border-bottom: 4px solid #6A5ACD; /* Thicker, deeper line */
    margin-bottom: 25px;
}
.stMetric > div {
    background-color: #F8F4FF;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #6A5ACD; /* Slate Blue */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
}
/* Style for the main plot container */
.stPlot > div {
    border-radius: 12px;
    padding: 20px;
    background-color: #FFFFFF;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Population Time Series Forecasting & Environmental Analysis</p>', unsafe_allow_html=True)

# --- Welcome and Introduction ---
st.write("""
Welcome to the Time Series Analyst! This application allows you to forecast population trends based on historical data and analyze the environmental conditions (temperature, rainfall, habitat index) that correlate with periods of high population.
""")

with st.expander("📚 **Learn the Science: Technical Terms Explained**"):
    st.markdown("""
    ### Time Series Forecasting Models
    Time series data is indexed by time (like annual data). Forecasting predicts future values based on past observations.

    * **ARIMA (AutoRegressive Integrated Moving Average):** A statistical model that looks at dependencies between an observation and a number of lagged observations (AR component), the difference between raw observations to make the series stationary (I component), and the dependency between an observation and a residual error from a moving average model (MA component).
    * **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** An extension of ARIMA that explicitly supports the modeling of time series data with a seasonal component (like monthly or quarterly data), although here we treat the annual data simply.
    * **LSTM (Long Short-Term Memory):** A specialized type of Recurrent Neural Network (RNN) in deep learning. LSTMs are designed to remember sequential information over long periods, making them highly effective for complex, non-linear time series patterns.

    ### Data Processing & Evaluation Metrics
    * **MinMaxScaler:** A data transformation technique that scales features to a fixed range, typically 0 to 1. This is crucial for Neural Networks like LSTM to prevent features with larger values from dominating the learning process.
    * **MSE (Mean Squared Error):** A common metric to measure the quality of a forecast. It calculates the average of the squares of the errors (the difference between the actual value and the predicted value). Lower MSE is better.
    $$MSE = \frac{1}{n}\sum_{t=1}^{n} (A_t - F_t)^2$$
    * **RMSE (Root Mean Squared Error):** The square root of the MSE. It has the same units as the forecasted quantity (e.g., population count), making it easier to interpret than MSE. Lower RMSE is better.
    $$RMSE = \sqrt{MSE}$$
    """)

# --- File Uploader Starts Here ---
uploaded_file = st.file_uploader("Upload your annual time-series CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- Data Preprocessing ---
        df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
        df.dropna(subset=['year'], inplace=True)
        df.set_index('year', inplace=True)
        
        required_cols = ['population', 'temperature', 'rainfall', 'habitat_index']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Data validation failed. Missing column(s): **{', '.join(missing_cols)}**. Please ensure your CSV includes all columns: {required_cols}")
            st.stop()
        
        # Ensure data is sorted by index (year)
        df.sort_index(inplace=True)

        st.success(f"✅ Data loaded successfully. Historical range: **{df.index.year.min()}** to **{df.index.year.max()}**")
        
        # --- Model Selection and Data Split ---
        # 80/20 split for training and testing
        train_size = int(len(df) * 0.8)
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
        
        # Use columns for model choice and future steps input
        col1, col2 = st.columns([2, 1])
        with col1:
            algorithm_choice = st.selectbox(
                "Select a forecasting model:", 
                ["ARIMA", "SARIMA", "LSTM"],
                help="Choose the algorithm to model the population trend."
            )
        with col2:
            future_steps = st.slider("Forecast Years:", min_value=10, max_value=100, value=50, step=10, help="Number of years to forecast into the future.")

        st.info(f"Using **{algorithm_choice}** to forecast population over the next **{future_steps}** years.")
        
        # --- Forecasting Logic ---
        forecast_label = f'{algorithm_choice} Forecast'
        rmse, mse = 0, 0
        
        if algorithm_choice == 'LSTM':
            # Keras/TensorFlow model requires sequence creation and scaling
            scaler = MinMaxScaler()
            train_data_scaled = scaler.fit_transform(train_data[['population']])
            test_data_scaled = scaler.transform(test_data[['population']])
            window_size = 5
            
            @st.cache_data
            def create_sequences(data, window):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i:i+window])
                    y.append(data[i+window])
                return np.array(X), np.array(y)

            # Create sequences for training and testing
            X_train, y_train = create_sequences(train_data_scaled, window_size)
            
            # Check if we have enough data for test sequence creation
            if len(test_data_scaled) > window_size:
                X_test, y_test = create_sequences(test_data_scaled, window_size)
            else:
                # If test set is too small, skip test evaluation for visual clarity, train directly on full data
                st.warning(f"Test data size ({len(test_data_scaled)}) is too small for a window size of {window_size}. Skipping test set evaluation.")
                X_test = X_train # Dummy assignment to satisfy later block
                
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

            # Build and train the model on the full data for future prediction
            data_scaled_full = scaler.fit_transform(df[['population']])
            X_full, y_full = create_sequences(data_scaled_full, window_size)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
            
            model_full = Sequential([
                LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
                Dense(1)
            ])
            model_full.compile(optimizer='adam', loss='mse')
            
            with st.spinner(f"Training {algorithm_choice} model..."):
                model_full.fit(X_full, y_full, epochs=50, batch_size=8, verbose=0)
            
            # Prediction on test set for metric calculation
            if 'X_test' in locals() and len(X_test) > 0:
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                lstm_predictions_scaled = model_full.predict(X_test_reshaped, verbose=0)
                lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                
                # Align prediction indices with actual test data used
                actual_test_data = test_data['population'].iloc[window_size:]
                mse = mean_squared_error(actual_test_data, lstm_predictions)
                rmse = np.sqrt(mse)

            # Future forecasting
            last_seq_full = data_scaled_full[-window_size:]
            lstm_preds_full = []
            
            for _ in range(future_steps):
                input_seq_full = last_seq_full.reshape(1, window_size, 1)
                pred_full = model_full.predict(input_seq_full, verbose=0)
                lstm_preds_full.append(pred_full[0, 0])
                last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
                
            forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1)).flatten()

        elif algorithm_choice == 'ARIMA':
            with st.spinner(f"Fitting {algorithm_choice} model..."):
                arima_model = ARIMA(train_data['population'], order=(3,1,1)).fit()
                arima_forecast_test = arima_model.forecast(steps=len(test_data))
                
                mse = mean_squared_error(test_data['population'], arima_forecast_test)
                rmse = np.sqrt(mse)
                
                arima_model_full = ARIMA(df['population'], order=(3,1,1)).fit()
                forecast_full = arima_model_full.forecast(steps=future_steps).values

        elif algorithm_choice == 'SARIMA':
            with st.spinner(f"Fitting {algorithm_choice} model..."):
                sarima_model = SARIMAX(train_data['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
                sarima_forecast_test = sarima_model.forecast(steps=len(test_data))
                
                mse = mean_squared_error(test_data['population'], sarima_forecast_test)
                rmse = np.sqrt(mse)
                
                sarima_model_full = SARIMAX(df['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
                forecast_full = sarima_model_full.forecast(steps=future_steps).values

        # --- Extinction Check ---
        extinction_year = None
        for i, val in enumerate(forecast_full):
            if val <= 0:
                extinction_year = df.index.year.max() + i + 1
                break

        # --- Plotting ---
        years_future = np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set a modern style for the plot
        plt.style.use('seaborn-v0_8-whitegrid')
        
        ax.plot(df.index.year, df['population'], label='Historical Population', linewidth=3, color='#3498db') # Blue
        ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', linewidth=2, color='#e74c3c') # Red/Orange
        
        # Highlight test period
        if algorithm_choice != 'LSTM' or ('X_test' in locals() and len(X_test) > 0):
             # Ensure test data plot only runs if test data is sufficiently large
             if len(test_data) > (window_size if algorithm_choice == 'LSTM' else 0):
                ax.plot(test_data.index.year, test_data['population'], label='Actual Test Data', linewidth=3, color='#2ecc71') # Green

        # Highlight extinction point if found
        if extinction_year:
            ax.axvline(extinction_year, color='darkred', linestyle=':', linewidth=2, label=f'Predicted Extinction: {extinction_year}')
            ax.annotate(f'Extinction Year:\n{extinction_year}', 
                        (extinction_year, max(df['population']) * 0.9), 
                        textcoords="offset points", xytext=(-10,0), ha='right', color='darkred', fontsize=10)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Population Size', fontsize=12)
        ax.set_title(f'Population Time Series & {algorithm_choice} Forecast', fontsize=16)
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)

        # --- Results and Metrics ---
        st.markdown("---")
        st.subheader("Model Performance and Outlook")

        col_metric_1, col_metric_2, col_alert = st.columns([1, 1, 2])
        
        with col_metric_1:
            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
        
        with col_metric_2:
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")

        with col_alert:
            if extinction_year:
                st.error(f"⚠️ **CRITICAL FORECAST:** The model predicts population extinction in the year **{extinction_year}**.")
            else:
                st.success(f"✅ **Positive Outlook:** The model did not predict extinction within the next {future_steps} years.")

        # --- Environmental Analysis Section ---
        st.markdown("---")
        st.subheader("Environmental Analysis: Identifying Thriving Conditions")
        
        min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
        
        # Use columns for input fields
        col_input_1, col_input_2 = st.columns(2)
        with col_input_1:
            low_thriving = st.number_input("Lower Bound of Thriving Population:", min_value=min_pop, max_value=max_pop, value=int(max_pop * 0.9), step=100)
        with col_input_2:
            high_thriving = st.number_input("Upper Bound of Thriving Population:", min_value=low_thriving, max_value=max_pop, value=max_pop, step=100)

        # Filter data based on user input
        thriving_years = df[(df['population'] >= low_thriving) & (df['population'] <= high_thriving)]

        st.markdown("### Optimal Average Environmental Metrics:")
        
        if thriving_years.empty:
            st.info("No years found in the historical data matching that thriving population range.")
        else:
            environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
            
            col_env_1, col_env_2, col_env_3 = st.columns(3)
            
            with col_env_1:
                st.metric(
                    label="Optimal Temperature", 
                    value=f"{environmental_means['temperature']} °C",
                    delta="Average during thriving years"
                )
            with col_env_2:
                st.metric(
                    label="Optimal Rainfall", 
                    value=f"{environmental_means['rainfall']} mm",
                    delta="Average during thriving years"
                )
            with col_env_3:
                st.metric(
                    label="Optimal Habitat Index", 
                    value=f"{environmental_means['habitat_index']}",
                    delta="Average during thriving years"
                )
            
            st.caption(f"Based on {len(thriving_years)} historical data point(s) where population was between {low_thriving} and {high_thriving}.")

    except Exception as e:
        # Catch and display general errors clearly
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)
elif uploaded_file is None:
    st.info("⬆️ Please upload a CSV file to begin the analysis. Required columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`.")
    st.markdown("""
        **Example Data Structure:**
        | year | population | temperature | rainfall | habitat_index |
        |---|---|---|---|---|
        | 2000 | 1000 | 15.2 | 850 | 0.9 |
        | 2001 | 950 | 16.0 | 900 | 0.85 |
        | 2002 | 980 | 14.8 | 800 | 0.92 |
    """)
