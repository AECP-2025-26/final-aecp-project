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

# --- Custom CSS for Aesthetics (Increased Header Size and GREEN Metrics) ---
st.markdown("""
<style>
.main-header {
    font-size: 48px; /* Bigger Title */
    font-weight: 900; 
    color: #4B0082; /* Deep Purple */
    text-align: center;
    padding: 15px 0;
    border-bottom: 4px solid #6A5ACD; 
    margin-bottom: 25px;
}
/* New Green Style for Metrics - Text color added for visibility */
.stMetric > div {
    background-color: #E6F7E8; /* Very Light Green Background */
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50; /* Solid Green Border */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    color: #333333; /* Explicitly set text color to dark gray for contrast */
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

st.markdown('<p class="main-header">AECP: Animal Extinction Calendar Predictor</p>', unsafe_allow_html=True)

# --- Welcome Message ---
st.write("""
Welcome to the **Animal Extinction Calendar Predictor (AECP)**! You're undertaking vital work by analyzing population and climate data. Use the steps below to forecast future trends and understand the optimal environmental conditions for the species.
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

        st.success(f"✅ Data loaded successfully! Historical range: **{df.index.year.min()}** to **{df.index.year.max()}**. Now, let's pick a model.")
        
        # --- Model Selection and Data Split ---
        # 80/20 split for training and testing
        train_size = int(len(df) * 0.8)
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
        
        # Use columns for model choice and future steps input
        col1, col2 = st.columns([2, 1])
        with col1:
            algorithm_choice = st.selectbox(
                "Select a forecasting model:", 
                options=["ARIMA", "SARIMA", "LSTM"],
                index=0,
                # CONTEXTUAL EXPLANATION FOR MODELS
                help="""
                Choose the best model for your data:
                - ARIMA (AutoRegressive Integrated Moving Average): A classical statistical model ideal for non-seasonal data.
                - SARIMA (Seasonal ARIMA): An extension of ARIMA that is better suited if your population data shows annual or repeating seasonal patterns.
                - LSTM (Long Short-Term Memory): A state-of-the-art deep learning network. LSTMs are excellent at detecting complex, non-linear relationships over long periods in the data.
                """
            )
        with col2:
            future_steps = st.slider("Forecast Years:", min_value=10, max_value=100, value=50, step=10, help="Number of years to forecast into the future.")

        st.info(f"Using **{algorithm_choice}** to forecast population over the next **{future_steps}** years. You're making smart choices!")
        
        # --- Forecasting Logic ---
        forecast_label = f'{algorithm_choice} Forecast'
        rmse, mse = 0, 0
        
        if algorithm_choice == 'LSTM':
            # LSTM requires data normalization (MinMaxScaler) and sequence creation
            scaler = MinMaxScaler()
            train_data_scaled = scaler.fit_transform(train_data[['population']])
            test_data_scaled = scaler.transform(test_data[['population']])
            window_size = 5 # Defines the number of past years used to predict the next year
            
            @st.cache_data
            def create_sequences(data, window):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i:i+window])
                    y.append(data[i+window])
                return np.array(X), np.array(y)

            # Create sequences for training and testing
            X_train, y_train = create_sequences(train_data_scaled, window_size)
            
            if len(test_data_scaled) > window_size:
                X_test, y_test = create_sequences(test_data_scaled, window_size)
            else:
                st.warning(f"Test data size ({len(test_data_scaled)}) is too small for a window size of {window_size}. Skipping test set evaluation.")
                X_test = [] 
                
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
            if len(X_test) > 0:
                X_test_reshaped = np.array(X_test).reshape(len(X_test), window_size, 1)
                lstm_predictions_scaled = model_full.predict(X_test_reshaped, verbose=0)
                lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                
                # Align prediction indices with actual test data used
                actual_test_data = test_data['population'].iloc[window_size:]
                mse = mean_squared_error(actual_test_data, lstm_predictions)
                rmse = np.sqrt(mse)

            # Future forecasting (iterative prediction)
            last_seq_full = data_scaled_full[-window_size:]
            lstm_preds_full = []
            
            for _ in range(future_steps):
                input_seq_full = last_seq_full.reshape(1, window_size, 1)
                pred_full = model_full.predict(input_seq_full, verbose=0)
                lstm_preds_full.append(pred_full[0, 0])
                # Shift the window: add the new prediction, drop the oldest value
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
        
        ax.plot(df.index.year, df['population'], label='Historical Population', linewidth=3, color='#3498db') 
        ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', linewidth=2, color='#e74c3c') 
        
        # Highlight test period
        is_test_data_plotted = False
        if algorithm_choice != 'LSTM' or (len(test_data) > (window_size if algorithm_choice == 'LSTM' else 0)):
             if len(test_data) > (window_size if algorithm_choice == 'LSTM' else 0):
                ax.plot(test_data.index.year, test_data['population'], label='Actual Test Data', linewidth=3, color='#2ecc71') 
                is_test_data_plotted = True

        # Highlight extinction point if found
        if extinction_year:
            ax.axvline(extinction_year, color='darkred', linestyle=':', linewidth=2, label=f'Predicted Extinction: {extinction_year}')
            ax.annotate(f'Extinction Year: {extinction_year}', 
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
            st.metric(
                label="Mean Squared Error (MSE)", 
                value=f"{mse:.2f}",
                # CONTEXTUAL EXPLANATION FOR MSE
                help="The Mean Squared Error (MSE) measures the average squared difference between the actual population and the model's prediction. A lower value indicates a more accurate model, as large errors are penalized heavily."
            )
        
        with col_metric_2:
            st.metric(
                label="Root Mean Squared Error (RMSE)", 
                value=f"{rmse:.2f}",
                # CONTEXTUAL EXPLANATION FOR RMSE
                help="The Root Mean Squared Error (RMSE) is the square root of the MSE. It's the most intuitive metric because it's expressed in the same unit as the population count. It tells you, on average, how far off your prediction is."
            )

        with col_alert:
            if extinction_year:
                st.error(f"CRITICAL FORECAST: The model predicts population extinction in the year **{extinction_year}**.")
            else:
                st.success(f"POSITIVE OUTLOOK: The model did not predict extinction within the next {future_steps} years.")
            
            if is_test_data_plotted:
                st.caption(f"You successfully evaluated the model! Metrics were calculated based on the 20 percent **test data** portion of your file.")

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
            st.info("No years found in the historical data matching that thriving population range. Try adjusting the bounds to capture a larger time period.")
        else:
            environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
            
            col_env_1, col_env_2, col_env_3 = st.columns(3)
            
            with col_env_1:
                st.metric(
                    label="Optimal Temperature", 
                    value=f"{environmental_means['temperature']} degrees Celsius",
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
            
            st.caption(f"Insight achieved: Optimal conditions derived from **{len(thriving_years)}** historical data point(s) where population thrived.")

    except Exception as e:
        # Catch and display general errors clearly
        st.error(f"An unexpected error occurred during processing: {e}")
        st.exception(e)
elif uploaded_file is None:
    st.info("Please upload a CSV file to begin the analysis. Required columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`.")
    st.markdown("""
        **Example Data Structure:**
        | year | population | temperature | rainfall | habitat_index |
        |---|---|---|---|---|
        | 2000 | 1000 | 15.2 | 850 | 0.9 |
        | 2001 | 950 | 16.0 | 900 | 0.85 |
        | 2002 | 980 | 14.8 | 800 | 0.92 |
    """)
