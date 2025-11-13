# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

# === Streamlit App Title ===
st.title('Population Forecasting and Environmental Analysis')

# === Upload CSV File (Streamlit) ===
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None # Initialize df outside the if block

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # === Basic Cleaning ===
    # Convert 'year' to integer and set as index
    df['year'] = df['year'].astype(int)
    df.set_index('year', inplace=True)

    # Required columns
    required_cols = ['population', 'temperature', 'rainfall', 'habitat_index']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}. Please include all: {required_cols}")
            st.stop() # Stop execution if required column is missing

    # === Split Data into Training and Testing Sets ===
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # === Choose Forecasting Algorithm ===
    algorithm_choice = st.selectbox(
        "Select the forecasting algorithm:",
        ('ARIMA', 'SARIMA', 'LSTM')
    ).upper()

    if algorithm_choice == 'LSTM':
        # === Normalize population for LSTM ===
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data[['population']])
        test_data_scaled = scaler.transform(test_data[['population']]) # Use the same scaler fitted on training data

        window_size = 5

        def create_sequences(data, window):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i+window])
                y.append(data[i+window])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data_scaled, window_size)
        X_test, y_test = create_sequences(test_data_scaled, window_size)

        # === Build and Train LSTM Model ===
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

        # === Evaluate LSTM Model ===
        lstm_predictions_scaled = model.predict(X_test, verbose=0)
        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
        mse = mean_squared_error(test_data['population'].iloc[window_size:], lstm_predictions)
        rmse = np.sqrt(mse)
        forecast_label = 'LSTM Forecast'
        forecast_test = lstm_predictions # Use LSTM test predictions for plotting comparison


        # === Forecast with LSTM (Full Data) for future plot ===
        data_scaled_full = scaler.fit_transform(df[['population']]) # Re-fit scaler on full data
        X_full, y_full = create_sequences(data_scaled_full, window_size)
        model_full = Sequential([
            LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
            Dense(1)
        ])
        model_full.compile(optimizer='adam', loss='mse')
        model_full.fit(X_full, y_full, epochs=50, batch_size=8, verbose=0)

        future_steps = 100
        last_seq_full = data_scaled_full[-window_size:]
        lstm_preds_full = []
        for _ in range(future_steps):
            input_seq_full = last_seq_full.reshape(1, window_size, 1)
            pred_full = model_full.predict(input_seq_full, verbose=0)
            lstm_preds_full.append(pred_full[0, 0])
            last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
        forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1))

        # === Extinction Year Detection (LSTM) ===
        extinction_year = None
        # Corrected line to access the year from the index directly
        last_historical_year = df.index[-1]
        for i, val in enumerate(forecast_full):
            if val[0] <= 0:
                extinction_year = last_historical_year + i + 1
                break

    elif algorithm_choice == 'ARIMA':
        # === ARIMA Forecast ===
        arima_model = ARIMA(train_data['population'], order=(3,1,1)).fit()
        arima_forecast_test = arima_model.forecast(steps=len(test_data)) # Forecast for the test set period

        # === Evaluate ARIMA Model ===
        mse = mean_squared_error(test_data['population'], arima_forecast_test)
        rmse = np.sqrt(mse)
        forecast_label = 'ARIMA Forecast'
        forecast_test = arima_forecast_test # Use ARIMA test predictions for plotting comparison

        # === ARIMA Forecast (Full Data) for future plot ===
        future_steps = 100
        arima_model_full = ARIMA(df['population'], order=(3,1,1)).fit()
        forecast_full = arima_model_full.forecast(steps=future_steps)

        # === Extinction Year Detection (ARIMA) ===
        extinction_year = None
        # Corrected line to access the year from the index directly
        last_historical_year = df.index[-1]
        for i, val in enumerate(forecast_full):
            if val <= 0: # ARIMA forecast is a Series, values are accessed directly
                extinction_year = last_historical_year + i + 1
                break

    elif algorithm_choice == 'SARIMA':
        # === SARIMA Forecast ===
        sarima_model = SARIMAX(train_data['population'], order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
        sarima_forecast_test = sarima_model.forecast(steps=len(test_data)) # Forecast for the test set period

        # === Evaluate SARIMA Model ===
        mse = mean_squared_error(test_data['population'], sarima_forecast_test)
        rmse = np.sqrt(mse)
        forecast_label = 'SARIMA Forecast'
        forecast_test = sarima_forecast_test # Use SARIMA test predictions for plotting comparison

        # === SARIMA Forecast (Full Data) for future plot ===
        future_steps = 100
        sarima_model_full = SARIMAX(df['population'], order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
        forecast_full = sarima_model_full.forecast(steps=future_steps)

        # === Extinction Year Detection (SARIMA) ===
        extinction_year = None
        # Corrected line to access the year from the index directly
        last_historical_year = df.index[-1]
        for i, val in enumerate(forecast_full):
            if val <= 0: # SARIMA forecast is a Series, values are accessed directly
                extinction_year = last_historical_year + i + 1
                break

    else:
        st.error("Invalid algorithm choice. Please select ARIMA, SARIMA, or LSTM.")
        st.stop()


    # === Plot Forecast ===
    # Corrected line to access the year from the index directly
    years_future = np.arange(df.index[-1] + 1, df.index[-1] + 1 + future_steps)


    plt.figure(figsize=(14, 6))
    # Corrected line to access the year from the index directly
    plt.plot(df.index, df['population'], label='Historical', linewidth=2)
    # Plot the chosen model's full forecast
    plt.plot(years_future, forecast_full, label=forecast_label, linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Population Forecast ({algorithm_choice})')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


    # === Display Evaluation Metrics ===
    st.subheader("📈 Model Evaluation Metrics:")
    st.write(f"{algorithm_choice} - MSE: {mse:.2f}, RMSE: {rmse:.2f}")


    # === Show Predicted Extinction Year ===
    if extinction_year:
        st.warning(f"⚠️ {algorithm_choice} predicts extinction in the year: {extinction_year}")
    else:
        st.success(f"✅ {algorithm_choice} did not predict extinction within {future_steps} years.")

    # === Thriving Population Range Input ===
    st.subheader("Thriving Population Range")
    col1, col2 = st.columns(2)
    with col1:
        low_thriving = st.number_input("Enter lower bound of thriving population range:", value=100, step=10)
    with col2:
        high_thriving = st.number_input("Enter upper bound of thriving population range:", value=1000, step=10)

    # === Filter and Report Environmental Vectors ===
    thriving_years = df[(df['population'] >= low_thriving) & (df['population'] <= high_thriving)]

    if thriving_years.empty:
        st.write("No years found in that range.")
    else:
        environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
        st.subheader("🧠 Best Environmental and Climate Conditions for Survival:")
        st.write(f"Temperature: {environmental_means['temperature']} °C")
        st.write(f"Rainfall: {environmental_means['rainfall']} mm")
        st.write(f"Habitat Index: {environmental_means['habitat_index']}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
