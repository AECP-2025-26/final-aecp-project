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
    font-size: 96px; 
    font-weight: 900; 
    color: #57CA0F; 
    text-align: center;
    padding: 15px 0;
    border-bottom: 4px solid #85F341; 
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="main-header">AECP: Animal Extinction Calendar Predictor</p>', unsafe_allow_html=True)

# --- Welcome Message ---
st.write("Hi! Since you're already familiar with the project's overview, let's dive into the Data Science aspect of AECP. This application uses an extensive, pre-processed dataset containing up to 250 years of population, habitat, deforestation, and rainfall index data. The libraries Scikit-learn and Statsmodels use this data to predict future population. You'll see this visualized with a polynomial regression curve on the graphs. Critically, the app's interactive component, powered by Streamlit, allows you to modify model hyperparameters. By changing these, you instantly generate new forecast scenarios, letting you see how conservation efforts (like reducing the decline rate by 75%) directly shift the projected extinction date. \n\n Here are the full forms and meanings of a few important terms: ")
st.write("ARIMA: Autoregressive integrated moving average, and SARIMA: Seasonal Autoregressive integrated moving average. The autoregressive (AR) part of ARIMA indicates that the evolving variable of interest is regressed on its prior values. The moving average (MA) part indicates that the regression error is a linear combination of error terms whose values occurred contemporaneously and at various times in the past. The integrated (I) part indicates that the data values have been replaced with the difference between each value and the previous value.")
st.write("LSTM stands for Long Short-Term Memory. It's a type of recurrent neural network designed to capture long-term dependencies in sequential data, making it ideal for tasks such as time series forecasting and analysis.") 
st.write("The Mean Squared Error (MSE) is calculated by taking the average of the squared prediction errors. It serves as the loss function used internally during model training, where squaring the errors helps the learning algorithm focus on reducing large mistakes and outliers.")
st.write("The Root Mean Squared Error (RMSE) is the square root of the MSE, translating the error back into the original units of measurement. This makes RMSE the standard evaluation metric for users, as it clearly represents the average or typical magnitude of the error the model makes, providing an easily understood measure of accuracy.")
st.write("The Habitat Index measures the proportion of suitable habitats for a country's species that remain intact, relative to a baseline set in the year 2001.")

# --- Data Generation Function (Now parameterized) ---
@st.cache_data
def generate_custom_data(start_year, num_years, initial_pop, decline_rate):
    """Generates synthetic time-series data based on user-defined parameters."""
    end_year = start_year + num_years
    years = np.arange(start_year, end_year)
    
    # Population trend (linear decline with noise)
    population_base = initial_pop - decline_rate * (years - start_year)
    population_noise = np.random.normal(0, initial_pop * 0.05, len(years))
    # Ensure population doesn't go below a safe threshold (e.g., 50)
    population = np.maximum(50, population_base + population_noise) 

    # Environmental factors (mock data, correlated with year/decline)
    # Temperature rises slightly
    temperature = 10 + 0.1 * (years - start_year) + np.random.normal(0, 0.5, len(years))
    # Rainfall decreases slightly
    rainfall = 1000 - 5 * (years - start_year) + np.random.normal(0, 50, len(years))
    # Habitat index (declining from 1.0, influenced by decline_rate)
    habitat_index_base = 1.0 - (decline_rate / initial_pop) * 5 * (years - start_year)
    habitat_index = habitat_index_base + np.random.normal(0, 0.05, len(years))
    habitat_index = np.clip(habitat_index, 0.1, 1.0) # Clip between 0.1 and 1.0

    df = pd.DataFrame({
        'year': years,
        'population': population.astype(int),
        'temperature': temperature.round(1),
        'rainfall': rainfall.astype(int),
        'habitat_index': habitat_index.round(2)
    })
    return df

# --- File Selection Logic Starts Here ---
st.subheader("1. Data Source Selection")

data_source_choice = st.radio(
    "How would you like to obtain your data?",
    options=["Generate Custom Data", "Upload My Own CSV"],
    index=0,
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
    st.success("✅ Custom Synthetic Data generated successfully.")

elif data_source_choice == "Upload My Own CSV":
    uploaded_file = st.file_uploader(
        "Upload your annual time-series CSV file", 
        type=['csv'],
        help="The CSV must contain columns: `year`, `population`, `temperature`, `rainfall`, `habitat_index`."
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Custom CSV file loaded successfully!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            df = None
            
# --- Main Analysis Block ---
if df is not None:
    try:
        st.subheader("2. Data Preprocessing and Validation")
        
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

        st.info(f"Historical range: **{df.index.year.min()}** to **{df.index.year.max()}**. Total data points: **{len(df)}**.")
        
        # --- Model Selection and Data Split ---
        st.subheader("3. Model Configuration and Forecasting")
        
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
        window_size = 5 # Default window size for LSTM

        if algorithm_choice == 'LSTM':
            # LSTM requires data normalization (MinMaxScaler) and sequence creation
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

            # Create sequences for training and testing
            X_train, y_train = create_sequences(train_data_scaled, window_size)
            
            X_test, y_test = [], []
            if len(test_data_scaled) > window_size:
                X_test, y_test = create_sequences(test_data_scaled, window_size)
            else:
                # Need enough data for the window size
                st.warning(f"Test data size ({len(test_data_scaled)}) is too small for a window size of {window_size}. Skipping test set evaluation.")
            
            if len(X_train) == 0:
                st.error("Insufficient data points to create LSTM sequences. Try increasing the 'Historical Data Length' parameter.")
                st.stop()
                
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
                # Fit on training data for evaluation
                arima_model = ARIMA(train_data['population'], order=(3,1,1)).fit()
                arima_forecast_test = arima_model.forecast(steps=len(test_data))
                
                mse = mean_squared_error(test_data['population'], arima_forecast_test)
                rmse = np.sqrt(mse)
                
                # Fit on full data for future forecast
                arima_model_full = ARIMA(df['population'], order=(3,1,1)).fit()
                forecast_full = arima_model_full.forecast(steps=future_steps).values

        elif algorithm_choice == 'SARIMA':
            with st.spinner(f"Fitting {algorithm_choice} model..."):
                # Fit on training data for evaluation
                sarima_model = SARIMAX(train_data['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
                sarima_forecast_test = sarima_model.forecast(steps=len(test_data))
                
                mse = mean_squared_error(test_data['population'], sarima_forecast_test)
                rmse = np.sqrt(mse)
                
                # Fit on full data for future forecast
                sarima_model_full = SARIMAX(df['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
                forecast_full = sarima_model_full.forecast(steps=future_steps).values

        # --- Extinction Check ---
        extinction_year = None
        for i, val in enumerate(forecast_full):
            if val <= 0:
                extinction_year = df.index.year.max() + i + 1
                break

        # --- Plotting ---
        st.subheader("4. Forecast Visualization")
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
        st.subheader("5. Environmental Analysis: Identifying Thriving Conditions")
        
        min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
        
        # Use columns for input fields
        col_input_1, col_input_2 = st.columns(2)
        with col_input_1:
            # Adjust default value to a percentage of max_pop to avoid immediate empty set for small data
            low_thriving_default = max(min_pop, int(max_pop * 0.9)) if max_pop > 100 else min_pop 
            low_thriving = st.number_input("Lower Bound of Thriving Population:", min_value=min_pop, max_value=max_pop, value=low_thriving_default, step=100)
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


        # --- NEW SECTION: Independent Variable Trends (Section 6) ---
        st.markdown("---")
        st.subheader("6. Environmental Variable Trends")

        # Create a single figure with three subplots for the environmental variables
        fig_env, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        plt.style.use('seaborn-v0_8-whitegrid')

        variables = ['temperature', 'rainfall', 'habitat_index']
        titles = ['Annual Temperature Trend (Degrees Celsius)', 'Annual Rainfall Trend (mm)', 'Habitat Index Trend (Proportion)']
        colors = ['#e67e22', '#3498db', '#27ae60']

        for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
            ax = axes[i]
            # Plot the variable against the year index
            ax.plot(df.index.year, df[var], label=var.replace('_', ' ').title(), linewidth=3, color=color)
            ax.set_title(title, fontsize=16)
            ax.set_ylabel(var.replace('_', ' ').title(), fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)

        # Set X-axis label only for the bottom plot
        axes[-1].set_xlabel('Year', fontsize=12)
        axes[-1].tick_params(axis='x', rotation=45)

        st.pyplot(fig_env)
        
    except Exception as e:
        # Catch and display general errors clearly
        st.error(f"An unexpected error occurred during model processing. This usually happens if the generated or uploaded data is too small/sparse for the selected model (especially LSTM).")
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
