import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

FORECAST_MODELS = {
    "Agriculture": "SARIMA",
    "Mining & Quarrying": "Exponential Smoothing",
    "Manufacturing": "VAR",
    "Electricity & Water Supply": "SARIMAX",
    "Construction": "Prophet",
    "Wholesale And Retail Trade": "SARIMA",
    "Accommodation And Food Service Activities": "SARIMA",
    "Transportation And Storage": "VAR",
    "Information And Communication": "Holt-Winters",
    "Financial And Insurance": "ARIMA-Exog",
    "Other Services": "SARIMA",
    "Fisim": "ARIMA",
    "All Industries At Basic Prices": "VAR",
    "Taxes On Product": "ARIMA"
}

def prepare_ts_data(df, sector):
    ts_data = df[['Date', sector]].set_index('Date')
    ts_data = ts_data[sector].dropna()
    return ts_data

def forecast_sector(df, sector, model_type, forecast_periods=4):
    ts_data = prepare_ts_data(df, sector)
    
    if len(ts_data) < 8:  # Minimum data points for meaningful forecasting
        return None, None, f"Not enough data for {sector} to perform forecasting."

    try:
        # Prepare future dates
        last_date = ts_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(1), periods=forecast_periods, freq='QS')
        
        if model_type == "SARIMA":
            # Seasonal ARIMA
            model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            fit_model = model.fit(disp=False)
            forecast = fit_model.forecast(steps=forecast_periods)
            forecast.index = future_dates
            return ts_data, forecast, None

        elif model_type == "Exponential Smoothing":
            # Simple Exponential Smoothing
            model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=forecast_periods)
            forecast.index = future_dates
            return ts_data, forecast, None

        elif model_type == "Holt-Winters":
            # Holt-Winters Exponential Smoothing
            model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=4)
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=forecast_periods)
            forecast.index = future_dates
            return ts_data, forecast, None

        elif model_type == "VAR":
            # Vector Auto Regression
            # Use related sectors as additional variables
            related_sectors = ['Gdp At Market Prices', sector]
            var_data = df[['Date'] + related_sectors].set_index('Date').dropna()
            if len(var_data) < 8:
                return None, None, f"Not enough data for VAR on {sector}."
            model = VAR(var_data)
            fit_model = model.fit(maxlags=4)
            forecast = fit_model.forecast(var_data.values[-fit_model.k_ar:], steps=forecast_periods)
            forecast_df = pd.DataFrame(forecast, index=future_dates, columns=related_sectors)
            return ts_data, forecast_df[sector], None

        elif model_type == "SARIMAX":
            # SARIMAX with exogenous variable (e.g., Manufacturing for Electricity)
            exog = prepare_ts_data(df, 'Manufacturing')
            ts_data, exog = ts_data.align(exog, join='inner')
            if len(ts_data) < 8:
                return None, None, f"Not enough aligned data for SARIMAX on {sector}."
            model = SARIMAX(ts_data, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            fit_model = model.fit(disp=False)
            # Forecast with future exogenous (assume last known value)
            exog_future = pd.Series([exog[-1]] * forecast_periods, index=future_dates)
            forecast = fit_model.forecast(steps=forecast_periods, exog=exog_future)
            forecast.index = future_dates
            return ts_data, forecast, None

        elif model_type == "Prophet":
            # Facebook Prophet
            prophet_df = pd.DataFrame({'ds': ts_data.index, 'y': ts_data.values})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            model.fit(prophet_df)
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            forecast_series = pd.Series(forecast['yhat'].values, index=future_dates)
            return ts_data, forecast_series, None

        elif model_type == "ARIMA-Exog":
            # ARIMA with exogenous variable (e.g., GDP for Financial & Insurance)
            exog = prepare_ts_data(df, 'Gdp At Market Prices')
            ts_data, exog = ts_data.align(exog, join='inner')
            if len(ts_data) < 8:
                return None, None, f"Not enough aligned data for ARIMA-Exog on {sector}."
            model = ARIMA(ts_data, exog=exog, order=(1, 1, 1))
            fit_model = model.fit()
            # Forecast with future exogenous (assume last known value)
            exog_future = pd.Series([exog[-1]] * forecast_periods, index=future_dates)
            forecast = fit_model.forecast(steps=forecast_periods, exog=exog_future)
            forecast.index = future_dates
            return ts_data, forecast, None

        elif model_type == "ARIMA":
            # Simple ARIMA
            model = ARIMA(ts_data, order=(1, 1, 1))
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=forecast_periods)
            forecast.index = future_dates
            return ts_data, forecast, None

        else:
            return None, None, f"Model {model_type} not implemented for {sector}."

    except Exception as e:
        return None, None, f"Error forecasting {sector}: {str(e)}"

def plot_forecast(historical, forecast, sector, model_type):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecasted data
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{sector} Forecast ({model_type})",
        xaxis_title="Date",
        yaxis_title="GDP (KES Million)",
        xaxis_tickangle=45
    )
    
    return fig

def generate_forecast_interpretation(historical, forecast, sector, model_type):
    # Initialize interpretation list
    interpretation = ["**What This Forecast Means:**", ""]  # Add introductory line with a blank line for spacing
    
    # 1. Net Trend (based on last historical to last forecasted value)
    last_historical = historical.iloc[-1]
    last_forecast = forecast.values[-1]
    net_change = last_forecast - last_historical
    trend = "increasing" if net_change > 0 else "decreasing" if net_change < 0 else "stable"
    interpretation.append(f"- **Overall Trend**: The forecast for {sector} shows an {trend} trend over the next {len(forecast)} quarters, from the last historical value to the end of the forecast period.")
    interpretation.append("")  # Add blank line for spacing
    
    # 2. Magnitude of Change
    percentage_change = (net_change / last_historical) * 100
    change_direction = "increase" if percentage_change > 0 else "decrease" if percentage_change < 0 else "no change"
    percentage_change = abs(percentage_change)
    # Fix date formatting for the last forecast date
    last_forecast_date = forecast.index[-1]
    quarter = (last_forecast_date.month - 1) // 3 + 1  # Convert month to quarter (1-4)
    year = last_forecast_date.year
    interpretation.append(f"- **Change**: From the last recorded value ({last_historical:,.0f} KES Million), the GDP is expected to {change_direction} by {percentage_change:.1f}% to {last_forecast:,.0f} KES Million by {year} Q{quarter}.")
    interpretation.append("")  # Add blank line for spacing
    
    # 3. Fluctuations Within Forecast Period
    forecast_values = forecast.values
    if len(forecast_values) > 1:
        diffs = np.diff(forecast_values)
        if np.std(diffs) > np.mean(np.abs(diffs)) * 0.5:  # Check for significant fluctuations
            interpretation.append(f"- **Fluctuations**: Within the forecast period, there are noticeable ups and downs, likely due to seasonal patterns or short-term variations in {sector}.")
        else:
            interpretation.append(f"- **Fluctuations**: The forecast shows a relatively smooth trend with minor variations across the quarters.")
    interpretation.append("")  # Add blank line for spacing
    
    # 4. Seasonality (if applicable)
    if model_type in ["SARIMA", "Holt-Winters", "Prophet"]:
        interpretation.append(f"- **Seasonality**: The forecast captures seasonal patterns, which are expected for {sector} due to quarterly cycles (e.g., harvest seasons, tourism peaks). These patterns are modeled by {model_type} to reflect historical trends.")
        interpretation.append("")  # Add blank line for spacing
    
    # 5. Contextual Insight
    if sector == "Agriculture":
        interpretation.append(f"- **Insight**: Agriculture often experiences seasonal fluctuations due to planting and harvest cycles. The {model_type} model accounts for these patterns, suggesting potential growth or decline based on historical trends.")
    elif sector == "Mining & Quarrying":
        interpretation.append(f"- **Insight**: Mining & Quarrying can be influenced by global commodity prices and local production capacity. The {model_type} model smooths out short-term volatility to predict a general trend.")
    elif sector == "Manufacturing":
        interpretation.append(f"- **Insight**: Manufacturing is forecasted using {model_type}, which considers its relationship with overall GDP, reflecting dependencies on economic activity and demand.")
    elif sector == "Electricity & Water Supply":
        interpretation.append(f"- **Insight**: Electricity & Water Supply is forecasted with {model_type}, using Manufacturing as an influencing factor, as industrial activity drives energy and water demand.")
    elif sector == "Construction":
        interpretation.append(f"- **Insight**: Construction forecasts with {model_type} account for seasonal and yearly trends, often tied to infrastructure projects and government spending cycles.")
    elif sector == "Wholesale And Retail Trade":
        interpretation.append(f"- **Insight**: Wholesale and Retail Trade shows seasonal patterns due to consumer behavior (e.g., holiday seasons). The {model_type} model captures these cycles for better predictions.")
    elif sector == "Accommodation And Food Service Activities":
        interpretation.append(f"- **Insight**: Accommodation and Food Services are highly seasonal, often peaking during tourist seasons. The {model_type} model predicts based on these recurring patterns.")
    elif sector == "Transportation And Storage":
        interpretation.append(f"- **Insight**: Transportation and Storage forecasts with {model_type} consider its link to overall GDP, reflecting economic activity and trade volumes.")
    elif sector == "Information And Communication":
        interpretation.append(f"- **Insight**: Information and Communication often shows steady growth due to digital expansion. The {model_type} model captures both trends and seasonal effects in this sector.")
    elif sector == "Financial And Insurance":
        interpretation.append(f"- **Insight**: Financial and Insurance forecasts with {model_type} use overall GDP as an influencing factor, reflecting its sensitivity to economic conditions.")
    elif sector == "Other Services":
        interpretation.append(f"- **Insight**: Other Services may include various activities with seasonal patterns. The {model_type} model helps predict these fluctuations for better planning.")
    elif sector == "Fisim":
        interpretation.append(f"- **Insight**: FISIM (Financial Intermediation) can be volatile due to banking costs. The {model_type} model provides a smoothed prediction based on historical trends.")
    elif sector == "All Industries At Basic Prices":
        interpretation.append(f"- **Insight**: All Industries at Basic Prices reflects the broader economy. The {model_type} model uses its relationship with GDP to forecast overall economic activity.")
    elif sector == "Taxes On Product":
        interpretation.append(f"- **Insight**: Taxes on Product can fluctuate with policy changes and economic activity. The {model_type} model predicts based on historical patterns.")

    return "\n".join(interpretation)

def create_forecasting_tab(df):
    st.header("Forecasting")
    st.write("Select a sector to view its forecasted GDP values for the next 4 quarters.")

    # Sector selection for forecasting
    forecast_sectors = list(FORECAST_MODELS.keys())
    selected_forecast_sector = st.selectbox("Select Sector for Forecasting", forecast_sectors)

    # Get the model type
    model_type = FORECAST_MODELS[selected_forecast_sector]

    # Perform forecasting
    historical, forecast, error = forecast_sector(df, selected_forecast_sector, model_type)

    if error:
        st.error(error)
    else:
        # Plot the forecast
        fig = plot_forecast(historical, forecast, selected_forecast_sector, model_type)
        st.plotly_chart(fig, use_container_width=True, key=f"forecast_{selected_forecast_sector}_chart")

        # Display forecast values
        st.subheader("Forecasted Values")
        # Convert datetime index to Year_Quarter format
        year_quarter = [f"{date.year} Q{(date.month - 1) // 3 + 1}" for date in forecast.index]
        forecast_df = pd.DataFrame({
            "Date": year_quarter,
            "Forecasted GDP (KES Million)": forecast.values
        })
        st.write(forecast_df)

        # Display interpretation
        st.subheader("Interpretation")
        interpretation = generate_forecast_interpretation(historical, forecast, selected_forecast_sector, model_type)
        st.markdown(interpretation)