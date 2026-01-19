# Tesla Stock Price Prediction


## Introduction


Tesla, Inc., founded in 2003 by Martin Eberhard and Mark Tarpenning, has revolutionized the automotive industry under Elon Musk’s leadership. The company is positioned as a pioneer in electric vehicles (Roadster, Model S/X/3/Y, Cybertruck) and energy solutions, with a growing manufacturing footprint via Gigafactories in the US, China, and Germany. The notebook mentions 2024–2025 targets including 2.5M vehicle production capacity, Full Self-Driving deployment, and a $127B revenue projection, alongside Chinese market competition and stock volatility.

This project focuses on building a forecasting model for Tesla’s closing stock price.

Close is the last traded price during a trading day and reflects:

*   the final market agreement between buyers and sellers;
    
*   a support/resistance reference level for subsequent sessions.

---


## Data Description

The [dataset]() contains daily market data with the following features:


* Time Range: `2010-06-29` - `2024-12-24`
* Number of observations: 3,648

The notebook states that there are no missing values, and highlights strong price growth.

---

## What Is a Time Series?


A time series is a set of observations of a variable taken at different time points (e.g., sales trends, stock prices, weather forecasts).

Key components described:

*   **Trend**: long-term growth/decline over time.
    
*   **Seasonality**: short-term recurring movements (e.g., weather/festivities).
    
*   **Irregularity**: sudden non-repeating shocks (residual/random).
    
*   **Cyclic**: long-term oscillations (often 5–12+ years).
    
---

## Metric


To evaluate forecast quality, the notebook uses **MAPE** (Mean Absolute Percentage Error).

The notebook also notes:

*   if true values are close to 0, MAPE can become extremely large;
    
*   the direction of the error (over/under prediction) affects the value;
    
*   it is treated as an inverse measure of forecasting accuracy.
    
---

## Exploratory Analysis

### Price history (2010–2024)

A Plotly visualization is built for Close, with a horizontal dashed line showing the mean Close over the full period.

The notebook’s interpretation:

*   a clear upward trend, with uneven growth;
    
*   periods of stagnation and sharp spikes (notably since early 2020);
    
*   a hypothesis that seasonality may be linked to reporting periods or expectations of major events (e.g., model announcements/releases).
    

### Seasonal decomposition (additive, period=365)

The notebook applies seasonal\_decompose with period = 365 and interprets:

*   **Observed**: exponential-like growth and high volatility, especially around 2018–2020 and 2021–2022.
    
*   **Trend**: long-term upward movement with acceleration after 2020.
    
*   **Seasonal**: strong seasonal swings with peaks near year-end (Q4) and troughs mid-year (Q2), linked hypothetically to reporting and macro cycles.
    
*   **Residual**: relatively small noise (±0.5) with outliers attributed to exogenous events (e.g., tweets/pandemic).
    

### Distribution and autocorrelation

Using KDE and autocorrelation plots, the notebook concludes:

*   the Close distribution is not normal, skewed with a long right tail and multiple peaks;
    
*   autocorrelation is high at small lags, indicating temporal dependence;
    
*   stationarity testing and differencing are recommended.
    
---

## Modeling

The notebook prepares:

*   df\_arima (copy of the original dataframe),
    
*   df\_prophet (reset index and renamed columns: ds for Date and y for Close),
    
*   df\_xgboost (copy; used as a placeholder for later ideas).
    

### ARIMA

ARIMA is presented as AutoRegressive Integrated Moving Average, parameterized as (p, d, q).

Stationarity test (ADF):

*   ADF statistic: 0.2976
    
*   p-value: 0.9772
    
*   Conclusion: non-stationary, differencing required.
    

Differencing: d = 1After differencing, the notebook selects ARIMA(2, 1, 2) based on:

*   PACF cutoff at lag 2 → p = 2
    
*   ACF exponential decay → q = 2
    

Forecast horizon: 30 business days (steps = 30)

### Prophet

Prophet is described as an open-source forecasting approach that models a smooth curve as a sum of:

*   overall growth trend,
    
*   yearly seasonality,
    
*   weekly seasonality,
    
*   holiday effects (conceptually noted).
    

Prophet expects:

*   `ds` as datetime,
    
*   `y` as target.
    

Model configuration used in the notebook:

*   yearly\_seasonality=True, weekly\_seasonality=True, daily\_seasonality=True
    
*   seasonality\_mode='multiplicative'
    
*   interval\_width=0.95
    
*   future dataframe with periods=30
    

Notebook interpretation of Prophet components:

*   upward trend continues over the next 30 days (volatility remains high);
    
*   weekly: rise early week, decline toward weekend;
    
*   yearly: peak mid-year, lows end of year (hypothesis tied to quarterly reporting);
    
*   daily: rise in first half of day, decline in evening.
    

### Evaluation (MAPE on a 30-day window)

To compute MAPE on real future values, the notebook loads the second CSV (TSLA\_2010-06-29\_2025-02-13.csv) and defines:

*   last train date: 2024-12-24
    
*   forecast evaluation range: next 30 business days
    

MAPE results:

*   ARIMA: 0.1703
    
*   Prophet: 0.4828
    

Conclusion in the notebook: ARIMA achieves lower MAPE than Prophet, so its short-term forecasts are considered more trustworthy for this setup.

---

## Summary

The notebook’s key outcome:

*   ARIMA outperforms Prophet on the defined 30-day evaluation window (lower MAPE).

The notebook explains this by emphasizing ARIMA’s ability (after differencing) to capture short-term linear structure without relying on potentially unstable seasonal decomposition, while Prophet’s seasonal structure and sensitivity to outliers may lead to worse performance in a high-volatility series.
