import math
import requests
import numpy as np
import statsmodels.api as sm

IEX_CLOUD_API_TOKEN = "YOUR_IEX_API_TOKEN"
IEX_CLOUD_API_URL = "https://cloud.iexapis.com/stable"

def get_option_data(ticker, option_type):
    """Fetch option data for a given ticker and option type (call/put)"""
    url = f"{IEX_CLOUD_API_URL}/stock/{ticker}/options"
    params = {
        "token": IEX_CLOUD_API_TOKEN
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Filter for ATM options (assuming closest to current stock price)
    atm_option = min(data, key=lambda x: abs(x['strikePrice'] - x['lastPrice']))
    return atm_option if atm_option['side'] == option_type else None

def expected_move(call_price, put_price):
    """Calculate expected move based on ATM call and put prices."""
    return (call_price + put_price) * math.sqrt(2 * math.pi)

def zscore(series):
    """Calculate the Z-Score for a series."""
    return (series - series.mean()) / np.std(series)

ticker = input("Enter the ticker symbol: ")

# Fetch ATM call and put option data
call_option = get_option_data(ticker, 'call')
put_option = get_option_data(ticker, 'put')

# Calculate expected move
em = expected_move(call_option['lastPrice'], put_option['lastPrice'])

# Cointegration analysis
y = np.array([em for _ in range(100)])  # Mock series for expected move
x = np.array([call_option['delta'] for _ in range(100)])  # Mock series for Delta

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predicted_y = model.predict(x)

# Calculate the Z-Score for the residuals
residuals = y - predicted_y
z = zscore(residuals)[-1]

print(f"Expected Move: {em:.2f}")
print(f"Z-Score: {z:.2f}")

# Determine if the expected move is underpriced based on the Z-Score
alpha = 0.05  # 5% significance level
z_critical = abs(np.percentile(residuals, 100 * alpha))
if z > z_critical:
    print("The expected move is underpriced relative to the call option's Delta based on cointegration analysis.")
else:
    print("The expected move is not underpriced relative to the option's Delta based on cointegration analysis.")

