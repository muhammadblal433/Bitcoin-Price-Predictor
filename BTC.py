import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1) Download & normalize data
data = yf.download("BTC-USD", start="2014-01-01", auto_adjust=False, group_by="column")

# Flatten possible MultiIndex columns like ('Close','BTC-USD')
if isinstance(data.columns, pd.MultiIndex):
    try:
        data.columns = data.columns.get_level_values(0)
    except Exception:
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

# Ensure we have a 'Close' (fallback to 'Adj Close' if needed)
if "Close" not in data.columns and "Adj Close" in data.columns:
    data = data.rename(columns={"Adj Close": "Close"})

# Clean & snapshot
data = data.dropna(subset=["Close"]).copy()
data.reset_index(inplace=True)               # 'Date' becomes a column
data.to_csv("BTC-USD.csv", index=False)      # optional snapshot

data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# 2) Feature engineering (technicals)
close = data["Close"]

# Simple moving averages
data["SMA_7"] = close.rolling(7).mean()
data["SMA_21"] = close.rolling(21).mean()

# Exponential moving average
data["EMA_50"] = close.ewm(span=50, adjust=False).mean()

# Volatility (rolling std of daily returns)
ret_1d = close.pct_change()
data["Volatility_14"] = ret_1d.rolling(14).std()

# Bollinger Bands (20, 2)
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
data["BB_Width"] = (2 * bb_std) / bb_mid

# MACD (12,26,9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
data["MACD"] = macd
data["MACD_Signal"] = signal

# RSI(14)
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss.replace(0, np.nan))
data["RSI_14"] = 100 - (100 / (1 + rs))

# Lagged closes (short window for recency)
LAGS = 7
for i in range(1, LAGS + 1):
    data[f"Close_lag_{i}"] = close.shift(i)

# 3) Target = next-day return
# Predict the *next day* return; later we'll accumulate to price.
data["Return_t"] = close.pct_change()              # current day's return (t)
data["Return_t+1"] = data["Return_t"].shift(-1)    # next day's return (target)

# Drop NaNs from indicators/lagging and the last row (target NaN)
data_model = data.dropna().copy()

FEATURES = (
    [f"Close_lag_{i}" for i in range(1, LAGS + 1)]
    + ["SMA_7", "SMA_21", "EMA_50", "Volatility_14", "BB_Width",
       "MACD", "MACD_Signal", "RSI_14", "Return_t"]
)
TARGET = "Return_t+1"

X = data_model[FEATURES].values
y = data_model[TARGET].values
dates_all = data_model.index

# Time-based split (last 20% = test)
split_idx = int(len(data_model) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates_all[split_idx:]

# 4) Train the model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on returns
y_pred_ret = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ret)
mae = mean_absolute_error(y_test, y_pred_ret)
r2 = r2_score(y_test, y_pred_ret)
print(f"Test (returns) MSE: {mse:.6f}")
print(f"Test (returns) MAE: {mae:.6f}")
print(f"Test (returns) R^2: {r2:.4f}")

# Convert test return predictions back to price for plotting overlay
test_prices_true = data_model.loc[dates_test, "Close"]
# Rebuild predicted prices by compounding predicted returns starting from the first true price
test_prices_pred = [float(test_prices_true.iloc[0])]
for r in y_pred_ret[1:]:
    test_prices_pred.append(test_prices_pred[-1] * (1.0 + float(r)))
test_prices_pred = pd.Series(test_prices_pred, index=dates_test)

# 5) Forecast future by compounding returns
# How far to forecast
today = data.index.max()
target_date = pd.Timestamp("2026-12-31")
days_ahead_min = 500
days_to_forecast = max((target_date - today).days, days_ahead_min)

# Build a rolling feature row for each future step
work = data_model.copy()

future_dates = pd.date_range(start=today + pd.Timedelta(days=1),
                             periods=days_to_forecast, freq="D")
future_prices = []
last_price = float(close.iloc[-1])

for dt in future_dates:
    # Make a single-row feature frame from the latest known values in 'work'
    row = {}
    # lagged closes from latest 'Close'
    for i in range(1, LAGS + 1):
        row[f"Close_lag_{i}"] = float(work["Close"].iloc[-i])
    # indicators from latest row
    last_row = work.iloc[-1]
    for k in ["SMA_7", "SMA_21", "EMA_50", "Volatility_14", "BB_Width",
              "MACD", "MACD_Signal", "RSI_14", "Return_t"]:
        row[k] = float(last_row[k])

    x_next = np.array([row[f] for f in FEATURES], dtype=float).reshape(1, -1)
    pred_ret = float(model.predict(x_next)[0])     # predicted next-day return

    # Update price by compounding predicted return
    last_price = last_price * (1.0 + pred_ret)
    future_prices.append(last_price)

    # Append this "future" day to 'work' to update indicators for next iteration
    # We only have Close (simulated). We update features that depend on Close/returns.
    new = {
        "Close": last_price,
        "Return_t": pred_ret,  # treat predicted return as today's realized return
    }
    # recompute indicators that depend on history
    temp = pd.DataFrame([new], index=[dt])
    work = pd.concat([work[["Close", "Return_t"]], temp])  # keep minimal cols first
    # recompute derived indicators on the fly for last 60 days window
    wclose = work["Close"]
    work.loc[dt, "SMA_7"] = wclose.tail(7).mean()
    work.loc[dt, "SMA_21"] = wclose.tail(21).mean()
    work.loc[dt, "EMA_50"] = wclose.ewm(span=50, adjust=False).mean().iloc[-1]
    work.loc[dt, "Volatility_14"] = wclose.pct_change().tail(14).std()

    bb_mid = wclose.tail(20).mean() if len(wclose) >= 20 else wclose.mean()
    bb_std = wclose.tail(20).std() if len(wclose) >= 20 else wclose.std()
    work.loc[dt, "BB_Width"] = (2 * bb_std) / bb_mid if bb_mid != 0 else 0.0

    ema12 = wclose.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = wclose.ewm(span=26, adjust=False).mean().iloc[-1]
    macd_val = ema12 - ema26
    macd_signal = pd.Series([macd_val], index=[dt]).ewm(span=9, adjust=False).mean().iloc[-1]
    work.loc[dt, "MACD"] = macd_val
    work.loc[dt, "MACD_Signal"] = macd_signal

    # RSI
    delta = wclose.diff().iloc[-14:]
    gain = delta.clip(lower=0).mean()
    loss = (-delta.clip(upper=0)).mean()
    rs = (gain / loss) if loss not in (0, np.nan) else np.nan
    rsi = 100 - (100 / (1 + rs)) if pd.notna(rs) else work["RSI_14"].iloc[-1]
    work.loc[dt, "RSI_14"] = rsi

# 6) Plot & save
plt.figure(figsize=(14, 7))
plt.plot(data.index, data["Close"], label="Historical Close")
plt.plot(dates_test, test_prices_pred, label="Model (Test, price from returns)")
plt.plot(future_dates, future_prices, linestyle="dashed", label="Future Forecast")
plt.xlabel("Date")
plt.ylabel("BTC-USD Close Price")
plt.title("Bitcoin Price — Historical, Test (returns→price), and Future Forecast")
plt.legend()
plt.tight_layout()

# Save forecast & figure
forecast_df = pd.DataFrame({"Date": future_dates, "PredictedClose": future_prices})
forecast_df.to_csv("forecast_btcusd.csv", index=False)
plt.savefig("btc_forecast.png", dpi=150)
print("Saved forecast_btcusd.csv and btc_forecast.png")

plt.show()