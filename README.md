# 📈 Bitcoin Price Predictor

Welcome to the **Bitcoin Price Predictor** project!  
This repository contains code to **fetch live Bitcoin data**, engineer features (lags + technical indicators), and train a machine learning model to forecast Bitcoin prices.  

We use a **Random Forest Regressor** on lagged prices and technical indicators, then generate **future forecasts** by iteratively predicting daily returns and compounding them into prices. The project also automatically saves both a **forecast CSV** and a **visualization PNG**.

---

## Table of Contents
- [Introduction](#introduction)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Notes](#notes)

---

## Introduction
Predicting Bitcoin’s price is both fascinating and challenging due to its volatility.  
This project demonstrates how to apply machine learning to **time-series forecasting** of Bitcoin, using:
- Historical price data from [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/).  
- Technical indicators (moving averages, volatility, RSI, MACD, Bollinger Bands).  
- Lagged features of previous daily closes.  
- Random Forest for regression on returns.  

The result is a model that can **forecast prices into the future (e.g., 2025/2026)** while saving outputs for analysis and visualization.

---

## Features
- **Automatic data download** using `yfinance` (no manual CSV needed).  
- **Feature engineering**: lagged closes, SMA, EMA, RSI, MACD, Bollinger Bands, volatility.  
- **Future forecasting**: iterative prediction of daily returns → compounded into future prices.  
- **Visualization**: plot of historical vs test predictions vs forecast.  
- **Outputs saved automatically**:  
  - `forecast_btcusd.csv` → future predictions in CSV format.  
  - `btc_forecast.png` → forecast plot image.  

---

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/muhammadblal433/Bitcoin-Price-Predictor.git
   cd Bitcoin-Price-Predictor

2. Create and activate a virtual environment (recommended):
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate    # Mac/Linux
   .venv\Scripts\activate       # Windows

3. Install the required dependencies:
   pip install -r requirements.txt

## Usage

1. Run the script:
   python BTC.py

## The script will:
- Fetch fresh Bitcoin price data from Yahoo Finance.
- Train the Random Forest model on historical features.
- Evaluate performance on recent data.
- Generate a future forecast up to your chosen horizon (default: Dec 2026).
- Save results to: forecast_btcusd.csv and btc_forecast.png

## Results:
- Model evaluation is printed in the terminal (MSE, MAE, R² on returns).
- Example forecast plot (your own will update with each run).

## Notes:
- Forecasts are purely illustrative — this is a machine learning demo, not financial advice.
- Accuracy decreases the further you forecast into the future due to compounding uncertainty.