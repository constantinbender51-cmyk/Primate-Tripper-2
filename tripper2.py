#!/usr/bin/env python3
"""
lr_live_sma.py - Live trading bot with SMA-based strategy
Key features:
1. Train on Binance data (Jan 2022 - Sep 2023) once at startup
2. Use 4 SMA features: 7-day & 365-day price SMA, 5-day & 10-day volume SMA
3. Predict 1 day ahead
4. Daily at 00:00 UTC: close position, wait 3 minutes, predict, open new position
5. No leverage, no stop loss, no HOLD state
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import subprocess
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc
import binance_ohlc

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
SYMBOL_OHLC_BINANCE = "BTCUSDT"
INTERVAL_KRAKEN = 1440          # Kraken uses minutes (1440 = 1 day)
INTERVAL_BINANCE = "1d"         # Binance uses string format
LOOKBACK = 3                     # 3-day lookback for features
LEV = 1.0                        # No leverage
TEST_SIZE_BTC = 0.0001
WAIT_FOR_CANDLE_SEC = 180       # Wait 3 minutes for candle to form

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("lr_live_sma")


class ModelBundle:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.theta   = None
        self.mu      = None
        self.sigma   = None

    def fit(self, df: pd.DataFrame):
        """Train on 50% of data, store mu/sigma/theta."""
        d = df.copy()
        
        # Calculate target: actual price 'horizon' days ahead
        d["y"] = d["close"].shift(-self.horizon)
        d = d.dropna(subset=["y"])
        
        # Build features
        feats = self._build_features(d)
        
        # 50/50 split
        split = int(len(feats) * 0.5)
        X = feats[:split]
        y = d["y"].values[:split]
        
        # Store normalization parameters
        self.mu    = X.mean(axis=0)
        self.sigma = np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
        
        # Normalize and train
        Xz = (X - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        
        log.info(f"Model trained on {len(X)} samples (50% of {len(feats)} total)")

    def predict_last(self, df: pd.DataFrame) -> float:
        """Return prediction for the last row."""
        feats = self._build_features(df).iloc[[-1]]
        Xz = (feats - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        return float(Xb @ self.theta)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix with 4 SMA features:
        - 7-day SMA of close price
        - 365-day SMA of close price
        - 5-day SMA of volume
        - 10-day SMA of volume
        """
        close = df["close"].values
        volume = df["volume"].values
        
        # Calculate SMAs
        sma_7 = pd.Series(close).rolling(window=7).mean().values
        sma_365 = pd.Series(close).rolling(window=365).mean().values
        vol_sma_5 = pd.Series(volume).rolling(window=5).mean().values
        vol_sma_10 = pd.Series(volume).rolling(window=10).mean().values
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'sma_7': sma_7,
            'sma_365': sma_365,
            'vol_sma_5': vol_sma_5,
            'vol_sma_10': vol_sma_10
        })
        
        # Drop rows with NaN values
        return features_df.dropna()


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")


def cancel_all(api: kf.KrakenFuturesApi):
    log.info("Cancelling all orders")
    try:
        api.cancel_all_orders()
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)


def flatten_position(api: kf.KrakenFuturesApi):
    """Close all open positions."""
    pos = api.get_open_positions()
    for p in pos.get("openPositions", []):
        if p["symbol"] != SYMBOL_FUTS_UC:
            continue
        side = "sell" if p["side"] == "long" else "buy"
        size = abs(float(p["size"]))
        log.info("Flatten %s position %.4f BTC", p["side"], size)
        api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
        })


def smoke_test(api: kf.KrakenFuturesApi, model: ModelBundle):
    """Quick test to verify everything works."""
    log.info("=== Smoke-test start ===")
    
    # Get live data from Kraken
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    usd = portfolio_usd(api)
    mp  = mark_price(api)
    log.info("Fetched %d days  flex=%.2f USD  mark=%.2f", len(df), usd, mp)

    # Make prediction
    pred = model.predict_last(df)
    log.info("3-day price prediction: %.2f", pred)
    
    # Determine signal
    signal = "SHORT" if pred < mp else "LONG"
    log.info("Signal: %s (predicted=%.2f, current=%.2f)", signal, pred, mp)

    # Test market order
    log.info("Sending TEST market buy %.4f BTC", TEST_SIZE_BTC)
    ord = api.send_order({
        "orderType": "mkt",
        "symbol": SYMBOL_FUTS_LC,
        "side": "buy",
        "size": TEST_SIZE_BTC,
    })
    log.info("Test order filled at: %.2f", float(ord.get("price", mp)))

    time.sleep(3)
    flatten_position(api)
    cancel_all(api)
    log.info("=== Smoke-test complete ===")


def trade_step(api: kf.KrakenFuturesApi, model: ModelBundle):
    """
    Trading logic (executed every 3 days):
    1. Close any existing position
    2. Wait 3 minutes for candle to form
    3. Get fresh data and predict price 3 days ahead
    4. Open new position based on prediction
    """
    log.info("=== Starting trade step (3-day cycle) ===")
    
    # Step 1: Close existing position
    log.info("Step 1: Flattening any existing positions")
    cancel_all(api)
    flatten_position(api)
    
    # Step 2: Wait for candle to form
    log.info("Step 2: Waiting %d seconds for candle to form...", WAIT_FOR_CANDLE_SEC)
    time.sleep(WAIT_FOR_CANDLE_SEC)
    
    # Step 3: Get fresh data and predict
    log.info("Step 3: Fetching fresh Kraken data and making 3-day prediction")
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    predicted_price = model.predict_last(df)
    
    # Get current market price
    current_price = mark_price(api)
    predicted_price = pred_pct
    
    log.info("Current price: %.2f", current_price)
    log.info("Predicted price (3 days ahead): %.2f", predicted_price)
    
    # Step 4: Determine signal and open position
    if predicted_price > current_price:
        signal = "SHORT"
        side = "sell"
        log.info("Signal: SHORT (prediction > current)")
    else:
        signal = "LONG"
        side = "buy"
        log.info("Signal: LONG (prediction < current)")
    
    # Calculate position size (no leverage)
    collateral = portfolio_usd(api)
    notional = collateral * LEV
    size_btc = round(notional / current_price, 4)
    
    log.info("Opening %s position: %.4f BTC (collateral=%.2f USD)", signal, size_btc, collateral)
    
    if dry:
        log.info("DRY-RUN: Would %s %.4f BTC at market", signal, size_btc)
    else:
        ord = api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": size_btc,
        })
        fill_price = float(ord.get("price", current_price))
        log.info("Position opened at %.2f", fill_price)
    
    log.info("=== Trade step complete ===")


def wait_until_00_00_utc():
    """Wait until next 00:00 UTC."""
    now = datetime.utcnow()
    next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:00 UTC (%s), sleeping %.0f seconds", 
             next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)


def wait_three_days():
    """Wait exactly 3 days until next trade execution."""
    wait_sec = 3 * 24 * 60 * 60  # 3 days in seconds
    next_run = datetime.utcnow() + timedelta(days=3)
    log.info("Next trade execution at %s (in 3 days), sleeping %.0f seconds", 
             next_run.strftime("%Y-%m-%d %H:%M UTC"), wait_sec)
    time.sleep(wait_sec)


def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_API_SECRET missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)

    # ============================================================
    # Training Phase: Use Binance data from Jan 2022 - Sep 2023
    # ============================================================
    log.info("Fetching Binance historical data for training...")
    df_all = binance_ohlc.get_ohlc_for_training(
        symbol=SYMBOL_OHLC_BINANCE,
        interval=INTERVAL_BINANCE
    )
    
    # Convert timestamp to datetime and set as index
    log.info("Converting timestamp to datetime index...")
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_all.set_index('timestamp', inplace=True)
    
    # Filter to training period: Jan 1, 2022 - Sep 30, 2023
    df_train = df_all[
        (df_all.index >= "2022-01-01") & 
        (df_all.index <= "2023-09-30")
    ].copy()
    
    log.info(f"Filtered training data: {len(df_train)} days (Jan 2022 - Sep 2023)")
    
    if len(df_train) < 400:  # Need at least 365 days for 365-day SMA
        log.error("Insufficient training data (need at least 400 days for 365-day SMA)")
        sys.exit(1)
    
    # Train model (3-day ahead prediction)
    log.info("Training model with 3-day ahead prediction...")
    model = ModelBundle(horizon=3)
    model.fit(df_train)
    log.info("Model training complete!")
    log.info("Model will predict 3 days ahead on Kraken live data, trained once on Binance historical data")

    # Run smoke test
    smoke_test(api, model)

    # Optional: Run trade immediately if flag is set
    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade_step once")
        try:
            trade_step(api, model)
        except Exception as exc:
            log.exception("Immediate trade_step failed: %s", exc)

    # Start web dashboard
    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    subprocess.Popen([sys.executable, "web_state.py"])

    # Main loop: execute trade every 3 days
    log.info("Entering main loop (trade execution every 3 days)")
    while True:
        wait_three_days()
        try:
            trade_step(api, model)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
            break
        except Exception as exc:
            log.exception("Trade step failed: %s", exc)


if __name__ == "__main__":
    main()
