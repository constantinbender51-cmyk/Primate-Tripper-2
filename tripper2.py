#!/usr/bin/env python3
"""
lr_live_sma.py - Live trading bot with SMA-based strategy
Key features:
1. Train on Binance data (Jan 2022 - Sep 2023) scaled to recent price levels
2. Use 4 SMA features: 7-day & 365-day price SMA, 5-day & 10-day volume SMA
3. Predict actual price 3 days ahead
4. Every 3 days: compare prediction made 3 days ago vs actual, trade accordingly
5. No leverage, no stop loss
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

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Backtest on the second 50% of data.
        Returns dict with performance metrics.
        """
        d = df.copy()
        d["y"] = d["close"].shift(-self.horizon)
        d = d.dropna(subset=["y"])
        
        feats = self._build_features(d)
        split = int(len(feats) * 0.5)
        
        # Test on second half
        X_test = feats[split:]
        y_test = d["y"].values[split:]
        
        # Make predictions
        Xz = (X_test - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        predictions = Xb @ self.theta
        
        # Calculate strategy performance
        capital = 1000.0
        trades = []
        
        for i in range(len(predictions)):
            pred_price = predictions[i]
            actual_price = y_test[i]
            
            # Get the price 3 days before this prediction
            # This is the "entry" price
            idx = split + i
            if idx < self.horizon:
                continue
            entry_price = d["close"].iloc[idx - self.horizon]
            
            # Calculate actual return
            actual_return = (actual_price - entry_price) / entry_price
            
            # Trading logic: if prediction > actual → SHORT (negative return)
            #                if prediction < actual → LONG (positive return)
            if pred_price > actual_price:
                # We were too optimistic → SHORT
                ret = -actual_return
            else:
                # We were too pessimistic → LONG
                ret = actual_return
            
            capital *= (1 + ret)
            trades.append({
                'pred': pred_price,
                'actual': actual_price,
                'return': ret,
                'capital': capital
            })
        
        final_return = (capital - 1000) / 1000 * 100
        
        return {
            'initial_capital': 1000,
            'final_capital': capital,
            'return_pct': final_return,
            'num_trades': len(trades),
            'trades': trades
        }

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

    # Simulate what we would have predicted 3 days ago
    if len(df) < 4:
        log.warning("Not enough data for smoke test")
        return
    
    three_days_ago_idx = len(df) - 4
    df_historical = df.iloc[:three_days_ago_idx]
    
    pred = model.predict_last(df_historical)
    actual_today = df["close"].iloc[-1]
    
    log.info("3-day prediction (made 3 days ago): %.2f", pred)
    log.info("Actual price today: %.2f", actual_today)
    
    # Determine signal
    signal = "SHORT" if pred > actual_today else "LONG"
    log.info("Signal: %s", signal)

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
    3. Get data and simulate prediction made 3 days ago
    4. Compare to actual price today
    5. Open position based on comparison
    """
    log.info("=== Starting trade step (3-day cycle) ===")
    
    # Step 1: Close existing position
    log.info("Step 1: Flattening any existing positions")
    cancel_all(api)
    flatten_position(api)
    
    # Step 2: Wait for candle to form
    log.info("Step 2: Waiting %d seconds for candle to form...", WAIT_FOR_CANDLE_SEC)
    time.sleep(WAIT_FOR_CANDLE_SEC)
    
    # Step 3: Get fresh data
    log.info("Step 3: Fetching fresh Kraken data")
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    
    if len(df) < 4:
        log.error("Not enough data to make 3-day-old prediction")
        return
    
    # Step 4: Simulate what we would have predicted 3 days ago for today
    three_days_ago_idx = len(df) - 4
    df_historical = df.iloc[:three_days_ago_idx]
    
    predicted_price = model.predict_last(df_historical)
    actual_price_today = df["close"].iloc[-1]
    
    log.info("Price predicted 3 days ago for today: %.2f", predicted_price)
    log.info("Actual price today: %.2f", actual_price_today)
    
    # Step 5: Determine signal and open position
    if predicted_price > actual_price_today:
        signal = "SHORT"
        side = "sell"
        log.info("Signal: SHORT (prediction was too high - we were too optimistic)")
    else:
        signal = "LONG"
        side = "buy"
        log.info("Signal: LONG (prediction was too low - we were too pessimistic)")
    
    # Calculate position size (no leverage)
    collateral = portfolio_usd(api)
    notional = collateral * LEV
    current_price = mark_price(api)
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
    
    # ============================================================
    # Scale training data to match recent price levels
    # ============================================================
    log.info("Scaling training data to recent price levels...")
    
    # Get recent 30 days of data to calculate scaling factor
    df_recent = df_all.tail(30)
    avg_price_recent = df_recent['close'].mean()
    avg_price_training = df_train['close'].mean()
    scale_factor = avg_price_recent / avg_price_training
    
    log.info(f"Recent 30-day avg price: ${avg_price_recent:.2f}")
    log.info(f"Training period avg price: ${avg_price_training:.2f}")
    log.info(f"Scale factor: {scale_factor:.4f}")
    
    # Scale ALL price and volume columns
    df_train['open'] = df_train['open'] * scale_factor
    df_train['high'] = df_train['high'] * scale_factor
    df_train['low'] = df_train['low'] * scale_factor
    df_train['close'] = df_train['close'] * scale_factor
    df_train['volume'] = df_train['volume'] * scale_factor
    
    log.info(f"Scaled training data - new avg price: ${df_train['close'].mean():.2f}")
    
    # Train model (3-day ahead prediction)
    log.info("Training model with 3-day ahead prediction...")
    model = ModelBundle(horizon=3)
    model.fit(df_train)
    log.info("Model training complete!")
    
    # ============================================================
    # Backtest on scaled data
    # ============================================================
    log.info("=== Running backtest on test set ===")
    backtest_results = model.backtest(df_train)
    
    log.info(f"Backtest Results:")
    log.info(f"  Initial Capital: ${backtest_results['initial_capital']:.2f}")
    log.info(f"  Final Capital: ${backtest_results['final_capital']:.2f}")
    log.info(f"  Return: {backtest_results['return_pct']:.2f}%")
    log.info(f"  Number of Trades: {backtest_results['num_trades']}")
    
    # Show first 5 and last 5 trades
    trades = backtest_results['trades']
    if len(trades) > 0:
        log.info("First 5 trades:")
        for i, trade in enumerate(trades[:5]):
            log.info(f"  Trade {i+1}: pred=${trade['pred']:.2f} actual=${trade['actual']:.2f} "
                    f"return={trade['return']*100:.2f}% capital=${trade['capital']:.2f}")
        
        if len(trades) > 5:
            log.info("...")
            log.info("Last 5 trades:")
            for i, trade in enumerate(trades[-5:]):
                idx = len(trades) - 5 + i + 1
                log.info(f"  Trade {idx}: pred=${trade['pred']:.2f} actual=${trade['actual']:.2f} "
                        f"return={trade['return']*100:.2f}% capital=${trade['capital']:.2f}")
    
    log.info("=== Backtest complete ===")

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
