#!/usr/bin/env python3
"""
lr_live.py
Kraken-Futures live trader that uses the 6-day & 10-day
linear-regression strategy from lr_strategy.py.

Stop-loss is placed immediately after entry:
    distance (%) = 0.80 * |6-day prediction at entry|
Exchange will trigger the stop-order for us – no daily re-check needed.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
#  remove argparse --dry completely

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC    = "XBTUSD"
INTERVAL       = 1440          # 1-day candles
LOOKBACK       = 10            # lagged days
LEV            = 3.0           # 3× leverage
STOP_FAC       = 0.80          # 80 % of |pred6| (fraction, not percent)
STATE_FILE     = Path("lr_state.json")
TEST_SIZE_BTC  = 0.0001        # smoke-test size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("lr_live")

# ------------------------------------------------------------------
# 1.  INDICATORS
# ------------------------------------------------------------------
def stoch_rsi(close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    roll_gain = pd.Series(gain).rolling(rsi_period).mean()
    roll_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = roll_gain / roll_loss
    rsi = 100 - (100 / (1 + rs))
    stoch = (rsi - rsi.rolling(stoch_period).min()) / \
            (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
    return stoch.values

def ema(arr: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

# ------------------------------------------------------------------
# 2.  MODEL BUNDLE
# ------------------------------------------------------------------
class ModelBundle:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.theta   = None
        self.mu      = None
        self.sigma   = None

    def fit(self, df: pd.DataFrame):
        """Train on 80 % of data, store mu/sigma/theta."""
        d = df.copy()
        d["y"] = (d["close"].shift(-self.horizon) / d["close"] - 1) * 100
        d = d.dropna(subset=["y"])
        split = int(len(d) * 0.8)
        feats = self._build_features(d)
        X = feats[:split]
        y = d["y"].values[:split]
        self.mu    = X.mean(axis=0)
        self.sigma = np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
        Xz = (X - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]

    def predict_last(self, df: pd.DataFrame) -> float:
        """Return prediction for the last row."""
        feats = self._build_features(df).iloc[[-1]]
        Xz = (feats - self.mu) / self.sigma
        Xb = np.c_[np.ones(Xz.shape[0]), Xz]
        return float(Xb @ self.theta)

    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix (no NaNs)."""
        close = df["close"].values
        stoch = stoch_rsi(close)
        pct   = np.concatenate([[np.nan], np.diff(close) / close[:-1] * 100])
        vol_pct = df["volume"].pct_change().values * 100

        macd_line = ema(close, 12) - ema(close, 26)
        macd_sig  = ema(macd_line, 9)
        macd_diff = macd_line - macd_sig

        df = df.assign(
            stoch_rsi=stoch,
            pct_chg=pct,
            vol_pct_chg=vol_pct,
            macd_signal=macd_diff,
        )

        # lagged columns
        for i in range(LOOKBACK):
            df[f"stoch_{i}"] = df["stoch_rsi"].shift(LOOKBACK - i)
            df[f"pct_{i}"]   = df["pct_chg"].shift(LOOKBACK - i)
            df[f"vol_{i}"]   = df["vol_pct_chg"].shift(LOOKBACK - i)
            df[f"macd_{i}"]  = df["macd_signal"].shift(LOOKBACK - i)

        feature_cols = [f"{pre}_{i}" for pre in ["stoch", "pct", "vol", "macd"] for i in range(LOOKBACK)]
        return df[feature_cols].dropna()

# ------------------------------------------------------------------
# 3.  PORTFOLIO / PRICE HELPERS
# ------------------------------------------------------------------
def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])

def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")

# ------------------------------------------------------------------
# 4.  ORDER HELPERS
# ------------------------------------------------------------------
def cancel_all(api: kf.KrakenFuturesApi):
    log.info("Cancelling all orders")
    try:
        api.cancel_all_orders()
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)

def flatten_position(api: kf.KrakenFuturesApi):
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

def place_stop(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float, pred6_abs: float):
    """Place stop-order 0.80 * |pred6| % away from fill_price."""
    allowed_move = STOP_FAC * pred6_abs / 100.0        # pred6 is in %
    if side == "buy":      # long
        stop_price = fill_price * (1 - allowed_move)
    else:                  # short
        stop_price = fill_price * (1 + allowed_move)
    limit_price = stop_price * (0.9999 if side == "buy" else 1.0001)
    stop_side = "sell" if side == "buy" else "buy"
    log.info("Placing stop %s at %.2f (%.2f %% away)", stop_side, stop_price, allowed_move*100)
    api.send_order({
        "orderType": "stp",
        "symbol": SYMBOL_FUTS_LC,
        "side": stop_side,
        "size": round(size_btc, 4),
        "stopPrice": int(round(stop_price)),
        "limitPrice": int(round(limit_price)),
    })

# ------------------------------------------------------------------
# 5.  STATE FILE
# ------------------------------------------------------------------
def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st))

# ------------------------------------------------------------------
# 6.  SMOKE TEST
# ------------------------------------------------------------------
def smoke_test(api: kf.KrakenFuturesApi, model6: ModelBundle, model10: ModelBundle):
    log.info("=== Smoke-test start ===")
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL)
    usd = portfolio_usd(api)
    mp  = mark_price(api)
    log.info("Fetched %d days  flex=%.2f USD  mark=%.2f", len(df), usd, mp)

    p6  = model6.predict_last(df)
    p10 = model10.predict_last(df)
    log.info("6-day pred=%.2f %%  10-day pred=%.2f %%", p6, p10)

    log.info("Sending TEST market buy %.4f BTC", TEST_SIZE_BTC)
    ord = api.send_order({
        "orderType": "mkt",
        "symbol": SYMBOL_FUTS_LC,
        "side": "buy",
        "size": TEST_SIZE_BTC,
    })
    fill_p = float(ord.get("price", mp))
    place_stop(api, "buy", TEST_SIZE_BTC, fill_p, abs(p6))

    time.sleep(3)
    flatten_position(api)
    cancel_all(api)
    log.info("=== Smoke-test complete ===")

# ------------------------------------------------------------------
# 7.  DAILY SIGNAL
# ------------------------------------------------------------------
def scan_signal(api: kf.KrakenFuturesApi, model6: ModelBundle, model10: ModelBundle) -> str:
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL)
    p6  = model6.predict_last(df)
    p10 = model10.predict_last(df)
    log.info("Pred 6-day=%.2f %%  10-day=%.2f %%", p6, p10)

    state = load_state()
    last = state.get("last_signal")

    if p6 > 0 and p10 > 0:
        new = "BUY"
    elif p6 < 0 and p10 < 0:
        new = "SELL"
    else:
        new = "HOLD"

    if new != "HOLD" and new != last:
        log.info("Signal change -> %s", new)
        state["last_signal"] = new
        save_state(state)
        return new
    return "HOLD"

# ------------------------------------------------------------------
# 8.  TRADE STEP
# ------------------------------------------------------------------
def trade_step(api: kf.KrakenFuturesApi, model6: ModelBundle, model10: ModelBundle, dry: bool):
    signal = scan_signal(api, model6, model10)
    if signal == "HOLD":
        log.info("No trade")
        return

    cancel_all(api)
    collateral = portfolio_usd(api)
    notional   = collateral * LEV
    price      = mark_price(api)
    size_btc   = round(notional / price, 4)

    flatten_position(api)          # start flat

    if dry:
        log.info("DRY-RUN: %s %.4f BTC market", signal, size_btc)
        return

    side = "buy" if signal == "BUY" else "sell"
    log.info("Market %s %.4f BTC", side, size_btc)
    ord = api.send_order({
        "orderType": "mkt",
        "symbol": SYMBOL_FUTS_LC,
        "side": side,
        "size": size_btc,
    })
    fill_p = float(ord.get("price", price))

    # pred6 used for stop distance
    pred6_today = model6.predict_last(kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL))
    place_stop(api, side, size_btc, fill_p, abs(pred6_today))

# ------------------------------------------------------------------
# 9.  SCHEDULER
# ------------------------------------------------------------------
def wait_until_00_05_utc():
    now = datetime.utcnow()
    next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:05 UTC (%s), sleeping %.0f s", next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)

# ------------------------------------------------------------------
# 10.  MAIN
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kraken-Futures 6d/10d LinReg bot")
    ap.add_argument("--dry", action="store_true", help="Dry-run (no orders)")
    args = ap.parse_args()

    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_API_SECRET missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)

    # fit models once at start
    log.info("Fitting 6-day & 10-day models…")
    df0 = kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL)
    model6  = ModelBundle(6)
    model10 = ModelBundle(10)
    model6.fit(df0)
    model10.fit(df0)
    log.info("Models ready")

    smoke_test(api, model6, model10)

    # daily loop
    while True:
        wait_until_00_05_utc()
        try:
            trade_step(api, model6, model10, args.dry)
        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily step failed: %s", exc)

if __name__ == "__main__":
    main()
