#!/usr/bin/env python3
"""
macd_live.py
MACD cross-over bot for Kraken-Futures inverse perpetual PF_XBTUSD.

0. On start-up: full connectivity & capability test
   - Fetch 200-day OHLC, compute MACD, print last 5
   - Get accounts.flex.portfolioValue
   - Get tickers → PF_XBTUSD markPrice
   - Send tiny MARKET order (5×) + 2% stop
   - Immediately flatten + cancel all
1. Enter daily loop: check signal at 00:05 UTC
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

import kraken_futures as kf
import kraken_ohlc

SYMBOL_FUTS_UC = "PF_XBTUSD"   # upper-case for comparisons
SYMBOL_FUTS_LC = "pf_xbtusd"   # lower-case for send_order only
SYMBOL_OHLC = "XBTUSD"
INTERVAL = 1440
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
LEVERAGE = 5
STOP_PCT = 0.02
STATE_FILE = Path("macd_state.json")
TEST_SIZE_BTC = 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("macd_live")


# ------------------------------------------------------------------
# indicators
# ------------------------------------------------------------------
def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = df["close"].ewm(span=fast).mean()
    ema_s = df["close"].ewm(span=slow).mean()
    macd_l = ema_f - ema_s
    sig_l = macd_l.ewm(span=signal).mean()
    hist = macd_l - sig_l
    return macd_l, sig_l, hist


# ------------------------------------------------------------------
# price & portfolio
# ------------------------------------------------------------------
def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


# ------------------------------------------------------------------
# order helpers
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
        size_btc = abs(float(p["size"]))
        log.info("Flatten %s position %.4f BTC", p["side"], size_btc)
        api.send_order(
            {
                "orderType": "mkt",
                "symbol": SYMBOL_FUTS_LC,  # <— lower-case
                "side": side,
                "size": round(size_btc, 4),
            }
        )


def place_stop(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    stop_price = fill_price * (1 - STOP_PCT) if side == "buy" else fill_price * (1 + STOP_PCT)
    stop_side = "sell" if side == "buy" else "buy"
    log.info("Placing stop %s at %.2f", stop_side, stop_price)
    api.send_order(
        {
            "orderType": "stp",
            "symbol": SYMBOL_FUTS_LC,  # <— lower-case
            "side": stop_side,
            "size": round(size_btc, 4),
            "stopPrice": stop_price,
        }
    )


# ------------------------------------------------------------------
# initial smoke test
# ------------------------------------------------------------------
def smoke_test(api: kf.KrakenFuturesApi):
    log.info("=== Smoke-test start ===")

    # 1. OHLC + MACD
    log.info("Fetching 200-day OHLC for %s", SYMBOL_OHLC)
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL)
    macd_l, sig_l, hist = macd(df, MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = macd_l
    df["signal"] = sig_l
    df["hist"] = hist
    log.info("Last 5 MACD rows:\n%s", df[["close", "macd", "signal", "hist"]].tail())

    # 2. Portfolio
    usd = portfolio_usd(api)
    log.info("flex.portfolioValue = %.2f USD", usd)

    # 3. Mark price
    mp = mark_price(api)
    log.info("PF_XBTUSD markPrice = %.2f", mp)

    # 4. Tiny market order + stop
    log.info("Sending TEST market buy %.4f BTC", TEST_SIZE_BTC)
    ord = api.send_order(
        {"orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": "buy", "size": TEST_SIZE_BTC}
    )
    fill_p = float(ord.get("price", mp))
    place_stop(api, "buy", TEST_SIZE_BTC, fill_p)

    # ------------------ 5-minute delay ---------------------------------
    log.info("Waiting 5 min before flattening smoke-test orders…")
    time.sleep(300)          # 300 s = 5 min
    # --------------------------------------------------------------------
    flatten_position(api)
    cancel_all(api)
    log.info("=== Smoke-test complete ===")

# ------------------------------------------------------------------
# signal
# ------------------------------------------------------------------
def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st))


def scan_signal(api: kf.KrakenFuturesApi) -> str:
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC, INTERVAL)
    macd_l, sig_l, _ = macd(df, MACD_FAST, MACD_SLOW, MACD_SIG)

    prev_macd, prev_sig = macd_l.iloc[-2], sig_l.iloc[-2]
    curr_macd, curr_sig = macd_l.iloc[-1], sig_l.iloc[-1]

    log.info("MACD %.4f  Signal %.4f  (prev %.4f %.4f)", curr_macd, curr_sig, prev_macd, prev_sig)

    state = load_state()
    last = state.get("last_signal")

    if prev_macd <= prev_sig and curr_macd > prev_sig:
        new = "BUY"
    elif prev_macd >= prev_sig and curr_macd < prev_sig:
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
# trade step
# ------------------------------------------------------------------
def trade_step(api: kf.KrakenFuturesApi, dry: bool):
    signal = scan_signal(api)
    if signal == "HOLD":
        log.info("No trade")
        return

    cancel_all(api)
    collateral = portfolio_usd(api)
    notional_usd = collateral * LEVERAGE
    price = mark_price(api)
    size_btc = round(notional_usd / price, 4)

    log.info(
        "Collateral=%.2f USD  notional=%.2f  BTC size=%.4f at %.2f",
        collateral,
        notional_usd,
        size_btc,
        price,
    )

    flatten_position(api)

    if dry:
        log.info("DRY-RUN: %s %.4f BTC at market", signal, size_btc)
        return

    side = "buy" if signal == "BUY" else "sell"
    log.info("Market %s %.4f BTC", side, size_btc)
    ord = api.send_order(
        {"orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": size_btc}
    )
    fill_p = float(ord.get("price", price))
    place_stop(api, side, size_btc, fill_p)


# ------------------------------------------------------------------
# scheduler
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
# CLI
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kraken-Futures MACD bot")
    ap.add_argument("--dry", action="store_true", help="Dry-run (no orders)")
    args = ap.parse_args()

    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_SECRET_KEY missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)

    # initial capability test
    smoke_test(api)

    # daily loop
    while True:
        wait_until_00_05_utc()
        try:
            trade_step(api, args.dry)
        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily step failed: %s", exc)


if __name__ == "__main__":
    main()
