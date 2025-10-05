import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 0. USER CONTROLS – change these four lines only
# ------------------------------------------------------------------
# These are the SMAs for market regime classification
SMA_SHORT_PERIOD = 30
SMA_LONG_PERIOD  = 190

LEVERAGE  = 4.0
STOP_FRAC = 0.05

# ------------------------------------------------------------------
# 1. load hourly csv -> daily candles + SMAs
# ------------------------------------------------------------------
def load_daily(path='xbtusd_1h_8y.csv'):
    df = pd.read_csv(path)
    time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    daily = (df.resample('D')
               .agg({'open': 'first',
                     'high': 'max',
                     'low':  'min',
                     'close':'last'})
               .dropna())

    daily[f'sma{SMA_SHORT_PERIOD}'] = daily['close'].rolling(SMA_SHORT_PERIOD).mean()
    daily[f'sma{SMA_LONG_PERIOD}'] = daily['close'].rolling(SMA_LONG_PERIOD).mean()
    return daily.dropna()

# ------------------------------------------------------------------
# 2. single engine – returns summary dict
# ------------------------------------------------------------------
def _engine(daily, lev=LEVERAGE, fee=0.0025, stop=STOP_FRAC, cash=100):
    pos = 0
    entry = None
    balance = cash
    trades = 0
    trade_log = []

    sma_short_col = f'sma{SMA_SHORT_PERIOD}'
    sma_long_col = f'sma{SMA_LONG_PERIOD}'

    # Calculate previous SMA values for slope and crossover detection
    prev_sma_short = daily[sma_short_col].shift(1)
    prev_sma_long = daily[sma_long_col].shift(1)

    for i, (date, r) in enumerate(daily.iterrows()):
        # Determine market regime
        is_bull_market = r[sma_short_col] > r[sma_long_col]

        # 1. Stop exit (applies regardless of market regime if a position is open)
        if pos:
            st = entry * (1 - stop) if pos > 0 else entry * (1 + stop)
            if (pos > 0 and r.low <= st) or (pos < 0 and r.high >= st):
                pnl_btc = pos * (st - entry)  # realised P&L (BTC)
                bal_before = balance
                balance += pnl_btc - abs(pnl_btc) * fee
                ret_pct_tot = (balance / cash - 1) * 100
                pnl_pct = (pnl_btc / bal_before) * 100  # % vs balance before exit
                print(f"{date.date()}  STOP  side={'LONG' if pos > 0 else 'SHORT'}  "
                      f"price={st:.2f}  pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                      f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                trade_log.append({'date': date, 'side': 'STOP',
                                  'price': st, 'pnl': pnl_btc, 'balance': balance})
                pos = 0
                trades += 1

        # Trading logic based on market regime
        if is_bull_market:
            # In bull market, trade based on SMA_SHORT_PERIOD slope
            current_sma_short = r[sma_short_col]
            previous_sma_short = prev_sma_short.iloc[i]

            slope_positive = current_sma_short > previous_sma_short
            slope_negative = current_sma_short < previous_sma_short

            sig = 0
            if slope_positive and pos <= 0: # Buy if slope is positive and not already long
                sig = 1
            elif slope_negative and pos >= 0: # Sell if slope is negative and not already short
                # This means close long position, or open short if allowed (but concept says stay out of market)
                # For now, let's interpret this as closing long positions only in bull market
                # The concept says "sell whenever the 30 sma slope is smaller than 1" which implies closing existing long positions.
                sig = -1

            # Print slope events
            if slope_positive or slope_negative:
                print(f"{date.date()}  SLOPE  {'POS' if slope_positive else 'NEG'}  "
                      f"sma30={current_sma_short:.2f}  prev_sma30={previous_sma_short:.2f}")

            # 3. enter / flip – LEVERAGE-sized
            if sig and balance > 0:
                if pos:                         # close old position first
                    pnl_btc = pos * (r.close - entry)
                    bal_before = balance
                    balance += pnl_btc - abs(pnl_btc) * fee
                    ret_pct_tot = (balance / cash - 1) * 100
                    pnl_pct = (pnl_btc / bal_before) * 100
                    print(f"{date.date()}  FLIP  side={'LONG' if pos > 0 else 'SHORT'}->"
                          f"{'LONG' if sig > 0 else 'SHORT'}  price={r.close:.2f}  "
                          f"pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                          f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                    trade_log.append({'date': date, 'side': 'EXIT',
                                      'price': r.close, 'pnl': pnl_btc, 'balance': balance})
                    trades += 1

                # Only enter long positions in bull market with positive slope
                if sig == 1: # Only buy (go long)
                    notional = balance * lev
                    max_size = notional / (r.close * (1 + fee))
                    pos = sig * max_size
                    entry = r.close
                    balance -= abs(pos * entry) * fee
                    trade_log.append({'date': date, 'side': 'ENTRY',
                                      'price': entry, 'pnl': 0, 'balance': balance})
                    trades += 1
                elif sig == -1 and pos > 0: # Close long position if slope is negative
                    pnl_btc = pos * (r.close - entry)
                    bal_before = balance
                    balance += pnl_btc - abs(pnl_btc) * fee
                    ret_pct_tot = (balance / cash - 1) * 100
                    pnl_pct = (pnl_btc / bal_before) * 100
                    print(f"{date.date()}  CLOSE_LONG  price={r.close:.2f}  "
                          f"pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                          f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                    trade_log.append({'date': date, 'side': 'EXIT',
                                      'price': r.close, 'pnl': pnl_btc, 'balance': balance})
                    pos = 0
                    trades += 1

        else: # Bear market: stay out of the market
            if pos: # If there's an open position, close it immediately
                pnl_btc = pos * (r.close - entry)
                bal_before = balance
                balance += pnl_btc - abs(pnl_btc) * fee
                ret_pct_tot = (balance / cash - 1) * 100
                pnl_pct = (pnl_btc / bal_before) * 100
                print(f"{date.date()}  BEAR_EXIT  side={'LONG' if pos > 0 else 'SHORT'}  "
                      f"price={r.close:.2f}  pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                      f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                trade_log.append({'date': date, 'side': 'BEAR_EXIT',
                                  'price': r.close, 'pnl': pnl_btc, 'balance': balance})
                pos = 0
                trades += 1

    # 4. final exit
    if pos:
        pnl_btc = pos * (daily["close"].iloc[-1] - entry)
        balance += pnl_btc - abs(pnl_btc) * fee
        trade_log.append({'date': daily.index[-1], 'side': 'FINAL_EXIT',
                          'price': daily["close"].iloc[-1], 'pnl': pnl_btc, 'balance': balance})
        trades += 1

    return {'final': balance,
            'return_%' : (balance / cash - 1) * 100,
            'trades': trades}

# ------------------------------------------------------------------
# 3. run one simulation
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()
    result = _engine(daily)
    print('One-run result:')
    for k, v in result.items():
        print(f'  {k}: {v}')