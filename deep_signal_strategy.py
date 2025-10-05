import pandas as pd
from datetime import datetime
import numpy as np # Adding numpy just in case

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
    """
    Loads hourly data, resamples it to daily candles, and calculates SMAs.
    
    NOTE: This function requires a file named 'xbtusd_1h_8y.csv' to run, 
    or you must provide a path to your own data.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{path}'. Cannot load daily data.")
        # Create a dummy DataFrame for demonstration if the file is missing
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start_date, periods=200, freq='D')
        data = {
            'open': np.linspace(5000, 6000, 200),
            'high': np.linspace(5050, 6050, 200),
            'low': np.linspace(4950, 5950, 200),
            'close': np.linspace(5020, 6020, 200) + np.sin(np.arange(200) / 10) * 50
        }
        daily = pd.DataFrame(data, index=dates)
        daily.index.name = 'timestamp'
        
        print("Using dummy data instead. Please replace with actual data for real backtest results.")
        
        # Calculate SMAs for dummy data
        daily[f'sma{SMA_SHORT_PERIOD}'] = daily['close'].rolling(SMA_SHORT_PERIOD).mean()
        daily[f'sma{SMA_LONG_PERIOD}'] = daily['close'].rolling(SMA_LONG_PERIOD).mean()
        return daily.dropna()

    time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    daily = (df.resample('D')
               .agg({'open': 'first',
                     'high': 'max',
                     'low':  'min',
                     'close':'last'})).dropna()

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

    # FIX: Removed the line continuation character (\) and ensured strings are properly terminated
    sma_short_col = f'sma{SMA_SHORT_PERIOD}'
    sma_long_col = f'sma{SMA_LONG_PERIOD}'

    # Calculate previous SMA values for slope and crossover detection
    prev_sma_short = daily[sma_short_col].shift(1)
    # prev_sma_long = daily[sma_long_col].shift(1) # Not used in current logic, but kept for context

    # Track previous market regime for printing changes
    prev_is_bull_market = None

    # Ensure the dataframe has enough data after dropping initial NaNs from rolling means
    if len(daily) < SMA_LONG_PERIOD:
         print("Not enough data points to run backtest with given SMA periods.")
         return {'final': cash, 'return_%': 0.0, 'trades': 0}

    for i, (date, r) in enumerate(daily.iterrows()):
        # Determine current market regime
        current_is_bull_market = r[sma_short_col] > r[sma_long_col]

        # Print market regime change
        if prev_is_bull_market is not None and current_is_bull_market != prev_is_bull_market:
            if current_is_bull_market:
                print(f"{date.date()}  MARKET_CHANGE: BULL MARKET")
            else:
                print(f"{date.date()}  MARKET_CHANGE: BEAR MARKET")
        prev_is_bull_market = current_is_bull_market

        # 1. Stop exit (applies regardless of market regime if a position is open)
        if pos:
            st = entry * (1 - stop) if pos > 0 else entry * (1 + stop)
            if (pos > 0 and r.low <= st) or (pos < 0 and r.high >= st):
                # Calculate P&L based on stop price (st)
                pnl_btc = pos * (st - entry)
                
                # Check if balance would drop below zero before applying P&L
                if balance + pnl_btc - abs(pnl_btc) * fee <= 0:
                    print(f"{date.date()}  LIQUIDATION (STOP) - Account drained.")
                    balance = 0
                    pos = 0
                    break

                bal_before = balance
                balance += pnl_btc - abs(pnl_btc) * fee
                ret_pct_tot = (balance / cash - 1) * 100
                pnl_pct = (pnl_btc / (bal_before * lev)) * 100 # % P&L vs notional used
                
                print(f"{date.date()}  CLOSE (STOP)  side={'LONG' if pos > 0 else 'SHORT'}  "
                      f"price={st:.2f}  pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                      f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                
                trade_log.append({'date': date, 'side': 'STOP',
                                  'price': st, 'pnl': pnl_btc, 'balance': balance})
                pos = 0
                trades += 1

        # Trading logic based on market regime
        if current_is_bull_market:
            # In bull market, trade based on SMA_SHORT_PERIOD slope
            current_sma_short = r[sma_short_col]
            
            # Use .iloc[i] to correctly access the previous day's SMA at the current loop index
            previous_sma_short = prev_sma_short.iloc[i] 

            # Only proceed if we have a previous SMA value (i.e., not the first day with SMAs)
            if not np.isnan(previous_sma_short):
                slope_positive = current_sma_short > previous_sma_short
                slope_negative = current_sma_short < previous_sma_short

                sig = 0
                if slope_positive and pos <= 0: # Buy if slope is positive and not already long
                    sig = 1
                elif slope_negative and pos > 0: # Sell if slope is negative and currently long
                    sig = -1

                # 3. enter / flip – LEVERAGE-sized
                if sig and balance > 0:
                    
                    # --- EXIT/FLIP LOGIC ---
                    if pos:                         # close old position first
                        pnl_btc = pos * (r.close - entry)
                        bal_before = balance
                        balance += pnl_btc - abs(pnl_btc) * fee
                        ret_pct_tot = (balance / cash - 1) * 100
                        pnl_pct = (pnl_btc / (bal_before * lev)) * 100
                        
                        print(f"{date.date()}  CLOSE (FLIP)  side={'LONG' if pos > 0 else 'SHORT'}  "
                              f"price={r.close:.2f}  pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                              f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                        
                        trade_log.append({'date': date, 'side': 'EXIT',
                                          'price': r.close, 'pnl': pnl_btc, 'balance': balance})
                        pos = 0
                        trades += 1
                        
                        # Handle case where closing the position results in account drain
                        if balance <= 0:
                            print(f"{date.date()}  LIQUIDATION (FLIP) - Account drained.")
                            break
                        
                    # --- ENTRY LOGIC ---
                    # Only enter long positions in bull market with positive slope
                    if sig == 1: # Only buy (go long)
                        notional = balance * lev
                        max_size = notional / (r.close * (1 + fee))
                        pos = sig * max_size
                        entry = r.close
                        
                        # Check for max leverage available to avoid negative balance from fees
                        # This simplified check ensures we don't over-leverage based on cash + entry fee
                        if balance - abs(pos * entry) * fee > 0:
                            balance -= abs(pos * entry) * fee
                            print(f"{date.date()}  LONG POSITION  price={entry:.2f}  bal={balance:.2f}")
                            trade_log.append({'date': date, 'side': 'ENTRY',
                                              'price': entry, 'pnl': 0, 'balance': balance})
                            trades += 1
                        else:
                             # If insufficient cash for fees, abort the entry
                             pos = 0
                             print(f"{date.date()}  FAILED ENTRY: Insufficient funds for fees.")


        else: # Bear market: stay out of the market
            if pos: # If there's an open position, close it immediately
                pnl_btc = pos * (r.close - entry)
                
                bal_before = balance
                balance += pnl_btc - abs(pnl_btc) * fee
                ret_pct_tot = (balance / cash - 1) * 100
                pnl_pct = (pnl_btc / (bal_before * lev)) * 100
                
                print(f"{date.date()}  CLOSE (BEAR MARKET)  side={'LONG' if pos > 0 else 'SHORT'}  "
                      f"price={r.close:.2f}  pnl_btc={pnl_btc:+.4f}  pnl_pct={pnl_pct:+.2f}%  "
                      f"bal={balance:.2f}  cum_ret={ret_pct_tot:+.2f}%")
                
                trade_log.append({'date': date, 'side': 'BEAR_EXIT',
                                  'price': r.close, 'pnl': pnl_btc, 'balance': balance})
                pos = 0
                trades += 1
                
                if balance <= 0:
                    print(f"{date.date()}  LIQUIDATION (BEAR) - Account drained.")
                    break


    # 4. final exit
    if pos and balance > 0:
        pnl_btc = pos * (daily["close"].iloc[-1] - entry)
        balance += pnl_btc - abs(pnl_btc) * fee
        trade_log.append({'date': daily.index[-1], 'side': 'FINAL_EXIT',
                          'price': daily["close"].iloc[-1], 'pnl': pnl_btc, 'balance': balance})
        trades += 1

    print("\nFINAL ANALYSIS:")
    print(f"  Final Balance: {balance:.2f}")
    print(f"  Total Return %: {(balance / cash - 1) * 100:+.2f}%")
    print(f"  Total Trades: {trades}")

    return {'final': balance,
            'return_%': (balance / cash - 1) * 100,
            'trades': trades}

# ------------------------------------------------------------------
# 3. run one simulation
# ------------------------------------------------------------------
if __name__ == '__main__':
    daily = load_daily()
    # Check if load_daily returned data before calling _engine
    if not daily.empty:
        result = _engine(daily)
    else:
        print("Backtest could not run due to missing data.")