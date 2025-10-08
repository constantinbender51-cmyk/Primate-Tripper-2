import pandas as pd
import numpy as np
import time

# ------------------------------------------------ read data ---------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD (proper warm-up) ----------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

# 1/-1 on cross
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# =====================  SINGLE RUN WITH 6.7 % STOP  ===========================
STOP_PCT = 6.7
LEVERAGE = 3.0
curve    = [10000]
in_pos   = 0
entry_p  = None
entry_d  = None
trades   = []

# 0: no block,  1: block longs, -1: block shorts
block_side = 0

for i in range(1, len(df)):
    p_prev = df['close'].iloc[i-1]
    p_now  = df['close'].iloc[i]
    pos_i  = pos.iloc[i]

    # ----- entry logic --------------------------------------------------------
    if in_pos == 0 and pos_i != 0 and pos_i != block_side:
        in_pos     = pos_i
        entry_p    = p_now
        entry_d    = df['date'].iloc[i]
        block_side = 0          # reset block once we are in again

        # ---------- 6.7 % intrabar stop ----------
    if in_pos != 0:
        if in_pos == 1:                           # long
            stop_price = entry_p * (1 - STOP_PCT/100)
            if df['low'].iloc[i] <= stop_price:
                trades.append((entry_d, df['date'].iloc[i],
                               -STOP_PCT/100 * LEVERAGE))
                in_pos     = 0
                block_side = 1
                continue                      # <-- skip cross logic this bar
        else:                                     # short
            stop_price = entry_p * (1 + STOP_PCT/100)
            if df['high'].iloc[i] >= stop_price:
                trades.append((entry_d, df['date'].iloc[i],
                               -STOP_PCT/100 * LEVERAGE))
                in_pos     = 0
                block_side = -1
                continue                      # <-- skip cross logic this bar

    # exit on opposite MACD cross (only reached if still in_pos)
    if in_pos != 0 and pos_i == -in_pos:
        ret = (p_now / entry_p - 1) * in_pos * LEVERAGE
        trades.append((entry_d, df['date'].iloc[i], ret))
        in_pos     = pos_i
        entry_p    = p_now
        entry_d    = df['date'].iloc[i]
        block_side = 0

    # ----- exit on opposite MACD cross ----------------------------------------
    if in_pos != 0 and pos_i == -in_pos:
        ret = (p_now / entry_p - 1) * in_pos * LEVERAGE
        trades.append((entry_d, df['date'].iloc[i], ret))
        in_pos     = pos_i
        entry_p    = p_now
        entry_d    = df['date'].iloc[i]
        block_side = 0

    # ----- equity update ------------------------------------------------------
    curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos * LEVERAGE))

curve = pd.Series(curve, index=df.index)

# ---------------------------  FULL METRICS  -----------------------------------
daily_ret = curve.pct_change().dropna()
trades_ret = pd.Series([t[2] for t in trades])
n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1
vol  = daily_ret.std() * np.sqrt(252)
sharpe = cagr / vol if vol else np.nan
drawdown = curve / curve.cummax() - 1
maxdd = drawdown.min()
calmar = cagr / abs(maxdd) if maxdd else np.nan

wins   = trades_ret[trades_ret > 0]
losses = trades_ret[trades_ret < 0]
win_rate = len(wins) / len(trades_ret) if trades_ret.size else 0
avg_win  = wins.mean()   if len(wins)   else 0
avg_loss = losses.mean() if len(losses) else 0
payoff   = abs(avg_win / avg_loss) if avg_loss else np.nan
profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() else np.nan
expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

kelly = expectancy / trades_ret.var() if trades_ret.var() > 0 else np.nan

time_in_mkt = (pos != 0).mean()
tail_ratio = (np.percentile(daily_ret, 95) /
              abs(np.percentile(daily_ret, 5))) if daily_ret.size else np.nan
trades_per_year = len(trades) / n_years
lose_streak = (trades_ret < 0).astype(int)
max_lose_streak = lose_streak.groupby(
                      lose_streak.diff().ne(0).cumsum()).sum().max()

# ---------------------------  PRINT  ------------------------------------------
final_macd = curve.iloc[-1]
final_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
worst      = min(trades, key=lambda x: x[2])

print(f'\n===== MACD + {STOP_PCT}% intrabar stop-loss ({LEVERAGE}× lev) =====')
print(f'MACD final:        €{final_macd:,.0f}')
print(f'Buy & Hold final:  €{final_hold:,.0f}')
print(f'Worst trade:       {worst[2]*100:.2f}% (exit {worst[1].strftime("%Y-%m-%d")})')
print(f'Max drawdown:      {maxdd*100:.2f}%')
time.sleep(0.01)

print('\n----- full performance stats -----')
print(f'CAGR:               {cagr*100:6.2f}%')
print(f'Ann. volatility:    {vol*100:6.2f}%')
print(f'Sharpe (rf=0):      {sharpe:6.2f}')
print(f'Max drawdown:       {maxdd*100:6.2f}%')
print(f'Calmar:             {calmar:6.2f}')
print(f'Trades/year:        {trades_per_year:6.1f}')
print(f'Win-rate:           {win_rate*100:6.1f}%')
print(f'Average win:        {avg_win*100:6.2f}%')
time.sleep(0.01)
print(f'Average loss:       {avg_loss*100:6.2f}%')
print(f'Payoff ratio:       {payoff:6.2f}')
print(f'Profit factor:      {profit_factor:6.2f}')
print(f'Expectancy/trade:   {expectancy*100:6.2f}%')
print(f'Kelly fraction:     {kelly*100:6.2f}%')
print(f'Time in market:     {time_in_mkt*100:6.1f}%')
print(f'Tail ratio (95/5):  {tail_ratio:6.2f}')
print(f'Max lose streak:    {max_lose_streak:6.0f}')
time.sleep(0.01)

# ---------------------  DAY-BY-DAY EQUITY CURVE  ------------------------------
print('\n----- equity curve (day-by-day) -----')
print('date       close      equity')
for idx, row in df.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d')}  "
          f"{row['close']:>10.2f}  "
          f"{curve[idx]:>10.2f}")
    time.sleep(0.01)
