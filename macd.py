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

cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# =======================  6.7 % STOP  ––  WITH 5× LEVERAGE  =================
STOP_PCT  = 6.7
LEVERAGE  = 5
curve_5x  = [10000]
in_pos    = 0
entry_p   = None
entry_d   = None
trades_5x = []

for i in range(1, len(df)):
    p_prev = df['close'].iloc[i-1]
    p_now  = df['close'].iloc[i]
    pos_i  = pos.iloc[i]

    # enter
    if in_pos == 0 and pos_i != 0:
        in_pos, entry_p, entry_d = pos_i, p_now, df['date'].iloc[i]

    # ---------- 6.7 % intrabar stop (checked BEFORE leverage) -----------------
    if in_pos != 0:
        if in_pos == 1:                           # long
            stop_price = entry_p * (1 - STOP_PCT/100)
            if df['low'].iloc[i] <= stop_price:
                trades_5x.append((entry_d, df['date'].iloc[i], -STOP_PCT/100))
                in_pos = 0
        else:                                     # short
            stop_price = entry_p * (1 + STOP_PCT/100)
            if df['high'].iloc[i] >= stop_price:
                trades_5x.append((entry_d, df['date'].iloc[i], -STOP_PCT/100))
                in_pos = 0

    # exit on opposite MACD cross
    if in_pos != 0 and pos_i == -in_pos:
        ret = (p_now / entry_p - 1) * in_pos
        trades_5x.append((entry_d, df['date'].iloc[i], ret))
        in_pos = pos_i
        entry_p, entry_d = p_now, df['date'].iloc[i]

    # equity update: raw return × leverage × position flag
    raw_ret = (p_now / p_prev - 1)
    curve_5x.append(curve_5x[-1] * (1 + raw_ret * LEVERAGE * in_pos))

curve_5x = pd.Series(curve_5x, index=df.index)

# ---------------------------  METRICS  ----------------------------------------
daily_ret_5x = curve_5x.pct_change().dropna()
trades_ret_5x = pd.Series([t[2] for t in trades_5x])
n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

cagr_5x = (curve_5x.iloc[-1] / curve_5x.iloc[0]) ** (1 / n_years) - 1
vol_5x  = daily_ret_5x.std() * np.sqrt(252)
sharpe_5x = cagr_5x / vol_5x if vol_5x else np.nan
drawdown_5x = curve_5x / curve_5x.cummax() - 1
maxdd_5x = drawdown_5x.min()
calmar_5x = cagr_5x / abs(maxdd_5x) if maxdd_5x else np.nan

wins_5x   = trades_ret_5x[trades_ret_5x > 0]
losses_5x = trades_ret_5x[trades_ret_5x < 0]
win_rate_5x = len(wins_5x) / len(trades_ret_5x) if trades_ret_5x.size else 0
avg_win_5x  = wins_5x.mean()   if len(wins_5x)   else 0
avg_loss_5x = losses_5x.mean() if len(losses_5x) else 0
payoff_5x   = abs(avg_win_5x / avg_loss_5x) if avg_loss_5x else np.nan
profit_factor_5x = wins_5x.sum() / abs(losses_5x.sum()) if losses_5x.sum() else np.nan
expectancy_5x = win_rate_5x * avg_win_5x - (1 - win_rate_5x) * abs(avg_loss_5x)
kelly_5x = expectancy_5x / trades_ret_5x.var() if trades_ret_5x.var() > 0 else np.nan
time_in_mkt = (pos != 0).mean()
tail_5x = (np.percentile(daily_ret_5x, 95) /
           abs(np.percentile(daily_ret_5x, 5))) if daily_ret_5x.size else np.nan
trades_per_year_5x = len(trades_5x) / n_years
lose_streak_5x = (trades_ret_5x < 0).astype(int)
max_lose_streak_5x = lose_streak_5x.groupby(
                          lose_streak_5x.diff().ne(0).cumsum()).sum().max()

# -------------------- BUY & HOLD 1× and 5× ------------------------------------
final_bnh_1x = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
daily_bnh  = df['close'].pct_change().dropna()
bnh_5x_curve = 10000 * (1 + daily_bnh * LEVERAGE).cumprod()
final_bnh_5x = bnh_5x_curve.iloc[-1]
bnh_5x_dd    = (bnh_5x_curve / bnh_5x_curve.cummax() - 1).min()

# -------------------- SIDE-BY-SIDE PRINT --------------------------------------
def fmt(d): return f"{d:.2f}"

print(f'\n========  6.7 % STOP  ––  1× vs 5× LEVERAGE  ========')
print(f"{'Metric':<20} {'1× MACD':<10} {'5× MACD':<10} {'1× B&H':<10} {'5× B&H':<10}")
print('-'*60)
print(f"{'Final equity':<20} {final_macd:>10,.0f} {curve_5x.iloc[-1]:>10,.0f} "
      f"{final_bnh_1x:>10,.0f} {final_bnh_5x:>10,.0f}")
print(f"{'CAGR %':<20} {fmt(cagr*100):>10} {fmt(cagr_5x*100):>10} "
      f"{fmt((final_bnh_1x/10000)**(1/n_years)*100-100):>10} {fmt((final_bnh_5x/10000)**(1/n_years)*100-100):>10}")
print(f"{'Ann. vol %':<20} {fmt(vol*100):>10} {fmt(vol_5x*100):>10} "
      f"{fmt(daily_bnh.std()*np.sqrt(252)*100):>10} {fmt(daily_bnh.std()*np.sqrt(252)*LEVERAGE*100):>10}")
print(f"{'Sharpe':<20} {fmt(sharpe):>10} {fmt(sharpe_5x):>10} "
      f"{fmt(((final_bnh_1x/10000)**(1/n_years)-1)/(daily_bnh.std()*np.sqrt(252))):>10} "
      f"{fmt(((final_bnh_5x/10000)**(1/n_years)-1)/(daily_bnh.std()*np.sqrt(252)*LEVERAGE)):>10}")
print(f"{'Max DD %':<20} {fmt(maxdd*100):>10} {fmt(maxdd_5x*100):>10} "
      f"{fmt((df['close']/df['close'].cummax()-1).min()*100):>10} {fmt(bnh_5x_dd*100):>10}")
print(f"{'Calmar':<20} {fmt(calmar):>10} {fmt(calmar_5x):>10} "
      f"{fmt(((final_bnh_1x/10000)**(1/n_years)-1)/abs((df['close']/df['close'].cummax()-1).min())):>10} "
      f"{fmt(((final_bnh_5x/10000)**(1/n_years)-1)/abs(bnh_5x_dd)):>10}")
print(f"{'Trades/year':<20} {fmt(trades_per_year):>10} {fmt(trades_per_year_5x):>10} {'-':>10} {'-':>10}")
print(f"{'Win-rate %':<20} {fmt(win_rate*100):>10} {fmt(win_rate_5x*100):>10} {'-':>10} {'-':>10}")
print(f"{'Expectancy/trade %':<20} {fmt(expectancy*100):>10} {fmt(expectancy_5x*100):>10} {'-':>10} {'-':>10}")
print(f"{'Kelly fraction %':<20} {fmt(kelly*100):>10} {fmt(kelly_5x*100):>10} {'-':>10} {'-':>10}")
print(f"{'Time in mkt %':<20} {fmt(time_in_mkt*100):>10} {fmt(time_in_mkt*100):>10} {'100.0':>10} {'100.0':>10}")
print('='*60)

# ---------------------  DAY-BY-DAY 5× EQUITY CURVE  ---------------------------
print('\n----- 5× leverage equity curve (day-by-day) -----')
print('date       close      equity')
for idx, row in df.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d')}  "
          f"{row['close']:>10.2f}  "
          f"{curve_5x[idx]:>10.2f}")
    time.sleep(0.01)          # 10 ms pacing
