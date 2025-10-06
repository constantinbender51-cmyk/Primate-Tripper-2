import pandas as pd
import numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD with proper warm-up -------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

# 1/-1 on cross, 0 before any signal
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# ---- equity curve -----------------------------------------------------------
ret = df['close'].pct_change()
curve = (1 + pos.shift() * ret).cumprod() * 10000

# ---- trades -----------------------------------------------------------------
trades = []          # (entry_date, exit_date, return)
in_pos = 0
for i, p in enumerate(pos):
    if in_pos == 0 and p != 0:
        in_pos, entry_p, entry_d = p, df['close'].iloc[i], df['date'].iloc[i]
    elif in_pos != 0 and p == -in_pos:
        r = (df['close'].iloc[i] / entry_p - 1) * in_pos
        trades.append((entry_d, df['date'].iloc[i], r))
        in_pos = p
        entry_p, entry_d = df['close'].iloc[i], df['date'].iloc[i]

# ---- metrics -----------------------------------------------------------------
final_macd = curve.iloc[-1]
final_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
worst      = min(trades, key=lambda x: x[2])
maxbal     = curve.cummax()

print(f"MACD final: €{final_macd:,.0f}")
print(f"B&H final:  €{final_hold:,.0f}")
print(f"Worst trade: {worst[2]*100:.1f}% (exit {worst[1].strftime('%Y-%m-%d')})")
print(f"Max drawdown: {(curve/maxbal - 1).min()*100:.1f}%")
