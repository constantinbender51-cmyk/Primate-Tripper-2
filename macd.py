import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD ----------------------------------------------------------
ema12 = df['close'].ewm(12, adjust=False).mean()
ema26 = df['close'].ewm(26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(9, adjust=False).mean()

# 1/-1 on cross
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
cross = pd.Series(cross, index=df.index)

# position series
pos = cross.replace(0, np.nan).ffill().fillna(0)

# ---- trades --------------------------------------------------------
trades = []
in_pos = 0
for i, p in enumerate(pos):
    if in_pos == 0 and p != 0:                       # enter
        in_pos, entry_p, entry_d = p, df['close'].iloc[i], df['date'].iloc[i]
    elif in_pos != 0 and p == -in_pos:               # exit/flip
        ret = (df['close'].iloc[i] / entry_p - 1) * in_pos
        trades.append((entry_d, df['date'].iloc[i], ret))
        in_pos = p
        entry_p, entry_d = df['close'].iloc[i], df['date'].iloc[i]

trades = pd.DataFrame(trades, columns=['entry_dt', 'exit_dt', 'ret'])

# ---- worst trade ---------------------------------------------------
worst = trades.loc[trades['ret'].idxmin()]
w_date = worst.exit_dt

# 5 trades around worst
idx = trades.index.get_loc(trades['ret'].idxmin())
around = trades.iloc[max(idx-2,0):idx+3]

print("5 trades around worst:")
for _, t in around.iterrows():
    print(f"{t.exit_dt.strftime('%Y-%m-%d')}  {t.ret*100:+.1f}%")

# ---- crossings around worst ----------------------------------------
print("\nCrossovers ±10 days around worst exit:")
mask = (df['date'] >= w_date - pd.Timedelta(days=10)) & \
       (df['date'] <= w_date + pd.Timedelta(days=10))
subs = df.loc[mask].copy()
subs['cross'] = cross.loc[mask]
for _, r in subs.iterrows():
    if r.cross != 0:
        print(f"{r.date.strftime('%Y-%m-%d')}  {'BUY' if r.cross == 1 else 'SELL'}")
