import pandas as pd
import numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# ---- MACD (proper warm-up) ----------------------------------------------------
ema12 = df['close'].ewm(span=12, min_periods=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, min_periods=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))
pos = pd.Series(cross, index=df.index).replace(0, np.nan).ffill().fillna(0)

# ---- back-test with 2 % stop --------------------------------------------------
curve = [10000]
in_pos = 0
entry_p = None
trades = []

# month-end balances
month_end_bal = []          # list of (date, equity)

for i in range(1, len(df)):
    p_prev, p_now = df['close'].iloc[i-1], df['close'].iloc[i]
    pos_i = pos.iloc[i]

    # enter
    if in_pos == 0 and pos_i != 0:
        in_pos, entry_p, entry_d = pos_i, p_now, df['date'].iloc[i]

    # stop / cross exit
    if in_pos != 0:
        ret = (p_now / entry_p - 1) * in_pos
        if ret <= -0.02:
            trades.append((entry_d, df['date'].iloc[i], -0.02))
            in_pos = 0
        elif pos_i == -in_pos:
            trades.append((entry_d, df['date'].iloc[i], ret))
            in_pos = pos_i
            entry_p, entry_d = p_now, df['date'].iloc[i]

    # equity update
    curve.append(curve[-1] * (1 + (p_now/p_prev - 1) * in_pos))

    # month-end snapshot
    if i == len(df)-1 or df['date'].iloc[i].month != df['date'].iloc[i+1].month:
        month_end_bal.append((df['date'].iloc[i], curve[-1]))

curve = pd.Series(curve, index=df.index)

# ---- metrics -----------------------------------------------------------------
final_macd = curve.iloc[-1]
final_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) * 10000
worst      = min(trades, key=lambda x: x[2])
maxbal     = curve.cummax()

# month-end stats
me_dates, me_eq = zip(*month_end_bal)
me_series = pd.Series(me_eq, index=me_dates)
mean_me   = me_series.mean()
worst_me  = me_series.min()

print(f"MACD+2%SL final: €{final_macd:,.0f}")
print(f"B&H final:       €{final_hold:,.0f}")
print(f"Worst trade:     {worst[2]*100:.1f}% (exit {worst[1].strftime('%Y-%m-%d')})")
print(f"Max drawdown:    {(curve/maxbal - 1).min()*100:.1f}%")
print(f"Mean month-end:  €{mean_me:,.0f}")
print(f"Worst month-end: €{worst_me:,.0f}")
