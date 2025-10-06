import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# MACD
ema12 = df['close'].ewm(12).mean()
ema26 = df['close'].ewm(26).mean()
macd = ema12 - ema26
signal = macd.ewm(9).mean()

# 1/-1 on cross
pos = np.where((macd > signal) & (macd.shift() <= signal.shift()), 1,
              np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, np.nan))
pos = pd.Series(pos, index=df.index).ffill().fillna(0)

# back-test with 2 % stop-loss
cash = 10_000
strat = [cash]
in_pos = 0
entry = np.nan
for i, p in enumerate(pos[1:], 1):
    if in_pos == 0 and p != 0:          # enter
        in_pos, entry = p, df.close.iloc[i]
    elif in_pos != 0:
        chg = df.close.iloc[i] / entry - 1
        if (in_pos == 1 and chg <= -0.02) or (in_pos == -1 and chg >= 0.02):
            in_pos = 0                  # stopped out
        elif p == -in_pos:              # opposite signal
            in_pos = p
            entry = df.close.iloc[i]
    strat.append(strat[-1] * (1 + (df.close.iloc[i]/df.close.iloc[i-1]-1)*in_pos))

strat = pd.Series(strat, index=df.index)
hold = (df.close / df.close.iloc[0]) * cash

maxbal = strat.cummax()
print(f"Final MACD: €{strat.iloc[-1]:,.0f}")
print(f"Final HOLD: €{hold.iloc[-1]:,.0f}")
print(f"Max drawdown: {(strat/maxbal-1).min()*100:.1f}%")
