import pandas as pd
import numpy as np

# 1. load data
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# 2. MACD(12,26,9)
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9).mean()

# 3. positions: 1 on bullish cross, -1 on bearish, hold otherwise
pos = np.where(macd > signal, 1, np.where(macd < signal, -1, np.nan))
pos = pd.Series(pos, index=df.index).ffill().fillna(0)

# 4. back-test
capital = 10_000
btc = capital / df['close'].iloc[0]          # buy & hold size
hold = btc * df['close']                     # buy & hold curve

pct = df['close'].pct_change()
strat = (1 + pos.shift() * pct).cumprod() * capital

# 5. metrics
max_bal = strat.cummax()
drawdown = (strat - max_bal) / max_bal
trades = strat[pos.diff().abs() == 2]        # round-trip exits
trade_ret = trades.pct_change()
max_neg = trade_ret.min() * 100              # worst trade

print(f"Final MACD: €{strat.iloc[-1]:,.0f}")
print(f"Final HOLD: €{hold.iloc[-1]:,.0f}")
print(f"Max trade loss: {max_neg:.1f}%")
print(f"Max drawdown: {drawdown.min()*100:.1f}%")
