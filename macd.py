import pandas as pd, numpy as np, time

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

ema12 = df['close'].ewm(12, adjust=False).mean()
ema26 = df['close'].ewm(26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(9, adjust=False).mean()

cross = np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))

for _, r in df.iterrows():
    if cross[_] != 0:
        print(f"{r.date.strftime('%Y-%m-%d')}  {'BUY' if cross[_] == 1 else 'SELL'}")
        time.sleep(0.01)
