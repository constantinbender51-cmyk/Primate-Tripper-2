import pandas as pd, numpy as np

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# MACD
macd = df['close'].ewm(12).mean() - df['close'].ewm(26).mean()
signal = macd.ewm(9).mean()

# crossings
cross = np.where((macd > signal) & (macd.shift() <= signal.shift()), 1,
                np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, 0))

last5 = df.loc[cross != 0, ['date']].assign(dir=cross[cross != 0]).tail(5)
for _, r in last5.iterrows():
    print(r.date.strftime('%Y-%m-%d'), 'BUY' if r.dir == 1 else 'SELL')
