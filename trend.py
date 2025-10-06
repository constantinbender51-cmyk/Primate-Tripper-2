import pandas as pd
df = pd.read_csv('XBTUSD_daily.csv', parse_dates=['date'])
df['SMA20'] = df['close'].rolling(20).mean()
df['trend'] = df['close'].gt(df['SMA20']).map({True: 'UP', False: 'DOWN'})
print(df[['date','close','trend']].tail())
