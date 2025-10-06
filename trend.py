# pip install ta
import pandas as pd
import ta

df = pd.read_csv('btc_daily.csv', parse_dates=['date'])

# 1. 20- and 200-period EMAs
df['EMA20']  = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['EMA200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

# 2. Trend direction vs EMA20
df['trend'] = df['close'].gt(df['EMA20']).map({True: 'UP', False: 'DOWN'})

# 3. ADX (14-period Wilder)
df['ADX'] = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).adx()

# 4. Consecutive-count of the same trend
grp = (df['trend'] != df['trend'].shift()).cumsum()
df['consolid'] = df.groupby(grp).cumcount() + 1   # 1,2,3… consecutive
# make it negative when in DOWN trend
df.loc[df['trend'] == 'DOWN', 'consolid'] *= -1

# quick look at the tail
print(df[['date', 'close', 'EMA20', 'EMA200', 'trend', 'ADX', 'consolid']].tail(10))
