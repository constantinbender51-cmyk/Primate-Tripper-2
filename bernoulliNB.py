import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB

# ---------- config ----------
ROLL = 252
FEATS = ['price_up', 'vol_up', 'sma2_up', 'sma5_up',
         'sma10_up', 'sma20_up', 'sma50_up']
BUY, SELL = 0.55, 0.45
PATH = 'xbtusd_1h_8y.csv'          # your hourly source
# ------------------------------

def load_daily(path):
    df = pd.read_csv(path)
    df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    daily = (df.resample('D')
               .agg({'open': 'first', 'high': 'max', 'low': 'min',
                     'close': 'last', 'volume': 'sum'}).dropna())
    return daily

def add_features(df):
    df = df.copy()
    df['price_up'] = (df['close'] > df['open']).astype(int)
    df['vol_up']   = (df['volume'] > df['volume'].shift(1)).astype(int)
    for n in [2, 5, 10, 20, 50]:
        sma = df['close'].rolling(n).mean()
        df[f'sma{n}_up'] = (sma > sma.shift(1)).astype(int)
    return df

def build_signals(df):
    df = df.dropna().copy()
    model = BernoulliNB()
    signals = []
    for i in range(ROLL, len(df)):
        X = df[FEATS].iloc[i-ROLL:i]
        y = df['price_up'].iloc[i-ROLL+1:i+1]
        model.fit(X, y)
        x_today = df[FEATS].iloc[i:i+1]
        prob = model.predict_proba(x_today)[0, 1]
        pos = 1 if prob > BUY else -1 if prob < SELL else 0
        signals.append({'date': df.index[i], 'prob': prob, 'pos': pos})
    sig = pd.DataFrame(signals).set_index('date')
    return sig

def main():
    daily = load_daily(PATH)
    data  = add_features(daily)
    sig   = build_signals(data)

    # merge signal + next-day prices for P&L
    audit = data.join(sig).join(data[['open','close']].shift(-1), rsuffix='_next')
    audit['pnl'] = audit['pos'] * (audit['close_next'] - audit['open_next'])
    audit = audit.dropna(subset=['pnl'])

    # slow print
    for ts, row in audit.iterrows():
        print(f"{ts.date()} | "
              f"O:{row['open']:7.2f} H:{row['high']:7.2f} L:{row['low']:7.2f} C:{row['close']:7.2f} | "
              f"prob:{row['prob']:.3f} pos:{row['pos']: 2.0f} | "
              f"P&L:{row['pnl']: 7.2f}")
        time.sleep(0.01)

if __name__ == "__main__":
    main()
