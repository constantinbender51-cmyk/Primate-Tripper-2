"""
21-Day Smart-Coin BernoulliNB BTC day-trader
steps 1-5 only  (no flip toggle)
"""
import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB

# ---------- config ----------
ROLL   = 21          # 1) 21-day rolling window
PATH   = 'xbtusd_1h_8y.csv'
BUY    = 0.55        # long threshold
SELL   = 0.45        # short threshold
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

# ---------- 2) 5 indicators + tercile bins ----------
def add_features(df):
    df = df.copy()

    # basic bits
    df['price_up'] = (df['close'] > df['open']).astype(int)
    df['vol_up']   = (df['volume'] > df['volume'].shift(1)).astype(int)

    # 1. Money-Flow Index (14)
    def mfi_series(h, l, c, v, n=14):
        typical = (h + l + c) / 3
        money   = typical * v
        pos_m   = money.where(typical >= typical.shift(1), 0).rolling(n).sum()
        neg_m   = money.where(typical <  typical.shift(1), 0).rolling(n).sum()
        return 100 - (100 / (1 + pos_m / (neg_m + 1e-6)))
    df['mfi'] = mfi_series(df['high'], df['low'], df['close'], df['volume'])

    # 2. Accumulation/Distribution
    ad = ((2 * df['close'] - df['high'] - df['low']) /
          (df['high'] - df['low'] + 1e-6) * df['volume'])
    df['ad'] = ad.cumsum()

    # 3. Bollinger-band width
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std(ddof=0)
    df['bbw'] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20

    # 4. Keltner-channel width  (20 EMA ± 2 ATR)
    ema20 = df['close'].ewm(span=20).mean()
    atr   = df[['high','low','close']].apply(
        lambda x: np.maximum(x['high']-x['low'],
                             np.maximum(abs(x['high']-x['close'].shift(1)),
                                        abs(x['low'] -x['close'].shift(1)))), axis=1).ewm(span=20).mean()
    df['kcw'] = (ema20 + 2 * atr - (ema20 - 2 * atr)) / ema20

    # 5. Parabolic-SAR direction  (0 = down, 1 = up)
    sar = df['close'].copy()
    af, ep, uptrend = 0.02, df['high'].iloc[0], True
    sar_vals, trend = [], []
    for i in range(len(df)):
        if i == 0:
            sar_vals.append(sar.iloc[0]); trend.append(1)
            continue
        if uptrend:
            sar_new = sar_vals[-1] + af * (ep - sar_vals[-1])
            if df['low'].iloc[i] < sar_new:          # flip down
                uptrend, af, ep = False, 0.02, df['low'].iloc[i]
            else:
                if df['high'].iloc[i] > ep: ep = df['high'].iloc[i]; af = min(af+0.02, 0.2)
        else:
            sar_new = sar_vals[-1] + af * (ep - sar_vals[-1])
            if df['high'].iloc[i] > sar_new:         # flip up
                uptrend, af, ep = True, 0.02, df['high'].iloc[i]
            else:
                if df['low'].iloc[i] < ep: ep = df['low'].iloc[i]; af = min(af+0.02, 0.2)
        sar_vals.append(sar_new)
        trend.append(int(uptrend))
    df['psar_up'] = trend

    # ---------- tercile bins ----------
    def bin3(ser, name):
        roll = ser.rolling(ROLL)
        q1 = roll.quantile(.33); q2 = roll.quantile(.67)
        cat = pd.cut(ser, bins=[-np.inf, q1, q2, np.inf],
                     labels=[f'{name}_L', f'{name}_M', f'{name}_H'])
        return pd.get_dummies(cat, prefix=name)

    df = pd.concat([df,
                    bin3(df['mfi'],  'mfi'),
                    bin3(df['ad'],   'ad'),
                    bin3(df['bbw'],  'bbw'),
                    bin3(df['kcw'],  'kcw'),
                    bin3(df['psar_up'], 'psar')], axis=1).fillna(0)

    return df

# ---------- build feature list ----------
def feat_cols(df):
    return [c for c in df.columns if c.startswith(('mfi_', 'ad_', 'bbw_', 'kcw_', 'psar_'))] + ['vol_up']

# ---------- walk-forward ----------
def walk_forward(df):
    df = df.dropna()
    model = BernoulliNB(alpha=0.1, fit_prior=True)   # 3) paper params
    feats = feat_cols(df)
    preds = []
    for i in range(ROLL, len(df)-1):
        # train
        X = df[feats].iloc[i-ROLL:i]
        y = (df['close'].diff() > 0).iloc[i-ROLL+1:i+1]  # close-to-close sign
        model.fit(X, y)
        # predict
        x_today = df[feats].iloc[i:i+1]
        prob = model.predict_proba(x_today)[0, 1]

        # 4) MFI filter
        mfi = df['mfi'].iloc[i]
        if mfi < 20 or mfi > 80:
            pos = 0
        else:
            pos = 1 if prob > BUY else -1 if prob < SELL else 0

        preds.append({'date': df.index[i], 'prob': prob, 'pos': pos})
    return pd.DataFrame(preds).set_index('date')

# ---------- slow audit print ----------
def audit_print(audit):
    correct = trades = equity = longs = shorts = flats = 0
    for ts, row in audit.iterrows():
        real_move = np.sign(row['close_next'] - row['close'])
        call_move = np.sign(row['pos'])
        if call_move != 0:
            trades += 1
            if call_move == real_move:
                correct += 1
        equity += row['pnl']
        if row['pos'] == 1:   longs += 1
        elif row['pos'] == -1: shorts += 1
        else:                  flats += 1

        hit_rate = 100 * correct / trades if trades else 0
        avg_pnl = equity / trades if trades else 0
        print(f"{ts.date()} | "
              f"O:{row['open']:7.2f} H:{row['high']:7.2f} L:{row['low']:7.2f} C:{row['close']:7.2f} | "
              f"prob:{row['prob']:.3f} pos:{row['pos']: 2.0f} | "
              f"P&L:{row['pnl']: 7.2f} | "
              f"Hit:{hit_rate:5.1f}%  Trades:{trades}  Eq:{equity: 10.2f}  Avg:{avg_pnl: 6.2f}")
        time.sleep(0.01)

    print("\n=== FINAL ===")
    print(f"Days:{len(audit)}  Long:{longs}  Short:{shorts}  Flat:{flats}")
    print(f"Trades:{trades}  Correct:{correct}  Hit-rate:{100*correct/trades:.2f}%")
    print(f"Equity:{equity:,.2f}  Avg/trade:{equity/trades:,.2f}")

# ---------- one-click ----------
if __name__ == "__main__":
    daily = load_daily(PATH)
    data  = add_features(daily)
    sig   = walk_forward(data)
    audit = data.join(sig).join(data[['open','close']].shift(-1), rsuffix='_next')
    audit['pnl'] = audit['pos'] * (audit['close_next'] - audit['close'])
    audit = audit.dropna(subset=['pnl'])
    audit_print(audit)
