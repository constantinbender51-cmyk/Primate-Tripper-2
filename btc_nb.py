#!/usr/bin/env python3
"""
btc_nb.py
Train BernoulliNB on BTC daily booleans, leveraged back-test,
print CSV to stdout with 0.01 s delay per line.
"""
import time, sys, warnings, io
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------ #
# 1. LOAD DAILY CSV
# ------------------------------------------------------------------ #
def load_daily(csv_path='btc_daily.csv'):
    df = pd.read_csv(csv_path, parse_dates=['date']).sort_values('date')
    df = df[df['date'].between('2017-09-07', '2025-09-05')].reset_index(drop=True)
    return df

# ------------------------------------------------------------------ #
# 2. BOOLEAN FEATURE ENGINEERING
# ------------------------------------------------------------------ #
def add_features(df):
    df = df.copy()
    df['ret'] = np.log(df['close']).diff()
    df['vol'] = df['volume'].rolling(21).median()
    
    # 1-sign features
    df['ret_sign_t0'] = (df['ret'] >= 0).astype(np.uint8)
    df['ret_sign_t1'] = df['ret_sign_t0'].shift(1).fillna(0).astype(np.uint8)
    
    # volume spike
    df['vol_spike'] = (df['volume'] > df['vol']).astype(np.uint8)
    
    # Bollinger touches (21,2)
    ma21 = df['close'].rolling(21).mean()
    std21 = df['close'].rolling(21).std()
    df['bb_hi'] = (df['high'] >= ma21 + 2*std21).astype(np.uint8)
    df['bb_lo'] = (df['low']  <= ma21 - 2*std21).astype(np.uint8)
    
    # MFI (simplified with typical price)
    tp = (df['high']+df['low']+df['close'])/3
    rmf = tp * df['volume']
    pos_flow = rmf.where(tp >= tp.shift(1), 0).rolling(14).sum()
    neg_flow = rmf.where(tp <  tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100/(1 + pos_flow/neg_flow)
    df['mfi_hi'] = (mfi >= 80).astype(np.uint8)
    df['mfi_lo'] = (mfi <= 20).astype(np.uint8)
    
    # Parabolic SAR bull (dummy: close > prev close 5 days)
    df['psar_bull'] = (df['close'] > df['close'].shift(1)).rolling(5).sum() == 5
    df['psar_bull'] = df['psar_bull'].astype(np.uint8)
    
    # optional flags (can be all 0)
    df['volatility_spike'] = 0
    df['funding_heat'] = 0
    
    return df

# ------------------------------------------------------------------ #
# 3. TRAIN / SPLIT
# ------------------------------------------------------------------ #
def split_and_fit(df):
    feature_cols = ['ret_sign_t0','ret_sign_t1','vol_spike','bb_hi','bb_lo','mfi_hi','mfi_lo','psar_bull','volatility_spike','funding_heat']
    df['y'] = (df['ret'].shift(-1) >= 0).astype(np.uint8)
    df = df.dropna(subset=feature_cols+['y'])
    
    train_mask = df['date'] < '2025-01-01'
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'y']
    
    # tiny grid-search on alpha
    best_pnl = -np.inf
    best_clf = None
    for alpha in [1e-3, 1e-2, 0.1, 1.0]:
        clf = BernoulliNB(alpha=alpha, fit_prior=True)
        clf.fit(X_train, y_train)
        # quick PNL eval on last 252 train days to pick alpha
        last252 = train_mask[train_mask].tail(252)
        pnl = simulate_pnl(df.loc[last252.index], clf, feature_cols, leverage=1)
        if pnl > best_pnl:
            best_pnl, best_clf = pnl, clf
    return best_clf, feature_cols

# ------------------------------------------------------------------ #
# 4. BACK-TEST SIMULATION
# ------------------------------------------------------------------ #
def simulate_pnl(df_slice, clf, feature_cols, leverage):
    """Return total PNL over slice (1× only, for alpha pick)"""
    if df_slice.empty: return 0
    X = df_slice[feature_cols]
    prob = clf.predict_proba(X)[:,1]
    signal = (prob >= 0.5).astype(np.uint8)
    ret = df_slice['ret'].shift(-1).fillna(0)
    fee = 0.002
    pnl = signal * (ret - fee) + (1-signal) * (-ret - fee)
    return float(pnl.sum())

# ------------------------------------------------------------------ #
# 5. LEVERAGED RUN WITH LIQUIDATION & STOP
# ------------------------------------------------------------------ #
def backtest_leveraged(df, clf, feature_cols):
    test = df[df['date'] >= '2025-01-01'].copy()
    X = test[feature_cols]
    prob = clf.predict_proba(X)[:,1]
    test['signal'] = (prob >= 0.5).astype(np.uint8)
    test['ret_btc'] = test['ret'].shift(-1).fillna(0)
    test['fee'] = 0.002
    test['funding'] = 0.0002   # ≈ 7 % APR in daily units
    
    lev_eq = {1: 1.0, 2: 1.0, 5: 1.0}
    peak = {1: 1.0, 2: 1.0, 5: 1.0}
    liquidated = {1: False, 2: False, 5: False}
    stopped = {1: False, 2: False, 5: False}
    cooldown = {1: 0, 2: 0, 5: 0}
    
    rows = []
    for _, r in test.iterrows():
        row = [r['date'].strftime('%Y-%m-%d'),
               r['signal'],
               f"{prob[len(rows)]:.4f}",   # <- numpy array, use [] not .iloc
               f"{r['ret_btc']:.6f}",
               f"{r['funding']:.6f}",
               f"{r['fee']:.6f}"]
        for lev in [1,2,5]:
            if liquidated[lev]:
                pnl, eq = 0.0, 0.0
            elif stopped[lev] and cooldown[lev] > 0:
                pnl, eq = 0.0, lev_eq[lev]
                cooldown[lev] -= 1
                if cooldown[lev] == 0: stopped[lev] = False
            else:
                raw = r['signal'] * (r['ret_btc'] - r['fee'] - r['funding']) + (1-r['signal']) * (-r['ret_btc'] - r['fee'] + r['funding'])
                pnl = lev * raw
                new_eq = lev_eq[lev] * (1 + pnl)
                # soft stop
                if new_eq / peak[lev] - 1 <= -0.10:
                    stopped[lev], cooldown[lev] = True, 3
                # hard liquidation
                if lev * abs(raw) >= 0.5:
                    liquidated[lev], new_eq = True, 0.0
                peak[lev] = max(peak[lev], new_eq)
                eq = new_eq
                lev_eq[lev] = eq
            row += [f"{pnl:.6f}", f"{eq:.6f}", "True" if (lev==2 and stopped[2] and cooldown[2]==3) else "False",
                    "True" if (lev==5 and stopped[5] and cooldown[5]==3) else "False", "True" if liquidated[5] else "False"]
        rows.append(row)
    return rows, {lev: lev_eq[lev] for lev in [1,2,5]}

# ------------------------------------------------------------------ #
# 6. MAIN
# ------------------------------------------------------------------ #
def main():
    df = load_daily()
    df = add_features(df)
    clf, feat_cols = split_and_fit(df)
    rows, finals = backtest_leveraged(df, clf, feat_cols)
    
    # CSV to stdout with 0.01 s delay
    header = "date,signal,prob_long,ret_btc,funding,fee,lev1_pnl,lev1_equity,lev2_pnl,lev2_equity,lev2_stop,lev5_pnl,lev5_equity,lev5_stop,lev5_liquidated"
    print(header)
    for r in rows:
        print(",".join(r))
        time.sleep(0.01)
    
    # trailer
    liq5 = sum(1 for r in rows if r[-1] == "True")
    stop5 = sum(1 for r in rows if r[-2] == "True")
    stop2 = sum(1 for r in rows if r[-3] == "True")
    trailer = f"SUMMARY,{finals[1]:.4f},{finals[2]:.4f},{finals[5]:.4f},0,{liq5},{stop2},{stop5}"
    time.sleep(0.01)
    print(trailer)

if __name__ == "__main__":
    main()
