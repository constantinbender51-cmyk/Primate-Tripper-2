# btc_bagging_nextbar_v2.py
import pandas as pd
import ta
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load data ----------------------------------------------------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date']).sort_values('date')
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()

# 2. Build requested indicators ----------------------------------------------
# 1. Accumulation / Distribution Index
df['ADI'] = ta.volume.AccDistIndexIndicator(
                high=df['high'], low=df['low'],
                close=df['close'], volume=df['volume']).acc_dist_index()

# 2. Money Flow Index
df['MFI'] = ta.volume.MFIIndicator(
                high=df['high'], low=df['low'],
                close=df['close'], volume=df['volume'],
                window=14).money_flow_index()

# 3. Bollinger Bands → keep width only (already normalised by close)
bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']

# 4. Keltner Channel Width
kc = ta.volatility.KeltnerChannel(
        high=df['high'], low=df['low'], close=df['close'],
        window=20, window_atr=10)
df['KC_width'] = (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / df['close']

# 5. Parabolic SAR (distance in %)
psar = ta.trend.PSARIndicator(
        high=df['high'], low=df['low'], close=df['close'],
        step=0.02, max_step=0.2)
df['PSAR_dist'] = (psar.psar() - df['close']) / df['close']

# 3. Feature list ------------------------------------------------------------
FEATS = ['open', 'high', 'low', 'close', 'volume',
         'ADI', 'MFI', 'BB_width', 'KC_width', 'PSAR_dist']

# 3 lags of every feature
for feat in FEATS:
    for lag in range(1, 4):
        df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

# target: next-bar return
df['y'] = df['close'].pct_change().shift(-1)
df = df.dropna()

# 4. Walk-forward split ------------------------------------------------------
test_split = int(len(df) * 0.8)
train_df = df.iloc[:test_split].copy()
test_df  = df.iloc[test_split:].copy()

X_cols = [c for c in df.columns if any(c.startswith(f) for f in FEATS)]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[X_cols])
y_train = train_df['y'].values

X_test = scaler.transform(test_df[X_cols])
y_test = test_df['y'].values

# 5. Model --------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

base = DecisionTreeRegressor(random_state=42)

grid = {
    'estimator__max_depth':        [4, 5],
    'estimator__min_samples_leaf': [20, 40],
    'estimator__max_features':     [0.7, 0.8],
    'n_estimators':                [500, 1000],
    'max_samples':                 [0.8],
    'max_features':                [0.8]
}          # 2×2×2×2 = 16 fits instead of 3 200

bag = BaggingRegressor(estimator=base, random_state=42, n_jobs=-1)

model = GridSearchCV(bag, grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
model.fit(X_train, y_train)

print('Best params:', model.best_params_)
model = model.best_estimator_

# 6. Predict ------------------------------------------------------------------
pred_ret = model.predict(X_test)
pred_close = test_df['close'].values * (1 + pred_ret)

# 7. Quick evaluation ---------------------------------------------------------
test_df = test_df.copy()
test_df['pred_ret'] = pred_ret
test_df['pred_close'] = pred_close

# directional accuracy
test_df['dir_real'] = np.sign(test_df['y'])
test_df['dir_pred'] = np.sign(test_df['pred_ret'])
dir_acc = (test_df['dir_real'] == test_df['dir_pred']).mean()
print(f"Directional accuracy on test: {dir_acc:.2%}")

# MAE of return forecast
mae = np.abs(test_df['y'] - test_df['pred_ret']).mean()
print(f"MAE next-bar return: {mae:.4f}")

# 8. Show last few predictions -----------------------------------------------
cols = ['date', 'close', 'y', 'pred_ret', 'pred_close']
print(test_df[cols].tail())
