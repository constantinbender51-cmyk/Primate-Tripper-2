# btc_bagging_enhanced.py
import pandas as pd
import ta
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load data -----------------------------------------------------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date']).sort_values('date')
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()

# 2. Base features -------------------------------------------------------------
df['ret'] = df['close'].pct_change()
df['EMA20']  = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['EMA200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

# trend vs EMA20 + consecutive counter
df['trend'] = df['close'].gt(df['EMA20']).map({True: 'UP', False: 'DOWN'})
grp = (df['trend'] != df['trend'].shift()).cumsum()
df['consolid'] = df.groupby(grp).cumcount() + 1
df.loc[df['trend'] == 'DOWN', 'consolid'] *= -1

# 3. Order-flow features -------------------------------------------------------
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
df['EOM'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()

# normalise order-flow
for col in ['OBV', 'CMF', 'EOM']:
    df[col] = df[col] / df[col].rolling(252).std()

# 4. Regime & calendar ---------------------------------------------------------
df['vol30'] = df['ret'].rolling(30).std()
median_vol = df['vol30'].expanding().median()
df['high_vol'] = (df['vol30'] > median_vol).astype(int)

df['dow']  = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# one-hot encode dow & month
df = pd.get_dummies(df, columns=['dow', 'month'], prefix=['dow', 'm'])

# 5. Feature list --------------------------------------------------------------
price_feats = ['ret', 'close-EMA20', 'close-EMA200', 'ADX', 'ATR', 'consolid']
flow_feats  = ['OBV', 'CMF', 'EOM']
regime_cal  = [c for c in df.columns if c.startswith(('high_vol', 'dow_', 'm_'))]

FEATS = price_feats + flow_feats + regime_cal

# create price distance features
df['close-EMA20']  = (df['close'] - df['EMA20']) / df['EMA20']
df['close-EMA200'] = (df['close'] - df['EMA200']) / df['EMA200']

# 6. Lags ----------------------------------------------------------------------
for feat in price_feats + flow_feats:
    for lag in range(1, 4):
        df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

# 7. Target transformation -----------------------------------------------------
df['y_raw'] = df['ret'].shift(-1)
df['y'] = np.sign(df['y_raw']) * np.sqrt(np.abs(df['y_raw']))

# 8. Drop NaN ------------------------------------------------------------------
df = df.dropna()
X_cols = [c for c in df.columns if c.startswith(tuple(price_feats + flow_feats + regime_cal))]

# 9. Walk-forward split --------------------------------------------------------
test_split = int(len(df) * 0.8)
train_df = df.iloc[:test_split].copy()
test_df  = df.iloc[test_split:].copy()

# 10. Recency oversample (last 6 months ×3) -----------------------------------
n = len(train_df)
ix = np.arange(n)
ix_recent = np.random.choice(ix[-180:], size=2*180, replace=True)
ix = np.concatenate([ix, ix_recent])
train_df = train_df.iloc[ix].copy()

# 11. Scale --------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[X_cols])
y_train = train_df['y'].values

X_test  = scaler.transform(test_df[X_cols])
y_test  = test_df['y'].values

# 12. Model --------------------------------------------------------------------
base = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
model = BaggingRegressor(
    estimator=base,
    n_estimators=500,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 13. Calibration --------------------------------------------------------------
pred_val = model.predict(X_train)
cal = LinearRegression().fit(pred_val.reshape(-1, 1), y_train)

# 14. Predict ------------------------------------------------------------------
pred = model.predict(X_test)
pred = cal.predict(pred.reshape(-1, 1))

# back-transform to return space
pred_ret = np.sign(pred) * (pred ** 2)

# 15. Evaluation ---------------------------------------------------------------
test_df = test_df.copy()
test_df['pred_ret'] = pred_ret
test_df['pred_close'] = test_df['close'] * (1 + pred_ret)

dir_acc = (np.sign(test_df['y_raw']) == np.sign(test_df['pred_ret'])).mean()
mae = np.abs(test_df['y_raw'] - test_df['pred_ret']).mean()

print(f"Directional accuracy on test: {dir_acc:.2%}")
print(f"MAE next-bar return: {mae:.4f}")

# 16. Last 10 rows -------------------------------------------------------------
cols = ['date', 'close', 'y_raw', 'pred_ret', 'pred_close']
print(test_df[cols].tail(10).to_string(index=False))
