import pandas as pd
import numpy as np
from itertools import product
import time, pathos.multiprocessing as mp   # pip install pathos (drop-in replacement for std-lib multiprocessing)

# ------------------------------------------------ data -------------------------------------------------
df = pd.read_csv('btc_daily.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ------------------------------------------------ MACD -------------------------------------------------
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
macd  = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()

#  1 / -1 / 0  on cross  (explicit dtype -> float64)
pos = pd.Series(
        np.where((macd > signal) & (macd.shift() <= signal.shift()),  1,
        np.where((macd < signal) & (macd.shift() >= signal.shift()), -1, np.nan)),
        index=df.index,
        dtype='float64'
      ).ffill().fillna(0)

# ------------------------------------------- single back-test -----------------------------------------
def one_run(stop_pct, leverage):
    """
    Returns a dict with all key metrics for ONE (stop, leverage) pair.
    Vectorised core – no python loop over 3650 rows.
    """
    stop_frac = stop_pct / 100
    # 1. raw signal position
    raw_pos = pos * leverage
    # 2. stop-trigger mask
    long_stop  = (df['low']  <= df['close'].shift() * (1 - stop_frac)) & (raw_pos > 0)
    short_stop = (df['high'] >= df['close'].shift() * (1 + stop_frac)) & (raw_pos < 0)
    stopped = long_stop | short_stop
    # 3. set position to 0 on stopped days
    pos_adj = raw_pos.where(~stopped, 0)
    # 4. daily returns
    day_ret = df['close'].pct_change() * pos_adj.shift().fillna(0)
    curve = (1 + day_ret).cumprod()
    curve.iloc[0] = 1
    # 5. trade list (entries & exits)
    flat = (pos_adj == 0)
    entries = (~flat & flat.shift(1).fillna(True))
    exits   = (flat & (~flat).shift(1).fillna(True))
    trades_ret = []
    for en, ex in zip(df.index[entries], df.index[exits]):
        trades_ret.append(day_ret.loc[en:ex].sum())
    trades_ret = pd.Series(trades_ret)
    # 6. metrics
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = curve.iloc[-1] ** (1/n_years) - 1
    vol  = day_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol else np.nan
    dd = (curve / curve.cummax() - 1)
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd else np.nan
    win_rate = (trades_ret > 0).mean() if len(trades_ret) else np.nan
    avg_win  = trades_ret[trades_ret>0].mean()
    avg_loss = trades_ret[trades_ret<0].mean()
    payoff   = abs(avg_win/avg_loss) if pd.notna(avg_loss) and avg_loss!=0 else np.nan
    pf = trades_ret[trades_ret>0].sum() / abs(trades_ret[trades_ret<0].sum()) if trades_ret[trades_ret<0].sum() else np.nan
    exp = win_rate*avg_win - (1-win_rate)*abs(avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) else np.nan
    kelly = exp / trades_ret.var() if trades_ret.var()>0 else np.nan
    time_mkt = (pos_adj != 0).mean()
    tail = np.percentile(day_ret.dropna(),95) / abs(np.percentile(day_ret.dropna(),5)) if len(day_ret.dropna()) else np.nan
    tpy = len(trades_ret) / n_years
    lose_streak = (trades_ret < 0).astype(int)
    max_lose = lose_streak.groupby(lose_streak.diff().ne(0).cumsum()).sum().max() if len(lose_streak) else 0

    return dict(
        stop=stop_pct, lev=leverage,
        CAGR=cagr, VOL=vol, Sharpe=sharpe,
        MaxDD=maxdd, Calmar=calmar,
        WinRate=win_rate, Payoff=payoff, PF=pf,
        Exp=exp, Kelly=kelly, TimeMkt=time_mkt,
        Tail=tail, TPY=tpy, MaxLS=max_lose
    )

# ------------------------------------------- parameter grid -------------------------------------------
stops = np.arange(0.5, 8.05, 0.1).round(1)   # 0.5 … 8.0
levs  = [1,2,3,4,5]

# ------------------------------------------- run (parallel) -------------------------------------------
if __name__ == '__main__':
    tasks = list(product(stops, levs))
    pool  = mp.Pool(processes=mp.cpu_count())
    print(f'scanning {len(tasks)} combinations …')
    tic  = time.time()
    recs = pool.starmap(one_run, tasks)
    pool.close(); pool.join()
    print(f'done in {time.time()-tic:.1f}s')

    # save
    pd.DataFrame(recs).to_csv('MACD_scan_stop_lev.csv', index=False, float_format='%.4f')
    print('saved → MACD_scan_stop_lev.csv')
    # ------------------------------------------------ pretty print ---------------------------------------
    print('\n----- full scan results -----')
    with open('MACD_scan_stop_lev.csv') as f:
        for line in f:
            print(line.rstrip())
            time.sleep(0.01)
                
  
