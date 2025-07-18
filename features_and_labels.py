import pandas as pd
import pandas_ta as ta

def add_indicators(df):
    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema'] = ta.ema(df['close'], length=21)
    df['ema100'] = ta.ema(df['close'], length=100)  # for regime filter

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    bbands = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']

    df['obv'] = ta.obv(df['close'], df['volume']).astype(float)

    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    return df

def label_signals(df, future_window=6, buy_thresh=0.01, sell_thresh=-0.01):
    df = df.copy()
    df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1
    # Regime-based adaptive thresholds
    if 'ema100' not in df.columns:
        raise ValueError('ema100 must be present in DataFrame for regime-based labeling.')
    df['signal'] = 0
    for idx, row in df.iterrows():
        price = row['close']
        ema100 = row['ema100']
        fut_ret = row['future_return']
        # Bull regime
        if price > ema100:
            bthresh = buy_thresh * 0.7  # more aggressive buy
            sthresh = sell_thresh * 1.5  # less aggressive sell
        # Bear regime
        elif price < ema100:
            bthresh = buy_thresh * 1.5  # less aggressive buy
            sthresh = sell_thresh * 0.7  # more aggressive sell
        else:
            bthresh = buy_thresh
            sthresh = sell_thresh
        if fut_ret >= bthresh:
            df.at[idx, 'signal'] = 1
        elif fut_ret <= sthresh:
            df.at[idx, 'signal'] = -1
        else:
            df.at[idx, 'signal'] = 0
    return df.drop(columns=['future_return'])
