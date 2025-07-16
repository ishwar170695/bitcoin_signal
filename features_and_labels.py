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
    df['signal'] = 0
    df.loc[df['future_return'] >= buy_thresh, 'signal'] = 1
    df.loc[df['future_return'] <= sell_thresh, 'signal'] = -1
    return df.drop(columns=['future_return'])
