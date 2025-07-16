import pandas as pd

def backtest_signals(df, initial_cash=10000, stop_loss_pct=0.03, take_profit_pct=0.06):
    cash = initial_cash
    position = None
    entry_price = 0
    equity_curve = []
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = prev['signal']  # signal from previous bar
        price_open = row['open']
        price_high = row['high']
        price_low = row['low']
        price_close = row['close']
        time = row['datetime']

        # Exit logic
        if position:
            sl_hit = False
            tp_hit = False
            if position['side'] == 'long':
                sl_price = entry_price * (1 - stop_loss_pct)
                tp_price = entry_price * (1 + take_profit_pct)
                if price_low <= sl_price:
                    cash += sl_price
                    trades.append({'datetime': time, 'type': 'stop_loss', 'price': sl_price})
                    position = None
                    continue
                elif price_high >= tp_price:
                    cash += tp_price
                    trades.append({'datetime': time, 'type': 'take_profit', 'price': tp_price})
                    position = None
                    continue
            elif position['side'] == 'short':
                sl_price = entry_price * (1 + stop_loss_pct)
                tp_price = entry_price * (1 - take_profit_pct)
                if price_high >= sl_price:
                    cash += entry_price - sl_price
                    trades.append({'datetime': time, 'type': 'stop_loss', 'price': sl_price})
                    position = None
                    continue
                elif price_low <= tp_price:
                    cash += entry_price - tp_price
                    trades.append({'datetime': time, 'type': 'take_profit', 'price': tp_price})
                    position = None
                    continue

        # Entry logic
        if not position:
            if signal == 1:  # Buy on next bar open
                entry_price = price_open
                cash -= entry_price
                position = {'side': 'long', 'entry': entry_price}
                trades.append({'datetime': time, 'type': 'buy', 'price': entry_price})
            elif signal == -1:  # Sell/Short on next bar open
                entry_price = price_open
                cash += entry_price
                position = {'side': 'short', 'entry': entry_price}
                trades.append({'datetime': time, 'type': 'sell', 'price': entry_price})

        # Track current equity
        if position:
            if position['side'] == 'long':
                equity = cash + price_close
            else:  # short
                equity = cash + (entry_price - price_close)
        else:
            equity = cash

        equity_curve.append({'datetime': time, 'equity': equity})

    # Final exit if holding position
    final_price = df.iloc[-1]['close']
    time = df.iloc[-1]['datetime']
    if position:
        if position['side'] == 'long':
            cash += final_price
            trades.append({'datetime': time, 'type': 'exit_long', 'price': final_price})
        elif position['side'] == 'short':
            cash += (entry_price - final_price)
            trades.append({'datetime': time, 'type': 'exit_short', 'price': final_price})

    print(f"\n[Realistic Backtest] Final Portfolio Value: ${cash:.2f}")
    print(f"Total Trades: {len(trades)}")

    return trades, cash, pd.DataFrame(equity_curve)
