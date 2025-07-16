import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\pythoncodes\codes\Summer_break\bitcoin_signal\data\btc_18_22_30m_with_signals.csv', parse_dates=['datetime'])

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['equity'], label='Equity Curve', color='green')
plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("equity_curve_plot.png")
plt.show()
