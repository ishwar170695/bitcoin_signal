import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from features_and_labels import add_indicators, label_signals
from simple_backtest import backtest_signals
import seaborn as sns

FEATURES = [
    'rsi', 'ema', 'macd', 'macd_signal', 'macd_hist', 'atr',
    'bb_upper', 'bb_middle', 'bb_lower', 'obv',
    'return_1', 'return_5', 'return_10'
]

label_map = {-1: 0, 0: 1, 1: 2}
inv_label_map = {0: -1, 1: 0, 2: 1}

def walk_forward_xgb(df, confidence_threshold=0.6, shuffle_labels=False):
    df = df.copy()
    df = df.dropna(subset=FEATURES + ['signal', 'ema100'])
    df['regime'] = (df['close'] > df['ema100']).astype(int)  # 1 = bull, 0 = bear
    X = df[FEATURES].values
    y = df['signal'].map(label_map).values
    regimes = df['regime'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    window_size = int(0.6 * len(df))
    step_size = int(0.05 * len(df))
    all_y_true, all_y_pred = [], []
    all_test_indices = []
    feature_importances = None

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        X_train, y_train = X_scaled[start:end], y[start:end]
        X_test, y_test = X_scaled[end:end+step_size], y[end:end+step_size]
        regime_train = regimes[start:end]
        regime_test = regimes[end:end+step_size]
        test_indices = df.index[end:end+step_size]

        if len(set(y_train)) < 2:
            continue

        if shuffle_labels:
            print(" Shuffling labels for control test...")
            import random
            random.shuffle(y_train)

        clf_bull = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                                 eval_metric='mlogloss', random_state=42)
        clf_bear = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                                 eval_metric='mlogloss', random_state=42)

        # Train separate models
        clf_bull.fit(X_train[regime_train == 1], y_train[regime_train == 1])
        clf_bear.fit(X_train[regime_train == 0], y_train[regime_train == 0])

        y_pred = []
        for i, idx in enumerate(test_indices):
            x_row = X_test[i].reshape(1, -1)
            regime = regime_test[i]

            clf = clf_bull if regime == 1 else clf_bear
            proba = clf.predict_proba(x_row)[0]
            max_p = max(proba)
            pred_class = proba.argmax()

            # Confidence filter
            if confidence_threshold is not None and max_p < confidence_threshold:
                pred_class = 1  # hold

            y_pred.append(pred_class)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_test_indices.extend(test_indices)

        acc = accuracy_score(y_test, y_pred)
        print(f"Window {start}-{end}: Accuracy={acc:.4f}")

        # Add importance only from bull model (or combine both if you prefer)
        if feature_importances is None:
            feature_importances = clf_bull.feature_importances_
        else:
            feature_importances += clf_bull.feature_importances_

    # Save predictions
    pred_signals = [inv_label_map[p] for p in all_y_pred]
    df['ml_signal'] = 0
    df.loc[all_test_indices, 'ml_signal'] = pred_signals

    print("\nWalk-forward Classification Report:")
    print(classification_report(all_y_true, all_y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))

    # Plot feature importances
    if feature_importances is not None:
        feature_importances /= (len(df) // step_size)
        plt.figure(figsize=(10, 6))
        sorted_idx = feature_importances.argsort()[::-1]
        plt.bar([FEATURES[i] for i in sorted_idx], feature_importances[sorted_idx])
        plt.title('XGBoost Feature Importances (Bull Model Avg)')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.show()

    return df

def plot_signal_vs_future(df):
    df['future_return_3h'] = df['close'].shift(-6) / df['close'] - 1
    df_sig = df[df['ml_signal'] != 0].copy()
    df_sig['signal_type'] = df_sig['ml_signal'].map({-1: 'sell', 1: 'buy'})
    if not df_sig.empty:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_sig, x='signal_type', y='future_return_3h')
        plt.title("Future 3h Returns After Each ML Signal")
        plt.axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig("signal_vs_future_return.png")
        plt.show()

def run_all(shuffle=False, confidence_threshold=0.6):
    df = pd.read_csv('data/btc_17-18_30m.csv', parse_dates=['datetime'])
    df = add_indicators(df)
    df = label_signals(df, future_window=6, buy_thresh=0.01, sell_thresh=-0.01)

    print("\n--- Walk-forward XGBoost ---")
    df = walk_forward_xgb(df, confidence_threshold=confidence_threshold, shuffle_labels=shuffle)

    print("\n--- Realistic Backtest (ML signals) ---")
    df_bt = df.copy()
    df_bt['signal'] = df_bt['ml_signal']
    trades, final_value, equity_df = backtest_signals(df_bt)
    equity_df.to_csv("equity_curve.csv", index=False)
    pd.DataFrame(trades).to_csv("trades.csv", index=False)

    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Trades: {len(trades)}")

    df = pd.merge(df, equity_df, on='datetime', how='left')
    df.to_csv("data/btc_18_22_30m_with_signals.csv", index=False)

    print("\n--- Signal vs Future Return Plot ---")
    plot_signal_vs_future(df)

if __name__ == "__main__":
    # You can change threshold or set it to None to disable
    run_all(shuffle=False, confidence_threshold=0.6)
