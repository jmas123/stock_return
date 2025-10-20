import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np


def download_stock_data(symbol, period):
    try:
        company = yf.Ticker(symbol)
        data = company.history(period)
        return data
    except Exception as e:
        print("Error")
        return None


def plot_stock_data(x_test, y_test, y_test_pred, symbol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f'{symbol} Stock Analysis', fontsize=16)
    
    ax1.scatter(y_test, y_test_pred, alpha=0.6)
    lo, hi = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', lw=2)
    ax1.set_xlabel('Actual Return')
    ax1.set_ylabel('Predicted Return')
    ax1.set_title(f'{symbol} - Actual vs Predicted Returns')
    ax1.grid(True, alpha=0.3)


    # Volume over time
    test_idx = x_test.index
    ax2.plot(test_idx, y_test, label='Actual', marker='o', markersize=3)
    ax2.plot(test_idx, y_test_pred, label='Predicted', marker='s', markersize=3)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return')
    ax2.set_title(f'{symbol} - Return Predictions Over Time')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    
    
    plt.tight_layout()
    plt.show()

def feature_importance (regressor):
    importances = pd.Series(regressor.feature_importances_, index=x.columns)

    importances.sort_values(ascending=False).plot.bar(figsize=(10,5))

    plt.title('Feature Importances')
    plt.show()

def prep_data(stock_data: pd.DataFrame):
    data = stock_data.copy()

    data['ret_1d'] = data['Close'].pct_change()
    data['ret_3d'] = data['Close'].pct_change(3)
    data['ret_5d'] = data['Close'].pct_change(5)

    data['ma_10'] = data['Close'].rolling(10).mean()
    data['ma_20'] = data['Close'].rolling(20).mean()

    

    data['ma_50'] = data['Close'].rolling(50).mean()
    data['ma_200'] = data['Close'].rolling(200).mean()


    data['ma_10_ratio'] = data['Close'] / data['ma_10'] - 1
    data['ma_20_ratio'] = data['Close'] / data['ma_20'] - 1


    data['golden_cross'] = 0
    data.loc[(data['ma_50'] > data['ma_200']) & (data['ma_50'].shift(1) <= data['ma_200'].shift(1)), 'golden_cross'] = 1

    data['trend'] = (data['ma_50'] > data['ma_200']).astype(int)



    data['avg_c_15'] = data['Close'].rolling(15).mean()
    data['avg_c_15_ratio'] = data['Close'] / data['avg_c_15'] - 1

    data['vol_z_20'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / (data['Volume'].rolling(20).std() + 1e-9)
    data['vol_z_10'] = (data['Volume'] - data['Volume'].rolling(10).mean()) / (data['Volume'].rolling(10).std() + 1e-9)

    change = data['Close'].diff()
    up = change.clip(lower = 0).rolling(15).mean()
    down = (-change.clip(upper = 0)).rolling(15).mean()
    rs = up/down
    data['rsi_14'] = 100 - (100 / (1 + rs))

    ema_12 = data['Close'].ewm(span = 12, adjust = False).mean()
    ema_26 = data['Close'].ewm(span = 26, adjust = False).mean()
    data['macd'] = ema_12-ema_26
    data['macd_signal'] = data['macd'].ewm(span = 9, adjust = False).mean()

    data['macd_normalized'] = data['macd'] / data['Close']
    data['macd_signal_normalized'] = data['macd_signal'] / data['Close']

    data['volatility'] = data['ret_1d'].rolling(20).std()

    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

    data['hl_spread'] = (data['High'] - data['Low']) / data['Close']

    data['y'] = data['Close'].shift(-1) / data['Close'] - 1.0

    data = data.dropna()
    feature_cols = [
        'ret_1d', 'ret_3d', 'ret_5d',
        'ma_10_ratio', 'ma_20_ratio', 'avg_c_15_ratio', 
        'vol_z_20', 'vol_z_10',
        'rsi_14', 'macd_normalized', 'macd_signal_normalized',
        'trend', 'volume_ratio',
        'volatility', 'hl_spread'

    ]
    x = data[feature_cols]
    y = data['y']

    return x, y

def time_split(x, y, test_size):
    n = len(x)
    cut = int(n * (1 - test_size))
    return x.iloc[:cut], x.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

def build_model(x, y):

    x_train, x_test, y_train, y_test = time_split(x, y, test_size=0.2)

    print(f"training samples{x_train.shape} testing samples{x_test.shape} total samples{x.shape}")
    regressor = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    regressor.fit(x_train, y_train)


    y_pred_train = regressor.predict(x_train)
    y_pred_test = regressor.predict(x_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    cv_mse = mean_squared_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    direction_accuracy = ((y_test > 0) == (y_pred_test > 0)).mean()
    print(f"Direction accuracy: {direction_accuracy:.2%}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"CV MSE: {cv_mse:.4f}")
    print(f"Training R^2: {train_r2:.4f}")
    print(f"CV R^2: {test_r2:.4f}")



    return y_pred_test, y_pred_train, x_test, y_test, regressor


def backtest_strategy(y_test, y_pred_test, threshold = 0.0005, cost = 0.0005 ):
    positions = pd.Series(0, index = y_test.index)
    positions[y_pred_test > threshold] = 1  
    positions[y_pred_test < -threshold] = -1

    strategy_returns = positions.shift(1) * y_test
    transaction_cost = positions.diff().abs() * cost
    net_return = strategy_returns - transaction_cost


    total_days = len(net_return)
    positive_days = (net_return > 0).sum()
    num_trades = positions.diff().abs().sum()


    annual_return = net_return.mean() * 252
    annual_vol = net_return.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    cum_returns = (1 + net_return).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()

    buy_hold_return = y_test
    buy_hold_cumulative = (1 + buy_hold_return).cumprod()
    buy_hold_total = buy_hold_cumulative.iloc[-1] - 1
    buy_hold_sharpe = (buy_hold_return.mean() * 252) / (buy_hold_return.std() * np.sqrt(252)) 

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_vol:.2%}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Win Rate: {positive_days/total_days:.2%}")
    print(f"Number of Trades: {num_trades:.0f}")
    print(f"\nBuy & Hold Sharpe: {buy_hold_sharpe:.2f}")
    print(f"Buy & Hold Return: {buy_hold_total:.2%}")
    print(f"Strategy vs B&H: {(total_return - buy_hold_total):.2%}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))


    cum_returns.plot(ax=axes[0], label='Strategy', color='blue')
    buy_hold_cumulative.plot(ax=axes[0], label='Buy & Hold', color='gray', alpha=0.7)
    axes[0].set_title('Cumulative Returns: Strategy vs Buy & Hold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    drawdown = (cum_returns / cum_returns.cummax() - 1)
    drawdown.plot(ax=axes[1], color='red', alpha=0.7)
    axes[1].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    axes[1].set_title('Strategy Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(True, alpha=0.3)
    
    position_counts = positions.value_counts().sort_index()
    axes[2].bar(position_counts.index, position_counts.values)
    axes[2].set_title('Position Distribution')
    axes[2].set_xlabel('Position (-1=Short, 0=Flat, 1=Long)')
    axes[2].set_ylabel('Days')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return net_return


if __name__ == "__main__":
    symbol = "SPY"
    period = "10y"

    data = download_stock_data(symbol, period)
    if data is not None:
        x, y = (prep_data(data))
        latest_feature = x.iloc[-1:].copy()
        y_pred_test, y_pred_train, x_test, y_test, regressor = build_model(x, y)
        plot_stock_data(x_test, y_test, y_pred_test, symbol)
        feature_importance(regressor)
        backtest_strategy(y_test,  y_pred_test)
        
        




