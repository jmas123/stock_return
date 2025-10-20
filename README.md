# stock_return
# Stock Price Prediction with Machine Learning

A machine learning project that predicts next-day stock returns using technical indicators and a Random Forest model, with comprehensive backtesting to evaluate trading strategy performance.

## Overview

This project builds a predictive model for stock price movements using historical price and volume data. It features extensive feature engineering with technical indicators, proper time-series validation, and a realistic backtesting framework that accounts for transaction costs.

## Features

- **Data Collection**: Automated stock data download using yfinance
- **Feature Engineering**: 15+ technical indicators including:
  - Price momentum (1-day, 3-day, 5-day returns)
  - Moving averages (10, 20, 50, 200-day) and ratios
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Volume z-scores and ratios
  - Volatility measures
  - High-Low spread
  - Golden cross detection
  
- **Model Training**: Random Forest Regressor with time-series split
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R² Score
  - Direction Accuracy
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate
  
- **Backtesting**: Realistic strategy simulation with:
  - Transaction costs
  - Long/short positions
  - Benchmark comparison (Buy & Hold)
  - Visual performance analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/jmas123/stock_return.git
cd stock-return

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

**Note**: On macOS/Linux, use `source venv/bin/activate`. On Windows, use `venv\Scripts\activate`.

### Requirements

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## Usage

**First time setup**: Make sure you've activated your virtual environment:
```bash
source venv/bin/activate
```

Run the main script to analyze a stock (default: SPY):

```bash
python stock_prediction.py
```

To analyze a different stock, modify the `symbol` variable in the `__main__` section:

```python
if __name__ == "__main__":
    symbol = "AAPL"  # Change to any valid ticker
    period = "10y"    # Adjust time period as needed
```

**When finished**: Deactivate the virtual environment:
```bash
deactivate
```

## How It Works

1. **Data Download**: Fetches historical stock data for the specified period
2. **Feature Creation**: Calculates technical indicators from raw OHLCV data
3. **Train/Test Split**: Uses time-based split (80/20) to prevent look-ahead bias
4. **Model Training**: Trains Random Forest to predict next-day returns
5. **Evaluation**: Assesses model performance using multiple metrics
6. **Backtesting**: Simulates trading strategy with realistic constraints
7. **Visualization**: Displays prediction accuracy, feature importance, and strategy performance

## Sample Results (SPY, 10-year period)

```
Direction Accuracy: ~52-55%
Sharpe Ratio: 0.8-1.2
Annual Return: 8-12%
Max Drawdown: -15-25%
```

*Note: Results vary based on market conditions and time period*

## Project Structure

```
stock-ml-prediction/
│
├── stock_prediction.py      # Main script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
└── venv/                    # Virtual environment (not tracked in git)
```

## Visualizations

The project generates three types of visualizations:

1. **Prediction Analysis**: Actual vs predicted returns scatter plot and time series
2. **Feature Importance**: Bar chart showing which features matter most
3. **Backtest Results**: Cumulative returns, drawdown, and position distribution

## Key Learnings

- **Time-series data requires special handling**: Random train/test splits leak future information
- **Direction accuracy matters more than MSE**: Correctly predicting up/down is more important than exact magnitude
- **Transaction costs are significant**: Even small costs (0.05%) substantially impact profitability
- **Feature engineering is crucial**: Technical indicators provide meaningful signals for price prediction
- **Markets are efficient**: Beating buy-and-hold consistently is challenging

## Limitations & Future Improvements

**Current Limitations:**
- Single-stock analysis (no portfolio diversification)
- No fundamental data incorporated
- Fixed hyperparameters (no tuning)
- Simple threshold-based trading rules

**Potential Enhancements:**
- [ ] Add command-line arguments for stock selection
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Compare multiple models (XGBoost, LSTM, etc.)
- [ ] Add sentiment analysis from news/social media
- [ ] Implement portfolio optimization
- [ ] Add regime detection (bull/bear markets)
- [ ] Export predictions to CSV for further analysis

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## Disclaimer

This project is for **educational purposes only**. It is not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT License - feel free to use this code for learning and experimentation.

## Author

Built as a first ML project to learn about financial machine learning, time-series prediction, and quantitative trading strategies.

---

**Questions or feedback?** Open an issue or reach out!