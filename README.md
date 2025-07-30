# My Futures Trading Bot

A Python-based trading bot implementing a **Consolidation Breakout with Momentum Ignition Strategy**, enhanced with **Quad Stochastics** and **MACD** confirmations, and dynamic **ATR-based trailing stops**. This project integrates with the Financial Modeling Prep (FMP) API for historical data and provides comprehensive visualizations for strategy analysis.

## Strategy Overview

This bot identifies periods of price consolidation (low volatility) and seeks "momentum ignition" (strong directional moves) in alignment with the long-term trend. Entry signals are confirmed by a custom "quad stochastics" setup (Fast Stochastic (9,3,3), (14,3,3) and Slow Stochastic (40,4,4), (60,10,10)) and MACD crossovers, providing a robust multi-factor confirmation system. Positions are managed with dynamic ATR-based trailing stops.

## Features

- **Consolidation Detection:** Uses Average True Range (ATR) to identify periods of low market volatility.
- **Momentum Ignition:** Utilizes Rate of Change (ROC) to detect impulsive price movements.
- **Trend Filtering:** Employs Simple Moving Average (SMA) to ensure trades are in the direction of the prevailing market trend.
- **Advanced Confirmations:** Integrates custom "Quad Stochastics" logic and MACD crossovers for high-conviction entry signals.
- **Dynamic Risk Management:** Implements ATR-based trailing stops for effective position management.
- **FMP Data Integration:** Fetches historical OHLCV data directly from Financial Modeling Prep API.
- **Comprehensive Visualizations:** Generates:
    - Candlestick charts with trade entry/exit signals.
    - Equity curve plots to track portfolio performance.
    - Histograms of individual trade returns for statistical analysis.

## Project Structure

```
my_futures_trading_bot/
├── .env                  # Environment variables (e.g., FMP_API_KEY) - NOT committed to Git!
├── .gitignore            # Specifies intentionally untracked files to ignore
├── README.md             # Project documentation (this file)
├── main.py               # Main entry point for running the backtest and visualizations
├── requirements.txt      # Lists all Python dependencies
├── config/               # (Future: For modular configuration files)
├── data/                 # (Future: For storing downloaded historical data or backtest results)
├── src/                  # Core Python source code
│   ├── __init__.py       # Makes 'src' a Python package
│   ├── indicators.py     # Functions for technical indicator calculations (ATR, ROC, SMA, Stochastic, MACD)
│   ├── strategy.py       # Core trading strategy logic and signal generation
│   ├── data_fetcher.py   # Handles fetching historical data from FMP API
│   └── visualizer.py     # Functions for generating charts and plots
└── tests/                # (Future: For unit and integration tests)
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/my-futures-trading-bot.git
    cd my-futures-trading-bot
    ```
    (Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username)

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up FMP API Key:**
    - Get your API Key from [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/).
    - Open the `.env` file in the root directory and ensure it contains your actual API key:
      ```
      FMP_API_KEY=your_actual_fmp_api_key
      ```

## How to Run the Bot

1.  **Configure `main.py`:**
    - Open `main.py` in your text editor.
    - Change `TARGET_SYMBOL = "SPY"` to your desired futures symbol (e.g., `"MES"`, `"MNQ"`, `"MYM"`).
    - Adjust `START_DATE` and `END_DATE` as needed.
    *(Note: FMP's free tier might have limitations on futures data. Please refer to FMP documentation for specific futures symbols and data access tiers.)*

2.  **Execute the Backtest:**
    ```bash
    python main.py
    ```

The script will fetch data, run the strategy, print trade signals to the console, and automatically generate three types of visualizations to help you analyze the strategy's performance.

## Strategy Parameters

The strategy is highly configurable through the `strategy_params` dictionary in `main.py`. Key parameters include:

- **ATR Settings:** `atr_period`, `atr_threshold_factor`, `atr_stop_multiple`
- **Momentum Settings:** `roc_period`, `roc_threshold`
- **Trend Settings:** `trend_ma_period`
- **Stochastic Settings:** Four different stochastic configurations for confirmation
- **MACD Settings:** `macd_fast_period`, `macd_slow_period`, `macd_signal_period`

## Visualization Output

When trade signals are generated, the bot automatically creates:

1. **Candlestick Chart with Signals** - Shows price action with entry/exit markers
2. **Equity Curve** - Tracks portfolio performance over time
3. **Trade Returns Histogram** - Statistical distribution of individual trade returns

## Future Enhancements

- Parameter optimization framework
- Real-time trading capabilities
- Multiple timeframe analysis
- Additional technical indicators
- Portfolio risk metrics
- Database integration for historical results

## Disclaimer

This trading bot is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk and may not be suitable for all investors. Please consult with a financial advisor before using this or any trading system with real money.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.