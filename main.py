import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Import modules from our src directory
from src.strategy import MomentumIgnitionStrategy
from src.data_fetcher import fetch_fmp_historical_data
from src.visualizer import plot_price_with_signals, plot_equity_curve, plot_trade_returns_histogram # NEW import

def run_backtest():
    """
    Orchestrates the backtest of the Momentum Ignition Strategy using FMP data.
    """
    # Load environment variables from .env file
    load_dotenv()
    fmp_api_key = os.getenv("FMP_API_KEY")

    if not fmp_api_key:
        print("Error: FMP_API_KEY is missing in your .env file. Please add it and ensure it's loaded.")
        return

    # --- 1. Configure Data Fetching ---
    TARGET_SYMBOL = "SPY"  # <<< IMPORTANT: Change to your desired futures symbol (e.g., "MES", "MNQ")
    # Adjust your desired historical range
    START_DATE = "2023-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d') # Today's date

    # --- 2. Fetch real data from FMP ---
    print(f"Attempting to fetch data for {TARGET_SYMBOL}...")
    data = fetch_fmp_historical_data(
        symbol=TARGET_SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE
    )

    if data is None or data.empty:
        print("Could not fetch valid data or data is empty. Exiting backtest.")
        return

    print(f"\nSuccessfully loaded {len(data)} bars for {TARGET_SYMBOL}.")
    print("Data Head:\n", data.head())
    print("\nData Tail:\n", data.tail())

    # --- 3. Define your strategy parameters ---
    # These parameters are crucial for the strategy's performance.
    # You will likely want to optimize these later.
    strategy_params = {
        'atr_period': 14,
        'atr_threshold_factor': 0.6, # Tighter consolidation
        'roc_period': 3,             # Shorter ROC for faster ignition
        'roc_threshold': 0.8,        # Percentage change (0.8% price move)
        'trend_ma_period': 50,       # Shorter trend for example
        'atr_stop_multiple': 2.5,    # 2.5 times ATR for stop loss

        # Quad Stochastics Parameters
        'fast_stoch_k_period_1': 9, 'fast_stoch_d_period_1': 3, 'fast_stoch_smoothing_1': 3,
        'fast_stoch_k_period_2': 14, 'fast_stoch_d_period_2': 3, 'fast_stoch_smoothing_2': 3,
        'slow_stoch_k_period_1': 40, 'slow_stoch_d_period_1': 4, 'slow_stoch_smoothing_1': 4,
        'slow_stoch_k_period_2': 60, 'slow_stoch_d_period_2': 10, 'slow_stoch_smoothing_2': 10,
        'stoch_oversold': 20,       # Stochastic oversold threshold
        'stoch_overbought': 80,      # Stochastic overbought threshold

        # MACD Parameters
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        'macd_cross_threshold': 0   # Threshold for MACD line crossing signal line (e.g., 0 for simple cross)
    }

    # --- 4. Initialize the strategy ---
    strategy = MomentumIgnitionStrategy(strategy_params)

    # --- 5. Iterate through your data bar by bar to simulate backtest ---
    print("\nRunning strategy simulation with confirmations using FMP data...")
    for i in range(len(data)):
        # Pass a slice of the DataFrame up to the current bar to simulate real-time data flow
        strategy.process_bar(data.iloc[:i+1]) 

    # --- 6. Get and analyze signals ---
    trade_signals_df = strategy.get_signals()
    if not trade_signals_df.empty:
        print("\n--- Generated Trade Signals (with confirmations from FMP data) ---")
        print(trade_signals_df.to_string()) # Use to_string() to prevent truncation
        print(f"\nTotal trades generated: {len(trade_signals_df)}")

        # --- 7. Generate Visualizations (NEW) ---
        print("\nGenerating visualizations...")
        plot_price_with_signals(data, trade_signals_df, symbol=TARGET_SYMBOL)
        plot_equity_curve(trade_signals_df)
        plot_trade_returns_histogram(trade_signals_df)

    else:
        print("\nNo trade signals generated with current parameters and FMP data. No visualizations will be generated.")

    print("\nBacktest simulation complete.")

if __name__ == "__main__":
    run_backtest() 