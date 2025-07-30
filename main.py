import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Import modules from our src directory
from src.strategy import MomentumIgnitionStrategy
# from src.data_fetcher import fetch_fmp_historical_data # This line MUST be commented out or removed
from src.databento_fetcher import fetch_databento_historical_data # This is the correct import for Databento
from src.visualizer import plot_price_with_signals, plot_equity_curve, plot_trade_returns_histogram
from src.backtester import Backtester
from src.optimizer import StrategyOptimizer

def run_backtest():
    # ... (API key loading and other config) ...

    # --- 1. Configure Data Fetching & Backtester Parameters ---
    # IMPORTANT: Changed to a SPECIFIC futures contract symbol for September 2025.
    TARGET_SYMBOL = "ESU25"  # <<< IMPORTANT: Changed to "ESU25"

    # Set a very short, specific date range for quick testing (e.g., 1 hour of data from a recent trading day)
    target_date = datetime(2025, 7, 29) # Target specific trading day
    START_DATE = target_date.strftime('%Y-%m-%dT09:30:00') # Start of NYSE session (common for futures activity)
    END_DATE = target_date.strftime('%Y-%m-%dT10:30:00')   # 1 hour later
    
    INITIAL_CAPITAL = 100000.0
    COMMISSION_PER_TRADE = 0.005

    # --- 2. Fetch real data from Databento ---
    print(f"Attempting to fetch data for {TARGET_SYMBOL} from Databento for {START_DATE} to {END_DATE}...")
    data = fetch_databento_historical_data( # This is the correct function call
        symbol=TARGET_SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        schema="ohlcv-1m" # Make sure this schema is explicitly passed
    )

    if data is None or data.empty:
        print("Could not fetch valid data from Databento or data is empty. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(data)} 1-minute OHLCV bars for {TARGET_SYMBOL}.")
    print("OHLCV Data Head:\n", data.head())
    print("OHLCV Data Tail:\n", data.tail())

    # --- The old raw trade aggregation block has been removed ---
    # This section is no longer needed because fetch_databento_historical_data directly returns OHLCV-1m.

    # --- 3. Define Strategy Parameter Ranges for Optimization ---
    param_ranges = {
        'atr_period': [10, 14, 20],
        'atr_threshold_factor': [0.5, 0.6, 0.7],
        'roc_period': [3, 5],
        'roc_threshold': [0.5, 1.0],
        'trend_ma_period': [50, 100], # Note: For 1-min data, this is 50/100 minutes, not days. Adjust as needed.
        'atr_stop_multiple': [2.0, 2.5, 3.0],
        'fast_stoch_k_period_1': [9],
        'fast_stoch_d_period_1': [3],
        'fast_stoch_smoothing_1': [3],
        'fast_stoch_k_period_2': [14],
        'fast_stoch_d_period_2': [3],
        'fast_stoch_smoothing_2': [3],
        'slow_stoch_k_period_1': [40],
        'slow_stoch_d_period_1': [4],
        'slow_stoch_smoothing_1': [4],
        'slow_stoch_k_period_2': [60],
        'slow_stoch_d_period_2': [10],
        'slow_stoch_smoothing_2': [10],
        'stoch_oversold': [20],
        'stoch_overbought': [80],
        'stoch_oversold_60_10_10_alert': [10, 15, 20],
        'stoch_overbought_60_10_10_alert': [80, 85, 90],
        'macd_fast_period': [12],
        'macd_slow_period': [26],
        'macd_signal_period': [9],
        'macd_cross_threshold': [0]
    }

    # --- 4. Initialize and Run the Optimizer ---
    print("\n--- Starting Parameter Optimization ---")
    optimizer = StrategyOptimizer(
        data=data, # Pass the aggregated OHLCV data
        backtester_initial_capital=INITIAL_CAPITAL,
        backtester_commission_per_trade=COMMISSION_PER_TRADE
    )
    
    optimization_results_df = optimizer.run_grid_search(
        param_ranges=param_ranges,
        optimize_metric='sharpe_ratio' # Choose your primary optimization metric here
    )

    # --- 5. Display Optimization Results ---
    if not optimization_results_df.empty:
        print("\n--- Top 5 Best Performing Parameter Sets (by Sharpe Ratio) ---")
        best_results = optimizer.get_best_results(top_n=5, sort_by='sharpe_ratio', ascending=False)
        print(best_results.to_string())

        # OPTIONAL: Run a single backtest with the very best parameters for detailed visualization
        if not best_results.empty:
            print("\n--- Running detailed backtest for the BEST parameter set ---")
            best_params = best_results.iloc[0][list(param_ranges.keys())].to_dict()
            
            strategy_best = MomentumIgnitionStrategy(best_params)
            for i in range(len(data)):
                strategy_best.process_bar(data.iloc[:i+1])
            trade_signals_best_df = strategy_best.get_signals()

            if not trade_signals_best_df.empty:
                backtester_best = Backtester(data, initial_capital=INITIAL_CAPITAL, commission_per_trade=COMMISSION_PER_TRADE)
                backtester_best.run_backtest(trade_signals_best_df)
                
                trade_log_best_df, equity_curve_best_series = backtester_best.get_results()
                performance_metrics_best = backtester_best.calculate_metrics(trade_log_best_df, equity_curve_best_series)

                print("\n--- Performance Metrics for BEST Set ---")
                for metric, value in performance_metrics_best.items():
                    if isinstance(value, (int, float)):
                        if 'percent' in metric or 'return' in metric or 'drawdown' in metric:
                            print(f"{metric.replace('_', ' ').title()}: {value:.2f}%")
                        elif 'ratio' in metric or 'factor' in metric:
                            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"{metric.replace('_', ' ').title()}: {value}")

                print("\nGenerating visualizations for the BEST parameter set...")
                plot_price_with_signals(data, trade_signals_best_df, symbol=TARGET_SYMBOL, title="Price Chart (Best Params)")
                plot_equity_curve(equity_curve_best_series, initial_capital=INITIAL_CAPITAL, title="Equity Curve (Best Params)")
                plot_trade_returns_histogram(trade_log_best_df, title="Trade PnL Histogram (Best Params)")

            else:
                print("No trade signals generated for the best parameter set. No detailed backtest or visualizations.")

    else:
        print("\nNo successful backtests during optimization. No best parameters to display.")

    print("\nOptimization and backtest complete.")

if __name__ == "__main__":
    run_backtest()