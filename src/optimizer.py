import pandas as pd
from tqdm import tqdm
import itertools # Used for generating parameter combinations
import numpy as np # Ensure numpy is imported for np.inf

# Import our strategy and backtester modules
from .strategy import MomentumIgnitionStrategy
from .backtester import Backtester

class StrategyOptimizer:
    def __init__(self, data, backtester_initial_capital, backtester_commission_per_trade):
        """
        Initializes the strategy optimizer.

        Args:
            data (pd.DataFrame): Historical OHLCV data.
            backtester_initial_capital (float): Initial capital to use for each backtest simulation.
            backtester_commission_per_trade (float): Commission per trade for backtest simulations.
        """
        self.data = data.copy()
        self.backtester_initial_capital = backtester_initial_capital
        self.backtester_commission_per_trade = backtester_commission_per_trade
        self.results = [] # To store optimization results (parameters + metrics)

    def _evaluate_params(self, strategy_params):
        """
        Runs a single backtest with the given strategy parameters and returns its performance metrics.

        Args:
            strategy_params (dict): A dictionary of parameters for MomentumIgnitionStrategy.

        Returns:
            dict: A dictionary of performance metrics from the backtest, or None if an error occurs.
        """
        try:
            # 1. Instantiate the strategy with the current parameters
            strategy = MomentumIgnitionStrategy(strategy_params)

            # 2. Run the strategy to generate signals
            # Note: We need to pass the full data slice for strategy.process_bar
            # This is different from the main.py loop where we pass slices.
            # For optimization, we process the whole data at once for speed.
            for i in range(len(self.data)):
                strategy.process_bar(self.data.iloc[:i+1])
            trade_signals_df = strategy.get_signals()

            if trade_signals_df.empty:
                # print("No signals generated for this parameter set, skipping evaluation.")
                return None # No trades, no meaningful metrics

            # 3. Instantiate and run the backtester
            backtester = Backtester(
                self.data,
                initial_capital=self.backtester_initial_capital,
                commission_per_trade=self.backtester_commission_per_trade
            )
            backtester.run_backtest(trade_signals_df)

            # 4. Get results and calculate metrics
            trade_log_df, equity_curve_series = backtester.get_results()

            if trade_log_df.empty or equity_curve_series.empty or len(equity_curve_series) < 2:
                # print("Not enough data or trades for metric calculation, skipping evaluation.")
                return None

            metrics = backtester.calculate_metrics(trade_log_df, equity_curve_series)
            
            # Ensure essential metrics are present, handle NaNs if any specific metric calc failed
            if pd.isna(metrics.get('sharpe_ratio')) or pd.isna(metrics.get('total_return')):
                return None

            return metrics

        except Exception as e:
            # print(f"Error evaluating parameters {strategy_params}: {e}")
            return None # Return None if any error occurs during evaluation

    def run_grid_search(self, param_ranges, optimize_metric='sharpe_ratio'):
        """
        Performs a grid search over the given parameter ranges.

        Args:
            param_ranges (dict): A dictionary where keys are parameter names
                                 and values are lists of values to test for that parameter.
                                 Example: {'atr_period': [10, 14, 20], 'roc_threshold': [0.5, 1.0]}
            optimize_metric (str): The name of the metric to optimize (e.g., 'sharpe_ratio', 'total_return').

        Returns:
            pd.DataFrame: A DataFrame containing all tested parameter sets and their performance metrics.
        """
        # Generate all combinations of parameters
        keys = param_ranges.keys()
        values = param_ranges.values()
        
        # itertools.product creates an iterator of tuples, each tuple is a combination
        all_combinations = list(itertools.product(*values))
        
        print(f"\nStarting Grid Search with {len(all_combinations)} combinations...")
        
        # Use tqdm for a progress bar
        for combo in tqdm(all_combinations, desc="Optimizing"):
            current_params = dict(zip(keys, combo))
            
            # Evaluate this set of parameters
            metrics = self._evaluate_params(current_params)
            
            if metrics: # Only store if evaluation was successful and produced metrics
                result_entry = {**current_params, **metrics} # Merge params and metrics
                self.results.append(result_entry)
        
        print("Grid Search complete.")
        return pd.DataFrame(self.results)

    def get_best_results(self, top_n=5, sort_by='sharpe_ratio', ascending=False):
        """
        Retrieves and sorts the best optimization results.
        """
        if not self.results:
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        
        # Ensure the sorting column exists, fill NaN with a value that puts them at the end if ascending=False
        # For 'sharpe_ratio', NaN is bad, so for descending sort, NaN should come last.
        # For ascending sort (e.g., minimizing drawdown), NaN should come first.
        sort_value_for_nan = -np.inf if not ascending else np.inf
        
        # Handle cases where optimize_metric might be NaN in some results
        if sort_by in results_df.columns:
            results_df[sort_by] = results_df[sort_by].fillna(sort_value_for_nan)
        else:
            print(f"Warning: Optimization metric '{sort_by}' not found in results. Sorting by Total Return instead.")
            sort_by = 'total_return'
            results_df['total_return'] = results_df['total_return'].fillna(sort_value_for_nan)


        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
        return results_df.head(top_n)