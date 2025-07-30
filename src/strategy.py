import pandas as pd
import numpy as np
from .indicators import calculate_atr, calculate_roc, calculate_sma, calculate_stochastic, calculate_macd

# --- Strategy Logic Helper Functions ---

def is_consolidating(df, atr_period, atr_threshold_factor=0.7, window_for_avg_atr=20):
    """
    Determines if the price is in a consolidation phase.
    Checks if current ATR is below a factor of its recent average ATR.
    df: Pandas DataFrame with OHLC data.
    atr_period: Period for ATR calculation.
    atr_threshold_factor: Factor to multiply the average ATR by to set the threshold.
                          e.g., 0.7 means current ATR must be < 70% of its average.
    window_for_avg_atr: Window to calculate the average ATR for comparison.
    """
    atr_series = calculate_atr(df, period=atr_period)
    if atr_series.empty or len(atr_series) < window_for_avg_atr:
        return False # Not enough data

    # Get the latest ATR value and its rolling average
    current_atr = atr_series.iloc[-1]
    # Ensure there are enough values for the rolling mean calculation
    if len(atr_series) < window_for_avg_atr:
        return False
    avg_atr = atr_series.iloc[-window_for_avg_atr:].mean() # Average of recent ATR

    # Consolidation if current ATR is significantly lower than its recent average
    return current_atr < (avg_atr * atr_threshold_factor)

def get_momentum_ignition_signal(df, roc_period, roc_threshold=0.5):
    """
    Generates a momentum ignition signal (long/short/none).
    df: Pandas DataFrame with 'Close' column.
    roc_period: Period for ROC calculation.
    roc_threshold: Percentage change required to trigger a signal.
    """
    roc_series = calculate_roc(df, period=roc_period)
    if roc_series.empty:
        return "none"

    current_roc = roc_series.iloc[-1]

    if current_roc > roc_threshold:
        return "long"
    elif current_roc < -roc_threshold:
        return "short"
    return "none"

def get_trend(df, trend_ma_period):
    """
    Determines the long-term trend based on SMA.
    df: Pandas DataFrame with 'Close' column.
    trend_ma_period: Period for the long-term SMA.
    """
    sma_series = calculate_sma(df, period=trend_ma_period)
    if sma_series.empty:
        return "sideways"

    current_close = df['Close'].iloc[-1]
    current_sma = sma_series.iloc[-1]

    if current_close > current_sma:
        return "uptrend"
    elif current_close < current_sma:
        return "downtrend"
    return "sideways"

# --- Main Strategy Class ---

class MomentumIgnitionStrategy:
    def __init__(self, params):
        """
        Initializes the strategy with given parameters.
        params: Dictionary of strategy parameters.
            e.g., {'atr_period': 14, 'atr_threshold_factor': 0.7, 'roc_period': 5,
                   'roc_threshold': 0.5, 'trend_ma_period': 200, 'atr_stop_multiple': 2.0,
                   'fast_stoch_k_period_1': 9, 'fast_stoch_d_period_1': 3, 'fast_stoch_smoothing_1': 3,
                   'fast_stoch_k_period_2': 14, 'fast_stoch_d_period_2': 3, 'fast_stoch_smoothing_2': 3,
                   'slow_stoch_k_period_1': 40, 'slow_stoch_d_period_1': 4, 'slow_stoch_smoothing_1': 4,
                   'slow_stoch_k_period_2': 60, 'slow_stoch_d_period_2': 10, 'slow_stoch_smoothing_2': 10,
                   'stoch_oversold': 20, 'stoch_overbought': 80, 'macd_cross_threshold': 0 # For MACD line crossing signal line
                   }
        """
        self.params = params
        self.current_position = None # None, 'long', or 'short'
        self.entry_price = None
        self.trailing_stop_price = None
        self.signals = [] # To store all generated trade signals

    def _check_stochastic_confirmations(self, df):
        """
        Checks if the stochastic conditions for entry are met.
        Uses the 'quad stochastics' approach.
        """
        # Fast Stochastic (9, 3, 3)
        fast_k_9_3_3, fast_d_9_3_3 = calculate_stochastic(
            df, self.params['fast_stoch_k_period_1'], self.params['fast_stoch_d_period_1'], self.params['fast_stoch_smoothing_1']
        )
        # Fast Stochastic (14, 3, 3)
        fast_k_14_3_3, fast_d_14_3_3 = calculate_stochastic(
            df, self.params['fast_stoch_k_period_2'], self.params['fast_stoch_d_period_2'], self.params['fast_stoch_smoothing_2']
        )
        # Slow Stochastic (40, 4, 4)
        slow_k_40_4_4, slow_d_40_4_4 = calculate_stochastic(
            df, self.params['slow_stoch_k_period_1'], self.params['slow_stoch_d_period_1'], self.params['slow_stoch_smoothing_1']
        )
        # Slow Stochastic (60, 10, 10)
        slow_k_60_10_10, slow_d_60_10_10 = calculate_stochastic(
            df, self.params['slow_stoch_k_period_2'], self.params['slow_stoch_d_period_2'], self.params['slow_stoch_smoothing_2']
        )

        # Ensure all series have enough data and are not NaN at the latest point
        if any(s.empty or s.iloc[-1] is np.nan for s in [fast_k_9_3_3, fast_d_9_3_3, fast_k_14_3_3, fast_d_14_3_3,
                                                        slow_k_40_4_4, slow_d_40_4_4, slow_k_60_10_10, slow_d_60_10_10]):
            return {'long': False, 'short': False}

        # Your specific stochastic confirmation logic goes here.
        # Based on your mean reversion strategy ("over extended two to three standard VWAP deviations"),
        # for a LONG, you'd want stochastics to be oversold and turning up.
        # For a SHORT, you'd want stochastics to be overbought and turning down.

        # Example logic: All %K and %D lines are oversold for long, or overbought for short.
        # And %K crosses above %D for long, or below %D for short.
        
        # Long confirmation: All K/D are below oversold threshold and K crosses above D
        long_stoch_ok = (
            (fast_k_9_3_3.iloc[-1] < self.params['stoch_oversold'] and fast_d_9_3_3.iloc[-1] < self.params['stoch_oversold']) and
            (fast_k_14_3_3.iloc[-1] < self.params['stoch_oversold'] and fast_d_14_3_3.iloc[-1] < self.params['stoch_oversold']) and
            (slow_k_40_4_4.iloc[-1] < self.params['stoch_oversold'] and slow_d_40_4_4.iloc[-1] < self.params['stoch_oversold']) and
            (slow_k_60_10_10.iloc[-1] < self.params['stoch_oversold'] and slow_d_60_10_10.iloc[-1] < self.params['stoch_oversold'])
        )
        # Add cross-up confirmation (K crosses above D) for long
        long_cross_ok = (
            (fast_k_9_3_3.iloc[-2] < fast_d_9_3_3.iloc[-2] and fast_k_9_3_3.iloc[-1] > fast_d_9_3_3.iloc[-1]) or
            (fast_k_14_3_3.iloc[-2] < fast_d_14_3_3.iloc[-2] and fast_k_14_3_3.iloc[-1] > fast_d_14_3_3.iloc[-1])
            # You might want to be more specific here, e.g., only the slowest stochastics need to cross, or all of them.
            # For simplicity, let's say at least one fast stochastic cross-up.
        )
        
        # Short confirmation: All K/D are above overbought threshold and K crosses below D
        short_stoch_ok = (
            (fast_k_9_3_3.iloc[-1] > self.params['stoch_overbought'] and fast_d_9_3_3.iloc[-1] > self.params['stoch_overbought']) and
            (fast_k_14_3_3.iloc[-1] > self.params['stoch_overbought'] and fast_d_14_3_3.iloc[-1] > self.params['stoch_overbought']) and
            (slow_k_40_4_4.iloc[-1] > self.params['stoch_overbought'] and slow_d_40_4_4.iloc[-1] > self.params['stoch_overbought']) and
            (slow_k_60_10_10.iloc[-1] > self.params['stoch_overbought'] and slow_d_60_10_10.iloc[-1] > self.params['stoch_overbought'])
        )
        # Add cross-down confirmation (K crosses below D) for short
        short_cross_ok = (
            (fast_k_9_3_3.iloc[-2] > fast_d_9_3_3.iloc[-2] and fast_k_9_3_3.iloc[-1] < fast_d_9_3_3.iloc[-1]) or
            (fast_k_14_3_3.iloc[-2] > fast_d_14_3_3.iloc[-2] and fast_k_14_3_3.iloc[-1] < fast_d_14_3_3.iloc[-1])
        )

        return {'long': long_stoch_ok and long_cross_ok, 'short': short_stoch_ok and short_cross_ok}

    def _check_macd_confirmation(self, df):
        """
        Checks if the MACD conditions for entry are met.
        For long: MACD line crosses above Signal line (or is already above).
        For short: MACD line crosses below Signal line (or is already below).
        """
        macd_line, signal_line, _ = calculate_macd(
            df, self.params['macd_fast_period'], self.params['macd_slow_period'], self.params['macd_signal_period']
        )

        if macd_line.empty or signal_line.empty or macd_line.iloc[-1] is np.nan or signal_line.iloc[-1] is np.nan:
            return {'long': False, 'short': False}
        
        # Check for cross-up for long (MACD crosses above Signal)
        long_macd_ok = (macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]) or \
                       (macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-1] > self.params['macd_cross_threshold']) # Already above and positive

        # Check for cross-down for short (MACD crosses below Signal)
        short_macd_ok = (macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]) or \
                        (macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-1] < -self.params['macd_cross_threshold']) # Already below and negative

        return {'long': long_macd_ok, 'short': short_macd_ok}

    def process_bar(self, current_data_slice):
        """
        Processes a new bar of data and updates strategy state.
        current_data_slice: A Pandas DataFrame containing historical data up to the current bar.
                            This simulates receiving new data incrementally.
        """
        # Determine the maximum lookback period required by all indicators
        max_lookback = max(
            self.params['trend_ma_period'],
            self.params['atr_period'],
            self.params['roc_period'],
            self.params['fast_stoch_k_period_1'], self.params['fast_stoch_d_period_1'], self.params['fast_stoch_smoothing_1'],
            self.params['fast_stoch_k_period_2'], self.params['fast_stoch_d_period_2'], self.params['fast_stoch_smoothing_2'],
            self.params['slow_stoch_k_period_1'], self.params['slow_stoch_d_period_1'], self.params['slow_stoch_smoothing_1'],
            self.params['slow_stoch_k_period_2'], self.params['slow_stoch_d_period_2'], self.params['slow_stoch_smoothing_2'],
            self.params['macd_fast_period'], self.params['macd_slow_period'], self.params['macd_signal_period']
        ) + 1 # Add 1 for shift operations or initial valid data point

        if len(current_data_slice) < max_lookback:
            # Not enough historical data for all indicators to be valid
            return

        # Calculate core indicators
        current_atr = calculate_atr(current_data_slice, self.params['atr_period']).iloc[-1]
        consolidating = is_consolidating(
            current_data_slice, self.params['atr_period'], self.params['atr_threshold_factor']
        )
        momentum_signal = get_momentum_ignition_signal(
            current_data_slice, self.params['roc_period'], self.params['roc_threshold']
        )
        trend = get_trend(current_data_slice, self.params['trend_ma_period'])
        
        current_close = current_data_slice['Close'].iloc[-1]
        current_high = current_data_slice['High'].iloc[-1]
        current_low = current_data_slice['Low'].iloc[-1]
        current_timestamp = current_data_slice.index[-1]

        # Calculate confirmation indicators
        stoch_confirmations = self._check_stochastic_confirmations(current_data_slice)
        macd_confirmations = self._check_macd_confirmation(current_data_slice)

        # --- Exit Logic (Check before Entry) ---
        if self.current_position == "long":
            # Update trailing stop for long position
            new_stop = current_close - (current_atr * self.params['atr_stop_multiple'])
            self.trailing_stop_price = max(self.trailing_stop_price, new_stop) # Stop only moves up

            if current_low <= self.trailing_stop_price:
                self.signals.append({
                    'timestamp': current_timestamp,
                    'type': 'exit_long',
                    'price': self.trailing_stop_price, # Assumed fill at stop level
                    'reason': 'trailing_stop'
                })
                self.current_position = None
                self.entry_price = None
                self.trailing_stop_price = None

        elif self.current_position == "short":
            # Update trailing stop for short position
            new_stop = current_close + (current_atr * self.params['atr_stop_multiple'])
            self.trailing_stop_price = min(self.trailing_stop_price, new_stop) # Stop only moves down

            if current_high >= self.trailing_stop_price:
                self.signals.append({
                    'timestamp': current_timestamp,
                    'type': 'exit_short',
                    'price': self.trailing_stop_price, # Assumed fill at stop level
                    'reason': 'trailing_stop'
                })
                self.current_position = None
                self.entry_price = None
                self.trailing_stop_price = None

        # --- Entry Logic (with new confirmations) ---
        if self.current_position is None: # Only enter if not already in a position
            # Long Entry Condition
            if (consolidating and
                momentum_signal == "long" and
                trend == "uptrend" and
                stoch_confirmations['long'] and # Stochastic confirmation
                macd_confirmations['long']):   # MACD confirmation
                
                self.signals.append({
                    'timestamp': current_timestamp,
                    'type': 'entry_long',
                    'price': current_close,
                    'reason': 'consolidation_breakout_long_confirmed'
                })
                self.current_position = "long"
                self.entry_price = current_close
                self.trailing_stop_price = self.entry_price - (current_atr * self.params['atr_stop_multiple'])
                
            # Short Entry Condition
            elif (consolidating and
                  momentum_signal == "short" and
                  trend == "downtrend" and
                  stoch_confirmations['short'] and # Stochastic confirmation
                  macd_confirmations['short']):   # MACD confirmation
                
                self.signals.append({
                    'timestamp': current_timestamp,
                    'type': 'entry_short',
                    'price': current_close,
                    'reason': 'consolidation_breakout_short_confirmed'
                })
                self.current_position = "short"
                self.entry_price = current_close
                self.trailing_stop_price = self.entry_price + (current_atr * self.params['atr_stop_multiple'])
    
    def get_signals(self):
        return pd.DataFrame(self.signals) 