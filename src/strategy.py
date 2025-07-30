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
                   'stoch_oversold': 20, 'stoch_overbought': 80,
                   'stoch_oversold_60_10_10_alert': 15, # NEW
                   'stoch_overbought_60_10_10_alert': 85, # NEW
                   'macd_cross_threshold': 0
                   }
        """
        self.params = params
        self.current_position = None # None, 'long', or 'short'
        self.entry_price = None
        self.trailing_stop_price = None
        self.signals = [] # To store all generated trade signals

    def _check_stochastic_confirmations(self, df):
        """
        Checks if the stochastic conditions for entry are met based on user's refined logic.
        Looks for all stochastics in OS/OB and 60,10,10 K% cross D%.
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
        if any(s.empty or s.iloc[-1] is np.nan or (len(s) < 2 or s.iloc[-2] is np.nan) for s in [
            fast_k_9_3_3, fast_d_9_3_3, fast_k_14_3_3, fast_d_14_3_3,
            slow_k_40_4_4, slow_d_40_4_4, slow_k_60_10_10, slow_d_60_10_10
        ]):
            return {'long': False, 'short': False}

        # --- Long Confirmation for Stochastics ---
        # 1. All K/D are below oversold threshold
        all_stochs_oversold = (
            (fast_k_9_3_3.iloc[-1] < self.params['stoch_oversold'] and fast_d_9_3_3.iloc[-1] < self.params['stoch_oversold']) and
            (fast_k_14_3_3.iloc[-1] < self.params['stoch_oversold'] and fast_d_14_3_3.iloc[-1] < self.params['stoch_oversold']) and
            (slow_k_40_4_4.iloc[-1] < self.params['stoch_oversold'] and slow_d_40_4_4.iloc[-1] < self.params['stoch_oversold']) and
            (slow_k_60_10_10.iloc[-1] < self.params['stoch_oversold'] and slow_d_60_10_10.iloc[-1] < self.params['stoch_oversold'])
        )
        
        # 2. 60,10,10 K% crosses above D% (safer trade entry)
        # We also check the alert level as per user's input, if 60,10,10 K% drops below it.
        stoch_60_10_10_k = slow_k_60_10_10.iloc[-1]
        stoch_60_10_10_d = slow_d_60_10_10.iloc[-1]
        stoch_60_10_10_k_prev = slow_k_60_10_10.iloc[-2]
        stoch_60_10_10_d_prev = slow_d_60_10_10.iloc[-2]

        stoch_60_10_10_cross_up = (stoch_60_10_10_k_prev < stoch_60_10_10_d_prev and stoch_60_10_10_k > stoch_60_10_10_d)
        
        # Optional: Alert level check - can be an alert, but for entry, combined with cross
        stoch_60_10_10_at_alert_level_long = stoch_60_10_10_k <= self.params['stoch_oversold_60_10_10_alert']

        long_stoch_ok = all_stochs_oversold and stoch_60_10_10_cross_up and stoch_60_10_10_at_alert_level_long

        # --- Short Confirmation for Stochastics ---
        # 1. All K/D are above overbought threshold
        all_stochs_overbought = (
            (fast_k_9_3_3.iloc[-1] > self.params['stoch_overbought'] and fast_d_9_3_3.iloc[-1] > self.params['stoch_overbought']) and
            (fast_k_14_3_3.iloc[-1] > self.params['stoch_overbought'] and fast_d_14_3_3.iloc[-1] > self.params['stoch_overbought']) and
            (slow_k_40_4_4.iloc[-1] > self.params['stoch_overbought'] and slow_d_40_4_4.iloc[-1] > self.params['stoch_overbought']) and
            (slow_k_60_10_10.iloc[-1] > self.params['stoch_overbought'] and slow_d_60_10_10.iloc[-1] > self.params['stoch_overbought'])
        )
        
        # 2. 60,10,10 K% crosses below D% (safer trade entry)
        stoch_60_10_10_cross_down = (stoch_60_10_10_k_prev > stoch_60_10_10_d_prev and stoch_60_10_10_k < stoch_60_10_10_d)

        # Optional: Alert level check - if 60,10,10 K% goes above it.
        stoch_60_10_10_at_alert_level_short = stoch_60_10_10_k >= self.params['stoch_overbought_60_10_10_alert']

        short_stoch_ok = all_stochs_overbought and stoch_60_10_10_cross_down and stoch_60_10_10_at_alert_level_short

        return {'long': long_stoch_ok, 'short': short_stoch_ok}

    def _check_macd_confirmation(self, df):
        """
        Checks if the MACD conditions for entry are met based on user's refined logic.
        Looks for MACD below/above 0, moving towards 0, and histogram flip.
        """
        macd_line, signal_line, histogram = calculate_macd(
            df, self.params['macd_fast_period'], self.params['macd_slow_period'], self.params['macd_signal_period']
        )

        if macd_line.empty or signal_line.empty or histogram.empty or \
           macd_line.iloc[-1] is np.nan or signal_line.iloc[-1] is np.nan or histogram.iloc[-1] is np.nan or \
           (len(histogram) < 2 or histogram.iloc[-2] is np.nan):
            return {'long': False, 'short': False}
        
        # --- Long Confirmation for MACD ---
        # 1. MACD line AND Signal line are both under 0 (deep sweep visible)
        macd_under_zero = (macd_line.iloc[-1] < 0 and signal_line.iloc[-1] < 0)
        
        # 2. Histogram flips from negative to positive
        histogram_flip_pos = (histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0)
        
        long_macd_ok = macd_under_zero and histogram_flip_pos

        # --- Short Confirmation for MACD ---
        # 1. MACD line AND Signal line are both above 0
        macd_above_zero = (macd_line.iloc[-1] > 0 and signal_line.iloc[-1] > 0)
        
        # 2. Histogram flips from positive to negative
        histogram_flip_neg = (histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0)
        
        short_macd_ok = macd_above_zero and histogram_flip_neg

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
        ) + 2 # Add 2 for shift operations (e.g., histogram[-2]) or initial valid data point

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