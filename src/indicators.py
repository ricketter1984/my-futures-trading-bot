import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR).
    df: Pandas DataFrame with 'High', 'Low', 'Close' columns.
    period: Lookback period for ATR.
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close' columns for ATR calculation.")

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def calculate_roc(df, period=5):
    """
    Calculates the Rate of Change (ROC).
    df: Pandas DataFrame with 'Close' column.
    period: Lookback period for ROC.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for ROC calculation.")
    return df['Close'].pct_change(periods=period) * 100 # Percentage change

def calculate_sma(df, period=200):
    """
    Calculates the Simple Moving Average (SMA).
    df: Pandas DataFrame with 'Close' column.
    period: Lookback period for SMA.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for SMA calculation.")
    return df['Close'].rolling(window=period).mean()

def calculate_stochastic(df, k_period=14, d_period=3, smoothing_period=3):
    """
    Calculates the Stochastic Oscillator (%K and %D).
    df: Pandas DataFrame with 'High', 'Low', 'Close' columns.
    k_period: Lookback period for %K.
    d_period: Smoothing period for %D (SMA of %K).
    smoothing_period: Smoothing for %K (Fast %K is unsmoothed, Slow %K is smoothed).
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close' columns for Stochastic calculation.")

    # Calculate %K (Fast %K)
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    
    # Avoid division by zero
    range_hl = (highest_high - lowest_low)
    fast_k = 100 * ((df['Close'] - lowest_low) / range_hl)
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero if range is 0

    # Smooth %K to get Slow %K
    slow_k = fast_k.rolling(window=smoothing_period).mean()

    # Calculate %D (SMA of Slow %K)
    slow_d = slow_k.rolling(window=d_period).mean()

    return slow_k, slow_d

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    df: Pandas DataFrame with 'Close' column.
    fast_period: Period for the fast EMA.
    slow_period: Period for the slow EMA.
    signal_period: Period for the signal line EMA.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for MACD calculation.")

    exp1 = df['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram 