import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import numpy as np

# Set aesthetic style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8) # Default figure size

def plot_price_with_signals(df, signals_df, symbol="Symbol", title="Price Chart with Trade Signals"):
    """
    Plots candlestick chart with trade entry/exit signals.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame (must have Open, High, Low, Close, Volume, and DatetimeIndex).
        signals_df (pd.DataFrame): DataFrame of trade signals (must have 'timestamp', 'type', 'price').
        symbol (str): The trading symbol.
        title (str): Title for the plot.
    """
    if df.empty:
        print("Data for plotting is empty.")
        return
    if signals_df.empty:
        print("No signals to plot.")

    # Convert timestamps in signals_df to match df's index type (datetime)
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])

    # Prepare addplot arguments for signals
    # Use lists of NaNs to create gaps where no signal exists
    apds = []
    
    if not signals_df.empty:
        entry_long_dates = signals_df[signals_df['type'] == 'entry_long']['timestamp'].tolist()
        entry_short_dates = signals_df[signals_df['type'] == 'entry_short']['timestamp'].tolist()
        exit_long_dates = signals_df[signals_df['type'] == 'exit_long']['timestamp'].tolist()
        exit_short_dates = signals_df[signals_df['type'] == 'exit_short']['timestamp'].tolist()

        # Create numpy arrays for markers, aligned with df index
        entry_long_markers = np.full(len(df.index), np.nan)
        entry_short_markers = np.full(len(df.index), np.nan)
        exit_long_markers = np.full(len(df.index), np.nan)
        exit_short_markers = np.full(len(df.index), np.nan)

        for i, date in enumerate(df.index):
            if date in entry_long_dates:
                entry_long_markers[i] = df.loc[date, 'Low'] * 0.99 # Slightly below low
            if date in entry_short_dates:
                entry_short_markers[i] = df.loc[date, 'High'] * 1.01 # Slightly above high
            if date in exit_long_dates:
                exit_long_markers[i] = df.loc[date, 'High'] * 1.01 # Slightly above high
            if date in exit_short_dates:
                exit_short_markers[i] = df.loc[date, 'Low'] * 0.99 # Slightly below low

        # Addplots for entry/exit signals
        apds.append(mpf.make_addplot(pd.Series(entry_long_markers, index=df.index), type='scatter', marker='^', markersize=100, color='green', panel=0, label='Long Entry'))
        apds.append(mpf.make_addplot(pd.Series(entry_short_markers, index=df.index), type='scatter', marker='v', markersize=100, color='red', panel=0, label='Short Entry'))
        apds.append(mpf.make_addplot(pd.Series(exit_long_markers, index=df.index), type='scatter', marker='X', markersize=100, color='orange', panel=0, label='Long Exit'))
        apds.append(mpf.make_addplot(pd.Series(exit_short_markers, index=df.index), type='scatter', marker='X', markersize=100, color='blue', panel=0, label='Short Exit'))

    # Plot candlestick chart with volume and signals
    mpf.plot(df,
             type='candle',
             style='yahoo', # Or 'charles', 'binance', etc.
             volume=True,
             addplot=apds,
             title=f"{title} ({symbol})",
             ylabel='Price',
             ylabel_lower='Volume',
             figscale=1.5,
             figsize=(16, 9))
    
    plt.show()

def plot_equity_curve(signals_df, initial_capital=10000, title="Equity Curve"):
    """
    Plots the equity curve based on trade signals.
    This is a simplified P&L calculation for visualization purposes.
    
    Args:
        signals_df (pd.DataFrame): DataFrame of trade signals.
        initial_capital (float): Starting capital for the simulation.
        title (str): Title for the plot.
    """
    if signals_df.empty:
        print("No signals to plot equity curve.")
        return

    # Sort signals by timestamp
    signals_df = signals_df.sort_values(by='timestamp').reset_index(drop=True)

    # Simplified P&L calculation - assuming 1 unit traded per signal
    # This needs to be replaced with a proper backtesting framework for accurate P&L
    equity = [initial_capital]
    current_position = 0
    last_entry_price = 0

    for i, row in signals_df.iterrows():
        if row['type'] == 'entry_long':
            current_position += 1 # Assume buying 1 unit
            last_entry_price = row['price']
        elif row['type'] == 'exit_long' and current_position > 0:
            pnl = (row['price'] - last_entry_price) * 1 # Price difference * units
            equity.append(equity[-1] + pnl)
            current_position = 0
            last_entry_price = 0 # Reset for next trade
        elif row['type'] == 'entry_short':
            current_position -= 1 # Assume shorting 1 unit
            last_entry_price = row['price']
        elif row['type'] == 'exit_short' and current_position < 0:
            pnl = (last_entry_price - row['price']) * 1 # Price difference * units
            equity.append(equity[-1] + pnl)
            current_position = 0
            last_entry_price = 0 # Reset for next trade
    
    if len(equity) == 1 and equity[0] == initial_capital: # No trades generated profit/loss
        print("No completed trades to plot equity curve from.")
        return

    equity_series = pd.Series(equity, index=signals_df[signals_df['type'].isin(['exit_long', 'exit_short'])]['timestamp'])
    
    # Prepend initial capital at the start date if no exits
    if equity_series.empty:
        equity_series = pd.Series([initial_capital], index=[signals_df['timestamp'].min() if not signals_df.empty else pd.Timestamp.now()])
    else:
        first_exit_date = signals_df[signals_df['type'].isin(['exit_long', 'exit_short'])]['timestamp'].min()
        if first_exit_date > signals_df['timestamp'].min():
            equity_series = pd.concat([pd.Series([initial_capital], index=[signals_df['timestamp'].min()]), equity_series])


    plt.figure(figsize=(12, 6))
    plt.plot(equity_series.index, equity_series.values, label='Equity Curve', color='purple')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trade_returns_histogram(signals_df, title="Histogram of Individual Trade Returns"):
    """
    Plots a histogram of individual trade returns.
    
    Args:
        signals_df (pd.DataFrame): DataFrame of trade signals.
        title (str): Title for the plot.
    """
    if signals_df.empty:
        print("No signals to plot trade returns histogram.")
        return

    trade_returns = []
    
    # Group signals by trade (entry followed by exit)
    in_position = False
    entry_price = 0
    trade_type = ''

    for i, row in signals_df.sort_values(by='timestamp').iterrows():
        if not in_position and (row['type'] == 'entry_long' or row['type'] == 'entry_short'):
            in_position = True
            entry_price = row['price']
            trade_type = row['type']
        elif in_position and ((row['type'] == 'exit_long' and trade_type == 'entry_long') or \
                              (row['type'] == 'exit_short' and trade_type == 'entry_short')):
            if trade_type == 'entry_long':
                pnl = (row['price'] - entry_price) / entry_price * 100 # Percentage return
            else: # entry_short
                pnl = (entry_price - row['price']) / entry_price * 100 # Percentage return
            
            trade_returns.append(pnl)
            in_position = False
            entry_price = 0
            trade_type = ''
            
    if not trade_returns:
        print("No completed trades to plot returns histogram from.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(trade_returns, bins=20, kde=True, color='teal')
    plt.title(title)
    plt.xlabel('Trade Return (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# You can add more visualization functions here later, e.g., for
# - Drawdown plot
# - Indicator-specific plots
# - Heatmaps for parameter optimization results (later when we do optimization) 