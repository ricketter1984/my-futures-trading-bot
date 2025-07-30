import pandas as pd
import databento as db
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

def fetch_databento_historical_data(symbol, start_date, end_date, schema="ohlcv-1m", limit_rows=None):
    """
    Fetches historical data from Databento for a given symbol and date range.
    Can fetch trades or OHLCV bars directly.
    
    Args:
        symbol (str): The trading symbol (e.g., "ES.c.0", "MNQ.c.0", or "ES" for root).
        start_date (str): Start date/time in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format.
        end_date (str): End date/time in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format.
        schema (str): The data schema to request. e.g., 'trades', 'ohlcv-1m'.
        limit_rows (int, optional): Maximum number of rows to fetch. Useful for testing/sampling.

    Returns:
        pd.DataFrame: DataFrame with fetched data, or None if fetching fails.
                      Format depends on schema ('trades' for raw trades, OHLCV for 'ohlcv-1m').
    """
    load_dotenv()
    api_key = os.getenv("DATABENTO_API_KEY")

    if not api_key:
        print("Error: DATABENTO_API_KEY is missing in your .env file. Please add it.")
        return None

    try:
        print(f"Connecting to Databento and fetching '{schema}' data for {symbol} from {start_date} to {end_date}...")
        client = db.Historical(key=api_key)

        data_stream = client.timeseries.get_range(
            dataset="GLBX.MDP3", # CME Globex MDP 3.0 dataset
            schema=schema,
            symbols=[symbol], # Use raw symbol directly
            start=start_date,
            end=end_date,
            limit=limit_rows
        )
        
        df = data_stream.to_df()

        if df.empty:
            print(f"No {schema} data found for {symbol} in the specified range.")
            return None

        # Process DataFrame based on schema
        if schema == 'trades':
            # For trades, we need to aggregate to OHLCV bars
            df = df[['ts_event', 'price', 'size']]
            df.columns = ['timestamp', 'price', 'size']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df['size'] = df['size'].astype(float)
            df['price'] = df['price'].astype(float)
            
            # Aggregate trades to 1-minute OHLCV bars
            ohlc = df['price'].resample('1min').ohlc()
            volume = df['size'].resample('1min').sum()
            df = pd.concat([ohlc, volume], axis=1)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.dropna()
            
        elif schema.startswith('ohlcv'): # Handles 'ohlcv-1m', 'ohlcv-1h', etc.
            df = df[['ts_event', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
        else:
            print(f"Warning: Unhandled schema '{schema}'. Returning raw DataFrame.")
        
        print(f"Successfully fetched {len(df)} rows of {schema} data for {symbol}.")
        return df

    except Exception as e:
        print(f"Databento API error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Databento fetch: {e}")
        return None