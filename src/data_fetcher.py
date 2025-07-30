import pandas as pd
import requests
import time # For exponential backoff
from datetime import datetime
import os
from dotenv import load_dotenv # For loading environment variables

def fetch_fmp_historical_data(symbol, start_date=None, end_date=None, retry_attempts=3, initial_delay=1):
    """
    Fetches historical daily price data from Financial Modeling Prep (FMP) API.
    Reads API key from .env file.
    
    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "SPY").
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to 5 years ago.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        retry_attempts (int): Number of times to retry the request with exponential backoff.
        initial_delay (int): Initial delay in seconds for exponential backoff.

    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' and DateTimeIndex,
                      or None if data fetching fails.
    """
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")

    if not api_key:
        print("Error: FMP_API_KEY is missing in your .env file. Please add it.")
        return None

    if start_date is None:
        start_date = (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    url = f"https://financialmodelingprep.com/api/v3/historical-price/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    
    for attempt in range(retry_attempts):
        try:
            print(f"Fetching data for {symbol} from {start_date} to {end_date} (Attempt {attempt + 1}/{retry_attempts})...")
            response = requests.get(url, timeout=10) # 10-second timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if not data or 'historical' not in data or not data['historical']:
                print(f"No historical data found for {symbol} or invalid response structure.")
                return None

            df = pd.DataFrame(data['historical'])
            
            # FMP returns data in reverse chronological order, so reverse it
            df = df.iloc[::-1].reset_index(drop=True)

            # Convert 'date' to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Rename columns to match expected format
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna() # Drop rows with any NaN values that might result from coercion

            if df.empty:
                print(f"Fetched data for {symbol} is empty after processing.")
                return None

            print(f"Successfully fetched {len(df)} bars for {symbol}.")
            return df

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching data for {symbol}: {e}")
            if response.status_code == 429: # Too Many Requests
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None # For other HTTP errors, don't retry
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching data for {symbol}: {e}")
            delay = initial_delay * (2 ** attempt)
            print(f"Connection issue. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.exceptions.Timeout:
            print(f"Timeout error fetching data for {symbol}.")
            delay = initial_delay * (2 ** attempt)
            print(f"Timeout. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"An unexpected request error occurred for {symbol}: {e}")
            return None
        except ValueError as e:
            print(f"Data processing error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"An unknown error occurred while fetching data for {symbol}: {e}")
            return None
            
    print(f"Failed to fetch data for {symbol} after {retry_attempts} attempts.")
    return None 