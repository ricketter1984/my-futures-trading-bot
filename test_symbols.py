import databento as db
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DATABENTO_API_KEY")

if not api_key:
    print("Error: DATABENTO_API_KEY is missing")
    exit(1)

client = db.Historical(key=api_key)

print("Testing symbol resolution...")

# Try different symbol formats
test_symbols = ['ES', 'ES.c.0', 'AAPL', 'SPY', 'ESZ4', 'ESF5']

for symbol in test_symbols:
    try:
        print(f"\nTrying symbol: {symbol}")
        result = client.symbology.resolve(
            dataset='GLBX.MDP3', 
            symbols=[symbol], 
            stype_in='raw_symbol', 
            stype_out='instrument_id',
            start_date='2024-12-01',
            end_date='2024-12-02'
        )
        print(f"Success for {symbol}: {result}")
    except Exception as e:
        print(f"Error for {symbol}: {e}")

print("\nTrying to get available symbols...")
try:
    # Try to get some basic info about the dataset
    metadata = client.metadata.list_datasets()
    print(f"Available datasets: {metadata}")
except Exception as e:
    print(f"Error getting dataset info: {e}") 