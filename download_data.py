import yfinance as yf
import pandas as pd
import os
import datetime as dt

# --- Configuration ---
SYMBOLS = ["AAPL", "MSFT"]
DATA_DIR = "data"
INTERVAL = "1m"

def download_data_in_chunks(start_date, end_date):
    """
    Downloads historical 1-minute data for the specified symbols.
    The Yahoo Finance API limits this to the most recent 30 days.
    """
    print("--- Starting Data Download ---")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure the start date is no more than 30 days in the past due to API limits
    thirty_days_ago = dt.datetime.now() - dt.timedelta(days=30)
    if start_date < thirty_days_ago:
        print(f"Warning: 1m data is only available for the last 30 days. Adjusting start date from {start_date.strftime('%Y-%m-%d')} to {thirty_days_ago.strftime('%Y-%m-%d')}.")
        start_date = thirty_days_ago

    # Define the chunk size to respect the API limit
    chunk_size = dt.timedelta(days=7)

    for symbol in SYMBOLS:
        print(f"Downloading {symbol} data...")
        current_start = start_date
        
        # Loop to download data in 7-day chunks
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            print(f"  Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")

            try:
                # Download data from Yahoo Finance
                data = yf.download(tickers=symbol, start=current_start, end=current_end, interval=INTERVAL)

                if data.empty:
                    print(f"  No data downloaded for {symbol} for this period.")
                    current_start += chunk_size
                    continue

                # The column names from yfinance are a MultiIndex. We need to flatten them and make them lowercase.
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.columns = [str(col).lower() for col in data.columns]
                
                # The 'datetime' column is in the index. We need to make it a regular column.
                data.reset_index(inplace=True)
                
                # Rename the datetime column to 'datetime' (lowercase)
                if 'Datetime' in data.columns:
                    data.rename(columns={'Datetime': 'datetime'}, inplace=True)
                elif 'index' in data.columns:
                    data.rename(columns={'index': 'datetime'}, inplace=True)
                
                # Append to existing CSV or create a new one
                file_path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
                if os.path.exists(file_path):
                    data.to_csv(file_path, mode='a', header=False, index=False)
                    print(f"  Appended {symbol} data to {file_path}")
                else:
                    data.to_csv(file_path, index=False)
                    print(f"  Saved {symbol} data to {file_path}")

            except Exception as e:
                print(f"An error occurred while downloading data for {symbol}: {e}")
                
            current_start += chunk_size

    print("--- Data Download Complete ---")

if __name__ == "__main__":
    # Specify the full date range you want to download (e.g., 90 days)
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=90)
    download_data_in_chunks(start_date, end_date)