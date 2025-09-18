import yfinance as yf
import pandas as pd
import os
import datetime as dt

# --- Configuration ---
SYMBOLS = ["AAPL", "MSFT"]
DATA_DIR = "data"
INTERVAL = "1m"

def download_data():
    """
    Downloads historical 1-minute data for the specified symbols and saves it to the data directory.
    """
    print("--- Starting Data Download ---")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Define the start and end dates for the download
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=7)

    for symbol in SYMBOLS:
        print(f"Downloading {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        try:
            # Download data from Yahoo Finance
            data = yf.download(tickers=symbol, start=start_date, end=end_date, interval=INTERVAL)

            if data.empty:
                print(f"No data downloaded for {symbol}. It's possible the ticker is invalid or there's no data for the requested period.")
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

            # Save to CSV
            file_path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved {symbol} data to {file_path}")

        except Exception as e:
            print(f"An error occurred while downloading data for {symbol}: {e}")

    print("--- Data Download Complete ---")

if __name__ == "__main__":
    download_data()
