import pandas as pd
from ml_train_dual import train_dual_side

# --- 1. Define Symbols and Load Data ---
symbols = ["AAPL", "MSFT"]
data = {s: pd.read_csv(f"data/{s}_1min.csv", parse_dates=["datetime"]).set_index("datetime") for s in symbols}

# --- 2. Run Training for Each Symbol ---
for symbol in symbols:
    print(f"--- Training for {symbol} ---")
    df = data[symbol]
    train_dual_side(df, symbol=symbol)
    print(f"--- Finished Training for {symbol} ---")

print("All training complete. Artifacts saved to 'artifacts' directory.")
