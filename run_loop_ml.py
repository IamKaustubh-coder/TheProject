# run_loop_ml.py (Final Version)
import os
import pandas as pd
from core.event_queue import EventQueue
from core.data import CSVDataHandler
from core.order_sizer import FixedSizeOrderSizer
from core.execution import SimulatedExecutionHandler
from core.commission import FixedPercentageCommission
from core.slippage import FixedBasisPointsSlippage
from core.events import MarketEvent, OrderEvent, FillEvent
from core.portfolio import Portfolio
from core.metrics import summarize_performance
# --- CHANGE 1: Import the correct dual-sided strategy ---
from core.strategies.ml_dual_proba_strategy import MLDualProbaStrategy

def main():
    eq = EventQueue()
    symbols = ["AAPL", "MSFT"]
    data = CSVDataHandler(
        event_queue=eq,
        symbol_to_csv={s: f"data/{s}_1min.csv" for s in symbols},
        datetime_col="datetime",
    )

    # --- CHANGE 2: Load the dual-sided probability and threshold files ---
    # Load the out-of-sample probabilities for both up and down sides
    pfeeds = {
        s: pd.read_csv(f"artifacts/{s}_oos_dual.csv", parse_dates=["timestamp"]).set_index("timestamp")
        for s in symbols
    }
    # Load the separate thresholds for up and down signals
    thr_up = {s: float(open(f"artifacts/{s}_thr_up.txt").read().strip()) for s in symbols}
    thr_dn = {s: float(open(f"artifacts/{s}_thr_dn.txt").read().strip()) for s in symbols}

    # --- CHANGE 3: Instantiate the correct strategy with the new arguments ---
    strategy = MLDualProbaStrategy(symbol_to_df=pfeeds, thr_up=thr_up, thr_dn=thr_dn)
    
    sizer = FixedSizeOrderSizer(quantity=10)
    exec_handler = SimulatedExecutionHandler(
        event_queue=eq,
        commission_model=FixedPercentageCommission(0.001),
        slippage_model=FixedBasisPointsSlippage(5.0)
    )
    portfolio = Portfolio(initial_cash=100_000.0)

    print("Starting backtest loop...")
    while data.has_data():
        data.update_bars()
        while not eq.empty():
            evt = eq.get()

            if isinstance(evt, MarketEvent):
                portfolio.on_market(evt)
                exec_handler.on_market(evt)
                signals = strategy.on_market(evt)
                if signals:
                    orders = sizer.on_signals(signals)
                    for o in orders:
                        eq.put(o)

            elif isinstance(evt, OrderEvent):
                exec_handler.on_order(evt)

            elif isinstance(evt, FillEvent):
                portfolio.on_fill(evt)

    print("Backtest complete. Calculating performance...")
    equity_df = pd.DataFrame(portfolio.equity_curve)
    
    if equity_df.empty:
        print("No trades were made. Cannot calculate KPIs.")
        return

    # For minute bars in US equities: ~252 trading days * 390 minutes per day
    kpis = summarize_performance(equity_df, rf_rate=0.0, periods_per_year=252 * 390)
    
    print("\n--- Backtest Results ---")
    for key, value in kpis.items():
        if isinstance(value, float):
            print(f"{key:<20}: {value:.4f}")
        else:
            print(f"{key:<20}: {value}")
    print("------------------------")


if __name__ == "__main__":
    main()