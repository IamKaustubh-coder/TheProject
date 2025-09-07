# run_loop_ml.py
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
from core.strategies.ml_proba_strategy import MLProbaStrategy

def main():
    eq = EventQueue()
    symbols = ["AAPL", "MSFT"]
    data = CSVDataHandler(
        event_queue=eq,
        symbol_to_csv={s: f"data/{s}_1min.csv" for s in symbols},
        datetime_col="datetime",
    )

    # Load OOS probabilities and thresholds
    pfeeds = {s: pd.read_csv(f"artifacts/{s}_oos_proba.csv", parse_dates=["timestamp"]).set_index("timestamp") for s in symbols}
    th_map = {s: float(open(f"artifacts/{s}_threshold.txt").read().strip()) for s in symbols}

    strategy = MLProbaStrategy(symbol_to_proba=pfeeds, threshold_map=th_map)
    sizer = FixedSizeOrderSizer(quantity=10)
    exec_handler = SimulatedExecutionHandler(
        event_queue=eq,
        commission_model=FixedPercentageCommission(0.001),
        slippage_model=FixedBasisPointsSlippage(5.0)
    )
    portfolio = Portfolio(initial_cash=100_000.0)

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

    import pandas as pd
    equity_df = pd.DataFrame(portfolio.equity_curve)
    kpis = summarize_performance(equity_df, rf_rate=0.0, periods_per_year=252*390)
    print("KPIs:", kpis)

if __name__ == "__main__":
    main()

