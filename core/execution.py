# core/execution.py
from typing import Dict
from .events import MarketEvent, OrderEvent, FillEvent
from .event_queue import EventQueue
from .commission import CommissionModel
from .slippage import SlippageModel

class SimulatedExecutionHandler:
    """
    A simulated execution handler for backtesting.
    It consumes OrderEvents and produces FillEvents, modeling
    commission and slippage.
    """
    def __init__(
        self,
        event_queue: EventQueue,
        commission_model: CommissionModel,
        slippage_model: SlippageModel
    ):
        self.event_queue = event_queue
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        # Cache of the most recent market event per symbol
        self._latest_market_events: Dict[str, MarketEvent] = {}

    def on_market(self, event: MarketEvent):
        """Updates the latest market data for a symbol."""
        self._latest_market_events[event.symbol] = event

    def on_order(self, event: OrderEvent):
        """Simulates the execution of an order."""
        sym = event.symbol
        if sym not in self._latest_market_events:
            print(f"WARN: No market data for {sym} to execute order.")
            return

        latest_data = self._latest_market_events[sym]
        
        # Assumption: MARKET orders fill at the next bar's open price.
        # This is a common and reasonably realistic assumption.
        base_fill_price = float(latest_data.ohlcv["open"])

        # 1. Apply slippage model
        final_fill_price = self.slippage_model.calculate(event, base_fill_price)

        # 2. Apply commission model
        commission = self.commission_model.calculate(event.quantity, final_fill_price)

        # 3. Create and queue the FillEvent
        fill = FillEvent(
            timestamp=latest_data.timestamp,
            symbol=sym,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=final_fill_price,
            commission=commission
        )
        self.event_queue.put(fill)