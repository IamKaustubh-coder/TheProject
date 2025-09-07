# core/order_sizer.py
from typing import List
from .events import SignalEvent, OrderEvent

class FixedSizeOrderSizer:
    """
    Temporary component: converts SignalEvent -> OrderEvent (MARKET) with fixed size.
    Replace with Portfolio later.
    """
    def __init__(self, quantity: int = 10):
        self.quantity = quantity

    def on_signals(self, signals: List[SignalEvent]) -> List[OrderEvent]:
        orders: List[OrderEvent] = []
        for sig in signals:
            if sig.direction == "LONG":
                orders.append(OrderEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    order_type="MARKET",
                    quantity=self.quantity,
                    direction="BUY"
                ))
            elif sig.direction == "SHORT":
                orders.append(OrderEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    order_type="MARKET",
                    quantity=self.quantity,
                    direction="SELL"
                ))
            elif sig.direction == "EXIT":
                # In a real Portfolio, compute net position to flatten; here we send opposite side
                # as a placeholder with same size.
                orders.append(OrderEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    order_type="MARKET",
                    quantity=self.quantity,
                    direction="SELL"  # simplistic exit; Portfolio will replace this logic
                ))
        return orders
