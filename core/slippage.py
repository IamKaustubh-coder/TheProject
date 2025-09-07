# core/slippage.py
from abc import ABC, abstractmethod
from .events import OrderEvent

class SlippageModel(ABC):
    """Abstract interface for slippage models."""
    @abstractmethod
    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Returns the new price after applying slippage."""
        raise NotImplementedError

class NoSlippage(SlippageModel):
    """Ideal model with zero slippage."""
    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return fill_price

class FixedBasisPointsSlippage(SlippageModel):
    """Applies a fixed basis point slippage cost to the fill price."""
    def __init__(self, basis_points: float = 5.0): # 5 bps
        self.bps = basis_points / 10000.0

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        if order.direction == "BUY":
            return fill_price * (1.0 + self.bps) # Buys at a slightly higher price
        elif order.direction == "SELL":
            return fill_price * (1.0 - self.bps) # Sells at a slightly lower price
        return fill_price