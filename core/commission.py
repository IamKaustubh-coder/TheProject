# core/commission.py
from abc import ABC, abstractmethod

class CommissionModel(ABC):
    """Abstract interface for commission models."""
    @abstractmethod
    def calculate(self, quantity: int, price: float) -> float:
        raise NotImplementedError

class FixedPercentageCommission(CommissionModel):
    """Calculates commission as a fixed percentage of total trade value."""
    def __init__(self, percentage: float = 0.001): # 0.1% = 10 bps
        self.percentage = percentage

    def calculate(self, quantity: int, price: float) -> float:
        return abs(quantity * price) * self.percentage