# core/events.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

class Event:
    """Base class for all events."""
    pass

@dataclass
class MarketEvent(Event):
    """New market data is available."""
    timestamp: datetime
    symbol: str
    ohlcv: Dict[str, Any]  # e.g., {"open":..., "high":..., "low":..., "close":..., "volume":...}

@dataclass
class SignalEvent(Event):
    """Strategy signal: LONG/SHORT/EXIT, with optional strength."""
    timestamp: datetime
    symbol: str
    direction: str  # "LONG" | "SHORT" | "EXIT"
    strength: float = 1.0

@dataclass
class OrderEvent(Event):
    """Order to be executed: MARKET/LIMIT/STOP with size."""
    timestamp: datetime
    symbol: str
    order_type: str  # "MARKET" | "LIMIT" | "STOP"
    quantity: int
    direction: str   # "BUY" | "SELL"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

@dataclass
class FillEvent(Event):
    """Result of order execution: price, quantity, fees, slippage."""
    timestamp: datetime
    symbol: str
    quantity: int
    direction: str   # "BUY" | "SELL"
    fill_price: float
    commission: float = 0.0
    slippage: float = 0.0
