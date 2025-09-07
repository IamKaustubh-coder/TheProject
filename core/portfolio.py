# core/portfolio.py
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime

from .events import MarketEvent, FillEvent

@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def update_with_fill(self, direction: str, qty: int, price: float, commission: float):
        """
        Update position with a new fill. Buys increase qty, sells decrease qty.
        Realize PnL when crossing or reducing position on the opposite side.
        """
        signed_qty = qty if direction == "BUY" else -qty

        # If adding to same side or opening new, update VWAP
        if self.quantity == 0 or (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
            new_qty = self.quantity + signed_qty
            if new_qty == 0:
                # Fully flattened; realize nothing extra here
                pass
            else:
                # VWAP update for same-side adds
                self.avg_price = (abs(self.quantity) * self.avg_price + abs(signed_qty) * price) / abs(new_qty)
            self.quantity = new_qty
        else:
            # Reducing or flipping side: realize PnL on the reduced portion
            if self.quantity > 0 and signed_qty < 0:
                close_qty = min(self.quantity, -signed_qty)
                self.realized_pnl += close_qty * (price - self.avg_price)
                self.quantity -= close_qty
                signed_qty += close_qty  # remaining (negative) to apply
                if self.quantity == 0:
                    self.avg_price = 0.0
            elif self.quantity < 0 and signed_qty > 0:
                close_qty = min(-self.quantity, signed_qty)
                self.realized_pnl += close_qty * (self.avg_price - price)
                self.quantity += close_qty
                signed_qty -= close_qty  # remaining (positive) to apply
                if self.quantity == 0:
                    self.avg_price = 0.0
            # If still remaining signed_qty after closing, it opens on the other side
            if signed_qty != 0:
                self.avg_price = price
                self.quantity += signed_qty

        # Subtract commission from realized PnL
        self.realized_pnl -= commission

    def market_value(self, last_price: float) -> float:
        return float(self.quantity) * float(last_price)

    def unrealized_pnl(self, last_price: float) -> float:
        if self.quantity > 0:
            return self.quantity * (last_price - self.avg_price)
        elif self.quantity < 0:
            return (-self.quantity) * (self.avg_price - last_price)
        return 0.0


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        self.last_prices: Dict[str, float] = {}
        self.equity_curve: List[Dict] = []  # rows: {timestamp, cash, holdings, equity}

    def _get_or_create_pos(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def on_fill(self, evt: FillEvent):
        """
        Update cash and positions based on an executed fill.
        """
        pos = self._get_or_create_pos(evt.symbol)
        gross = float(evt.fill_price) * int(evt.quantity)
        if evt.direction == "BUY":
            self.cash -= gross
        elif evt.direction == "SELL":
            self.cash += gross
        self.cash -= float(evt.commission)
        pos.update_with_fill(evt.direction, int(evt.quantity), float(evt.fill_price), float(evt.commission))

    def on_market(self, evt: MarketEvent):
        """
        Mark-to-market portfolio using the latest tradeable price.
        """
        sym = evt.symbol
        px = float(evt.ohlcv["close"])
        self.last_prices[sym] = px

        # Compute snapshot at this timestamp
        holdings = sum(self.positions[s].market_value(self.last_prices.get(s, px)) for s in self.positions)
        equity = self.cash + holdings
        self.equity_curve.append({
            "timestamp": evt.timestamp,
            "cash": self.cash,
            "holdings": holdings,
            "equity": equity,
        })

    def current_equity(self) -> float:
        if not self.equity_curve:
            holdings = sum(self.positions[s].market_value(self.last_prices.get(s, 0.0)) for s in self.positions)
            return self.cash + holdings
        return float(self.equity_curve[-1]["equity"])
