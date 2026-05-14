from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Position:
    instrument_name: str = ''
    option_type: str = ''
    strike: float = 0.0
    entry_price: float = 0.0
    quantity: float = 0.0
    status: str = 'open'


@dataclass
class Trade:
    instrument_name: str = ''
    pnl: float = 0.0
    status: str = 'closed'


@dataclass
class Portfolio:
    initial_cash: float = 10000.0
    cash: float = 10000.0
    open_positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)

    def buy_option(self, instrument_name: str, option_type: str, strike: float, price: float, quantity: float) -> bool:
        cost = price * quantity
        if cost <= 0 or cost > self.cash:
            return False
        self.cash -= cost
        self.open_positions.append(Position(instrument_name, option_type, strike, price, quantity))
        return True


def print_portfolio_summary(portfolio: Portfolio) -> None:
    print(f'Current cash: ${portfolio.cash:,.2f}; Open positions: {len(portfolio.open_positions)}')
