"""
backtesting/portfolio.py

Portfolio and trade tracking for ETH options backtesting.

This module tracks:
- cash
- open positions
- closed trades
- portfolio equity
- realized profit/loss

This is a simplified research portfolio.
It does not connect to an exchange and does not place real trades.
"""

from dataclasses import dataclass, field


@dataclass
class Position:
    """
    Represents an open option position.
    """

    option_type: str
    entry_price: float
    quantity: float
    strike_price: float
    days_to_expiry: int
    direction: str = "long"

    def market_value(self, current_option_price: float) -> float:
        """
        Calculate current market value of the position.
        """

        if current_option_price < 0:
            raise ValueError("current_option_price cannot be negative.")

        return float(current_option_price * self.quantity)

    def unrealized_pnl(self, current_option_price: float) -> float:
        """
        Calculate unrealized profit/loss.
        """

        if self.direction == "long":
            return float((current_option_price - self.entry_price) * self.quantity)

        if self.direction == "short":
            return float((self.entry_price - current_option_price) * self.quantity)

        raise ValueError("direction must be either 'long' or 'short'.")


@dataclass
class Trade:
    """
    Represents a completed trade.
    """

    option_type: str
    entry_price: float
    exit_price: float
    quantity: float
    strike_price: float
    direction: str
    pnl: float


@dataclass
class Portfolio:
    """
    Simple portfolio for backtesting.
    """

    initial_cash: float
    cash: float = field(init=False)
    open_positions: list[Position] = field(default_factory=list)
    closed_trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Initialize portfolio cash.
        """

        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0.")

        self.cash = float(self.initial_cash)

    def open_position(
        self,
        option_type: str,
        entry_price: float,
        quantity: float,
        strike_price: float,
        days_to_expiry: int,
        direction: str = "long",
    ) -> None:
        """
        Open a new option position.

        For long positions:
            cash decreases by premium paid.

        For short positions:
            cash increases by premium received.
        """

        if entry_price <= 0:
            raise ValueError("entry_price must be greater than 0.")

        if quantity <= 0:
            raise ValueError("quantity must be greater than 0.")

        if strike_price <= 0:
            raise ValueError("strike_price must be greater than 0.")

        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be greater than 0.")

        direction = direction.lower().strip()

        if direction not in {"long", "short"}:
            raise ValueError("direction must be either 'long' or 'short'.")

        cost = entry_price * quantity

        if direction == "long":
            if cost > self.cash:
                raise ValueError("Not enough cash to open long position.")

            self.cash -= cost

        elif direction == "short":
            self.cash += cost

        position = Position(
            option_type=option_type.lower().strip(),
            entry_price=float(entry_price),
            quantity=float(quantity),
            strike_price=float(strike_price),
            days_to_expiry=int(days_to_expiry),
            direction=direction,
        )

        self.open_positions.append(position)

    def close_position(
        self,
        position_index: int,
        exit_price: float,
    ) -> Trade:
        """
        Close an open position by index.
        """

        if exit_price < 0:
            raise ValueError("exit_price cannot be negative.")

        if position_index < 0 or position_index >= len(self.open_positions):
            raise IndexError("Invalid position_index.")

        position = self.open_positions.pop(position_index)

        exit_value = exit_price * position.quantity

        if position.direction == "long":
            self.cash += exit_value
            pnl = (exit_price - position.entry_price) * position.quantity

        elif position.direction == "short":
            self.cash -= exit_value
            pnl = (position.entry_price - exit_price) * position.quantity

        else:
            raise ValueError("Invalid position direction.")

        trade = Trade(
            option_type=position.option_type,
            entry_price=float(position.entry_price),
            exit_price=float(exit_price),
            quantity=float(position.quantity),
            strike_price=float(position.strike_price),
            direction=position.direction,
            pnl=float(pnl),
        )

        self.closed_trades.append(trade)

        return trade

    def total_unrealized_pnl(
        self,
        current_option_prices: list[float],
    ) -> float:
        """
        Calculate total unrealized PnL for all open positions.

        current_option_prices must match open_positions order.
        """

        if len(current_option_prices) != len(self.open_positions):
            raise ValueError(
                "current_option_prices length must match number of open positions."
            )

        total = 0.0

        for position, current_price in zip(self.open_positions, current_option_prices):
            total += position.unrealized_pnl(current_price)

        return float(total)

    def equity(
        self,
        current_option_prices: list[float] | None = None,
    ) -> float:
        """
        Calculate portfolio equity.

        If no current prices are provided, equity is cash only.
        """

        if current_option_prices is None:
            return float(self.cash)

        if len(current_option_prices) != len(self.open_positions):
            raise ValueError(
                "current_option_prices length must match number of open positions."
            )

        open_value = 0.0

        for position, current_price in zip(self.open_positions, current_option_prices):
            if position.direction == "long":
                open_value += position.market_value(current_price)
            elif position.direction == "short":
                open_value -= position.market_value(current_price)
            else:
                raise ValueError("Invalid position direction.")

        return float(self.cash + open_value)

    def realized_pnl(self) -> float:
        """
        Calculate total realized PnL from closed trades.
        """

        return float(sum(trade.pnl for trade in self.closed_trades))

    def number_of_open_positions(self) -> int:
        """
        Return number of open positions.
        """

        return len(self.open_positions)

    def number_of_closed_trades(self) -> int:
        """
        Return number of closed trades.
        """

        return len(self.closed_trades)

    def summary(self) -> dict[str, float | int]:
        """
        Return portfolio summary.
        """

        return {
            "initial_cash": float(self.initial_cash),
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl()),
            "open_positions": self.number_of_open_positions(),
            "closed_trades": self.number_of_closed_trades(),
            "cash_return": float((self.cash - self.initial_cash) / self.initial_cash),
        }


def print_portfolio_summary(portfolio: Portfolio) -> None:
    """
    Print portfolio summary.
    """

    summary = portfolio.summary()

    print("========== PORTFOLIO SUMMARY ==========")
    print(f"Initial cash:        ${summary['initial_cash']:,.2f}")
    print(f"Current cash:        ${summary['cash']:,.2f}")
    print(f"Realized PnL:        ${summary['realized_pnl']:,.2f}")
    print(f"Open positions:      {summary['open_positions']}")
    print(f"Closed trades:       {summary['closed_trades']}")
    print(f"Cash return:         {summary['cash_return']:.2%}")
    print("=======================================")


if __name__ == "__main__":
    print("portfolio.py is running")

    portfolio = Portfolio(initial_cash=10_000.0)

    portfolio.open_position(
        option_type="call",
        entry_price=200.0,
        quantity=0.5,
        strike_price=3200.0,
        days_to_expiry=30,
        direction="long",
    )

    print("\nAfter opening position:")
    print_portfolio_summary(portfolio)

    trade = portfolio.close_position(
        position_index=0,
        exit_price=260.0,
    )

    print("\nClosed trade:")
    print(trade)

    print("\nAfter closing position:")
    print_portfolio_summary(portfolio)
