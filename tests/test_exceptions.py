"""
Exhaustive tests for the exception hierarchy.

Covers:
- Every subclass is catchable as BacktestError
- Every exception stores all passed kwargs as attributes
- str(exc) surfaces all non-None context fields without raising
- portfolio_snapshot round-trips through str() safely
- __repr__ produces a non-empty string
- _enrich correctly backfills missing fields without overwriting existing ones
- RiskRejectionError is catchable as RiskModelError and BacktestError
- PositionSizeLimitExceeded is a RiskRejectionError (soft rejection)
- MaxDrawdownExceeded is NOT a RiskRejectionError (hard halt)
- order_id field on OrderError and its subclasses
"""

from __future__ import annotations

from typing import Any

import pytest

from backtester.exceptions import (
    BacktestError,
    ConfigurationError,
    DataCorruptionError,
    DataFeedError,
    DataNotFoundError,
    ExecutionError,
    InsufficientFundsError,
    InsufficientPositionError,
    InvalidOrderError,
    InvalidSignalError,
    MaxDrawdownExceeded,
    NegativeCashError,
    OptimizationError,
    OrderError,
    OrderRejectedError,
    PortfolioError,
    PositionSizeLimitExceeded,
    RiskModelError,
    RiskRejectionError,
    StrategyError,
    WFOError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNAPSHOT: dict[str, Any] = {
    "cash": 95_000.0,
    "total_equity": 100_000.0,
    "positions": {
        "AAPL": {
            "quantity": 10.0,
            "avg_entry_price": 500.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
        }
    },
}

# All concrete exception classes that can be raised in the engine.
ALL_LEAF_EXCEPTIONS: list[type[BacktestError]] = [
    DataFeedError,
    DataNotFoundError,
    DataCorruptionError,
    StrategyError,
    InvalidSignalError,
    OrderError,
    InvalidOrderError,
    OrderRejectedError,
    ExecutionError,
    InsufficientFundsError,
    InsufficientPositionError,
    PortfolioError,
    NegativeCashError,
    RiskModelError,
    RiskRejectionError,
    MaxDrawdownExceeded,
    PositionSizeLimitExceeded,
    OptimizationError,
    WFOError,
    ConfigurationError,
]


# ---------------------------------------------------------------------------
# Inheritance / catchability
# ---------------------------------------------------------------------------


class TestInheritance:
    @pytest.mark.parametrize("exc_cls", ALL_LEAF_EXCEPTIONS)
    def test_catchable_as_backtest_error(self, exc_cls: type[BacktestError]) -> None:
        """Every exception in the engine should be caught by `except BacktestError`."""
        try:
            raise exc_cls("test message")
        except BacktestError:
            pass

    def test_risk_rejection_is_risk_model_error(self) -> None:
        assert issubclass(RiskRejectionError, RiskModelError)

    def test_position_size_limit_is_risk_rejection(self) -> None:
        assert issubclass(PositionSizeLimitExceeded, RiskRejectionError)

    def test_max_drawdown_is_not_risk_rejection(self) -> None:
        assert not issubclass(MaxDrawdownExceeded, RiskRejectionError)

    def test_max_drawdown_is_risk_model_error(self) -> None:
        assert issubclass(MaxDrawdownExceeded, RiskModelError)

    def test_position_size_limit_not_caught_by_max_drawdown_handler(self) -> None:
        """Soft and hard risk exceptions must be distinguishable."""
        with pytest.raises(RiskRejectionError):
            raise PositionSizeLimitExceeded("too big", requested_pct=0.2, limit_pct=0.1)

        # MaxDrawdownExceeded is NOT a RiskRejectionError
        with pytest.raises(MaxDrawdownExceeded):
            raise MaxDrawdownExceeded("drawdown", current_drawdown=0.3, limit=0.2)

        # Trying to catch MaxDrawdown as RiskRejection should re-raise
        try:
            try:
                raise MaxDrawdownExceeded("halt")
            except RiskRejectionError:
                pass  # should NOT reach here
            else:
                pass  # exception was not caught as RiskRejectionError (correct)
        except MaxDrawdownExceeded:
            pass  # correct — MaxDrawdownExceeded escapes the RiskRejectionError handler


# ---------------------------------------------------------------------------
# Context field storage
# ---------------------------------------------------------------------------


class TestContextFieldStorage:
    def test_base_fields_stored(self) -> None:
        exc = BacktestError(
            "msg",
            strategy_name="Foo",
            symbol="AAPL",
            bar_index=42,
            portfolio_snapshot=_SNAPSHOT,
        )
        assert exc.strategy_name == "Foo"
        assert exc.symbol == "AAPL"
        assert exc.bar_index == 42
        assert exc.portfolio_snapshot is _SNAPSHOT

    def test_defaults_are_none(self) -> None:
        exc = BacktestError("msg")
        assert exc.strategy_name is None
        assert exc.symbol is None
        assert exc.bar_index is None
        assert exc.portfolio_snapshot is None

    def test_order_error_stores_order_id(self) -> None:
        exc = OrderError("bad order", order_id="abc-123")
        assert exc.order_id == "abc-123"

    def test_order_rejected_error_stores_reason(self) -> None:
        exc = OrderRejectedError("rejected", order_id="xyz", reason="risk gate")
        assert exc.order_id == "xyz"
        assert exc.reason == "risk gate"

    def test_insufficient_funds_stores_required_available(self) -> None:
        exc = InsufficientFundsError("broke", required=5000.0, available=3000.0)
        assert exc.required == pytest.approx(5000.0)
        assert exc.available == pytest.approx(3000.0)

    def test_insufficient_position_stores_requested_held(self) -> None:
        exc = InsufficientPositionError("short", requested=10.0, held=5.0)
        assert exc.requested == pytest.approx(10.0)
        assert exc.held == pytest.approx(5.0)

    def test_max_drawdown_stores_drawdown_and_limit(self) -> None:
        exc = MaxDrawdownExceeded("big drop", current_drawdown=0.35, limit=0.20)
        assert exc.current_drawdown == pytest.approx(0.35)
        assert exc.limit == pytest.approx(0.20)

    def test_position_size_limit_stores_pcts(self) -> None:
        exc = PositionSizeLimitExceeded("huge order", requested_pct=0.25, limit_pct=0.10)
        assert exc.requested_pct == pytest.approx(0.25)
        assert exc.limit_pct == pytest.approx(0.10)

    def test_portfolio_snapshot_forwarded_through_kwargs(self) -> None:
        """portfolio_snapshot must survive the **kwargs chain in subclasses."""
        exc = InsufficientFundsError(
            "no cash",
            required=1000.0,
            available=100.0,
            portfolio_snapshot=_SNAPSHOT,
        )
        assert exc.portfolio_snapshot is _SNAPSHOT


# ---------------------------------------------------------------------------
# str() surfaces context fields
# ---------------------------------------------------------------------------


class TestStrOutput:
    def test_str_includes_message(self) -> None:
        assert "hello world" in str(BacktestError("hello world"))

    def test_str_includes_strategy_name(self) -> None:
        exc = BacktestError("msg", strategy_name="MyStrategy")
        assert "MyStrategy" in str(exc)

    def test_str_includes_symbol(self) -> None:
        exc = BacktestError("msg", symbol="TSLA")
        assert "TSLA" in str(exc)

    def test_str_includes_bar_index(self) -> None:
        exc = BacktestError("msg", bar_index=99)
        assert "99" in str(exc)

    def test_portfolio_snapshot_in_str_does_not_raise(self) -> None:
        exc = BacktestError("msg", portfolio_snapshot=_SNAPSHOT)
        result = str(exc)
        assert isinstance(result, str)  # must not raise

    def test_portfolio_snapshot_indicated_in_str(self) -> None:
        exc = BacktestError("msg", portfolio_snapshot=_SNAPSHOT)
        assert "portfolio_snapshot" in str(exc)

    def test_none_fields_not_in_str(self) -> None:
        exc = BacktestError("only message")
        s = str(exc)
        assert "strategy=" not in s
        assert "symbol=" not in s
        assert "bar_index=" not in s

    def test_order_id_in_order_error_str(self) -> None:
        exc = OrderError("bad", order_id="my-id")
        assert "my-id" in str(exc)

    def test_order_rejected_reason_in_str(self) -> None:
        exc = OrderRejectedError("rejected", reason="risk gate")
        assert "risk gate" in str(exc)

    def test_insufficient_funds_amounts_in_str(self) -> None:
        exc = InsufficientFundsError("broke", required=5000.0, available=100.0)
        s = str(exc)
        assert "5000" in s
        assert "100" in s

    def test_max_drawdown_pcts_in_str(self) -> None:
        exc = MaxDrawdownExceeded("ouch", current_drawdown=0.3, limit=0.2)
        s = str(exc)
        assert "30" in s   # "30.00%" form
        assert "20" in s

    def test_position_size_pcts_in_str(self) -> None:
        exc = PositionSizeLimitExceeded("too big", requested_pct=0.25, limit_pct=0.10)
        s = str(exc)
        assert "25" in s
        assert "10" in s

    def test_all_none_snapshot_not_in_str(self) -> None:
        exc = BacktestError("msg")  # no portfolio_snapshot
        assert "portfolio_snapshot" not in str(exc)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    @pytest.mark.parametrize("exc_cls", ALL_LEAF_EXCEPTIONS)
    def test_repr_is_non_empty_string(self, exc_cls: type[BacktestError]) -> None:
        exc = exc_cls("test")
        r = repr(exc)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_repr_contains_class_name(self) -> None:
        exc = MaxDrawdownExceeded("big drop")
        assert "MaxDrawdownExceeded" in repr(exc)

    def test_repr_does_not_raise_with_snapshot(self) -> None:
        exc = BacktestError("msg", portfolio_snapshot=_SNAPSHOT)
        repr(exc)  # must not raise


# ---------------------------------------------------------------------------
# _enrich (Backtester helper)
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

from backtester.core.backtester import Backtester
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import (
    BacktestResult,
    Bar,
    DataFeed,
    Iterator,
    Reporter,
    Strategy,
)
from backtester.risk.risk_models import NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy


class _MinimalFeed(DataFeed):
    def __iter__(self) -> Iterator[Bar]:
        return iter([])

    def reset(self) -> None:
        pass

    @property
    def symbols(self) -> list[str]:
        return []


class _MinimalReporter(Reporter):
    def report(self, result: BacktestResult) -> None:
        pass


def _make_bt() -> Backtester:
    slip = ZeroSlippage()
    comm = ZeroCommission()
    return Backtester(
        feed=_MinimalFeed(),
        strategy=EventDrivenStrategy(),
        execution=NextOpenExecution(slip, comm),
        slippage=slip,
        commission=comm,
        risk=NoRisk(),
        reporter=_MinimalReporter(),
    )


_BAR = Bar(
    symbol="AAPL",
    timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
    open=100.0,
    high=105.0,
    low=98.0,
    close=102.0,
    volume=10_000.0,
)


class TestEnrich:
    def test_enrich_sets_strategy_name_when_none(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg")
        bt._enrich(exc, _BAR, 5)
        assert exc.strategy_name == "EventDrivenStrategy"

    def test_enrich_sets_symbol_when_none(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg")
        bt._enrich(exc, _BAR, 5)
        assert exc.symbol == "AAPL"

    def test_enrich_sets_bar_index_when_none(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg")
        bt._enrich(exc, _BAR, 7)
        assert exc.bar_index == 7

    def test_enrich_does_not_overwrite_existing_strategy_name(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg", strategy_name="Original")
        bt._enrich(exc, _BAR, 5)
        assert exc.strategy_name == "Original"

    def test_enrich_does_not_overwrite_existing_symbol(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg", symbol="MSFT")
        bt._enrich(exc, _BAR, 5)
        assert exc.symbol == "MSFT"

    def test_enrich_does_not_overwrite_existing_bar_index(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg", bar_index=3)
        bt._enrich(exc, _BAR, 99)
        assert exc.bar_index == 3

    def test_enrich_returns_same_exception_object(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg")
        result = bt._enrich(exc, _BAR, 1)
        assert result is exc

    def test_enrich_with_none_bar_leaves_symbol_none(self) -> None:
        bt = _make_bt()
        exc = BacktestError("msg")
        bt._enrich(exc, None, 1)
        assert exc.symbol is None
