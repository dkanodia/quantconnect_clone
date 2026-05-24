"""
Strategy API — base classes for event-driven and vectorized strategies.
"""

from backtester.strategy.base import BaseStrategy
from backtester.strategy.event_driven import EventDrivenStrategy
from backtester.strategy.vectorized import VectorizedStrategy

__all__ = ["BaseStrategy", "EventDrivenStrategy", "VectorizedStrategy"]
