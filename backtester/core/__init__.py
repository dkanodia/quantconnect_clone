"""
Core engine components — event bus and backtester orchestrator.
"""

from backtester.core.backtester import Backtester
from backtester.core.event_bus import SimpleEventBus

__all__ = ["Backtester", "SimpleEventBus"]
