"""CISD° — Powell rejection-block model, ported from Pine to Python.

The port exists so the model can be backtested, unit-tested and run headless.
The Pine original is an `indicator()`, which means it has never produced a
measured win rate, an expectancy, or a single number that could be checked.

Import surface:

    from cisd import Config, CisdEngine, Signal

Everything else is internal to the pipeline and can be imported from its module
directly when a test needs it.
"""

from .bars import Bar, resample, filter_rth
from .config import Config, pine_bug_compat
from .engine import CisdEngine, EngineStats
from .gamma import GammaContext
from .sessions import EventCalendar
from .signal import Signal, Target

__all__ = [
    "Bar",
    "CisdEngine",
    "Config",
    "EngineStats",
    "EventCalendar",
    "GammaContext",
    "Signal",
    "Target",
    "filter_rth",
    "pine_bug_compat",
    "resample",
]

__version__ = "0.1.0"
