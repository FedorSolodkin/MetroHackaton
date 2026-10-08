"""
Пакет диспетчерского оптимизатора и СППР Линии 1.
Содержит правила принятия решений, расчет провозной способности,
выдачу горячего/холодного резервов и оптимизацию зонного оборота.
"""

from .decision_engine import (
    DispatchParams,
    SlotContext,
    DispatchDecision,
    decide_line_dispatch,
    robust_stats
)

__all__ = [
    "DispatchParams",
    "SlotContext",
    "DispatchDecision",
    "decide_line_dispatch",
    "robust_stats"
]
