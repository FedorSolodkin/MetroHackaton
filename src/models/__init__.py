"""
ML-ядро предиктивной аналитики пассажиропотока Санкт-Петербургского метрополитена (Линия 1).
Включает:
- HistoricalMedianBaseline: бейзлайн исторической медианы без утечек данных
- MultiHorizonPredictor: градиентный бустинг (LightGBM / CatBoost) на горизонты 15, 30, 60, 120 мин
- ExplainablePredictor: SHAP-интерпретация влияния погодных и пространственных факторов
"""

from .baseline import HistoricalMedianBaseline

__all__ = ["HistoricalMedianBaseline"]
