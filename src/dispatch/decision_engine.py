"""
Интеллектуальный диспетчерский оптимизатор (Decision Engine) СППР Линии 1.

Решение = Робастная статистика аномалий (MAD) 
        + Асимметричная экономика потерь (квантиль Newsvendor Cu/Co)
        + Физика движения и парности (113 с, 32 пары/час, вместимость 1478 чел)
        + Операционные ограничения (горячий резерв 15 мин, зонный оборот по тупикам).

Основано на регламентах Петербургского метрополитена и данных ментора.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, sqrt
from statistics import NormalDist, median
from typing import Optional, Sequence, Dict, Any, List

# Попытка импорта официальных констант линии, либо дефолтные параметры
try:
    from configs.metro_config import METRO_CONFIG, ROLLING_STOCK, ENERGY_CONFIG, COST_CONFIG
    DEFAULT_CAPACITY = ROLLING_STOCK["Baltiec"]["capacity_comfort"]
    DEFAULT_MAX_TRAINS = METRO_CONFIG["max_pairs_per_hour"]
    DEFAULT_LEAD_HOT = METRO_CONFIG["reserve_times"]["hot_reserve_deploy_min"]
    DEFAULT_LEAD_COLD = METRO_CONFIG["reserve_times"]["cold_reserve_deploy_min"]
    DEFAULT_TARIFF = ENERGY_CONFIG["tariff_rub_per_kwh"]
except ImportError:
    DEFAULT_CAPACITY = 1478
    DEFAULT_MAX_TRAINS = 32
    DEFAULT_LEAD_HOT = 15
    DEFAULT_LEAD_COLD = 30
    DEFAULT_TARIFF = 6.80


@dataclass
class DispatchParams:
    """Параметры оптимизатора насыщенности линии подвижным составом."""
    capacity: int = DEFAULT_CAPACITY        # Пассажиров в составе при норме комфорта (1478 для «Балтийца»)
    target_load: float = 0.85              # Целевая комфортная наполняемость (85% номинала)
    min_trains: int = 12                   # Мин. парность (поездов/час, интервал ~5 мин в межпик)
    max_trains: int = DEFAULT_MAX_TRAINS   # Макс. парность (32 поезда/час, мин. интервал 113 с)
    max_trains_on_line: int = 53           # Физический лимит составов на линии одновременно
    
    # Newsvendor: Cu / (Cu + Co)
    # Cu (цена дефицита: давка, риск безопасности, задержка дверей) в 3 раза выше Co (цена лишнего поезда)
    cost_shortage: float = 3.0
    cost_extra: float = 1.0
    
    # Статистический порог аномалии (робастный Z-score через MAD)
    z_on: float = 2.5
    
    # Операционные лаги депо и оборота
    lead_minutes_hot: int = DEFAULT_LEAD_HOT     # Лаг вывода горячего резерва (15 мин)
    lead_minutes_cold: int = DEFAULT_LEAD_COLD   # Лаг вывода холодного резерва (30 мин)
    confirm_slots: int = 2                       # Устойчивость аномалии (2 слота по 15 мин = 30 мин)
    min_run_minutes: int = 45                    # Минимальный рейс: состав отрабатывает минимум 1 полуоборот
    slot_minutes: int = 15
    
    # Экономические константы
    tariff_kwh: float = DEFAULT_TARIFF           # Руб за кВт·ч (6.80 ₽)
    kwh_full_circle: float = 1450.4              # Расход на полный круг 59.2 км (24.5 кВт·ч/км * 59.2)
    kwh_short_circle: float = 695.8              # Расход на зонный круг ~28.4 км (до «Пл. Ленина»)
    toir_rub_per_km: float = 120.0               # Амортизация ресурса ТОиР на км пробега


@dataclass
class SlotContext:
    """Контекст одного 15-минутного временного слота."""
    time: str                                    # Например: "08:15"
    forecast: float                              # Прогноз пассажиропотока (чел/слот на лимитирующем сечении)
    history: Sequence[float]                     # Исторические значения того же слота (тот же тип дня)
    trains_now: int                              # Текущая парность (поездов/час в действующем графике)
    is_central_concentrated: bool = False        # Приток локализован в центре («Пл. Ленина» – «Пл. Восстания»)
    primary_depot: str = "avtovo"                # Ближайшее депо к очагу перегруза ("avtovo" или "severnoe")


@dataclass
class DispatchDecision:
    """Управляющая директива поездного диспетчера ЦУП."""
    action: str                                  # "release_hot" | "release_cold" | "short_turn" | "return" | "keep"
    count: int                                   # Количество рекомендуемых составов
    start_time: Optional[str]                    # Время ввода команды в действие
    end_time: Optional[str]                      # Ожидаемое окончание интервала воздействия
    location: str                                # Точка реализации ("ТЧ-1 'Автово'", "ст. 'Площадь Ленина'" и т.д.)
    z_score: float                               # Максимальный уровень аномалии в серии
    economic_impact_rub: float                   # Финансовый эффект (экономия со знаком + или затраты со знаком -)
    reason: str                                  # Понятное объяснение (XAI) для диспетчера


def robust_stats(history: Sequence[float]) -> tuple[float, float]:
    """
    Расчет робастной медианы и разброса через MAD (Median Absolute Deviation).
    Защищен от выбросов и имеет пуассоновский пол sqrt(median).
    """
    if not history:
        return 0.0, 1.0
    med = float(median(history))
    mad = float(median(abs(x - med) for x in history))
    # Магический множитель 1.4826 приводит MAD к стандартному отклонению нормального распределения
    sigma = max(1.4826 * mad, sqrt(max(med, 1.0)), 1.0)
    return med, sigma


def slot_delta(slot: SlotContext, p: DispatchParams) -> tuple[int, float]:
    """
    Расчет необходимого изменения числа поездов/час для отдельного слота.
    Использует квантиль Newsvendor для покрытия пиковых рисков.
    """
    med, sigma = robust_stats(slot.history)
    z = (slot.forecast - med) / sigma

    # Оптимальный квантиль готовности Newsvendor: Cu / (Cu + Co) = 3 / (3 + 1) = 0.75
    q = p.cost_shortage / (p.cost_shortage + p.cost_extra)
    plan_flow = max(slot.forecast + NormalDist().inv_cdf(q) * sigma, 0.0)
    flow_per_hour = plan_flow * (60.0 / p.slot_minutes)

    # Физический расчет потребного числа поездов
    need = ceil(flow_per_hour / (p.capacity * p.target_load))
    need = max(p.min_trains, min(p.max_trains, need))
    delta = need - slot.trains_now

    # Статистический фильтр: меняем график только при подтвержденной аномалии нужного знака
    if (delta > 0 and z < p.z_on) or (delta < 0 and z > -p.z_on):
        delta = 0
        
    return delta, z


def decide_line_dispatch(slots: Sequence[SlotContext], p: DispatchParams = DispatchParams()) -> DispatchDecision:
    """
    Формирует оперативное диспетчерское решение для последовательности слотов.
    
    Алгоритм:
    1. Учитывает лаг депо: до наступления lead_time поезда физически не успеют выйти.
    2. Фильтрует кратковременный шум (требуется серия >= confirm_slots).
    3. Различает типы управляющих воздействий:
       - Зонный оборот (short_turn) по ст. «Площадь Ленина» при локализации перегруза в центре;
       - Выдача горячего резерва (release_hot) за 15 минут из ТЧ-1 / ТЧ-4 при общем росте;
       - Снятие состава (return) при спаде спроса для экономии тяговой электроэнергии и ТОиР.
    """
    if not slots:
        return DispatchDecision("keep", 0, None, None, "Линия в норме", 0.0, 0.0, "Нет данных для анализа")

    lead_hot_slots = ceil(p.lead_minutes_hot / p.slot_minutes)
    need_len = max(p.confirm_slots, ceil(p.min_run_minutes / p.slot_minutes))
    
    # Расчет дельт для каждого слота
    deltas = [slot_delta(s, p) for s in slots]
    
    # Поиск первой устойчивой серии слотов
    i = lead_hot_slots
    while i < len(slots):
        d = deltas[i][0]
        if d == 0:
            i += 1
            continue
            
        j = i
        while j < len(slots) and deltas[j][0] * d > 0:
            j += 1
            
        # Серия подтвердилась
        if j - i >= need_len:
            run_deltas = deltas[i:j]
            count = min(abs(x[0]) for x in run_deltas)
            z_peak = max((x[1] for x in run_deltas), key=abs)
            
            start_t = slots[i].time
            end_t = slots[j - 1].time
            
            # 1. СИТУАЦИЯ РОСТА ПАССАЖИРОПОТОКА (d > 0)
            if d > 0:
                # Проверяем локализацию в центре
                is_center = any(slots[k].is_central_concentrated for k in range(i, j))
                
                if is_center:
                    action = "short_turn"
                    location = "Оборотные тупики ст. 'Площадь Ленина'"
                    # Зонный оборот экономит холостой пробег по сравнению с выдачей нового длинного поезда
                    # Экономим разницу между полным и зонным кругом
                    km_diff = 59.2 - 28.4
                    kwh_saved = km_diff * 24.5
                    econ = count * (kwh_saved * p.tariff_kwh + km_diff * p.toir_rub_per_km)
                    reason = (
                        f"Локальный пик центра (z={z_peak:+.1f}): "
                        f"назначить зонный оборот {count} сост. по ст. 'Площадь Ленина' "
                        f"с {start_t} до {end_t}. Экономия пробега хвостов: {econ:,.0f} руб."
                    )
                else:
                    action = "release_hot"
                    depot_name = "ТЧ-1 'Автово'" if slots[i].primary_depot == "avtovo" else "ТЧ-4 'Северное'"
                    location = depot_name
                    # Затраты на тягу дополнительного поезда
                    cost = count * (p.kwh_full_circle * p.tariff_kwh + 59.2 * p.toir_rub_per_km)
                    econ = -cost
                    reason = (
                        f"Общий всплеск потока (z={z_peak:+.1f}, +{count} пар/ч): "
                        f"выдать горячий резерв из {depot_name} на отправление {start_t} "
                        f"(таймер готовности {p.lead_minutes_hot} мин)."
                    )
                return DispatchDecision(action, count, start_t, end_t, location, round(z_peak, 2), round(econ, 2), reason)
            
            # 2. СИТУАЦИЯ СПАДА ПАССАЖИРОПОТОКА (d < 0)
            else:
                action = "return"
                location = "Электродепо ТЧ-1 / ТЧ-4"
                # Прямая экономия от снятия составов
                savings = count * (p.kwh_full_circle * p.tariff_kwh + 59.2 * p.toir_rub_per_km)
                reason = (
                    f"Устойчивый спад потока ниже нормы (z={z_peak:+.1f}, {j-i} слотов): "
                    f"снять {count} сост. в депо с {start_t}. "
                    f"Экономия тяги и ТОиР: +{savings:,.0f} руб."
                )
                return DispatchDecision(action, count, start_t, end_t, location, round(z_peak, 2), round(savings, 2), reason)
                
        i = j
        
    return DispatchDecision(
        action="keep",
        count=0,
        start_time=None,
        end_time=None,
        location="Линия 1",
        z_score=0.0,
        economic_impact_rub=0.0,
        reason="Пассажиропоток в пределах планового графика движения, вмешательство не требуется."
    )


# Демонстрационный запуск
if __name__ == "__main__":
    import random
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(42)
    print("=" * 70)
    print("ТЕСТ ДИСПЕТЧЕРСКОГО ОПТИМИЗАТОРА (DECISION ENGINE) СППР ЛИНИИ 1")
    print("=" * 70)
    
    hist_func = lambda m: [max(0, round(random.gauss(m, 0.04 * m))) for _ in range(8)]
    
    # Тест 1: Вечерний предпик и ливень в центре (17:00–18:00)
    base_flow = [5200, 5300, 5500, 5800, 7800, 8200, 8100, 7600]
    times = ["16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45", "18:00"]
    
    # Слоты с локализованным перегрузом центра
    test_slots = [
        SlotContext(
            time=t,
            forecast=f,
            history=hist_func(5400),
            trains_now=24,
            is_central_concentrated=(i >= 3)
        )
        for i, (t, f) in enumerate(zip(times, base_flow))
    ]
    
    decision = decide_line_dispatch(test_slots)
    print("\n[Результат симуляции: Ливень в центре 17:00]")
    for k, v in asdict(decision).items():
        print(f"  • {k:20s}: {v}")
