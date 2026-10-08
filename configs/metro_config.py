"""
Конфигурация параметров Линии 1 Петербургского метрополитена,
подвижного состава, ограничений графика движения и нормативов ТОиР.
Основано на официальных данных метрополитена Санкт-Петербурга и данных ментора.
"""

from typing import Dict, Any

METRO_CONFIG: Dict[str, Any] = {
    # Общие параметры Линии 1 (Кировско-Выборгская)
    "line_name": "Линия 1 (Кировско-Выборгская)",
    "line_number": 1,
    "stations_count": 19,
    # 24 вестибюля в 15-минутном датасете (5 станций имеют по 2 вестибюля); 
    # 23 в 8-месячном датасете (Технологический институт вынесен на Линию 2)
    "vestibules_count_operational": 24,
    "vestibules_count_annual": 23,
    
    # Временные нормативы движения и оборота
    "turnaround_time_minutes": 99,       # Время полного оборота (туда и обратно)
    "one_way_time_minutes": 49.5,        # Время хода в одну сторону
    "avg_interstation_minutes": 2.6,     # Среднее время перегона между станциями
    "passenger_turnover_ratio": 3.0,     # Коэффициент сменяемости пассажиров на Линии 1
    
    # Жесткие эксплуатационные ограничения линии
    "max_trains_on_line": 53,            # Максимум составов на линии одновременно (утренний пик)
    "min_interval_seconds": 113,         # Минимально допустимый интервал (1 мин 53 сек)
    "max_pairs_per_hour": 32,            # Максимальная парность (пар поездов в час)
    
    # Точки ввода и резервы подвижного состава
    "depots": {
        "avtovo": {
            "name": "Электродепо ТЧ-1 'Автово'",
            "location": "южная часть линии (ст. Автово)",
            "hot_reserves": 2,
        },
        "severnoe": {
            "name": "Электродепо ТЧ-4 'Северное'",
            "location": "северная часть линии (ст. Девяткино)",
            "hot_reserves": 2,
        }
    },
    
    # Временные нормативы готовности резерва (согласно документу ментора)
    "reserve_times": {
        "hot_reserve_deploy_min": 15,    # Время выхода горячего резерва на Линию (15-20 мин)
        "cold_reserve_deploy_min": 30,    # Время вывода холодного резерва
        "depot_exit_delay_min": 7,        # Ожидание состава после выезда из ворот депо до включения в график
    },
    
    # Промежуточные зонные станции оборота
    "turnaround_stations": [
        "Площадь Ленина",
        "Площадь Восстания",
        "Проспект Ветеранов",
        "Девяткино"
    ],
    
    # Характеристики подвижного состава (8-вагонное формирование)
    "rolling_stock": {
        "baltiec": {
            "model": "81-725.1/726.1/727.1 'Балтиец'",
            "cars_in_train": 8,
            "capacity_nominal_5pers_m2": 1478,  # Норма комфорта (порог для решения диспетчера)
            "capacity_dense_8pers_m2": 2174,
            "capacity_max_10pers_m2": 2640,     # Критическая давка
            "seats_count": 324,                 # 300 основных + 24 откидных
            "maintenance_km": {                 # Регламент ТОиР (Таблица № 4)
                "EO_period_hours": 24,          # Суточный эксплуатационный осмотр (20 +- 4 ч)
                "TO1_km": 10000,                # ТО-1 (+- 2000 км)
                "TO2_km": 35000,                # ТО-2 (+8000 / -5000 км)
                "PDR1_km": 140000,              # ПДР-1 (+- 20000 км)
                "PDR2_km": 280000,              # ПДР-2 (+- 40000 км)
                "PDR3_km": 560000,              # ПДР-3 (+- 80000 км)
                "PDR4_km": 1120000,             # ПДР-4 (+- 160000 км)
                "KR_km": 4300000,               # Капитальный ремонт (+- 800000 км)
            }
        },
        "yubileyny": {
            "model": "81-722.1/723.1/724.1 'Юбилейный'",
            "cars_in_train": 8,
            "capacity_nominal_5pers_m2": 1458,  # Норма комфорта
            "capacity_max_10pers_m2": 2608,
            "seats_count": 336,
            "maintenance_km": {                 # Регламент ТОиР (Таблица № 3)
                "TO1_km": 10000,                # ТО-1 (+- 2000 км)
                "TO2_km": 70000,                # ТО-2 (+- 10000 км)
                "TR1_km": 140000,               # ТР-1 (+- 20000 км)
                "TR2_km": 280000,               # ТР-2 (+- 40000 км)
                "TR3_km": 560000,               # ТР-3 (+- 80000 км)
                "SR_km": 1680000,               # Средний ремонт (+- 240000 км)
                "KR_km": 4300000,               # Капитальный ремонт (+- 800000 км)
            }
        }
    },
    
    # Пороги принятия решений диспетчером
    "thresholds": {
        "overcrowd_ratio": 1.15,         # Рост потока > +15% к плану -> Желтый уровень
        "critical_overcrowd_ratio": 1.25,# Рост потока > +25% к плану -> Красный уровень (выдать резерв!)
        "underload_ratio": 0.65,         # Спад потока < 65% от нормы -> Возможность снять состав
        "comfort_capacity_limit": 1460,  # Пассажиров на состав для комфортной поездки
    }
}

LINE1_CONFIG = {
    "name": "Линия 1 (Кировско-Выборгская)",
    "max_trains": METRO_CONFIG["max_trains_on_line"],
    "min_headway_sec": METRO_CONFIG["min_interval_seconds"],
    "max_pairs_per_hour": METRO_CONFIG["max_pairs_per_hour"],
    "turnaround_time_min": METRO_CONFIG["turnaround_time_minutes"],
    "length_km": 29.6,
    "passenger_turnover_ratio": METRO_CONFIG["passenger_turnover_ratio"]
}

ROLLING_STOCK = {
    "Baltiec": {
        "capacity_comfort": METRO_CONFIG["rolling_stock"]["baltiec"]["capacity_nominal_5pers_m2"],
        "capacity_dense": METRO_CONFIG["rolling_stock"]["baltiec"]["capacity_dense_8pers_m2"],
        "capacity_max": METRO_CONFIG["rolling_stock"]["baltiec"]["capacity_max_10pers_m2"],
        "cars": 8
    },
    "Yubileyny": {
        "capacity_comfort": METRO_CONFIG["rolling_stock"]["yubileyny"]["capacity_nominal_5pers_m2"],
        "capacity_max": METRO_CONFIG["rolling_stock"]["yubileyny"]["capacity_max_10pers_m2"],
        "cars": 8
    }
}

ENERGY_CONFIG = {
    # Удельный расход тяговой энергии 8-вагонного состава (~24.5 кВт⋅ч на поезд-км, ~1.45 МВт⋅ч на круг 59.2 км)
    "specific_kwh_per_train_km": 24.5,
    "consumption_kwh_per_train_round": 1450.4,
    "tariff_rub_per_kwh": 6.80
}

COST_CONFIG = {
    "toir_cost_per_km": 120.0,
    "penalty_per_passenger_los_e": 25.0
}

COMFORT_CRITERIA = {
    "los_threshold": 4.5
}

DAY_SCHEDULES = {
    "workday": {"max_pairs": 32, "min_interval": 113},
    "weekend": {"max_pairs": 24, "min_interval": 150}
}
