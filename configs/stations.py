"""
Справочник станций и вестибюлей Линии 1 Петербургского метрополитена.
Включает коды станций из системы РПДП (111-129), координаты, типы ТПУ и пересадки.
"""

from typing import Dict, Any, List

STATIONS_LINE_1: List[Dict[str, Any]] = [
    {
        "code": 111,
        "name": "Проспект Ветеранов",
        "vestibules": ["пр. Ветеранов-1", "пр. Ветеранов-2"],
        "lat": 59.8422,
        "lon": 30.2541,
        "type": "residential_hub",
        "description": "Южная конечная станция, крупнейший спальный хаб юго-запада (Красносельский р-н, Стрельна)",
        "order": 1,
        "has_turnaround": True
    },
    {
        "code": 112,
        "name": "Ленинский проспект",
        "vestibules": ["Ленинский пр.-1", "Ленинский пр.-2"],
        "lat": 59.8524,
        "lon": 30.2685,
        "type": "residential",
        "description": "Крупный жилой массив, высокая плотность утренней посадки",
        "order": 2,
        "has_turnaround": False
    },
    {
        "code": 113,
        "name": "Автово",
        "vestibules": ["Автово"],
        "lat": 59.8672,
        "lon": 30.2611,
        "type": "depot_hub",
        "description": "Примыкание к электродепо ТЧ-1 'Автово', точка выдачи и захода составов",
        "order": 3,
        "has_turnaround": True
    },
    {
        "code": 114,
        "name": "Кировский завод",
        "vestibules": ["Кировский завод"],
        "lat": 59.8797,
        "lon": 30.2620,
        "type": "industrial",
        "description": "Промышленный узел, пересадка на строящуюся Линию 6 (Путиловская)",
        "order": 4,
        "has_turnaround": False
    },
    {
        "code": 115,
        "name": "Нарвская",
        "vestibules": ["Нарвская"],
        "lat": 59.9011,
        "lon": 30.2748,
        "type": "hub",
        "description": "Линейный пункт (ЛП), крупный транспортный узел площади Стачек",
        "order": 5,
        "has_turnaround": False
    },
    {
        "code": 116,
        "name": "Балтийская",
        "vestibules": ["Балтийская"],
        "lat": 59.9072,
        "lon": 30.2997,
        "type": "railway_terminal",
        "description": "Балтийский вокзал (пригородные электрички Петергоф, Ораниенбаум, Гатчина)",
        "order": 6,
        "has_turnaround": False
    },
    {
        "code": 117,
        "name": "Технологический институт",
        "vestibules": ["Технологический институт", "Технологический институт-1"],
        "lat": 59.9164,
        "lon": 30.3186,
        "type": "interchange",
        "description": "Кросс-платформенная пересадка на Линию 2 (Московско-Петроградская)",
        "order": 7,
        "has_turnaround": False
    },
    {
        "code": 118,
        "name": "Пушкинская",
        "vestibules": ["Пушкинская"],
        "lat": 59.9205,
        "lon": 30.3297,
        "type": "interchange_railway",
        "description": "Витебский вокзал, пересадка на станцию 'Звенигородская' (Линия 5)",
        "order": 8,
        "has_turnaround": False
    },
    {
        "code": 119,
        "name": "Владимирская",
        "vestibules": ["Владимирская"],
        "lat": 59.9277,
        "lon": 30.3481,
        "type": "interchange",
        "description": "Пересадка на станцию 'Достоевская' (Линия 4)",
        "order": 9,
        "has_turnaround": False
    },
    {
        "code": 120,
        "name": "Площадь Восстания",
        "vestibules": ["пл.Восстания-1", "пл.Восстания-2", "пл. Восстания-1", "пл. Восстания-2"],
        "lat": 59.9317,
        "lon": 30.3606,
        "type": "central_hub",
        "description": "Московский вокзал ('Сапсаны', поезда дальнего следования), пересадка на ст. 'Маяковская' (Линия 3), оборотные пути",
        "order": 10,
        "has_turnaround": True
    },
    {
        "code": 121,
        "name": "Чернышевская",
        "vestibules": ["Чернышевская"],
        "lat": 59.9446,
        "lon": 30.3597,
        "type": "central",
        "description": "Центральный деловой район, глубокое заложение",
        "order": 11,
        "has_turnaround": False
    },
    {
        "code": 122,
        "name": "Площадь Ленина",
        "vestibules": ["пл.Ленина-1", "пл.Ленина-2", "пл. Ленина-1", "пл. Ленина-2"],
        "lat": 59.9555,
        "lon": 30.3556,
        "type": "railway_terminal",
        "description": "Финляндский вокзал (пригородные электрички Карельского перешейка), оборотные тупики, промежуточная точка ввода/отстоя",
        "order": 12,
        "has_turnaround": True
    },
    {
        "code": 123,
        "name": "Выборгская",
        "vestibules": ["Выборгская"],
        "lat": 59.9711,
        "lon": 30.3475,
        "type": "business_industrial",
        "description": "Линейный пункт (ЛП), Выборгская сторона",
        "order": 13,
        "has_turnaround": False
    },
    {
        "code": 124,
        "name": "Лесная",
        "vestibules": ["Лесная"],
        "lat": 59.9849,
        "lon": 30.3444,
        "type": "residential",
        "description": "Крупный жилой и студенческий узел (Лесотехнический университет)",
        "order": 14,
        "has_turnaround": False
    },
    {
        "code": 125,
        "name": "Площадь Мужества",
        "vestibules": ["пл.Мужества", "пл. Мужества"],
        "lat": 59.9996,
        "lon": 30.3661,
        "type": "residential_hub",
        "description": "Транспортный узел севера, пересадка на трамвайную и троллейбусную сеть",
        "order": 15,
        "has_turnaround": False
    },
    {
        "code": 126,
        "name": "Политехническая",
        "vestibules": ["Политехническая"],
        "lat": 60.0089,
        "lon": 30.3709,
        "type": "student",
        "description": "Политехнический университет Петра Великого",
        "order": 16,
        "has_turnaround": False
    },
    {
        "code": 127,
        "name": "Академическая",
        "vestibules": ["Академическая"],
        "lat": 60.0128,
        "lon": 30.3961,
        "type": "residential",
        "description": "Густонаселенный спальный район Гражданка",
        "order": 17,
        "has_turnaround": False
    },
    {
        "code": 128,
        "name": "Гражданский проспект",
        "vestibules": ["Гражданский", "Гражданский пр.", "Гражданский пр"],
        "lat": 60.0351,
        "lon": 30.4182,
        "type": "residential",
        "description": "Север Гражданки, высокий пассажиропоток в утренний пик",
        "order": 18,
        "has_turnaround": False
    },
    {
        "code": 129,
        "name": "Девяткино",
        "vestibules": ["Девяткино I", "Девяткино II", "Девяткино-1", "Девяткино-2", "Девяткино 1", "Девяткино 2"],
        "lat": 60.0503,
        "lon": 30.4431,
        "type": "regional_hub",
        "description": "Северная конечная станция, выход в Ленинградскую область (Мурино), ж/д станция Девяткино, депо ТЧ-4 'Северное'",
        "order": 19,
        "has_turnaround": True
    }
]

# Нормализованный маппинг
VESTIBULE_TO_STATION: Dict[str, Dict[str, Any]] = {}
for st in STATIONS_LINE_1:
    for vest in st["vestibules"]:
        clean_key = vest.strip().lower().replace("ё", "е")
        VESTIBULE_TO_STATION[clean_key] = {
            "station_code": st["code"],
            "station_name": st["name"],
            "vestibule": vest,
            "type": st["type"],
            "lat": st["lat"],
            "lon": st["lon"],
            "order": st["order"],
            "has_turnaround": st["has_turnaround"]
        }

# Дополнительные частые варианты написания
ALIASES = {
    "пр.ветеранов-1": "пр. ветеранов-1",
    "пр.ветеранов-2": "пр. ветеранов-2",
    "ленинский пр-1": "ленинский пр.-1",
    "ленинский пр-2": "ленинский пр.-2",
    "технологический институт-1": "технологический институт",
    "пл. восстания-1": "пл.восстания-1",
    "пл. восстания-2": "пл.восстания-2",
    "пл. ленина-1": "пл.ленина-1",
    "пл. ленина-2": "пл.ленина-2",
    "пл. мужества": "пл.мужества",
    "гражданский пр.": "гражданский",
    "гражданский пр": "гражданский",
    "девяткино-1": "девяткино i",
    "девяткино-2": "девяткино ii",
}

for alias, target in ALIASES.items():
    if target in VESTIBULE_TO_STATION:
        data_copy = dict(VESTIBULE_TO_STATION[target])
        data_copy["vestibule"] = alias
        VESTIBULE_TO_STATION[alias] = data_copy

def get_station_info(vestibule_raw: str) -> Dict[str, Any]:
    """Унифицированный поиск станции по названию вестибюля."""
    s = vestibule_raw.strip().lower().replace("ё", "е")
    if s in VESTIBULE_TO_STATION:
        res = dict(VESTIBULE_TO_STATION[s])
        res["has_turnout"] = res.get("has_turnaround", False)
        return res
    # Пытаемся по частичному совпадению
    for k, v in VESTIBULE_TO_STATION.items():
        if k in s or s in k:
            res = dict(v)
            res["has_turnout"] = res.get("has_turnaround", False)
            return res
    raise KeyError(f"Неизвестный вестибюль: {vestibule_raw}")

LINE1_STATIONS = [s["name"] for s in STATIONS_LINE_1]
LINE1_VESTIBULES = [
    'Автово', 'Академическая', 'Балтийская', 'Владимирская', 'Выборгская',
    'Кировский завод', 'Ленинский пр.-1', 'Ленинский пр.-2', 'Лесная',
    'Нарвская', 'Политехническая', 'Пушкинская', 'Чернышевская',
    'гражданский пр.', 'девяткино-1', 'девяткино-2', 'пл. восстания-1',
    'пл. восстания-2', 'пл. мужества', 'пл.Ленина-1', 'пл.Ленина-2',
    'пр.ветеранов-1', 'пр.ветеранов-2', 'технологический институт-1'
]

# Точные физические расстояния и время хода по 18 перегонам Линии 1
# Откалибровано по GPS-координатам станций, длине линии 29.6 км и графиковому времени хода 49.5 мин (35.9 км/ч)
INTERSTATION_SECTIONS: List[Dict[str, Any]] = [
    {"from": "Проспект Ветеранов", "to": "Ленинский проспект", "distance_km": 1.45, "travel_time_min": 2.4, "cumulative_min": 2.4},
    {"from": "Ленинский проспект", "to": "Автово", "distance_km": 1.77, "travel_time_min": 3.0, "cumulative_min": 5.4},
    {"from": "Автово", "to": "Кировский завод", "distance_km": 1.45, "travel_time_min": 2.4, "cumulative_min": 7.8},
    {"from": "Кировский завод", "to": "Нарвская", "distance_km": 2.59, "travel_time_min": 4.3, "cumulative_min": 12.1},
    {"from": "Нарвская", "to": "Балтийская", "distance_km": 1.61, "travel_time_min": 2.7, "cumulative_min": 14.8},
    {"from": "Балтийская", "to": "Технологический институт", "distance_km": 1.53, "travel_time_min": 2.6, "cumulative_min": 17.4},
    {"from": "Технологический институт", "to": "Пушкинская", "distance_km": 0.80, "travel_time_min": 1.3, "cumulative_min": 18.7},
    {"from": "Пушкинская", "to": "Владимирская", "distance_km": 1.36, "travel_time_min": 2.3, "cumulative_min": 21.0},
    {"from": "Владимирская", "to": "Площадь Восстания", "distance_km": 0.86, "travel_time_min": 1.4, "cumulative_min": 22.4},
    {"from": "Площадь Восстания", "to": "Чернышевская", "distance_km": 1.50, "travel_time_min": 2.5, "cumulative_min": 24.9},
    {"from": "Чернышевская", "to": "Площадь Ленина", "distance_km": 1.29, "travel_time_min": 2.2, "cumulative_min": 27.1},
    {"from": "Площадь Ленина", "to": "Выборгская", "distance_km": 1.87, "travel_time_min": 3.1, "cumulative_min": 30.2},
    {"from": "Выборгская", "to": "Лесная", "distance_km": 1.61, "travel_time_min": 2.7, "cumulative_min": 32.9},
    {"from": "Лесная", "to": "Площадь Мужества", "distance_km": 2.12, "travel_time_min": 3.5, "cumulative_min": 36.5},
    {"from": "Площадь Мужества", "to": "Политехническая", "distance_km": 1.11, "travel_time_min": 1.9, "cumulative_min": 38.3},
    {"from": "Политехническая", "to": "Академическая", "distance_km": 1.53, "travel_time_min": 2.6, "cumulative_min": 40.9},
    {"from": "Академическая", "to": "Гражданский проспект", "distance_km": 2.88, "travel_time_min": 4.8, "cumulative_min": 45.7},
    {"from": "Гражданский проспект", "to": "Девяткино", "distance_km": 2.28, "travel_time_min": 3.8, "cumulative_min": 49.5},
]

def get_interstation_travel_time(from_station: str, to_station: str) -> float:
    """Возвращает время хода в минутах между любыми двумя станциями Линии 1."""
    st_names = [s["name"] for s in STATIONS_LINE_1]
    if from_station not in st_names or to_station not in st_names:
        return 2.75
    idx1, idx2 = st_names.index(from_station), st_names.index(to_station)
    if idx1 == idx2:
        return 0.0
    start, end = min(idx1, idx2), max(idx1, idx2)
    return sum(sec["travel_time_min"] for sec in INTERSTATION_SECTIONS[start:end])

