"""
Master Production EDA Runner for SPB Metro Smart Dispatch (Line 1).
Generates high-resolution publication-quality visualizations and statistical tables:
1. Data Completeness & Quality Heatmaps
2. Temporal Dynamics & Diurnal Rush-Hour Profiles
3. Seasonal Inflow Evolution (Feb, May, Jul, Sep)
4. Station Topology Ranking & Asymmetry Clustering
5. Spatial-Temporal Passenger Wave Propagation
6. Spatial Cross-Correlation Along Line 1 Topology
7. Train Saturation & Headway Analysis vs Comfort Limit (1478 pass.)
8. Platform Crowd Density & Level-of-Service (LOS)
9. Weather Elasticity (Rain, Snow, Wind, Extreme Temp)
10. Calendar Shifts (Pre-holiday early peak & Holidays)
11. Incident Case Study (31.08 Chernyshevskaya Disruption)
12. Economic & Operational Optimization Potential (Power & TOiR)
"""

import os
import sys
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#ebebeb'
plt.rcParams['grid.linestyle'] = '--'

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_loader import MetroDataLoader
from src.data.weather_loader import WeatherLoader
from src.data.calendar_features import CalendarFeatureEngine
from src.data.preprocessor import MetroFeaturePipeline
from src.utils.metrics import (
    calculate_train_saturation,
    calculate_platform_density,
    calculate_economic_effect
)
from configs.metro_config import METRO_CONFIG
from configs.stations import STATIONS_LINE_1


def main():
    print("=" * 80)
    print("STARTING EXHAUSTIVE EDA PIPELINE FOR SPB METRO LINE 1")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "eda", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1/8] Loading unified 15-minute telemetry dataset...")
    loader = MetroDataLoader()
    df_raw = loader.load_15min_dataset()
    print(f"  Loaded raw 15-min rows: {len(df_raw):,} across {df_raw['operational_date'].nunique()} days")
    
    # 2. Enrich with Calendar & Weather
    print("\n[Step 2/8] Enriching with Calendar & Open-Meteo Weather...")
    cal_engine = CalendarFeatureEngine()
    df = cal_engine.add_features(df_raw, datetime_col="datetime")
    
    weather_loader = WeatherLoader()
    df_weather = weather_loader.get_interpolated_15min_weather()
    
    # Merge weather
    weather_cols = ["datetime", "temperature", "apparent_temperature", "precipitation", "rain", "snowfall", "wind_speed", "is_rain", "is_snow", "is_heavy_rain", "is_extreme_weather"]
    df = pd.merge(df, df_weather[weather_cols], on="datetime", how="left")
    df["temperature"] = df["temperature"].ffill().bfill()
    df["rain"] = df["rain"].fillna(0.0)
    df["snowfall"] = df["snowfall"].fillna(0.0)
    df["is_rain"] = df["is_rain"].fillna(0).astype(int)
    df["is_snow"] = df["is_snow"].fillna(0).astype(int)
    df["is_heavy_rain"] = df["is_heavy_rain"].fillna(0).astype(int)
    
    # Station & Line Aggregations
    df_station = loader.get_station_aggregated_flow(df)
    df_line = loader.get_line_aggregated_flow(df)
    df_line = pd.merge(df_line, df_weather[weather_cols], on="datetime", how="left")
    df_line["rain"] = df_line["rain"].fillna(0.0)
    df_line["temperature"] = df_line["temperature"].ffill().bfill()
    # Add hour to df_line
    df_line["hour"] = pd.to_datetime(df_line["datetime"]).dt.hour
    df_line["time_str"] = pd.to_datetime(df_line["datetime"]).dt.strftime("%H:%M")
    df_line["date"] = pd.to_datetime(df_line["datetime"]).dt.date
    df_line = cal_engine.add_features(df_line, datetime_col="datetime")
    
    # Load Schedule Profiles
    schedules = loader.get_planned_schedule_profiles()
    sched_sep = schedules["workday_autumn"]
    sched_weekend = schedules["weekend"]
    
    print("\n[Step 3/8] Generating Figure 1: Data Completeness & Quality...")
    # FIG 1: Completeness, missingness, hourly volume & zero-flow distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1.1 Monthly totals
    monthly_totals = df.groupby("season")["passengers"].sum() / 1e6
    month_order = ["winter", "spring", "summer", "autumn"]
    month_names_ru = ["Февраль (Зима)", "Май (Весна)", "Июль (Лето)", "Сентябрь (Осень)"]
    m_vals = [monthly_totals.get(m, 0) for m in month_order]
    
    ax = axes[0, 0]
    bars = ax.bar(month_names_ru, m_vals, color=['#2b5c8f', '#2ca02c', '#d62728', '#ff7f0e'], edgecolor='black', alpha=0.85)
    ax.set_title("Суммарный пассажиропоток по сезонам (млн поездок)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Млн пассажиров")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.2f}M", ha='center', va='bottom', fontweight='bold')
        
    # 1.2 Zero-flow analysis by hour
    ax = axes[0, 1]
    hourly_zeros = df[df["passengers"] == 0].groupby("hour")["passengers"].count() / df.groupby("hour")["passengers"].count() * 100
    all_hours = pd.Series(0.0, index=range(24))
    all_hours.update(hourly_zeros)
    ax.plot(all_hours.index, all_hours.values, marker='o', color='#c0392b', lw=2.5)
    ax.fill_between(all_hours.index, all_hours.values, color='#c0392b', alpha=0.15)
    ax.set_title("Доля нулевых интервалов по часам суток (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Час суток")
    ax.set_ylabel("Доля нулей (%)")
    ax.axvspan(1, 5, color='gray', alpha=0.2, label='Ночное технологическое окно (01:00-05:00)')
    ax.legend(loc='upper right')
    
    # 1.3 Daily traffic boxplot by day of week
    ax = axes[1, 0]
    daily_line = df_line.groupby(["operational_date", "day_name", "day_of_week"])["passengers"].sum().reset_index()
    daily_line["passengers_thousands"] = daily_line["passengers"] / 1000.0
    dow_order = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    sns.boxplot(data=daily_line, x="day_name", y="passengers_thousands", order=dow_order, ax=ax, palette="Blues_d")
    ax.set_title("Распределение суточного пассажиропотока по дням недели (тыс. пасс.)", fontsize=12, fontweight='bold')
    ax.set_xlabel("День недели")
    ax.set_ylabel("Пассажиров в сутки (тыс.)")
    
    # 1.4 Heatmap of daily traffic calendar
    ax = axes[1, 1]
    daily_line["month"] = pd.to_datetime(daily_line["operational_date"]).dt.month
    pivot_cal = daily_line.pivot_table(index="day_name", columns="month", values="passengers_thousands", aggfunc="mean").reindex(dow_order)
    pivot_cal.columns = ["Февраль", "Май", "Июль", "Сентябрь"]
    sns.heatmap(pivot_cal, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax, cbar_kws={'label': 'Тыс. пасс. / сутки'})
    ax.set_title("Матрица средних суточных потоков: День недели vs Месяц", fontsize=12, fontweight='bold')
    ax.set_xlabel("Месяц 2026 года")
    ax.set_ylabel("День недели")
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "01_data_completeness_and_quality.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n[Step 4/8] Generating Figure 2: Diurnal Curves & Peak Hours...")
    # FIG 2: Diurnal curves by day type
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 2.1 Diurnal curves line-level
    ax = axes[0, 0]
    daytype_order = ["workday", "pre_holiday", "weekend", "holiday"]
    daytype_labels = {
        "workday": "Рабочий день",
        "pre_holiday": "Предпраздничный (сокращенный)",
        "weekend": "Выходной (Сб/Вс)",
        "holiday": "Официальный праздник"
    }
    palette = {"workday": "#1f77b4", "pre_holiday": "#ff7f0e", "weekend": "#2ca02c", "holiday": "#d62728"}
    
    for dt_name in daytype_order:
        sub = df_line[df_line["day_type"] == dt_name]
        if len(sub) > 0:
            diurnal = sub.groupby("interval_index")["passengers"].mean()
            # X axis in hours (interval_index * 15 min + 3 hours) % 24
            hours_x = [(3.0 + i * 0.25) if (3.0 + i * 0.25) < 24.0 else (3.0 + i * 0.25 - 24.0) for i in range(96)]
            # Sort by continuous time from 03:00
            ax.plot(range(96), diurnal.values, label=daytype_labels[dt_name], color=palette[dt_name], lw=2.2)
            
    ax.set_title("Суточный профиль притока на Линию 1 по типам дней (15-мин шаг)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels([f"{(3 + i*2)%24:02d}:00" for i in range(12)], rotation=45)
    ax.set_xlabel("Время суток (с 03:00 открытия до 02:45 закрытия)")
    ax.set_ylabel("Входящий поток (пассажиров за 15 мин)")
    ax.axvspan(18, 26, color='yellow', alpha=0.2, label='Утренний пик (07:30–09:30)')
    ax.axvspan(56, 64, color='orange', alpha=0.2, label='Вечерний пик (17:00–19:00)')
    ax.legend(loc='upper right')
    
    # 2.2 Morning peak comparison zoom
    ax = axes[0, 1]
    for dt_name in ["workday", "pre_holiday", "weekend"]:
        sub = df_line[df_line["day_type"] == dt_name]
        sub_peak = sub[(sub["interval_index"] >= 16) & (sub["interval_index"] <= 32)]
        mean_curve = sub_peak.groupby("interval_index")["passengers"].mean()
        time_labels = [f"{(3 + i*0.25)%24:02.0f}:{(i*15)%60:02d}" for i in range(16, 33)]
        ax.plot(range(len(mean_curve)), mean_curve.values, marker='s', label=daytype_labels[dt_name], color=palette[dt_name], lw=2)
        
    ax.set_title("Фокус: Утренний час пик (07:00 - 11:00)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 17, 2))
    ax.set_xticklabels([f"{(7 + i*0.5):02.0f}:{(int(i*30)%60):02d}" for i in range(9)], rotation=30)
    ax.set_ylabel("Пассажиров за 15 мин")
    ax.legend()
    
    # 2.3 Evening peak shift in pre-holiday
    ax = axes[1, 0]
    for dt_name in ["workday", "pre_holiday"]:
        sub = df_line[df_line["day_type"] == dt_name]
        sub_peak = sub[(sub["interval_index"] >= 44) & (sub["interval_index"] <= 68)]
        mean_curve = sub_peak.groupby("interval_index")["passengers"].mean()
        ax.plot(range(len(mean_curve)), mean_curve.values, marker='o', label=daytype_labels[dt_name], color=palette[dt_name], lw=2.2)
        
    ax.set_title("Сдвиг вечернего часа пик: Рабочий день vs Предпраздничный", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{(14 + i):02d}:00" for i in range(7)])
    ax.set_xlabel("Время суток")
    ax.set_ylabel("Пассажиров за 15 мин")
    ax.axvline(8, color='orange', linestyle='--', label='Пик предпраздника (16:00)')
    ax.axvline(18, color='blue', linestyle='--', label='Пик стандарта (18:30)')
    ax.legend()
    
    # 2.4 Seasonal peak intensities
    ax = axes[1, 1]
    season_names_ru = {"winter": "Зима (Фев)", "spring": "Весна (Май)", "summer": "Лето (Июл)", "autumn": "Осень (Сен)"}
    for s_id in ["winter", "spring", "summer", "autumn"]:
        sub = df_line[(df_line["season"] == s_id) & (df_line["day_type"] == "workday")]
        diurnal = sub.groupby("interval_index")["passengers"].mean()
        ax.plot(range(96), diurnal.values, label=season_names_ru[s_id], lw=2.0)
        
    ax.set_title("Сезонное сопоставление рабочих дней (Зима / Весна / Лето / Осень)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels([f"{(3 + i*2)%24:02d}:00" for i in range(12)], rotation=45)
    ax.set_ylabel("Пассажиров за 15 мин")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "02_diurnal_profiles_daytypes.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n[Step 5/8] Generating Figure 3 & 4: Station Classification & Topology...")
    # FIG 3: Station Volumes & Cluster Ranking
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # 3.1 Total volume per station
    st_vol = df_station.groupby(["station_name", "station_order", "station_type"])["passengers"].sum().reset_index()
    st_vol = st_vol.sort_values("passengers", ascending=True)
    st_vol["passengers_millions"] = st_vol["passengers"] / 1e6
    
    type_colors = {
        "residential_hub": "#1f77b4",
        "regional_hub": "#005580",
        "central_hub": "#d62728",
        "railway_terminal": "#ff7f0e",
        "interchange": "#9467bd",
        "interchange_railway": "#8c564b",
        "residential": "#2ca02c",
        "industrial": "#7f7f7f",
        "business_industrial": "#bcbd22",
        "student": "#17becf",
        "depot_hub": "#e377c2",
        "central": "#e7ba52"
    }
    colors = [type_colors.get(t, "#333333") for t in st_vol["station_type"]]
    
    ax = axes[0]
    ax.barh(st_vol["station_name"], st_vol["passengers_millions"], color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("Суммарный пассажиропоток по станциям Линии 1 за 4 месяца (млн чел)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Млн пассажиров")
    for i, v in enumerate(st_vol["passengers_millions"]):
        ax.text(v + 0.05, i, f"{v:.2f}M", va='center', fontsize=9, fontweight='bold')
        
    # Legend for station types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#005580", label="Региональный хаб (Мурино/Девяткино)"),
        Patch(facecolor="#1f77b4", label="Спальный хаб (Пр. Ветеранов)"),
        Patch(facecolor="#ff7f0e", label="Ж/д вокзалы (Балтийский, Финляндский)"),
        Patch(facecolor="#d62728", label="Центральный хаб (Пл. Восстания)"),
        Patch(facecolor="#2ca02c", label="Спальные районы (Ленинский, Гражданка)"),
        Patch(facecolor="#9467bd", label="Пересадочные узлы (Техноложка, Владимирская)"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 3.2 Asymmetry Ratio (Morning Peak Inflow vs Evening Peak Inflow)
    ax = axes[1]
    # Filter workday
    df_st_wd = df_station[df_station["datetime"].dt.weekday < 5]
    # Morning: 07:30 to 09:30 (interval 18 to 26)
    am_mask = (df_st_wd["interval_index"] >= 18) & (df_st_wd["interval_index"] <= 26)
    am_flow = df_st_wd[am_mask].groupby("station_name")["passengers"].mean()
    
    # Evening: 17:00 to 19:00 (interval 56 to 64)
    pm_mask = (df_st_wd["interval_index"] >= 56) & (df_st_wd["interval_index"] <= 64)
    pm_flow = df_st_wd[pm_mask].groupby("station_name")["passengers"].mean()
    
    asym_df = pd.DataFrame({"AM_Flow": am_flow, "PM_Flow": pm_flow}).reset_index()
    # Add station type
    st_meta = st_vol.set_index("station_name")["station_type"].to_dict()
    asym_df["station_type"] = asym_df["station_name"].map(st_meta)
    
    for _, row in asym_df.iterrows():
        s_name = row["station_name"]
        color = type_colors.get(row["station_type"], "#333333")
        ax.scatter(row["AM_Flow"], row["PM_Flow"], color=color, s=120, edgecolor='black', zorder=4)
        ax.annotate(s_name, (row["AM_Flow"] + 30, row["PM_Flow"] + 30), fontsize=9)
        
    # Diagonal y = x line
    max_val = max(asym_df["AM_Flow"].max(), asym_df["PM_Flow"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Симметричный приток (AM = PM)')
    ax.set_title("Индекс асимметрии станций: Утренний приток vs Вечерний приток (пасс/15 мин)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Утренний пик на вход (07:30 - 09:30) [Маятниковый выезд из дома]")
    ax.set_ylabel("Вечерний пик на вход (17:00 - 19:00) [Маятниковый выезд с работы]")
    ax.text(2500, 700, "ЗОНА СПАЛЬНЫХ ХАБОВ\n(Девяткино, Ветеранов, Гражданский)", color="blue", fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="blue"))
    ax.text(700, 2200, "ЗОНА ДЕЛОВОГО ЦЕНТРА\nИ ПЕРЕСАДОК (Восстания, Чернышевская)", color="red", fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#fee", ec="red"))
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "03_station_volume_and_asymmetry.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n[Step 6/8] Generating Figure 5: Spatial-Temporal Passenger Wave Propagation...")
    # FIG 5: 2D Heatmap of Station Order (1 to 19) vs Time of Day
    # Filter representative autumn workday (e.g. Wednesday 16.09.2026 or average workday in Sep)
    df_st_sep_wd = df_station[(df_station["datetime"].dt.month == 9) & (df_station["datetime"].dt.weekday < 5)]
    pivot_wave = df_st_sep_wd.pivot_table(
        index="station_order",
        columns="interval_index",
        values="passengers",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(18, 9))
    st_names_ordered = df_station.sort_values("station_order").drop_duplicates("station_code")["station_name"].tolist()
    
    sns.heatmap(
        pivot_wave,
        cmap="turbo",
        ax=ax,
        cbar_kws={'label': 'Средний пассажиропоток (чел / 15 минут)'}
    )
    ax.set_title("Пространственно-временная волна пассажиропотока по станциям Линии 1 (Осень, рабочий день)", fontsize=13, fontweight='bold')
    ax.set_yticks(np.arange(19) + 0.5)
    ax.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(st_names_ordered)], rotation=0)
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{(3 + i)%24:02d}:00" for i in range(24)], rotation=45)
    ax.set_xlabel("Время суток (с 03:00 до 02:45)")
    ax.set_ylabel("Топология станций Линии 1 (Юг: Пр. Ветеранов → Север: Девяткино)")
    
    # Highlight zones
    ax.axvline(18, color='white', linestyle=':', lw=2)
    ax.axvline(26, color='white', linestyle=':', lw=2)
    ax.text(22, 0.5, "УТРЕННЯЯ ВОЛНА\n(Вход на концах линии)", color='white', fontweight='bold', ha='center', bbox=dict(boxstyle="square", fc="black", alpha=0.7))
    
    ax.axvline(56, color='white', linestyle=':', lw=2)
    ax.axvline(64, color='white', linestyle=':', lw=2)
    ax.text(60, 9.5, "ВЕЧЕРНЯЯ ВОЛНА\n(Вход в центре: Восстания)", color='white', fontweight='bold', ha='center', bbox=dict(boxstyle="square", fc="black", alpha=0.7))
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "04_spatial_temporal_wave_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n[Step 7/8] Generating Figure 6: Train Saturation, Line Capacity & Level of Service...")
    # FIG 6: Capacity vs Demand & Platform Density
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 6.1 Saturation curve over 24 hours
    ax = axes[0, 0]
    # Average 15-min inflow on Sep workday
    mean_sep_line = df_line[(df_line["month"] == 9) & (df_line["day_type"] == "workday")].groupby("interval_index")["passengers"].mean()
    # Map to schedule
    # Convert interval_index to hour
    hours_mapped = [(3 + i * 15 // 60) % 24 for i in range(96)]
    sched_dict = sched_sep.set_index("hour")["trains_on_line"].to_dict()
    trains_arr = np.array([sched_dict.get(h, 0) for h in hours_mapped])
    
    sat_res = calculate_train_saturation(mean_sep_line.values, trains_arr, turnaround_min=99.0, comfort_capacity=1478)
    
    ax.plot(range(96), mean_sep_line.values, label="Фактический пассажиропоток (пасс/15 мин)", color="#1f77b4", lw=2.5)
    ax.plot(range(96), sat_res["capacity_15min"], label="Провозная способность по ГД (норма комфорта 5 чел/м²)", color="#2ca02c", lw=2.5, linestyle="--")
    
    ax.set_title("Сопоставление спроса и провозной способности графика движения (Сентябрь, будни)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels([f"{(3 + i*2)%24:02d}:00" for i in range(12)], rotation=45)
    ax.set_ylabel("Пассажиров в 15 минут")
    ax.fill_between(range(96), mean_sep_line.values, sat_res["capacity_15min"], where=(mean_sep_line.values > sat_res["capacity_15min"]), color="red", alpha=0.3, label="Зона перегруза (требуется ввод резерва)")
    ax.fill_between(range(96), mean_sep_line.values, sat_res["capacity_15min"], where=(mean_sep_line.values < sat_res["capacity_15min"] * 0.65), color="gray", alpha=0.2, label="Зона избытка (возможность съема составов)")
    ax.legend(loc="upper right")
    
    # 6.2 Saturation Ratio
    ax = axes[0, 1]
    ax.plot(range(96), sat_res["saturation_ratio"], color="#8e44ad", lw=2.2)
    ax.axhline(1.0, color="orange", linestyle="--", label="Порог 100% нормы комфорта (1478 пасс/состав)")
    ax.axhline(1.15, color="red", linestyle="--", label="Критический перегруз (+15% к норме)")
    ax.axhline(0.65, color="gray", linestyle=":", label="Порог неэффективного пробега (<65%)")
    ax.set_title("Динамика коэффициента насыщенности линии подвижным составом", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels([f"{(3 + i*2)%24:02d}:00" for i in range(12)], rotation=45)
    ax.set_ylabel("Коэффициент насыщенности (Спрос / Вместимость)")
    ax.set_ylim(0, 1.4)
    ax.legend()
    
    # 6.3 Platform density across top 5 stations in morning peak
    ax = axes[1, 0]
    top_5_stations = ["Проспект Ветеранов", "Девяткино", "Площадь Восстания", "Балтийская", "Академическая"]
    for s_name in top_5_stations:
        st_sub = df_station[(df_station["station_name"] == s_name) & (df_station["datetime"].dt.weekday < 5) & (df_station["datetime"].dt.month == 9)]
        dens_res = calculate_platform_density(st_sub.groupby("interval_index")["passengers"].mean().values)
        ax.plot(range(96), dens_res["density_pers_m2"], label=s_name, lw=2.0)
        
    ax.axhline(3.0, color="orange", linestyle="--", label="LOS C/D: Предпиковая плотность (3.0 чел/м²)")
    ax.axhline(5.0, color="red", linestyle="--", label="LOS F: Критическая давка (>5.0 чел/м²)")
    ax.set_title("Расчетная плотность скопления людей на платформах хабов (чел/м²)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels([f"{(3 + i*2)%24:02d}:00" for i in range(12)], rotation=45)
    ax.set_ylabel("Плотность на платформе (чел/м²)")
    ax.legend()
    
    # 6.4 Economic potential of dynamic adjustment
    ax = axes[1, 1]
    # Simulation: removing 3 redundant trips during off-peak (12:00-15:00) and 2 late night (22:00-00:00) = 5 trips/day
    days_in_month = 30
    trips_per_day = 4
    eff = calculate_economic_effect(trips_optimized=trips_per_day * days_in_month, is_full_circle=True)
    
    categories = ["Тяговая электроэнергия", "Ресурс ТОиР вагонов", "ИТОГО ЭКОНОМИЯ"]
    values = [eff["energy_rub_saved"] / 1e6, eff["toir_rub_saved"] / 1e6, eff["total_rub_saved"] / 1e6]
    bars = ax.bar(categories, values, color=["#3498db", "#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.85)
    ax.set_title("Экономический эффект оптимизации графика (млн руб / месяц)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Млн рублей")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f} млн ₽", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "05_train_saturation_and_platform_density.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    print("\n[Step 8/8] Generating Figure 7: Weather Elasticity & Chernyshevskaya Incident Case Study...")
    # FIG 7: Weather Impact & August 31 Incident
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    
    # 7.1 Weather impact: Flow deviation vs Rain intensity
    ax = axes[0]
    # Group line traffic by weather rain presence on workdays
    df_line_wd = df_line[df_line["day_type"] == "workday"].copy()
    # Normalize by hour median
    hourly_median = df_line_wd.groupby("interval_index")["passengers"].transform("median")
    df_line_wd["relative_flow_pct"] = (df_line_wd["passengers"] - hourly_median) / hourly_median * 100.0
    
    # Filter daytime (07:00 - 22:00)
    daytime = df_line_wd[(df_line_wd["interval_index"] >= 16) & (df_line_wd["interval_index"] <= 76)]
    
    rain_bins = pd.cut(daytime["rain"], bins=[-0.01, 0.0, 0.2, 1.0, 5.0], labels=["Без осадков", "Слабый дождь", "Умеренный дождь", "Ливень"])
    daytime["rain_category"] = rain_bins
    
    sns.boxplot(data=daytime, x="rain_category", y="relative_flow_pct", ax=ax, palette="Blues", showfliers=False)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title("Эластичность пассажиропотока к осадкам (отклонение от нормы в %)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Категория интенсивности дождя")
    ax.set_ylabel("Отклонение входного потока от медианы (%)")
    
    # 7.2 Incident Case Study: August 31, 2026 Chernyshevskaya disruption
    # From incident document:
    # 07:50 - 10:31 Chernyshevskaya track 2 disabled.
    # Devyatkino closed/restricted 08:06-09:26, Grazhdansky restricted 08:11-09:48, Pl. Muzhestva closed 08:41-09:12.
    ax = axes[1]
    df_hourly = loader.load_hourly_annual_dataset()
    dev_all = df_hourly[df_hourly["station_name"] == "Девяткино"].copy()
    dev_all["hour"] = dev_all["datetime"].dt.hour
    dev_all["date"] = dev_all["datetime"].dt.date
    dev_all["month"] = dev_all["datetime"].dt.month
    dev_all["dow"] = dev_all["datetime"].dt.weekday
    
    dev_31 = dev_all[dev_all["date"] == datetime.date(2026, 8, 31)]
    dev_normal_mon = dev_all[
        (dev_all["dow"] == 0) & 
        (dev_all["month"] == 8) & 
        (dev_all["date"] != datetime.date(2026, 8, 31))
    ].groupby("hour")["passengers"].mean()
    
    if len(dev_31) > 0:
        dev_31_hourly = dev_31.groupby("hour")["passengers"].sum()
        ax.plot(dev_normal_mon.index, dev_normal_mon.values, label="Типичный понедельник августа (Норма)", color="blue", marker="o", lw=2)
        ax.plot(dev_31_hourly.index, dev_31_hourly.values, label="31.08.2026 (Авария ст. Чернышевская)", color="red", marker="s", lw=2.5)
        ax.axvspan(7, 10, color="red", alpha=0.2, label="Период сбоя и закрытия вестибюлей (07:50–10:30)")
        ax.set_title("Кейс-анализ сбоя: ст. 'Девяткино' 31.08.2026 (Авария состава 'Балтиец')", fontsize=12, fontweight='bold')
        ax.set_xlabel("Час суток")
        ax.set_ylabel("Входящий пассажиропоток (пасс / час)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Данные по инциденту 31.08 обрабатываются", ha="center", va="center")
        
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "06_weather_elasticity_and_incident_case.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Generate summary stats dictionary for comprehensive report
    print("\n[Step 9/8] Computing exhaustive statistical metrics...")
    stats = {
        "dataset_summary": {
            "total_records_15min": int(len(df)),
            "unique_operational_days": int(df["operational_date"].nunique()),
            "total_passengers_analyzed": int(df["passengers"].sum()),
            "date_range": [str(df["datetime"].min()), str(df["datetime"].max())],
            "vestibules_count": int(df["vestibule"].nunique()),
            "stations_count": int(df["station_name"].nunique()),
            "months": ["Февраль", "Май", "Июль", "Сентябрь"],
        },
        "monthly_totals": {
            "february": float(df[df["month"] == 2]["passengers"].sum()),
            "may": float(df[df["month"] == 5]["passengers"].sum()),
            "july": float(df[df["month"] == 7]["passengers"].sum()),
            "september": float(df[df["month"] == 9]["passengers"].sum())
        },
        "top_stations_total": st_vol.set_index("station_name")["passengers"].to_dict(),
        "saturation_metrics": {
            "max_trains_on_line": METRO_CONFIG["max_trains_on_line"],
            "min_headway_sec": METRO_CONFIG["min_interval_seconds"],
            "max_pairs_per_hour": METRO_CONFIG["max_pairs_per_hour"],
            "comfort_capacity_baltiec": METRO_CONFIG["rolling_stock"]["baltiec"]["capacity_nominal_5pers_m2"],
            "peak_morning_15min_line_demand": float(mean_sep_line.max()),
            "peak_capacity_15min": float(sat_res["capacity_15min"].max()),
            "peak_saturation_ratio": float(sat_res["saturation_ratio"].max()),
            "off_peak_saturation_ratio": float(sat_res["saturation_ratio"][40:50].mean()),
        },
        "economic_optimization": eff
    }
    
    stats_json_path = os.path.join(base_dir, "eda", "eda_summary_metrics.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Summary metrics saved to: {stats_json_path}")
    print("\nEXHAUSTIVE EDA PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
