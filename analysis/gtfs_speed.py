"""Индекс пробок по скорости наземного транспорта (GTFS-realtime архив СПб) вокруг станций Линии 1.
Для каждого снимка (каждые 10 с; берём каждый STRIDE-й) считаем транспорт в радиусе RADIUS_KM от каждой станции и
агрегируем в 15-минутные слоты: число машин, медианная скорость движущихся, средняя скорость, доля стоящих.
Запуск: python gtfs_speed.py <дата> ... -> _gtfs/speed_<дата>.csv. Из C:\metro."""
import os, sqlite3, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, "repo/events")
from stations import STATIONS
from google.transit import gtfs_realtime_pb2 as pb

RADIUS_KM = 1.5; STRIDE = 3
ST = [s for s, _, _ in STATIONS]; LAT = np.array([a for _, a, _ in STATIONS]); LON = np.array([b for _, _, b in STATIONS])


def dist_km(lat, lon):
    """матрица расстояний (машины x станции), км, равнопрямоугольная аппроксимация"""
    dy = (lat[:, None] - LAT[None, :]) * 111.2
    dx = (lon[:, None] - LON[None, :]) * 111.2 * np.cos(np.radians(59.95))
    return np.hypot(dx, dy)


def process(path):
    con = sqlite3.connect(path); rows = []
    ts_list = [r[0] for r in con.execute("select feed_timestamp from gtfs_realtime_data where data is not null order by feed_timestamp")][::STRIDE]
    for t in ts_list:
        blob = con.execute("select data from gtfs_realtime_data where feed_timestamp=?", (t,)).fetchone()[0]
        m = pb.FeedMessage(); m.ParseFromString(blob)
        v = [(e.vehicle.position.latitude, e.vehicle.position.longitude, e.vehicle.position.speed) for e in m.entity if e.HasField("vehicle")]
        if not v:
            continue
        a = np.array(v, dtype=float); D = dist_km(a[:, 0], a[:, 1]); near = D <= RADIUS_KM
        for j, s in enumerate(ST):
            sp = a[near[:, j], 2]
            if len(sp):
                mv = sp[sp > 0.5]
                rows.append((t, s, len(sp), np.median(mv) if len(mv) else 0.0, sp.mean(), (sp < 0.5).mean()))
    con.close()
    d = pd.DataFrame(rows, columns=["t", "station", "n_veh", "med_move", "mean_all", "share_stopped"])
    d["ts"] = pd.to_datetime(d.t, unit="s", utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None).dt.floor("15min")
    return d.groupby(["ts", "station"]).agg(n_veh=("n_veh", "mean"), med_move=("med_move", "mean"), mean_all=("mean_all", "mean"), share_stopped=("share_stopped", "mean")).reset_index()


if __name__ == "__main__":
    for day in sys.argv[1:]:
        out = f"_gtfs/speed_{day}.csv"
        if os.path.exists(out):
            print(day, "готово"); continue
        sq = f"_gtfs/x{day}/{day}/{day}.sqlite"
        if not os.path.exists(sq):
            os.makedirs(f"_gtfs/x{day}", exist_ok=True); subprocess.run(["7z", "x", "-y", f"-o_gtfs/x{day}", f"_gtfs/{day}.7z"], capture_output=True)
        process(sq).to_csv(out, index=False); print(day, "обработано", flush=True)
        try:
            os.remove(sq)
        except OSError:
            pass
