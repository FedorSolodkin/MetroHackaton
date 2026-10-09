"""Конвейер по дням: скачать архив (Яндекс.Диск) -> распаковать -> посчитать скорость вокруг станций -> удалить архив и базу.
python gtfs_pipeline.py <часть> <число_частей>   (дни наших 4 месяцев делятся между процессами по номеру части)
Результат: _gtfs/speed_<дата>.csv. Пропускает уже обработанные дни. Из C:\metro."""
import os, subprocess, sys, time
from datetime import date, timedelta
import requests
sys.path.insert(0, "MetroHackaton/analysis")
import gtfs_speed

PK = "https://disk.yandex.ru/d/3QzxcM9IxfJcog"; API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
PERIODS = [("2026-02-01", "2026-02-28"), ("2026-05-01", "2026-05-31"), ("2026-07-01", "2026-07-31"), ("2026-09-01", "2026-09-30")]


def days():
    out = []
    for a, b in PERIODS:
        d = date.fromisoformat(a)
        while d <= date.fromisoformat(b):
            out.append(d.isoformat()); d += timedelta(days=1)
    return out


def download(day):
    fn = f"_gtfs/{day}.7z"
    for k in range(4):
        try:
            r = requests.get(API, params={"public_key": PK, "path": f"/FromPortalRealtime/{day[:4]}/{day}.7z"}, timeout=40)
            if r.status_code != 200:
                return None if r.status_code == 404 else (time.sleep(10) or None) if k == 3 else time.sleep(10)
            with requests.get(r.json()["href"], stream=True, timeout=180) as s, open(fn + ".part", "wb") as f:
                for ch in s.iter_content(1 << 20):
                    f.write(ch)
            os.replace(fn + ".part", fn); return fn
        except requests.RequestException:
            time.sleep(10)
    return None


def main(part, parts):
    os.makedirs("_gtfs", exist_ok=True)
    mine = [d for i, d in enumerate(days()) if i % parts == part]
    for day in mine:
        out = f"_gtfs/speed_{day}.csv"
        if os.path.exists(out):
            continue
        t0 = time.time(); fn = download(day)
        if not fn:
            print(day, "нет файла", flush=True); continue
        sq = f"_gtfs/x{day}/{day}/{day}.sqlite"; os.makedirs(f"_gtfs/x{day}", exist_ok=True)
        subprocess.run(["7z", "x", "-y", f"-o_gtfs/x{day}", fn], capture_output=True)
        try:
            gtfs_speed.process(sq).to_csv(out, index=False); print(day, f"готово за {time.time()-t0:.0f} с", flush=True)
        except Exception as e:
            print(day, "ошибка:", type(e).__name__, e, flush=True)
        finally:
            for p in (fn, sq):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(f"_gtfs/x{day}/{day}"); os.rmdir(f"_gtfs/x{day}")
            except OSError:
                pass


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
