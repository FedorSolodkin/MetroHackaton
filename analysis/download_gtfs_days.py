"""Скачивание дневных архивов GTFS-realtime (позиции транспорта СПб) с публичной ссылки Яндекс.Диска.
python download_gtfs_days.py 2026-09-15 [другие даты...]   -> _gtfs/<дата>.7z (докачка не нужна, существующие пропускаются)"""
import os, sys, time
import requests

PK = "https://disk.yandex.ru/d/3QzxcM9IxfJcog"; API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
os.makedirs("_gtfs", exist_ok=True)
for d in sys.argv[1:]:
    fn = f"_gtfs/{d}.7z"
    if os.path.exists(fn) and os.path.getsize(fn) > 1_000_000:
        print(d, "уже есть", flush=True); continue
    r = requests.get(API, params={"public_key": PK, "path": f"/FromPortalRealtime/{d[:4]}/{d}.7z"}, timeout=40)
    if r.status_code != 200:
        print(d, "нет ссылки", r.status_code, r.text[:100], flush=True); continue
    href = r.json()["href"]; t0 = time.time(); n = 0
    with requests.get(href, stream=True, timeout=120) as s, open(fn + ".part", "wb") as f:
        for ch in s.iter_content(1 << 20):
            f.write(ch); n += len(ch)
    os.replace(fn + ".part", fn); print(d, f"{n/1e6:.0f} МБ за {time.time()-t0:.0f} с", flush=True)
