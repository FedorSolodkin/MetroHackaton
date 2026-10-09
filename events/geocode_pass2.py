"""Второй проход геокодирования: для адресов, которые Nominatim не нашёл, пробуем упрощённую строку
(без корпуса/литеры/помещения), затем только улицу (приближённо: центр улицы, точность ~1 км).
python geocode_pass2.py -> geocode_cache2.csv (q, lat, lon, display, level)"""
import csv, os, re, time
import pandas as pd, requests

H = {"User-Agent": "MetroHackaton/1.0 (student hackathon project)"}
URL = "https://nominatim.openstreetmap.org/search"; OUT = "geocode_cache2.csv"

def simplify(q):
    s = re.sub(r"\b(г\.?|город)\s+", "", q, flags=re.I)
    s = re.sub(r",?\s*(корп(ус)?\.?|к\.|лит(ера)?\.?|строение|стр\.?|пом(ещение)?\.?|оф(ис)?\.?|кв\.?)\s*[\w\-/]+", "", s, flags=re.I)
    s = re.sub(r"\bд(ом)?\.?\s*(?=\d)", "", s, flags=re.I)
    s = re.sub(r"(санкт-петербург)(.*?)(санкт-петербург)", r"\1\2", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" ,.;")

def street_only(q):
    s = simplify(q); m = re.search(r"^(.*?)[,\s]+\d", s)
    base = (m.group(1) if m else s).strip(" ,.;")
    return base if re.search(r"санкт|петербург", base, re.I) else base + ", Санкт-Петербург"

def ask(q):
    for _ in range(3):
        try:
            r = requests.get(URL, params=dict(q=q, format="jsonv2", limit=1, countrycodes="ru", viewbox="29.4,60.3,30.9,59.6", bounded=1), headers=H, timeout=30)
            if r.status_code == 200:
                j = r.json(); return (float(j[0]["lat"]), float(j[0]["lon"]), j[0]["display_name"][:80]) if j else None
            time.sleep(5)
        except requests.RequestException:
            time.sleep(5)
    return None

if __name__ == "__main__":
    c = pd.read_csv("geocode_cache.csv"); nf = c[c.lat.isna()].q.tolist()
    t = pd.read_csv("timepad_raw.csv"); import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geocode_nominatim import clean
    cnt = t[t.lat.isna() & t.venue.notna()].venue.map(clean).value_counts()
    nf = sorted(set(nf), key=lambda q: -cnt.get(q, 0))
    done = set(pd.read_csv(OUT).q) if os.path.exists(OUT) else set(); new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["q", "lat", "lon", "display", "level"])
        for i, q in enumerate(nf):
            if q in done: continue
            res, lvl = None, ""
            s1 = simplify(q)
            if s1 != q:
                res = ask(s1); lvl = "simplified"; time.sleep(1.1)
            if res is None:
                s2 = street_only(q)
                if s2 != s1:
                    res = ask(s2); lvl = "street"; time.sleep(1.1)
            w.writerow([q, *(res if res else (None, None, "")), lvl if res else "none"]); f.flush()
            if i % 100 == 0: print(i, "/", len(nf), flush=True)
    print("готово")
