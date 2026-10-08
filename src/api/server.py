"""
Веб-сервер СППР Линии 1 Петербургского метрополитена (FastAPI + WebSockets).
Транслирует телеметрию в реальном времени на диспетчерский пульт (SCADA Cockpit)
и принимает команды управления сценариями What-If.
"""

from typing import List, Dict, Any, Optional
import os
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.api.stream_replay import MetroStreamReplayer


# Инициализация глобального реплеера
replayer = MetroStreamReplayer()

# Менеджер активных WebSocket соединений
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        text = json.dumps(message, ensure_ascii=False)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(text)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


# Фоновый цикл авто-воспроизведения телеметрии
async def telemetry_playback_loop():
    while True:
        try:
            if replayer.is_playing:
                state = replayer.step()
                await manager.broadcast(state)
                # Время задержки в секундах: при скорости 15x шаг в 15 минут занимает 1 сек реального времени
                delay_sec = max(0.2, 15.0 / max(replayer.playback_speed, 1.0))
                await asyncio.sleep(delay_sec)
            else:
                # Если на паузе, отправляем heart-beat / обновление раз в 1 секунду
                await asyncio.sleep(1.0)
        except Exception as e:
            print(f"Ошибка в цикле телеметрии: {e}")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Старт фонового цикла
    task = asyncio.create_task(telemetry_playback_loop())
    yield
    # Остановка
    task.cancel()


app = FastAPI(
    title="СППР «МетроПульс-1» — Санкт-Петербургский Метрополитен",
    description="Система поддержки принятия решений по адаптивной насыщенности Линии 1 подвижным составом",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов (SCADA Cockpit UI)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "СППР 'МетроПульс-1' API активно. Загрузите файл static/index.html"}


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Сразу отправляем текущее состояние при подключении
        cur_state = replayer.get_current_state()
        await websocket.send_text(json.dumps(cur_state, ensure_ascii=False))
        while True:
            # Ожидаем входящие команды от UI
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "step":
                    new_state = replayer.step()
                    await manager.broadcast(new_state)
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# REST API Эндпоинты
@app.get("/api/state")
async def get_state():
    """Текущий срез состояния Линии 1."""
    return replayer.get_current_state()


@app.post("/api/playback/play")
async def playback_play():
    """Запуск воспроизведения."""
    replayer.is_playing = True
    return {"status": "playing", "speed": replayer.playback_speed}


@app.post("/api/playback/pause")
async def playback_pause():
    """Приостановка воспроизведения."""
    replayer.is_playing = False
    return {"status": "paused"}


@app.post("/api/playback/step")
async def playback_step():
    """Один шаг вперед (+15 минут)."""
    new_state = replayer.step()
    await manager.broadcast(new_state)
    return new_state


@app.post("/api/playback/speed")
async def set_playback_speed(speed: float = Query(..., ge=0.5, le=60.0)):
    """Изменение скорости воспроизведения (0.5x - 60x)."""
    replayer.playback_speed = float(speed)
    return {"status": "ok", "speed": replayer.playback_speed}


@app.post("/api/playback/seek")
async def seek_time_index(index: int = Query(..., ge=0, le=95)):
    """Перемотка на временной слот (0..95)."""
    replayer.set_time_index(index)
    new_state = replayer.get_current_state()
    await manager.broadcast(new_state)
    return new_state


@app.post("/api/simulate/weather")
async def set_weather_shock(rain_mm: float = Query(..., ge=0.0, le=30.0)):
    """
    Симуляция What-If сценария: Ливень в Санкт-Петербурге (0..30 мм/ч).
    Вызывает динамический всплеск пассажиропотока и реакцию Decision Engine.
    """
    replayer.weather_shock_mm = float(rain_mm)
    new_state = replayer.get_current_state()
    await manager.broadcast(new_state)
    return {
        "status": "ok",
        "weather_shock_mm": replayer.weather_shock_mm,
        "impact_pct": round(replayer.weather_shock_mm * 3.5, 1)
    }


@app.post("/api/dispatch/execute")
async def execute_dispatch_action():
    """
    Подтверждение директивы поездным диспетчером ЦУП.
    """
    cur_state = replayer.get_current_state()
    act = cur_state["decision"]["action"]
    replayer.shift_savings_rub += abs(cur_state["decision"]["economic_impact_rub"])
    return {
        "status": "confirmed",
        "action": act,
        "location": cur_state["decision"]["location"],
        "savings_total_rub": round(replayer.shift_savings_rub)
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("ЗАПУСК СЕРВЕРА СППР 'МЕТРОПУЛЬС-1' (ЦУП САНКТ-ПЕТЕРБУРГСКОГО МЕТРОПОЛИТЕНА)")
    print("Откройте браузер по адресу: http://localhost:8000")
    print("=" * 70)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
