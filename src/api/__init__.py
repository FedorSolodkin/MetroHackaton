"""
Модуль API реального времени и потокового воспроизведения телеметрии (Stream Replay).
Обеспечивает передачу данных по WebSocket на пульт поездного диспетчера (SCADA Cockpit).
"""

from .stream_replay import MetroStreamReplayer

__all__ = ["MetroStreamReplayer"]
