#!/usr/bin/env python3
"""
Server FINTO per testare SOLO l'interfaccia web (frontend), senza ROS2,
senza tellopy, senza drone reale. Genera uno stato simulato plausibile
(drone che si muove in cerchio dentro la stanza, batteria che scende
lentamente) e lo serve con lo STESSO formato JSON di position_controller_web.py.

USO:
    pip install fastapi "uvicorn[standard]" websockets
    python3 mock_server.py
    apri il browser su http://localhost:8080/static/index.html

Nessuna dipendenza da rclpy/tellopy/torch: gira su qualunque macchina con
Python 3, anche senza il container Docker.
"""

import asyncio
import json
import math
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_static")
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

ROOM_MIN = [-2.0, -2.0, 0.1]
ROOM_MAX = [2.0, 2.0, 3.0]

# -- stato finto, mutabile, condiviso tra il generatore e gli endpoint --
mock_state = {
    "session_state": "idle",   # idle | starting | flying | landing
    "dof_mask_mode": "full",
    "target_mode": "variabile",
    "advance_mode": "manual",
    "num_queues": -1,
    "queues_completed": 0,
    "wp_idx": 0,
    "n_waypoints": 4,
    "t0": time.time(),
}


def generate_fake_state():
    """Genera uno snapshot plausibile: drone che vola in cerchio a quota
    variabile quando session_state=='flying', altrimenti fermo al centro
    della stanza vicino al pavimento (come se fosse appena atterrato)."""
    now = time.time()
    elapsed = now - mock_state["t0"]

    flying = mock_state["session_state"] == "flying"
    if flying:
        radius = 1.0
        omega = 0.4  # rad/s, velocita' angolare del giro finto
        x = radius * math.cos(omega * elapsed)
        y = radius * math.sin(omega * elapsed)
        z = 1.2 + 0.3 * math.sin(0.2 * elapsed)
        yaw = (omega * elapsed + math.pi / 2) % (2 * math.pi)
        if yaw > math.pi:
            yaw -= 2 * math.pi
        roll = 8.0 * math.sin(0.5 * elapsed)     # gradi, finto
        pitch = 5.0 * math.cos(0.3 * elapsed)     # gradi, finto
        vx = -radius * omega * math.sin(omega * elapsed)
        vy = radius * omega * math.cos(omega * elapsed)
        vz = 0.3 * 0.2 * math.cos(0.2 * elapsed)
        wx, wy, wz = 0.05, 0.03, omega
        target = [1.0, -1.0, 1.5]
        battery = max(10, 100 - int(elapsed * 0.5))  # scende lentamente
    else:
        x, y, z = 0.0, 0.0, 0.15
        yaw = roll = pitch = 0.0
        vx = vy = vz = wx = wy = wz = 0.0
        target = None
        battery = 87

    return {
        "t": now,
        "session_state": mock_state["session_state"],
        "vicon_connected": True,
        "tello_connected": True,
        "battery": battery,
        "pos": [x, y, z],
        "yaw_deg": math.degrees(yaw),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "lin_vel_b": [vx, vy, vz],
        "ang_vel_b": [wx, wy, wz],
        "target": target,
        "room_min": ROOM_MIN,
        "room_max": ROOM_MAX,
        "wp_idx": mock_state["wp_idx"],
        "n_waypoints": mock_state["n_waypoints"],
        "queues_completed": mock_state["queues_completed"],
        "num_queues": mock_state["num_queues"],
        "dof_mask_mode": mock_state["dof_mask_mode"],
        "target_mode": mock_state["target_mode"],
        "advance_mode": mock_state["advance_mode"],
    }


class ParamsIn(BaseModel):
    dof_mask_mode: str
    target_mode: str
    advance_mode: str
    num_queues: int


app = FastAPI()
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/api/status")
def get_status():
    return generate_fake_state()


@app.post("/api/params")
def post_params(params: ParamsIn):
    if mock_state["session_state"] != "idle":
        return {"ok": False, "message": "Sessione non idle, impossibile cambiare parametri."}
    mock_state["dof_mask_mode"] = params.dof_mask_mode
    mock_state["target_mode"] = params.target_mode
    mock_state["advance_mode"] = params.advance_mode
    mock_state["num_queues"] = params.num_queues
    mock_state["queues_completed"] = 0
    mock_state["wp_idx"] = 0
    return {"ok": True, "message": "Parametri impostati (finto)."}


@app.post("/api/start")
def post_start():
    mock_state["session_state"] = "flying"
    mock_state["t0"] = time.time()
    return {"ok": True, "message": "Volo simulato avviato."}


@app.post("/api/land")
def post_land():
    mock_state["session_state"] = "idle"
    return {"ok": True, "message": "Atterraggio simulato."}


@app.post("/api/advance")
def post_advance():
    if mock_state["session_state"] != "flying":
        return {"ok": False}
    mock_state["wp_idx"] = (mock_state["wp_idx"] + 1) % mock_state["n_waypoints"]
    if mock_state["wp_idx"] == 0:
        mock_state["queues_completed"] += 1
    return {"ok": True}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(generate_fake_state()))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print(f"Mock server avviato: http://localhost:{WEB_PORT}/static/index.html")
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")