#!/usr/bin/env python3
"""
Nodo ROS2 'position_controller_web' (package tello_node): copia di
position_controller_VICON_VERSION.py con un'INTERFACCIA WEB integrata
nello stesso processo (FastAPI + uvicorn in un thread separato), invece
di parametri ROS2 da riga di comando e input da terminale.

ARCHITETTURA:
  - Un solo processo Python. rclpy gira nel thread principale (spin);
    il server web (FastAPI/uvicorn) gira in un thread separato, avviato
    dentro __init__. Comunicano tramite riferimenti Python diretti allo
    stesso oggetto nodo (nessun subprocess, nessuna pipe stdin/stdout).
  - tellopy si CONNETTE all'avvio del nodo (in un thread separato, non
    blocca l'apertura dell'interfaccia): batteria/IMU sono quindi visibili
    nel dashboard anche PRIMA di avviare l'algoritmo.
  - takeoff() reale scatta SOLO quando il frontend chiama POST /api/start
    (bottone "Avvia algoritmo" nell'interfaccia).
  - I parametri (dof_mask_mode, target_mode, advance_mode, num_queues)
    sono attributi dell'istanza, impostabili via POST /api/params SOLO
    quando session_state == "idle" (non in volo).
  - Dopo un atterraggio (manuale o num_queues raggiunto), session_state
    torna a "idle": si puo' fare un nuovo volo con parametri diversi
    SENZA riavviare il processo. La connessione tellopy resta viva tra
    un volo e l'altro (drone.quit() solo alla chiusura vera del nodo).
  - Stato del drone trasmesso al frontend via WebSocket (/ws/state) a
    circa 10Hz: posizione (coordinate Vicon grezze, la conversione per
    la vista 3D la fa il frontend), yaw/roll/pitch, velocita' lineari
    body-frame, velocita' angolari, batteria, target attivo, stato
    sessione, info coda waypoint.
  - NESSUN input da terminale (INVIO/q rimossi): tutto passa dai bottoni
    web. Ctrl+C sul processo resta l'UNICO modo per terminare il nodo
    (finally: land_sequence() + disconnessione tellopy), esattamente
    come nelle versioni precedenti.

CHECKPOINT: policy_pos_controller/2026-07-31_13-02-43_ppo_torch/
            checkpoints/best_agent.pt
  - head_key="policy_layer" CONFERMATO. Architettura [256,128,64] ELU
    CONFERMATA. drone_inertial.total_mass=0.087kg coerente col Tello.
    Parametri env.yaml confrontati e coincidenti con le costanti sotto.

AGGIORNAMENTO — FUNZIONI DI EMERGENZA:
  - NUOVO: emergency_land(reason), metodo UNIVERSALE per l'atterraggio
    d'emergenza. NON controlla e NON dipende da session_state prima di
    agire: chiama direttamente land_sequence() (gia' idempotente grazie
    a self._landing_started). E' il percorso UNICO usato ora da tutti i
    trigger di emergenza (bottone web, batteria critica, Vicon perso,
    tellopy perso).
  - NUOVO: il bottone "Land" (/api/land) ora chiama emergency_land()
    invece di request_land(): PRIMA poteva non fare nulla se
    session_state non era esattamente "starting"/"flying" (bug di stato
    -> bottone di emergenza silenziosamente inefficace). ORA atterra
    SEMPRE, incondizionatamente. request_land() resta nel file (non piu'
    usata dal bottone) per compatibilita', invariata.
  - NUOVO: escalation automatica se la pose Vicon manca (mai arrivata o
    persa a meta' volo) per oltre POSE_LOST_LAND_TIMEOUT_S secondi
    CONSECUTIVI -> atterraggio automatico via emergency_land(). Se il
    Vicon torna prima di quella soglia, il timer si azzera e non succede
    nulla di piu' del solito Twist nullo pubblicato nel frattempo.
  - NUOVO: stessa identica logica per i dati tellopy (IMU/batteria/quota):
    se non arriva nulla da tellopy per oltre TELLO_LOST_LAND_TIMEOUT_S
    secondi CONSECUTIVI -> atterraggio automatico. Il comportamento
    esistente di REQUIRE_IMU (blocco immediato se True, tolleranza nel
    breve termine se False) resta INVARIATO: l'escalation si applica IN
    AGGIUNTA, indipendentemente dal valore di REQUIRE_IMU.
  - Il failsafe batteria critica (gia' esistente) ora passa anch'esso da
    emergency_land(), invece di chiamare request_land() (stesso risultato
    pratico, ma ora e' garantito indipendentemente da session_state).

ASSUNZIONI DA VERIFICARE PRIMA DEL VOLO (invariate rispetto alle
versioni precedenti):
  1) VICON_POSE_TOPIC placeholder ("/vicon/tello/pose"): da confermare.
  2) CMD_VEL_TOPIC placeholder ("/tello/cmd_vel"): da confermare col
     nodo PID lato drone.
  3) origin_mocap: calibrare misurando il punto (0,0,0) Vicon reale.
  4) ROOM_MIN/ROOM_MAX: misurare i bordi reali della stanza.
  5) Unita' di imu.gyro_x/y/z, "height" EVENT_FLIGHT_DATA: da verificare.
  6) Convenzione yaw scipy vs Isaac Lab: mai confrontate esplicitamente.
  7) Modalita' di connessione tellopy (AP diretto vs stazione).
  8) WEB_HOST/WEB_PORT sotto: verificare che la porta scelta sia libera
     e raggiungibile dalla rete da cui apri il browser (stesso host del
     container se --network host, altrimenti serve un port mapping).
  9) POSE_LOST_LAND_TIMEOUT_S / TELLO_LOST_LAND_TIMEOUT_S (3.0s
     entrambe): soglie di default, da validare/tarare in laboratorio.
"""

import math
import os
import time
import json
import threading
import asyncio
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from scipy.spatial.transform import Rotation as R

import tellopy

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ============================================================
# CONFIG — deve rispecchiare params/env.yaml del checkpoint usato
# (2026-07-31_13-02-43_ppo_torch), verificato riga per riga.
# ============================================================
STEP_DT = 0.04
N_WAYPOINTS = 4
WP_PREVIEW_HORIZON = 4
INTEGRAL_TAU_S = 5.0
INTEGRAL_CLAMP = 1.0
INTEGRAL_OBS_SCALE = 0.5
VEL_REF_SCALE = torch.tensor([1.0, 1.0, 1.0, 1.5])

# Stanza REALE (metri, frame origin_mocap).
# <-- SOSTITUIRE con le misure reali della tua stanza di laboratorio
ROOM_MIN = torch.tensor([-2.0, -2.0, 0.1])
ROOM_MAX = torch.tensor([ 2.0,  2.0, 3.0])

TARGET_ROOM_MARGIN = 0.8
TARGET_REACH_THRESHOLD_M = 0.15
TARGET_REACH_YAW_THRESHOLD_RAD = 0.20
TARGET_HOLD_TIME_S = 1.2

DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),
}

VICON_POSE_TOPIC = "/vicon/tello/pose"     # <-- PLACEHOLDER, da confermare
CMD_VEL_TOPIC = "/tello/cmd_vel"           # <-- PLACEHOLDER, da confermare

POSE_TIMEOUT_S = 0.5
POSE_LOST_LAND_TIMEOUT_S = 3.0   # NUOVO: pose Vicon assente CONTINUATIVAMENTE oltre questo -> atterraggio automatico

IMU_TIMEOUT_S = 0.5
REQUIRE_IMU = False
TELLO_LOST_LAND_TIMEOUT_S = 3.0   # NUOVO: dati tellopy assenti CONTINUATIVAMENTE oltre questo -> atterraggio automatico

MAX_PLAUSIBLE_SPEED_MPS = 5.0
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1

MAX_LIN_VEL_MPS = 0.8
MAX_YAW_RATE_RADPS = 1.0

STATE_BROADCAST_PERIOD_S = 0.1   # ~10Hz verso il frontend

# ============================================================
# TELLOPY
# ============================================================
TELLOPY_CONNECT_TIMEOUT_S = 60.0
POST_TAKEOFF_SETTLE_S = 3.0
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_DM = 3              # <-- unita' presunte (decimetri), VERIFICARE
TELLOPY_LAND_WAIT_S = 5.0
BATTERY_FAILSAFE_PCT = 15

# ============================================================
# CHECKPOINT
# ============================================================
DEFAULT_CKPT_PATH = (
    "/ros_workspace/src/tello_node/policy_pos_controller/2026-07-31_13-02-43_ppo_torch/"
    "checkpoints/best_agent.pt"
)
CKPT_PATH = os.environ.get("POLICY_CKPT_PATH", DEFAULT_CKPT_PATH)

# ============================================================
# WEB SERVER
# ============================================================
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_static")


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_projected_gravity_b(qx, qy, qz, qw):
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(np.array([0.0, 0.0, -1.0]))


def compute_lin_vel_body(v_world: np.ndarray, qx, qy, qz, qw) -> np.ndarray:
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(v_world)


def sample_target_in_room():
    center = 0.5 * (ROOM_MIN + ROOM_MAX)
    half = 0.5 * (ROOM_MAX - ROOM_MIN) * TARGET_ROOM_MARGIN
    lo = center - half
    hi = center + half
    lo[2] = torch.clamp(lo[2], min=ROOM_MIN[2])
    hi[2] = torch.maximum(hi[2], lo[2] + 1e-3)
    pos = lo + torch.rand(3) * (hi - lo)
    yaw = float(torch.empty(1).uniform_(-math.pi, math.pi))
    return pos, yaw


class SkrlMlpPolicy(torch.nn.Module):
    def __init__(self, dims, act=torch.nn.ELU):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(dims):
            layers.append(torch.nn.Linear(a, b))
            if i < len(dims) - 1:
                layers.append(act())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_skrl_policy(ckpt_path, expected_in, expected_out, device="cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: '{ckpt_path}'.")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["policy"]
    running_mean = ckpt["state_preprocessor"]["running_mean"].to(device).float()
    running_var = ckpt["state_preprocessor"]["running_variance"].to(device).float()

    trunk_keys = sorted(
        (k for k in sd if k.startswith("net_container.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    head_key = "policy_layer.weight" if "policy_layer.weight" in sd else "mean_layer.weight"
    head_bias_key = head_key.replace("weight", "bias")

    trunk_dims = [tuple(sd[k].shape[::-1]) for k in trunk_keys]
    head_out, head_in = sd[head_key].shape
    layer_dims = trunk_dims + [(head_in, head_out)]

    in_dim, out_dim = layer_dims[0][0], layer_dims[-1][1]
    assert in_dim == expected_in, f"input_dim={in_dim}, atteso {expected_in}"
    assert out_dim == expected_out, f"output_dim={out_dim}, atteso {expected_out}"

    model = SkrlMlpPolicy(layer_dims).to(device)
    new_sd = {}
    lin_idx = [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]
    for orig, dest in zip(trunk_keys, lin_idx[:-1]):
        base = orig.rsplit(".", 1)[0]
        new_sd[f"net.{dest}.weight"] = sd[f"{base}.weight"]
        new_sd[f"net.{dest}.bias"] = sd[f"{base}.bias"]
    new_sd[f"net.{lin_idx[-1]}.weight"] = sd[head_key]
    new_sd[f"net.{lin_idx[-1]}.bias"] = sd[head_bias_key]
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, running_mean, running_var


# ============================================================
# NODO
# ============================================================
class PositionController(Node):
    def __init__(self):
        super().__init__("position_controller_web")

        # -- parametri modificabili da web (NON piu' argomenti ROS2) --
        self.dof_mask_mode = "full"
        self.target_mode = "variabile"
        self.advance_mode = "manual"
        self.num_queues = -1  # <=0 = infinito
        self.dof_mask = torch.tensor(DOF_MASKS[self.dof_mask_mode])

        # -- stato sessione: idle -> starting -> flying -> landing -> idle --
        self._session_lock = threading.Lock()
        self.session_state = "idle"

        self.policy, self.run_mean, self.run_var = load_skrl_policy(
            CKPT_PATH, expected_in=52, expected_out=4
        )
        self.get_logger().info(f"Policy caricata da: {CKPT_PATH}")

        # -- stato interno (come nelle versioni precedenti) --
        self.pos_env = torch.zeros(3)
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.lin_vel_b = torch.zeros(3)
        self.ang_vel_b = torch.zeros(3)
        self.proj_grav_b = np.array([0.0, 0.0, -1.0])
        self.prev_pos_world = None
        self.prev_time = None

        self.prev_hl_action = torch.zeros(4)
        self.err_integral = torch.zeros(4)
        self.alpha_leaky = math.exp(-STEP_DT / INTEGRAL_TAU_S)

        self._wp_lock = threading.Lock()
        self.wp_pos_queue = torch.zeros(N_WAYPOINTS, 3)
        self.wp_yaw_queue = torch.zeros(N_WAYPOINTS)
        self.wp_idx = 0
        self.hold_timer = 0.0
        self.queues_completed = 0
        self._fill_random_queue_locked()

        self.origin_mocap = torch.tensor([0.0, 0.0, 0.0])  # <-- calibrare!

        self._tello_lock = threading.Lock()
        self.imu_received = False
        self.last_imu_wall_time = None
        self._imu_warned = False
        self.battery_pct = None
        self.tello_alt_dm = None
        self._last_mvo_vel = None
        self.tello_connected = False

        self._mocap_rejected_count = 0

        # -- NUOVO: stato per l'escalation automatica verso emergency_land() --
        self._pose_lost_since = None
        self._tello_lost_since = None

        self.pose_sub = self.create_subscription(
            PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.pose_received = False
        self.last_pose_wall_time = None
        self.flight_ready = False
        self._landing_started = False

        self.drone = tellopy.Tello()
        self.drone.subscribe(self.drone.EVENT_LOG_DATA, self.tello_log_data_cb)
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.tello_flight_data_cb)

        # -- connessione tellopy in BACKGROUND, non blocca l'avvio del nodo --
        threading.Thread(target=self._connect_tellopy_async, daemon=True).start()

        self.timer = self.create_timer(STEP_DT, self.control_loop)

        # -- stato condiviso col server web (letto dal WebSocket broadcaster) --
        self._state_lock = threading.Lock()
        self._latest_state = {}
        self.state_timer = self.create_timer(STATE_BROADCAST_PERIOD_S, self._update_latest_state)

        # -- avvia il server web in un thread separato --
        threading.Thread(target=self._run_web_server, daemon=True).start()

        self.get_logger().info(
            f"Nodo position_controller_web avviato. Interfaccia su "
            f"http://<host>:{WEB_PORT}/ | Vicon topic: {VICON_POSE_TOPIC} | "
            f"cmd_vel: {CMD_VEL_TOPIC} | Checkpoint: {CKPT_PATH}"
        )

    # ==================================================================
    # -- coda waypoint (target random dentro la stanza) --
    # ==================================================================
    def _fill_random_queue_locked(self):
        if self.target_mode == "singolo":
            pos, yaw = sample_target_in_room()
            for k in range(N_WAYPOINTS):
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
        else:
            for k in range(N_WAYPOINTS):
                pos, yaw = sample_target_in_room()
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
        self.wp_idx = 0

    def _advance_waypoint_locked(self, source: str) -> bool:
        """Ritorna True se il volo deve terminare (num_queues raggiunto)."""
        if self.wp_idx < N_WAYPOINTS - 1:
            self.wp_idx += 1
            self.err_integral[:] = 0.0
            wp = self.wp_pos_queue[self.wp_idx].clone()
            self.get_logger().info(
                f"[{source}] Avanzato al waypoint {self.wp_idx + 1}/{N_WAYPOINTS}: "
                f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})m"
            )
            return False

        self.queues_completed += 1
        if self.num_queues > 0 and self.queues_completed >= self.num_queues:
            self.get_logger().info(
                f"[{source}] num_queues raggiunto ({self.queues_completed}/{self.num_queues}): "
                "avvio atterraggio."
            )
            return True

        self._fill_random_queue_locked()
        self.err_integral[:] = 0.0
        wp = self.wp_pos_queue[0].clone()
        self.get_logger().info(
            f"[{source}] Nuova coda random generata ({self.target_mode}). Primo target: "
            f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})m"
        )
        return False

    def manual_advance_from_web(self):
        if self.advance_mode != "manual" or not self.flight_ready:
            return
        with self._wp_lock:
            should_land = self._advance_waypoint_locked("web")
        if should_land:
            threading.Thread(target=self.land_sequence, daemon=True).start()

    # ==================================================================
    # -- gestione parametri (solo quando session_state == "idle") --
    # ==================================================================
    def set_params(self, dof_mask_mode, target_mode, advance_mode, num_queues):
        with self._session_lock:
            if self.session_state != "idle":
                return False, "Impossibile cambiare parametri: sessione non idle."
        if dof_mask_mode not in DOF_MASKS:
            return False, f"dof_mask_mode non valido: {dof_mask_mode}"
        if target_mode not in ("singolo", "variabile"):
            return False, f"target_mode non valido: {target_mode}"
        if advance_mode not in ("manual", "auto"):
            return False, f"advance_mode non valido: {advance_mode}"

        self.dof_mask_mode = dof_mask_mode
        self.target_mode = target_mode
        self.advance_mode = advance_mode
        self.num_queues = int(num_queues)
        self.dof_mask = torch.tensor(DOF_MASKS[dof_mask_mode])
        with self._wp_lock:
            self.queues_completed = 0
            self._fill_random_queue_locked()
        return True, "Parametri impostati."

    # ==================================================================
    # -- avvio/arresto algoritmo (chiamati dagli endpoint web, in thread
    # separati per non bloccare il server) --
    # ==================================================================
    def start_flight_sequence(self):
        with self._session_lock:
            if self.session_state != "idle":
                self.get_logger().warn("start_flight_sequence: sessione gia' attiva, ignorato.")
                return
            self.session_state = "starting"

        ok = self.takeoff_sequence()
        if not ok:
            with self._session_lock:
                self.session_state = "idle"
            return

        with self._session_lock:
            self.session_state = "flying"
        self.flight_ready = True

    def request_land(self):
        with self._session_lock:
            if self.session_state not in ("starting", "flying"):
                return
            self.session_state = "landing"
        self.land_sequence()
        with self._session_lock:
            self.session_state = "idle"

    # ==================================================================
    # -- CALLBACK TELLOPY --
    # ==================================================================
    def tello_log_data_cb(self, event, sender, data, **kwargs):
        imu = data.imu
        vals = (imu.gyro_x, imu.gyro_y, imu.gyro_z)
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            return
        with self._tello_lock:
            self.ang_vel_b = torch.tensor([imu.gyro_x, imu.gyro_y, imu.gyro_z])
            self.imu_received = True
            self.last_imu_wall_time = time.monotonic()
            self._last_mvo_vel = (data.mvo.vel_x, data.mvo.vel_y, data.mvo.vel_z)

    def tello_flight_data_cb(self, event, sender, data, **kwargs):
        battery = getattr(data, "battery_percentage", None)
        height = getattr(data, "height", None)
        with self._tello_lock:
            self.battery_pct = battery
            self.tello_alt_dm = height

        if (
            battery is not None
            and battery < BATTERY_FAILSAFE_PCT
            and self.flight_ready
            and not self._landing_started
        ):
            self.get_logger().error(f"[tellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza.")
            threading.Thread(
                target=self.emergency_land, args=(f"batteria critica ({battery}%)",), daemon=True
            ).start()

    # ==================================================================
    # -- connessione / takeoff / land tramite tellopy --
    # ==================================================================
    def _connect_tellopy_async(self):
        self.get_logger().info("[tellopy] connessione al drone (background)...")
        try:
            self.drone.connect()
            self.drone.wait_for_connection(TELLOPY_CONNECT_TIMEOUT_S)
            with self._tello_lock:
                self.tello_connected = True
            self.get_logger().info("[tellopy] connesso.")
        except Exception as e:
            self.get_logger().error(f"[tellopy] connessione fallita: {e}")

    def takeoff_sequence(self) -> bool:
        with self._tello_lock:
            connected = self.tello_connected
        if not connected:
            self.get_logger().error("[tellopy] non connesso: impossibile decollare.")
            return False

        self.get_logger().info("[tellopy] invio takeoff...")
        try:
            self.drone.takeoff()
        except Exception as e:
            self.get_logger().error(f"[tellopy] comando takeoff fallito: {e}")
            return False

        deadline = time.monotonic() + TAKEOFF_CONFIRM_TIMEOUT_S
        confirmed = False
        while time.monotonic() < deadline:
            with self._tello_lock:
                alt = self.tello_alt_dm
            if alt is not None and alt > TAKEOFF_MIN_ALT_DM:
                confirmed = True
                break
            time.sleep(0.2)

        if not confirmed:
            self.get_logger().error("[tellopy] decollo non confermato: ABORT.")
            return False

        self.get_logger().info(f"[tellopy] decollo confermato, assestamento {POST_TAKEOFF_SETTLE_S}s...")
        time.sleep(POST_TAKEOFF_SETTLE_S)
        return True

    def _publish_zero_twist(self):
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass

    def land_sequence(self):
        if self._landing_started:
            return
        self._landing_started = True

        self.flight_ready = False
        self._publish_zero_twist()
        time.sleep(0.2)

        self.get_logger().info("[tellopy] invio land...")
        try:
            self.drone.land()
        except Exception as e:
            self.get_logger().error(f"[tellopy] comando land fallito: {e}")

        time.sleep(TELLOPY_LAND_WAIT_S)
        self._landing_started = False  # pronto per un nuovo volo (connessione resta viva)

    def emergency_land(self, reason: str = "richiesta manuale"):
        """
        NUOVO: atterraggio d'EMERGENZA. NON controlla e NON dipende da
        session_state prima di agire: chiama direttamente land_sequence()
        (gia' idempotente grazie a self._landing_started), quindi e'
        sempre sicuro invocarla, da qualunque punto del codice e in
        qualunque momento. E' il percorso UNICO usato ora da tutti i
        trigger di emergenza (bottone web, batteria critica, Vicon perso,
        tellopy perso).
        """
        self.get_logger().error(f"[EMERGENZA] Atterraggio forzato: {reason}")
        self.flight_ready = False
        self.land_sequence()
        with self._session_lock:
            self.session_state = "idle"

    def disconnect_tellopy(self):
        """Chiamato SOLO alla chiusura vera del nodo (Ctrl+C), non ad ogni land."""
        try:
            self.drone.quit()
        except Exception:
            pass

    # -------------------- callback Vicon --------------------
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_world = torch.tensor([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        raw_vals = [p_world[0].item(), p_world[1].item(), p_world[2].item(), q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            return

        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            return

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt_check = t - self.prev_time
            if dt_check > 1e-3:
                implied_speed = float(torch.norm(p_world - self.prev_pos_world)) / dt_check
                if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
                    self._mocap_rejected_count += 1
                    return
        self._mocap_rejected_count = 0

        roll, pitch, yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        self.proj_grav_b = compute_projected_gravity_b(q.x, q.y, q.z, q.w)

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt = max(t - self.prev_time, 1e-3)
            v_world = (p_world - self.prev_pos_world) / dt
            v_body_raw = compute_lin_vel_body(v_world.numpy(), q.x, q.y, q.z, q.w)
            alpha_f = 0.3
            self.lin_vel_b = alpha_f * torch.tensor(v_body_raw, dtype=torch.float32) \
                              + (1 - alpha_f) * self.lin_vel_b

        self.prev_pos_world = p_world
        self.prev_time = t
        self.pos_env = p_world - self.origin_mocap
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch
        self.pose_received = True
        self.last_pose_wall_time = time.monotonic()

    # -------------------- ciclo di controllo @25Hz --------------------
    def control_loop(self):
        if not self.flight_ready:
            return

        # -- NUOVO: pose Vicon mai arrivata o persa a meta' volo. Se
        # l'assenza persiste CONTINUATIVAMENTE oltre POSE_LOST_LAND_TIMEOUT_S,
        # atterraggio automatico via emergency_land(). Se il Vicon torna
        # prima di quella soglia, il timer si azzera senza altre conseguenze. --
        now_mono = time.monotonic()
        pose_age = (now_mono - self.last_pose_wall_time) if self.pose_received else float("inf")
        if pose_age > POSE_TIMEOUT_S:
            self._publish_zero_twist()
            if self._pose_lost_since is None:
                self._pose_lost_since = now_mono
            elif now_mono - self._pose_lost_since > POSE_LOST_LAND_TIMEOUT_S:
                threading.Thread(
                    target=self.emergency_land,
                    args=(f"Vicon assente da oltre {POSE_LOST_LAND_TIMEOUT_S}s",),
                    daemon=True,
                ).start()
            return
        self._pose_lost_since = None

        # -- NUOVO: stessa logica di escalation per i dati tellopy
        # (IMU/batteria/quota). REQUIRE_IMU continua a comportarsi come
        # prima; l'escalation si applica IN AGGIUNTA. --
        with self._tello_lock:
            imu_ok = self.imu_received
            last_imu_t = self.last_imu_wall_time
        now_mono_tello = time.monotonic()
        imu_age = (now_mono_tello - last_imu_t) if (imu_ok and last_imu_t is not None) else float("inf")
        imu_stale = imu_age > IMU_TIMEOUT_S

        if imu_stale:
            if self._tello_lost_since is None:
                self._tello_lost_since = now_mono_tello
            elif now_mono_tello - self._tello_lost_since > TELLO_LOST_LAND_TIMEOUT_S:
                threading.Thread(
                    target=self.emergency_land,
                    args=(f"dati tellopy assenti da oltre {TELLO_LOST_LAND_TIMEOUT_S}s",),
                    daemon=True,
                ).start()
                return

            if REQUIRE_IMU:
                self._publish_zero_twist()
                return
            # REQUIRE_IMU=False: si continua comunque nel breve termine, come prima
        else:
            self._tello_lost_since = None

        pos_env = self.pos_env
        yaw = self.yaw

        with self._wp_lock:
            wp_idx = self.wp_idx
            w0_pos = self.wp_pos_queue[wp_idx].clone()
            w0_yaw = float(self.wp_yaw_queue[wp_idx])

        pos_err = pos_env - w0_pos
        yaw_err_signed = wrap_to_pi(yaw - w0_yaw)

        if self.advance_mode == "auto":
            dist_to_target = float(torch.norm(pos_err))
            yaw_controllable = float(self.dof_mask[3]) > 0.5
            yaw_err_eff = abs(yaw_err_signed) if yaw_controllable else 0.0
            converged = (
                dist_to_target < TARGET_REACH_THRESHOLD_M
                and yaw_err_eff < TARGET_REACH_YAW_THRESHOLD_RAD
            )
            self.hold_timer = self.hold_timer + STEP_DT if converged else 0.0

            if self.hold_timer >= TARGET_HOLD_TIME_S:
                self.hold_timer = 0.0
                with self._wp_lock:
                    should_land = self._advance_waypoint_locked("auto")
                    wp_idx = self.wp_idx
                    w0_pos = self.wp_pos_queue[wp_idx].clone()
                    w0_yaw = float(self.wp_yaw_queue[wp_idx])
                pos_err = pos_env - w0_pos
                yaw_err_signed = wrap_to_pi(yaw - w0_yaw)
                if should_land:
                    threading.Thread(target=self.request_land, daemon=True).start()
                    return

        with self._wp_lock:
            wp_idx = self.wp_idx
            wp_pos_queue = self.wp_pos_queue.clone()
            wp_yaw_queue = self.wp_yaw_queue.clone()

        self.err_integral[:3] = torch.clamp(
            self.alpha_leaky * self.err_integral[:3] + pos_err * STEP_DT,
            -INTEGRAL_CLAMP, INTEGRAL_CLAMP,
        )
        self.err_integral[3] = float(np.clip(
            self.alpha_leaky * self.err_integral[3] + yaw_err_signed * STEP_DT,
            -INTEGRAL_CLAMP, INTEGRAL_CLAMP,
        ))

        blocks = []
        e0 = wrap_to_pi(w0_yaw - yaw)
        blocks += [w0_pos - pos_env, torch.tensor([math.sin(e0), math.cos(e0)])]
        for k in range(1, WP_PREVIEW_HORIZON):
            i_c = min(wp_idx + k, N_WAYPOINTS - 1)
            i_p = min(wp_idx + k - 1, N_WAYPOINTS - 1)
            p_c, p_p = wp_pos_queue[i_c], wp_pos_queue[i_p]
            y_c, y_p = wp_yaw_queue[i_c], wp_yaw_queue[i_p]
            dyaw = wrap_to_pi(y_c - y_p)
            blocks += [p_c - p_p, torch.tensor([math.sin(dyaw), math.cos(dyaw)])]
        preview = torch.cat(blocks)

        clearance = torch.cat([ROOM_MAX - pos_env, pos_env - ROOM_MIN])
        proj_grav_b = torch.from_numpy(self.proj_grav_b).float()
        integral_norm = self.err_integral / INTEGRAL_OBS_SCALE

        with self._tello_lock:
            ang_vel_b = self.ang_vel_b.clone()

        obs = torch.cat([
            pos_env,
            torch.tensor([math.sin(yaw), math.cos(yaw)]),
            self.lin_vel_b,
            ang_vel_b,
            proj_grav_b,
            self.prev_hl_action,
            preview,
            clearance,
            self.dof_mask,
            integral_norm,
        ]).float()

        assert obs.numel() == 52, f"obs size={obs.numel()}, attesa 52"

        obs_n = (obs - self.run_mean) / torch.sqrt(self.run_var + 1e-8)
        with torch.no_grad():
            action = self.policy(obs_n.unsqueeze(0)).squeeze(0)
        action = action.clamp(-1.0, 1.0) * self.dof_mask
        self.prev_hl_action = action.clone()

        target_vel_ref = action * VEL_REF_SCALE

        vx = float(np.clip(float(target_vel_ref[0]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vy = float(np.clip(float(target_vel_ref[1]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vz = float(np.clip(float(target_vel_ref[2]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        wz = float(np.clip(float(target_vel_ref[3]), -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS))

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        twist.angular.z = wz
        self.cmd_vel_pub.publish(twist)

    # ==================================================================
    # -- stato per il frontend (WebSocket), aggiornato a ~10Hz --
    # ==================================================================
    def _update_latest_state(self):
        with self._wp_lock:
            wp_idx = self.wp_idx
            target = self.wp_pos_queue[wp_idx].tolist()
            queues_completed = self.queues_completed

        with self._tello_lock:
            battery = self.battery_pct
            tello_connected = self.tello_connected

        with self._session_lock:
            session_state = self.session_state

        state = {
            "t": time.time(),
            "session_state": session_state,
            "vicon_connected": self.pose_received,
            "tello_connected": tello_connected,
            "battery": battery,
            "pos": self.pos_env.tolist(),
            "yaw_deg": math.degrees(self.yaw),
            "roll_deg": math.degrees(self.roll),
            "pitch_deg": math.degrees(self.pitch),
            "lin_vel_b": self.lin_vel_b.tolist(),
            "ang_vel_b": self.ang_vel_b.tolist(),
            "target": target if self.flight_ready else None,
            "room_min": ROOM_MIN.tolist(),
            "room_max": ROOM_MAX.tolist(),
            "wp_idx": wp_idx,
            "n_waypoints": N_WAYPOINTS,
            "queues_completed": queues_completed,
            "num_queues": self.num_queues,
            "dof_mask_mode": self.dof_mask_mode,
            "target_mode": self.target_mode,
            "advance_mode": self.advance_mode,
        }
        with self._state_lock:
            self._latest_state = state

    def get_latest_state(self):
        with self._state_lock:
            return dict(self._latest_state)

    # ==================================================================
    # -- server web (FastAPI + uvicorn), gira in un thread separato --
    # ==================================================================
    def _run_web_server(self):
        node_ref = self

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
            return node_ref.get_latest_state()

        @app.post("/api/params")
        def post_params(params: ParamsIn):
            ok, msg = node_ref.set_params(
                params.dof_mask_mode, params.target_mode,
                params.advance_mode, params.num_queues,
            )
            return {"ok": ok, "message": msg}

        @app.post("/api/start")
        def post_start():
            threading.Thread(target=node_ref.start_flight_sequence, daemon=True).start()
            return {"ok": True, "message": "Avvio in corso."}

        @app.post("/api/land")
        def post_land():
            # NUOVO: chiama emergency_land() invece di request_land() —
            # atterra SEMPRE, incondizionatamente, indipendentemente da
            # session_state.
            threading.Thread(
                target=node_ref.emergency_land, args=("bottone Land",), daemon=True
            ).start()
            return {"ok": True, "message": "Atterraggio richiesto."}

        @app.post("/api/advance")
        def post_advance():
            node_ref.manual_advance_from_web()
            return {"ok": True}

        @app.websocket("/ws/state")
        async def ws_state(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    await websocket.send_text(json.dumps(node_ref.get_latest_state()))
                    await asyncio.sleep(STATE_BROADCAST_PERIOD_S)
            except WebSocketDisconnect:
                pass

        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")


def main(args=None):
    rclpy.init(args=args)
    node = PositionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruzione richiesta (Ctrl+C): avvio sequenza di atterraggio.")
    finally:
        node.land_sequence()
        node.disconnect_tellopy()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()