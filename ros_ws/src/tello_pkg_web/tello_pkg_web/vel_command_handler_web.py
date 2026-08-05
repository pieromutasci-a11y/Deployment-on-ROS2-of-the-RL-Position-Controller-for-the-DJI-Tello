#!/usr/bin/env python3
"""
Nodo ROS2 'vel_command_handler_web' (package tello_pkg_web): equivalente
WEB di tello_pkg/vel_command_handler.py + tello_pkg/mission_console.py,
fusi nello stesso processo.

PERCHE' FUSI (a differenza di tello_pkg, dove sono due processi separati):
mission_console e' un processo a parte SOLO perche' 'ros2 launch' non
inoltra lo stdin a piu' processi figli (vedi docstring di
tello_pkg/mission_console.py). Il server web qui dentro non ha questo
problema (HTTP/WebSocket, non stdin): puo' quindi girare nello stesso
'ros2 launch' di target_handler/observation_handler/policy_handler, senza
bisogno di un secondo terminale — vedi launch/web_pipeline.launch.py. Le
stesse identiche chiamate set_parameters usate da mission_console diventano
qui gli handler degli endpoint REST (/api/params, /api/custom_target,
/api/advance).

RIUSO (nessuna logica duplicata/reinventata):
  - attuazione/telemetria/watchdog/failsafe batteria/land/takeoff: portati
    1:1 da tello_pkg/vel_command_handler.py (stessi topic, stesse costanti,
    stesso gate esplicito su /tello/start_request — se non premi "Avvia
    algoritmo" nella dashboard il drone non si muove).
  - set_remote_param via servizio SetParameters: portato 1:1 da
    tello_pkg/mission_console.py.
  - calcoli di posa per la SOLA dashboard (pos/rpy/lin_vel_b/ang_vel_b):
    funzioni pure IMPORTATE da tello_pkg.observation_handler
    (compute_projected_gravity_b / compute_lin_vel_body /
    euler_rates_to_body_rates / wrap_to_pi) — questo nodo si iscrive PER
    CONTO SUO a VICON_POSE_TOPIC solo per popolare la dashboard web, NON
    alimenta mai la policy (quello resta compito esclusivo di
    observation_handler, altro processo).
  - limiti stanza (ROOM_MIN/ROOM_MAX), target_mode/advance_mode validi,
    ADVANCE_TOPIC: importati da tello_pkg.target_handler, stessa fonte di
    verita' gia' usata da target_handler/mission_console (il rifiuto
    fuori-stanza e' comunque garantito anche lato server dentro
    target_handler._on_param_change, indipendentemente da questo nodo).

NUOVO rispetto a tello_pkg_web/position_controller_web.py (monolitico,
lasciato invariato):
  - target_mode 'custom' esposto in dashboard: endpoint POST
    /api/custom_target {x,y,z} valida contro ROOM_MIN/ROOM_MAX e chiama
    set_remote_param su custom_target_x/y/z — applicabile in QUALSIASI
    momento (non solo a sessione idle), stessa semantica "a runtime" gia'
    prevista da target_handler (il target si aggiorna immediatamente,
    utile per "disegnare" il target nella scena 3D mentre il drone vola).
  - marker ArUco (ARUCO_POSE_TOPIC) sottoscritto SEMPRE, indipendentemente
    dal target_mode attivo, e trasmesso al frontend come 'aruco_pos' cosi'
    si vede muovere in scena anche fuori da target_mode='aruco_target'.

REGISTRAZIONE DATI: stessa logica di position_controller_web.py
(save_data_and_plots, CSV + PNG 4x2), qui alimentata dal pose_cb/telemetria
di QUESTO nodo (che possiede la connessione djitellopy).
"""

import csv
import math
import os
import time
import json
import signal
import threading
import asyncio
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Empty, Float32MultiArray
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('Agg')  # backend non interattivo per salvataggio figure senza server X
import matplotlib.pyplot as plt

from djitellopy import Tello as DJITello

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from tello_pkg.target_handler import (
    ROOM_MIN, ROOM_MAX, N_WAYPOINTS,
    VALID_TARGET_MODES, VALID_ADVANCE_MODES,
    VICON_POSE_TOPIC, ARUCO_POSE_TOPIC, TARGETS_TOPIC, ADVANCE_TOPIC,
)
from tello_pkg.observation_handler import (
    DOF_MASKS,
    wrap_to_pi, compute_projected_gravity_b, compute_lin_vel_body, euler_rates_to_body_rates,
)

# ============================================================
# CONFIG — stessi valori usati in tello_pkg/vel_command_handler.py
# ============================================================
VEL_REF_SCALE = np.array([1.0, 1.0, 1.0, 1.5])   # [vx,vy,vz,wz]
MAX_LIN_VEL_MPS = 0.8
MAX_YAW_RATE_RADPS = 1.0

ACTION_TIMEOUT_S = 0.2

PRE_TAKEOFF_SETTLE_S = 2.0
POST_TAKEOFF_SETTLE_S = 3.0
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_CM = 15

TELEMETRY_POLL_PERIOD_S = 1.0
TELLO_LOST_LAND_TIMEOUT_S = 3.0
BATTERY_FAILSAFE_PCT = 15

POLICY_ACTION_TOPIC = "/tello/policy_action"
FLIGHT_STATE_TOPIC = "/tello/flight_state"
LAND_REQUEST_TOPIC = "/tello/land_request"
START_REQUEST_TOPIC = "/tello/start_request"

TARGET_HANDLER_SET_PARAMS = "/target_handler/set_parameters"
OBSERVATION_HANDLER_SET_PARAMS = "/observation_handler/set_parameters"
SERVICE_WAIT_TIMEOUT_S = 5.0
SERVICE_CALL_TIMEOUT_S = 3.0

# -- pose per la SOLA dashboard (nessun impatto sulla pipeline di controllo) --
POSE_TIMEOUT_S = 0.5
MAX_PLAUSIBLE_SPEED_MPS = 5.0
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1
VEL_FILTER_ALPHA = 0.3

STATE_BROADCAST_PERIOD_S = 0.1   # ~10Hz verso il frontend

# ============================================================
# WEB SERVER
# ============================================================
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_static")


class VelCommandHandlerWeb(Node):
    def __init__(self):
        super().__init__("vel_command_handler_web")

        self.declare_parameter("enable_terminal_input", True)
        self.enable_terminal_input = self.get_parameter("enable_terminal_input").get_parameter_value().bool_value

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.declare_parameter("output_dir", script_dir)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("save_plot", True)
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        self.save_csv_flag = self.get_parameter("save_csv").get_parameter_value().bool_value
        self.save_plot_flag = self.get_parameter("save_plot").get_parameter_value().bool_value
        self.start_time = time.time()
        self.vicon_history = []
        self.flight_history = []

        # ---------------- attuazione djitellopy (da vel_command_handler.py) ----------------
        self.flight_ready = False
        self._landing_started = False
        self._shutdown_requested = False
        self._takeoff_in_progress = False
        self.connected = False

        self._action_lock = threading.Lock()
        self.last_action = np.zeros(4)
        self.last_action_wall_time = None

        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.tello_alt_cm = None
        self._last_state_snapshot = None
        self.last_state_change_time = None
        self._state_warned = False

        self.drone = DJITello()

        self.action_sub = self.create_subscription(Twist, POLICY_ACTION_TOPIC, self.policy_action_cb, 10)
        self.land_request_sub = self.create_subscription(Empty, LAND_REQUEST_TOPIC, self.land_request_cb, 10)
        self.start_request_sub = self.create_subscription(Empty, START_REQUEST_TOPIC, self.start_request_cb, 10)

        flight_state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.flight_state_pub = self.create_publisher(Bool, FLIGHT_STATE_TOPIC, flight_state_qos)

        self.watchdog_timer = self.create_timer(ACTION_TIMEOUT_S, self.watchdog_cb)
        self.telemetry_timer = self.create_timer(TELEMETRY_POLL_PERIOD_S, self.telemetry_cb)

        # ---------------- ruolo "mission console" via HTTP (da mission_console.py) ----------------
        self.advance_pub = self.create_publisher(Empty, ADVANCE_TOPIC, 10)
        self.target_handler_client = self.create_client(SetParameters, TARGET_HANDLER_SET_PARAMS)
        self.observation_handler_client = self.create_client(SetParameters, OBSERVATION_HANDLER_SET_PARAMS)

        # mirror locale degli ultimi parametri applicati con successo (stessi
        # default di declare_parameter in target_handler/observation_handler)
        self._params_lock = threading.Lock()
        self.dof_mask_mode = "full"
        self.target_mode = "variabile"
        self.advance_mode = "manual"
        self.num_queues = -1
        self.custom_target = None  # [x, y, z] correnti, None se mai impostato

        # ---------------- coda target (da target_handler, per la dashboard) ----------------
        self._targets_lock = threading.Lock()
        self.wp_idx = 0
        self._last_wp_idx = None
        self.queues_completed = 0   # stimato localmente: +1 ad ogni wrap di wp_idx (solo display, non safety-critical)
        self.wp_pos_queue = np.zeros((N_WAYPOINTS, 3))
        self.wp_yaw_queue = np.zeros(N_WAYPOINTS)
        self.targets_received = False
        self.targets_sub = self.create_subscription(Float32MultiArray, TARGETS_TOPIC, self.targets_cb, 10)

        # ---------------- pose Vicon PER LA SOLA DASHBOARD (nessun impatto sul controllo) ----------------
        self.pos_env = np.zeros(3)
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.lin_vel_b = np.zeros(3)
        self.ang_vel_b = np.zeros(3)
        self.proj_grav_b = np.array([0.0, 0.0, -1.0])
        self.prev_pos_world = None
        self.prev_euler = None
        self.prev_time = None
        self.pose_received = False
        self.last_pose_wall_time = None
        self.pose_sub = self.create_subscription(PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10)

        # ---------------- marker ArUco, SEMPRE live (indipendente da target_mode) ----------------
        self.last_aruco_pos = None
        self.aruco_sub = self.create_subscription(PoseStamped, ARUCO_POSE_TOPIC, self.aruco_cb, 10)

        # ---------------- stato condiviso col server web ----------------
        self._state_lock = threading.Lock()
        self._latest_state = {}
        self.state_timer = self.create_timer(STATE_BROADCAST_PERIOD_S, self._update_latest_state)

        threading.Thread(target=self._run_web_server, daemon=True).start()

        self.get_logger().info(
            f"vel_command_handler_web avviato. Interfaccia su http://<host>:{WEB_PORT}/ | "
            f"Sottoscritto a '{POLICY_ACTION_TOPIC}', '{LAND_REQUEST_TOPIC}', '{TARGETS_TOPIC}', "
            f"Vicon '{VICON_POSE_TOPIC}', ArUco '{ARUCO_POSE_TOPIC}'."
        )

    # ==================================================================
    # -- attuazione (invariata da vel_command_handler.py) --
    # ==================================================================
    def _publish_flight_state(self):
        msg = Bool()
        msg.data = self.flight_ready
        self.flight_state_pub.publish(msg)

    def policy_action_cb(self, msg: Twist):
        with self._action_lock:
            self.last_action = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])
            self.last_action_wall_time = time.monotonic()

        if not self.flight_ready or self._shutdown_requested:
            return

        action = np.clip(self.last_action, -1.0, 1.0)
        target_vel_ref = action * VEL_REF_SCALE

        vx = float(np.clip(target_vel_ref[0], -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vy = float(np.clip(target_vel_ref[1], -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vz = float(np.clip(target_vel_ref[2], -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        wz = float(np.clip(target_vel_ref[3], -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS))
        self._send_vel_command(vx, vy, vz, wz)

    def watchdog_cb(self):
        if not self.flight_ready or self._shutdown_requested:
            return
        with self._action_lock:
            age = (
                time.monotonic() - self.last_action_wall_time
                if self.last_action_wall_time is not None
                else float("inf")
            )
        if age > ACTION_TIMEOUT_S:
            self._send_stick_zero()

    def land_request_cb(self, msg: Empty):
        self.get_logger().warn("[land_request] Richiesta di atterraggio ricevuta da un nodo a valle.")
        threading.Thread(target=self.emergency_land, args=("land_request esterno",), daemon=True).start()

    def start_request_cb(self, msg: Empty):
        if not self.connected:
            self.get_logger().warn("[start_request] Drone non ancora connesso, richiesta ignorata.")
            return
        if self.flight_ready or self._takeoff_in_progress:
            self.get_logger().warn("[start_request] Takeoff gia' in corso o drone gia' in volo, richiesta ignorata.")
            return
        if self._shutdown_requested:
            return
        self._takeoff_in_progress = True
        threading.Thread(target=self._do_takeoff, daemon=True).start()

    def _do_takeoff(self):
        try:
            took_off = self.takeoff_sequence()
            if not took_off:
                self.get_logger().error("[start_request] Takeoff fallito. Puoi ritentare premendo di nuovo 'Avvia algoritmo'.")
        finally:
            self._takeoff_in_progress = False

    def telemetry_cb(self):
        self._poll_djitellopy_state()

        now_mono = time.monotonic()
        state_age = (
            (now_mono - self.last_state_change_time)
            if self.last_state_change_time is not None
            else float("inf")
        )
        if state_age > TELLO_LOST_LAND_TIMEOUT_S and self.flight_ready and not self._landing_started:
            if not self._state_warned:
                self.get_logger().error(
                    f"Telemetria drone (state djitellopy) ferma da {state_age:.1f}s: avvio atterraggio."
                )
                self._state_warned = True
            threading.Thread(
                target=self.emergency_land,
                args=(f"telemetria drone assente da oltre {TELLO_LOST_LAND_TIMEOUT_S}s",),
                daemon=True,
            ).start()
        elif state_age <= TELLO_LOST_LAND_TIMEOUT_S:
            self._state_warned = False

    def _poll_djitellopy_state(self):
        try:
            state = self.drone.get_current_state()
        except Exception:
            state = {}

        now_mono = time.monotonic()
        snapshot = tuple(sorted(state.items())) if state else None
        if snapshot is not None and snapshot != self._last_state_snapshot:
            self._last_state_snapshot = snapshot
            self.last_state_change_time = now_mono

        battery = state.get("bat")
        height_cm = state.get("h")
        with self._tello_lock:
            self.battery_pct = battery
            self.tello_alt_cm = height_cm
            if state:
                t_rel = time.time() - self.start_time
                self.flight_history.append({
                    "time": t_rel, "battery": battery, "height_cm": height_cm,
                    "tof_cm": state.get("tof"),
                    "vgx": state.get("vgx"), "vgy": state.get("vgy"), "vgz": state.get("vgz"),
                })

        if battery is not None and battery < BATTERY_FAILSAFE_PCT and self.flight_ready and not self._landing_started:
            self.get_logger().error(f"[djitellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza.")
            threading.Thread(target=self.emergency_land, args=(f"batteria critica ({battery}%)",), daemon=True).start()

    def connect_sequence(self) -> bool:
        self.get_logger().info("[djitellopy] connessione al drone (comandi/rc)...")
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] connessione fallita: {e}")
            return False

        self.connected = True
        self._poll_djitellopy_state()
        with self._tello_lock:
            bat = self.battery_pct
        self.get_logger().info(f"[djitellopy] connesso. batteria={bat}% | in attesa del tasto 'Avvia algoritmo'.")
        return True

    def takeoff_sequence(self) -> bool:
        self.get_logger().info(f"[djitellopy] start_request ricevuto: assestamento {PRE_TAKEOFF_SETTLE_S}s prima del takeoff...")
        time.sleep(PRE_TAKEOFF_SETTLE_S)

        self.get_logger().info("[djitellopy] invio takeoff...")
        try:
            self.drone.takeoff()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] comando takeoff fallito: {e}")
            return False

        self._landing_started = False
        self.flight_ready = True
        self._publish_flight_state()

        deadline = time.monotonic() + TAKEOFF_CONFIRM_TIMEOUT_S
        confirmed = False
        while time.monotonic() < deadline:
            self._poll_djitellopy_state()
            with self._tello_lock:
                alt = self.tello_alt_cm
            if alt is not None and alt > TAKEOFF_MIN_ALT_CM:
                confirmed = True
                break
            time.sleep(0.2)

        if not confirmed:
            self.get_logger().error(
                "[djitellopy] Nessuna variazione di quota plausibile rilevata dopo il takeoff: "
                "il drone resta comunque considerato IN VOLO per sicurezza."
            )
        else:
            self.get_logger().info(f"[djitellopy] decollo confermato, assestamento {POST_TAKEOFF_SETTLE_S}s...")

        time.sleep(POST_TAKEOFF_SETTLE_S)
        self.get_logger().info("[djitellopy] controllo di posizione ATTIVATO.")
        return True

    def _send_stick_zero(self):
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            self.get_logger().error(f"Errore azzerando i comandi rc via djitellopy: {e}")

    def _send_vel_command(self, vx, vy, vz, wz):
        forward_backward = int(round(np.clip(vx, -1.0, 1.0) * 100))
        left_right = int(round(np.clip(-vy, -1.0, 1.0) * 100))
        up_down = int(round(np.clip(vz, -1.0, 1.0) * 100))
        yaw = int(round(np.clip(-wz, -1.0, 1.0) * 100))
        try:
            self.drone.send_rc_control(left_right, forward_backward, up_down, yaw)
        except Exception as e:
            self.get_logger().error(f"Errore inviando comando di movimento via djitellopy: {e}")

    def land_sequence(self):
        if self._landing_started:
            return
        self._landing_started = True

        self.flight_ready = False
        self._publish_flight_state()
        self._send_stick_zero()
        time.sleep(0.2)

        self.get_logger().info("[djitellopy] invio land...")
        try:
            self.drone.land()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] comando land fallito: {e}")
            try:
                self.drone.send_command_without_return("land")
            except Exception:
                pass

        self._landing_started = False  # pronto per un nuovo volo, connessione resta viva

    def emergency_land(self, reason: str = "richiesta manuale"):
        self.get_logger().error(f"[EMERGENZA] Atterraggio forzato: {reason}")
        self.flight_ready = False
        self.land_sequence()

    def disconnect_drone(self):
        try:
            self.drone.end()
        except Exception:
            pass

    @property
    def session_state(self) -> str:
        if self._landing_started:
            return "landing"
        if self._takeoff_in_progress:
            return "starting"
        if self.flight_ready:
            return "flying"
        return "idle"

    # ==================================================================
    # -- ruolo "mission console": set_parameters remoto (da mission_console.py) --
    # ==================================================================
    def set_remote_param(self, client, node_label: str, name: str, value) -> bool:
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_S):
            self.get_logger().error(f"{node_label} non raggiungibile (servizio set_parameters assente), '{name}' NON impostato.")
            return False

        param = Parameter(name, value=value).to_parameter_msg()
        request = SetParameters.Request(parameters=[param])
        future = client.call_async(request)

        deadline = time.monotonic() + SERVICE_CALL_TIMEOUT_S
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)

        if not future.done():
            self.get_logger().error(f"Timeout impostando '{name}' su {node_label}.")
            return False

        result = future.result().results[0]
        if not result.successful:
            self.get_logger().error(f"{node_label} ha rifiutato '{name}={value}': {result.reason}")
            return False
        return True

    def set_params(self, dof_mask_mode, target_mode, advance_mode, num_queues):
        if self.session_state != "idle":
            return False, "Impossibile cambiare parametri: sessione non idle."
        if dof_mask_mode not in DOF_MASKS:
            return False, f"dof_mask_mode non valido: {dof_mask_mode}"
        if target_mode not in VALID_TARGET_MODES:
            return False, f"target_mode non valido: {target_mode}"
        if advance_mode not in VALID_ADVANCE_MODES:
            return False, f"advance_mode non valido: {advance_mode}"
        try:
            num_queues = int(num_queues)
        except (TypeError, ValueError):
            return False, f"num_queues non valido: {num_queues}"

        ok_dof = self.set_remote_param(self.observation_handler_client, "observation_handler", "dof_mask_mode", dof_mask_mode)
        ok_tm = self.set_remote_param(self.target_handler_client, "target_handler", "target_mode", target_mode)
        ok_am = self.set_remote_param(self.target_handler_client, "target_handler", "advance_mode", advance_mode)
        ok_nq = self.set_remote_param(self.target_handler_client, "target_handler", "num_queues", num_queues)
        if not (ok_dof and ok_tm and ok_am and ok_nq):
            return False, "Uno o piu' parametri non sono stati applicati (vedi log del nodo)."

        with self._params_lock:
            self.dof_mask_mode = dof_mask_mode
            self.target_mode = target_mode
            self.advance_mode = advance_mode
            self.num_queues = num_queues
        with self._targets_lock:
            self.queues_completed = 0
            self._last_wp_idx = None
        return True, "Parametri impostati."

    def set_custom_target(self, x: float, y: float, z: float):
        """Applicabile in QUALSIASI momento (non solo a sessione idle):
        stessa semantica 'a runtime' di target_handler (il target si
        aggiorna immediatamente). Il rifiuto fuori-stanza e' comunque
        garantito anche lato server in target_handler._on_param_change;
        qui il controllo e' ripetuto solo per un messaggio d'errore
        immediato senza fare nessuna chiamata parziale."""
        for name, v, lo, hi in (
            ("x", x, ROOM_MIN[0], ROOM_MAX[0]),
            ("y", y, ROOM_MIN[1], ROOM_MAX[1]),
            ("z", z, ROOM_MIN[2], ROOM_MAX[2]),
        ):
            if not (lo <= v <= hi):
                return False, f"custom_target_{name}={v} fuori dai limiti della stanza [{lo:.2f}, {hi:.2f}]"

        ok_x = self.set_remote_param(self.target_handler_client, "target_handler", "custom_target_x", float(x))
        ok_y = self.set_remote_param(self.target_handler_client, "target_handler", "custom_target_y", float(y))
        ok_z = self.set_remote_param(self.target_handler_client, "target_handler", "custom_target_z", float(z))
        if not (ok_x and ok_y and ok_z):
            return False, "custom_target non applicato del tutto (vedi log del nodo)."

        with self._params_lock:
            self.custom_target = [float(x), float(y), float(z)]
        return True, "custom_target impostato."

    def advance(self):
        self.advance_pub.publish(Empty())

    # ==================================================================
    # -- coda target (da target_handler, per la dashboard) --
    # ==================================================================
    def targets_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) != 1 + N_WAYPOINTS * 3 + N_WAYPOINTS:
            return
        wp_idx = int(round(data[0]))
        wp_pos_queue = np.array(data[1:1 + N_WAYPOINTS * 3]).reshape(N_WAYPOINTS, 3)
        wp_yaw_queue = np.array(data[1 + N_WAYPOINTS * 3:])
        with self._targets_lock:
            if self._last_wp_idx is not None and wp_idx < self._last_wp_idx:
                self.queues_completed += 1
            self._last_wp_idx = wp_idx
            self.wp_idx = wp_idx
            self.wp_pos_queue = wp_pos_queue
            self.wp_yaw_queue = wp_yaw_queue
            self.targets_received = True

    # ==================================================================
    # -- pose Vicon PER LA SOLA DASHBOARD (nessun impatto sul controllo) --
    # ==================================================================
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_world = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        raw_vals = [p_world[0], p_world[1], p_world[2], q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            return
        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            return

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt_check = t - self.prev_time
            if dt_check > 1e-3:
                implied_speed = float(np.linalg.norm(p_world - self.prev_pos_world)) / dt_check
                if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
                    return

        euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
        self.proj_grav_b = compute_projected_gravity_b(q.x, q.y, q.z, q.w)

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt = max(t - self.prev_time, 1e-3)
            v_world = (p_world - self.prev_pos_world) / dt
            v_body_raw = compute_lin_vel_body(v_world, q.x, q.y, q.z, q.w)
            self.lin_vel_b = VEL_FILTER_ALPHA * v_body_raw + (1 - VEL_FILTER_ALPHA) * self.lin_vel_b

            if self.prev_euler is not None:
                roll_dot = wrap_to_pi(roll - self.prev_euler[0]) / dt
                pitch_dot = wrap_to_pi(pitch - self.prev_euler[1]) / dt
                yaw_dot = wrap_to_pi(yaw - self.prev_euler[2]) / dt
                w_body_raw = euler_rates_to_body_rates(roll_dot, pitch_dot, yaw_dot, roll, pitch)
                if np.all(np.isfinite(w_body_raw)) and float(np.max(np.abs(w_body_raw))) < MAX_PLAUSIBLE_ANG_SPEED_RADPS:
                    self.ang_vel_b = VEL_FILTER_ALPHA * w_body_raw + (1 - VEL_FILTER_ALPHA) * self.ang_vel_b

        self.prev_pos_world = p_world
        self.prev_euler = (roll, pitch, yaw)
        self.prev_time = t
        self.pos_env = p_world
        self.yaw, self.roll, self.pitch = yaw, roll, pitch
        self.pose_received = True
        self.last_pose_wall_time = time.monotonic()

        t_rel = time.time() - self.start_time
        self.vicon_history.append({
            "time": t_rel,
            "x": p_world[0], "y": p_world[1], "z": p_world[2],
            "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
            "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw),
            "proj_grav_x": self.proj_grav_b[0], "proj_grav_y": self.proj_grav_b[1], "proj_grav_z": self.proj_grav_b[2],
            "lin_vel_b_x": self.lin_vel_b[0], "lin_vel_b_y": self.lin_vel_b[1], "lin_vel_b_z": self.lin_vel_b[2],
            "ang_vel_b_x": self.ang_vel_b[0], "ang_vel_b_y": self.ang_vel_b[1], "ang_vel_b_z": self.ang_vel_b[2],
        })

    def aruco_cb(self, msg: PoseStamped):
        """Aggiorna SEMPRE self.last_aruco_pos, indipendentemente dal
        target_mode attivo: serve solo a far vedere il marker muoversi
        nella scena 3D (richiesta esplicita di visualizzazione), il suo
        uso come TARGET vero resta compito esclusivo di target_handler."""
        p = msg.pose.position
        if any(math.isnan(v) or math.isinf(v) for v in (p.x, p.y, p.z)):
            return
        self.last_aruco_pos = [p.x, p.y, p.z]

    # ==================================================================
    # -- salvataggio CSV/PLOT (identico a position_controller_web.py) --
    # ==================================================================
    def save_data_and_plots(self):
        self.get_logger().info("Avvio salvataggio dati e generazione grafici...")
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        vicon_data = list(self.vicon_history)
        with self._tello_lock:
            flight_data = list(self.flight_history)

        if len(vicon_data) + len(flight_data) == 0:
            self.get_logger().warn("Nessun dato registrato durante la sessione, skip salvataggio.")
            return

        if self.save_csv_flag:
            if vicon_data:
                vicon_csv = os.path.join(self.output_dir, f"sensor_vicon_{timestamp_str}.csv")
                try:
                    with open(vicon_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=vicon_data[0].keys())
                        writer.writeheader()
                        writer.writerows(vicon_data)
                    self.get_logger().info(f"Dati Vicon salvati in: {vicon_csv}")
                except Exception as e:
                    self.get_logger().error(f"Errore salvataggio Vicon CSV: {e}")
            if flight_data:
                flight_csv = os.path.join(self.output_dir, f"sensor_tello_flight_{timestamp_str}.csv")
                try:
                    with open(flight_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=flight_data[0].keys())
                        writer.writeheader()
                        writer.writerows(flight_data)
                    self.get_logger().info(f"Dati Tello Flight salvati in: {flight_csv}")
                except Exception as e:
                    self.get_logger().error(f"Errore salvataggio Tello Flight CSV: {e}")

        if self.save_plot_flag:
            plot_png = os.path.join(self.output_dir, f"sensor_plots_{timestamp_str}.png")
            try:
                fig = plt.figure(figsize=(16, 18))
                fig.suptitle(f"Sensor Data Overview ({timestamp_str})", fontsize=16, fontweight='bold')

                ax1 = fig.add_subplot(4, 2, 1)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax1.plot(t_vicon, [d["x"] for d in vicon_data], label="Pos X (m)", color="r")
                    ax1.plot(t_vicon, [d["y"] for d in vicon_data], label="Pos Y (m)", color="g")
                    ax1.plot(t_vicon, [d["z"] for d in vicon_data], label="Pos Z (m)", color="b")
                    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Position (m)")
                    ax1.set_title("Vicon Position"); ax1.grid(True); ax1.legend()
                else:
                    ax1.set_title("Vicon Position (No Data)")

                ax2 = fig.add_subplot(4, 2, 2, projection='3d')
                if vicon_data:
                    x_v = [d["x"] for d in vicon_data]; y_v = [d["y"] for d in vicon_data]; z_v = [d["z"] for d in vicon_data]
                    ax2.plot(x_v, y_v, z_v, label="3D Path", color="purple")
                    ax2.scatter(x_v[0], y_v[0], z_v[0], color="green", s=40, label="Start")
                    ax2.scatter(x_v[-1], y_v[-1], z_v[-1], color="red", s=40, label="End")
                    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
                    ax2.set_title("Vicon 3D Trajectory"); ax2.legend()
                else:
                    ax2.set_title("Vicon 3D Trajectory (No Data)")

                ax3 = fig.add_subplot(4, 2, 3)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax3.plot(t_vicon, [d["roll_deg"] for d in vicon_data], label="Roll (deg)", color="r")
                    ax3.plot(t_vicon, [d["pitch_deg"] for d in vicon_data], label="Pitch (deg)", color="g")
                    ax3.plot(t_vicon, [d["yaw_deg"] for d in vicon_data], label="Yaw (deg)", color="b")
                    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Angle (deg)")
                    ax3.set_title("Vicon Euler Angles"); ax3.grid(True); ax3.legend()
                else:
                    ax3.set_title("Vicon Euler Angles (No Data)")

                ax4 = fig.add_subplot(4, 2, 4)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax4.plot(t_vicon, [d["proj_grav_x"] for d in vicon_data], label="Proj Grav X", color="m")
                    ax4.plot(t_vicon, [d["proj_grav_y"] for d in vicon_data], label="Proj Grav Y", color="c")
                    ax4.plot(t_vicon, [d["proj_grav_z"] for d in vicon_data], label="Proj Grav Z", color="y")
                    ax4.axhline(-1.0, color="grey", linestyle="--", linewidth=0.8, label="atteso Z=-1")
                    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Gravity (body frame)")
                    ax4.set_title("Projected Gravity Body"); ax4.grid(True); ax4.legend(fontsize=8)
                else:
                    ax4.set_title("Projected Gravity Body (No Data)")

                ax5 = fig.add_subplot(4, 2, 5)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax5.plot(t_vicon, [d["lin_vel_b_x"] for d in vicon_data], label="LinVel X (m/s)", color="r")
                    ax5.plot(t_vicon, [d["lin_vel_b_y"] for d in vicon_data], label="LinVel Y (m/s)", color="g")
                    ax5.plot(t_vicon, [d["lin_vel_b_z"] for d in vicon_data], label="LinVel Z (m/s)", color="b")
                if flight_data and any(d.get("vgx") is not None for d in flight_data):
                    t_fl = [d["time"] for d in flight_data]
                    ax5.plot(t_fl, [d["vgx"] for d in flight_data], label="SDK vgx (raw)", color="r", linestyle=":")
                    ax5.plot(t_fl, [d["vgy"] for d in flight_data], label="SDK vgy (raw)", color="g", linestyle=":")
                    ax5.plot(t_fl, [d["vgz"] for d in flight_data], label="SDK vgz (raw)", color="b", linestyle=":")
                ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Vel (m/s | raw SDK)")
                ax5.set_title("Linear Velocity (Vicon body vs SDK vg raw)"); ax5.grid(True); ax5.legend(fontsize=8)

                ax6 = fig.add_subplot(4, 2, 6)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax6.plot(t_vicon, [d["ang_vel_b_x"] for d in vicon_data], label="wx", color="darkred")
                    ax6.plot(t_vicon, [d["ang_vel_b_y"] for d in vicon_data], label="wy", color="darkgreen")
                    ax6.plot(t_vicon, [d["ang_vel_b_z"] for d in vicon_data], label="wz", color="darkblue")
                    ax6.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
                    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Ang vel (rad/s)")
                    ax6.set_title("Body Angular Velocity ang_vel_b"); ax6.grid(True); ax6.legend(fontsize=8)
                else:
                    ax6.set_title("Body Angular Velocity (No Data)")

                ax7 = fig.add_subplot(4, 2, 7)
                if vicon_data:
                    for key, lab, col in (("ang_vel_b_x", "wx", "darkred"), ("ang_vel_b_y", "wy", "darkgreen"), ("ang_vel_b_z", "wz", "darkblue")):
                        ax7.hist([d[key] for d in vicon_data], bins=60, alpha=0.5, label=lab, color=col)
                    ax7.set_xlabel("Ang vel (rad/s)"); ax7.set_ylabel("Occorrenze")
                    ax7.set_title("ang_vel_b distribution"); ax7.grid(True); ax7.legend(fontsize=8)
                else:
                    ax7.set_title("ang_vel_b distribution (No Data)")

                ax8 = fig.add_subplot(4, 2, 8)
                if flight_data:
                    t_fl = [d["time"] for d in flight_data]
                    if any(d["battery"] is not None for d in flight_data):
                        ax8.plot(t_fl, [d["battery"] for d in flight_data], label="Battery (%)", color="orange")
                    if any(d["height_cm"] is not None for d in flight_data):
                        h_vals = [d["height_cm"] / 100.0 if d["height_cm"] is not None else None for d in flight_data]
                        ax8.plot(t_fl, h_vals, label="Height SDK (m)", color="teal")
                    if vicon_data:
                        ax8.plot([d["time"] for d in vicon_data], [d["z"] for d in vicon_data], label="Height Vicon Z (m)", color="purple", linestyle="--")
                    ax8.set_xlabel("Time (s)"); ax8.set_ylabel("Value")
                    ax8.set_title("Tello State (Battery & Height)"); ax8.grid(True); ax8.legend(fontsize=8)
                else:
                    ax8.set_title("Tello State (No Data)")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(plot_png, dpi=200)
                plt.close(fig)
                self.get_logger().info(f"Grafici dei sensori salvati in: {plot_png}")
            except Exception as e:
                self.get_logger().error(f"Errore durante il plot dei grafici: {e}")

    # ==================================================================
    # -- stato per il frontend (WebSocket), aggiornato a ~10Hz --
    # ==================================================================
    def _update_latest_state(self):
        with self._targets_lock:
            wp_idx = self.wp_idx
            queues_completed = self.queues_completed
            target = self.wp_pos_queue[wp_idx].tolist() if self.targets_received else None

        with self._tello_lock:
            battery = self.battery_pct

        with self._params_lock:
            dof_mask_mode = self.dof_mask_mode
            target_mode = self.target_mode
            advance_mode = self.advance_mode
            num_queues = self.num_queues
            custom_target = list(self.custom_target) if self.custom_target is not None else None

        pose_age = (time.monotonic() - self.last_pose_wall_time) if self.pose_received else float("inf")
        vicon_connected = pose_age <= POSE_TIMEOUT_S

        state = {
            "t": time.time(),
            "session_state": self.session_state,
            "vicon_connected": vicon_connected,
            "tello_connected": self.connected,
            "battery": battery,
            "pos": self.pos_env.tolist(),
            "yaw_deg": math.degrees(self.yaw),
            "roll_deg": math.degrees(self.roll),
            "pitch_deg": math.degrees(self.pitch),
            "lin_vel_b": self.lin_vel_b.tolist(),
            "ang_vel_b": self.ang_vel_b.tolist(),
            "target": target if self.flight_ready else None,
            "aruco_pos": self.last_aruco_pos,
            "custom_target": custom_target,
            "room_min": ROOM_MIN.tolist(),
            "room_max": ROOM_MAX.tolist(),
            "wp_idx": wp_idx,
            "n_waypoints": N_WAYPOINTS,
            "queues_completed": queues_completed,
            "num_queues": num_queues,
            "dof_mask_mode": dof_mask_mode,
            "target_mode": target_mode,
            "advance_mode": advance_mode,
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

        class CustomTargetIn(BaseModel):
            x: float
            y: float
            z: float

        app = FastAPI()
        if os.path.isdir(STATIC_DIR):
            app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

        @app.get("/")
        def root():
            return RedirectResponse(url="/static/index.html")

        @app.get("/api/status")
        def get_status():
            return node_ref.get_latest_state()

        @app.post("/api/params")
        def post_params(params: ParamsIn):
            ok, msg = node_ref.set_params(params.dof_mask_mode, params.target_mode, params.advance_mode, params.num_queues)
            return {"ok": ok, "message": msg}

        @app.post("/api/custom_target")
        def post_custom_target(t: CustomTargetIn):
            ok, msg = node_ref.set_custom_target(t.x, t.y, t.z)
            return {"ok": ok, "message": msg}

        @app.post("/api/start")
        def post_start():
            threading.Thread(target=node_ref.start_request_cb, args=(Empty(),), daemon=True).start()
            return {"ok": True, "message": "Avvio in corso."}

        @app.post("/api/land")
        def post_land():
            threading.Thread(target=node_ref.emergency_land, args=("bottone Land",), daemon=True).start()
            return {"ok": True, "message": "Atterraggio richiesto."}

        @app.post("/api/advance")
        def post_advance():
            node_ref.advance()
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


def terminal_input_loop(node: VelCommandHandlerWeb):
    import select
    import sys

    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  vel_command_handler_web — TERMINALE (facoltativo, la dashboard web fa lo stesso)\n"
        "  start / s        -> decolla\n"
        "  q / quit / exit   -> atterra e chiudi il nodo\n"
        "=======================================================================\n"
    )
    while rclpy.ok() and not node._shutdown_requested:
        if select.select([sys.stdin], [], [], 0.2)[0]:
            comando = sys.stdin.readline().strip()
        else:
            continue
        if comando.lower() in ("q", "quit", "exit"):
            node.get_logger().info("[terminale] Comando 'q' ricevuto: avvio atterraggio e chiusura del nodo.")
            node._shutdown_requested = True
            node.land_sequence()
            rclpy.shutdown()
            break
        elif comando.lower() in ("start", "s"):
            node.start_request_cb(Empty())
        elif comando != "":
            print(f"[terminale] Comando non riconosciuto: '{comando}'")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VelCommandHandlerWeb()

    sigint_received = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: sigint_received.set())

    connected = node.connect_sequence()
    if not connected:
        node.get_logger().error("Connessione al drone fallita: chiudo il nodo.")
        node.destroy_node()
        rclpy.shutdown()
        return

    if node.enable_terminal_input:
        input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
        input_thread.start()
    else:
        node.get_logger().info("enable_terminal_input=false: thread stdin interno disattivato (usa la dashboard web).")

    try:
        while rclpy.ok() and not node._shutdown_requested and not sigint_received.is_set():
            rclpy.spin_once(node, timeout_sec=0.5)
        if sigint_received.is_set():
            node.get_logger().warn("Ctrl+C ricevuto: atterraggio di emergenza in corso.")
    finally:
        node._shutdown_requested = True
        try:
            node.land_sequence()
        except Exception as e:
            node.get_logger().error(f"[land] eccezione durante l'atterraggio: {e}")

        if getattr(node.drone, "is_flying", False):
            try:
                node.drone.send_command_without_return("land")
            except Exception:
                pass

        try:
            node.save_data_and_plots()
        except Exception as e:
            node.get_logger().error(f"Errore durante il salvataggio dati: {e}")

        node.disconnect_drone()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
