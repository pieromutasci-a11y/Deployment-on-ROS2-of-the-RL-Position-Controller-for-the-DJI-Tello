#!/usr/bin/env python3
"""
Nodo ROS2 'position_controller' (package tello_node) che esegue in
inferenza la policy HL (controllore di posizione) addestrata in Isaac Lab.

ARCHITETTURA COMANDI (AGGIORNATA):
  - tellopy: usato SOLO per connect()/takeoff()/land(), IMU (EVENT_LOG_DATA)
    e batteria/quota (EVENT_FLIGHT_DATA). NON viene piu' usato per il
    movimento (niente set_pitch/set_roll/set_throttle/set_yaw).
  - I comandi di velocita' (vx, vy, vz, wz calcolati dalla policy, in
    UNITA' FISICHE REALI m/s e rad/s) vengono pubblicati come
    geometry_msgs/Twist sul topic CMD_VEL_TOPIC. Un nodo/driver SEPARATO,
    con i PID lato drone, e' responsabile di leggere questo topic e
    tradurlo nei comandi reali verso il Tello (stesso pattern del nodo
    'posture_regulation.py' che pubblica su /go2/cmd_vel per il cane
    robot: qui e' l'equivalente per il Tello).

CHECKPOINT: policy_pos_controller/2026-07-31_13-02-43_ppo_torch/
            checkpoints/best_agent.pt
  - head_key="policy_layer" CONFERMATO (ispezionato ckpt["policy"].keys()).
  - architettura net_container [256,128,64] ELU CONFERMATA dai pesi.
  - drone_inertial.total_mass=0.087kg coerente col Tello.
  - parametri env.yaml confrontati e coincidenti con le costanti sotto.

PARAMETRI ROS2 SELEZIONABILI DA TERMINALE (--ros-args -p nome:=valore):
    dof_mask_mode := full | uniciclo   (default: full)
        Maschera [vx,vy,vz,wz] applicata HARD sull'azione della policy.
        Valori ESATTI col training di questo checkpoint (dof_mask_set in
        env.yaml): full=(1,1,1,1), uniciclo=(1,0,1,1) (niente vy/strafe).
        NOTA: in training uniciclo_vy_hard_mask=False, quindi vy in
        uniciclo era SOLO penalizzato in reward, non azzerato fisicamente
        nell'env. Qui invece l'azione viene mascherata hard (vedi
        action*self.dof_mask sotto): scelta deliberata, piu' sicura su
        hardware reale (garanzia di zero strafe), ma e' un comportamento
        piu' stretto di quello visto durante il training.
    target_mode := singolo | variabile   (default: variabile)
        singolo: un SOLO target random per coda, ripetuto su tutti gli
        N_WAYPOINTS slot (stesso target finche' non si passa alla coda
        successiva). variabile: N_WAYPOINTS target random distinti in
        sequenza. In ENTRAMBI i casi i target sono generati random dentro
        ROOM_MIN/ROOM_MAX (con margine TARGET_ROOM_MARGIN dal muro,
        stessa logica di _sample_in_room in pos_controller_env.py).
    advance_mode := manual | auto   (default: manual)
        manual: avanzamento SOLO da terminale (INVIO = prossimo waypoint,
        se e' l'ultimo genera una nuova coda random). auto: avanzamento
        automatico con l'ESATTO criterio di training (_update_waypoint in
        pos_controller_env.py): il drone deve restare con
        dist<target_reach_threshold E yaw_err<target_reach_yaw_threshold
        per target_hold_time_s CONSECUTIVI, poi avanza (o rigenera una
        nuova coda se era l'ultimo waypoint). In questa modalita' INVIO da
        terminale viene ignorato (solo 'q' resta attivo per l'atterraggio
        di emergenza).
    num_queues := intero   (default: -1 = infinito)
        Quante CODE (rigenerazioni di N_WAYPOINTS target) completare
        prima di atterrare e chiudere il nodo da soli, automaticamente
        (stesso significato di --num_queues in evaluate_pos_waypoint.py).
        Valori <= 0 = nessun limite: si vola finche' non arriva 'q'/Ctrl+C.
        Con N_WAYPOINTS=4 (fisso, baked nella policy addestrata a 52
        osservazioni), num_queues=3 significa "fai atterrare il drone dopo
        aver completato 3 code, cioe' 3*4=12 waypoint in tutto".

Nel terminale dove gira il nodo (thread stdin non bloccante, select()):
    INVIO (riga vuota)  -> avanza al PROSSIMO waypoint (solo advance_mode=manual)
    q / quit / exit     -> atterra e chiude il nodo (sempre attivo)

RESTA INVARIATO rispetto alle versioni precedenti:
  - struttura osservazione a 52 elementi, ordine e normalizzazione.
  - proj_grav_b e lin_vel_b calcolate con rotazione ESATTA (quaternione
    completo) dal Vicon, non piu' approssimazione solo-yaw.
  - sanity check mocap (NaN/Inf, quaternione degenere, jump).
  - watchdog pose Vicon e IMU (alimentato da EVENT_LOG_DATA tellopy).
  - failsafe batteria critica -> land automatico.
  - cap di sicurezza assoluto sulle velocita' comandate.
  - takeoff/land/IMU tramite tellopy.

ASSUNZIONI DA VERIFICARE PRIMA DEL VOLO:
  1) VICON_POSE_TOPIC e' un PLACEHOLDER ("/vicon/tello/pose"): da
     confermare col nome reale del subject Tello in Vicon Tracker.
  2) CMD_VEL_TOPIC e' un PLACEHOLDER ("/tello/cmd_vel"): da confermare
     col nome esatto ascoltato dal nodo PID lato drone (pattern osservato
     per il cane robot: /go2/cmd_vel).
  3) origin_mocap: calibrare misurando dove si trova fisicamente il punto
     (0,0,0) riportato dal Vicon nella TUA stanza.
  4) ROOM_MIN/ROOM_MAX: misurare i bordi reali della stanza in quel frame
     (i target random vengono campionati SOLO dentro questi limiti, con
     margine TARGET_ROOM_MARGIN dal muro).
  5) I guadagni/comportamento del PID lato drone non sono visibili da
     qui: MAX_LIN_VEL_MPS/MAX_YAW_RATE_RADPS restano un cap di sicurezza
     A MONTE del PID, non sostituiscono eventuali limiti che il PID
     stesso applica.
  6) Unita' di imu.gyro_x/y/z ("rad/s presunto", non documentato con
     certezza da tellopy): verificare con rotazione nota.
  7) Unita' di "height" in EVENT_FLIGHT_DATA (presumibilmente decimetri):
     TAKEOFF_MIN_ALT_DM va verificato.
  8) Convenzione yaw: scipy as_euler("xyz") vs Isaac Lab
     euler_xyz_from_quat, mai confrontate esplicitamente per angoli non
     piccoli.
  9) Modalita' di connessione tellopy (AP diretto vs stazione).
  10) POLICY_CKPT_PATH: gia' verificato raggiungibile nel container.
  11) advance_mode=uniciclo con hard-mask sull'azione: comportamento piu'
      stretto di quanto visto in training (vedi nota su dof_mask_mode
      sopra), da validare in volo a bassa quota prima di fidarsene.
"""

import math
import os
import time
import select
import sys
import threading
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from scipy.spatial.transform import Rotation as R

import tellopy

# ============================================================
# CONFIG — deve rispecchiare params/env.yaml del checkpoint usato
# (2026-07-31_13-02-43_ppo_torch), verificato riga per riga.
# ============================================================
STEP_DT = 0.04                     # 25 Hz HL loop (decimation=4, sim.dt=0.01)
N_WAYPOINTS = 4                    # env.yaml: n_waypoints
WP_PREVIEW_HORIZON = 4             # env.yaml: wp_preview_horizon
INTEGRAL_TAU_S = 5.0                # env.yaml: integral_tau_s
INTEGRAL_CLAMP = 1.0                 # env.yaml: integral_clamp
INTEGRAL_OBS_SCALE = 0.5             # env.yaml: integral_obs_scale
VEL_REF_SCALE = torch.tensor([1.0, 1.0, 1.0, 1.5])
# env.yaml: target_lin_vel_xy_scale=1.0, target_lin_vel_z_scale=1.0,
# target_yaw_vel_scale=1.5 -> [vx_scale, vy_scale, vz_scale, wz_scale]

# Stanza REALE (metri, frame origin_mocap).
# <-- SOSTITUIRE con le misure reali della tua stanza di laboratorio
ROOM_MIN = torch.tensor([-2.0, -2.0, 0.1])
ROOM_MAX = torch.tensor([ 2.0,  2.0, 3.0])

# -- generazione target RANDOM dentro la stanza. Valori ESATTI del
# training (env.yaml del checkpoint), vedi _sample_in_room/_update_waypoint
# in pos_controller_env.py. --
TARGET_ROOM_MARGIN = 0.8           # env.yaml: target_room_margin (0<m<=1, restringe il box verso il centro)
TARGET_REACH_THRESHOLD_M = 0.15    # env.yaml: target_reach_threshold
TARGET_REACH_YAW_THRESHOLD_RAD = 0.20  # env.yaml: target_reach_yaw_threshold
TARGET_HOLD_TIME_S = 1.2           # env.yaml: target_hold_time_s

# -- maschere DoF [vx,vy,vz,wz]. DEVONO combaciare ESATTAMENTE con
# cfg.dof_mask_set del training (env.yaml, righe dof_mask_set). Questo
# checkpoint e' stato addestrato SOLO su queste due. --
DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),  # vx + vz + wz, no vy (strafe)
}

# -- PLACEHOLDER: nome reale del subject Vicon del Tello ANCORA DA
# CONFERMARE (pattern: /vicon/<nome>/<nome>, es. /vicon/CART/CART). --
VICON_POSE_TOPIC = "/vicon/tello/pose"

# -- PLACEHOLDER: topic cmd_vel ascoltato dal nodo PID lato drone.
# Pattern osservato per il cane robot: /go2/cmd_vel. DA CONFERMARE. --
CMD_VEL_TOPIC = "/tello/cmd_vel"

POSE_TIMEOUT_S = 0.5   # safety: se non arriva pose Vicon entro questo tempo, ferma il drone

# -- watchdog IMU (alimentato da EVENT_LOG_DATA di tellopy) --
IMU_TIMEOUT_S = 0.5
REQUIRE_IMU = False

# -- sanity check sui dati mocap --
MAX_PLAUSIBLE_SPEED_MPS = 5.0
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1

# -- cap di sicurezza assoluto sui comandi (m/s, rad/s) PUBBLICATI su
#    cmd_vel, A MONTE di qualsiasi PID lato drone --
MAX_LIN_VEL_MPS = 0.8
MAX_YAW_RATE_RADPS = 1.0

STATUS_PRINT_PERIOD_S = 1.0

# -- lettura comandi da terminale (non bloccante, come nello script di eval) --
TERMINAL_POLL_TIMEOUT_S = 0.2

# ============================================================
# TELLOPY — SOLO connessione, takeoff/land, IMU, batteria (NON movimento)
# ============================================================
TELLOPY_CONNECT_TIMEOUT_S = 60.0
POST_TAKEOFF_SETTLE_S = 3.0

TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_DM = 3            # <-- unita' presunte (decimetri), VERIFICARE

TELLOPY_LAND_WAIT_S = 5.0

BATTERY_FAILSAFE_PCT = 15

# ============================================================
# CHECKPOINT — path letto da env var POLICY_CKPT_PATH, con default sotto.
# ============================================================
DEFAULT_CKPT_PATH = (
    "/ros_workspace/src/tello_node/policy_pos_controller/2026-07-31_13-02-43_ppo_torch/"
    "checkpoints/best_agent.pt"
)
CKPT_PATH = os.environ.get("POLICY_CKPT_PATH", DEFAULT_CKPT_PATH)


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_projected_gravity_b(qx, qy, qz, qw):
    """
    Calcola projected_gravity_b esattamente come Isaac Lab, dal quaternione
    VICON: ruota il vettore gravita' mondo [0,0,-1] nel frame corpo tramite
    l'inverso del quaternione di orientamento (world -> body).
    """
    rot_body_to_world = R.from_quat([qx, qy, qz, qw])
    rot_world_to_body = rot_body_to_world.inv()
    g_world = np.array([0.0, 0.0, -1.0])
    return rot_world_to_body.apply(g_world)


def compute_lin_vel_body(v_world: np.ndarray, qx, qy, qz, qw) -> np.ndarray:
    """
    Proietta un vettore VELOCITA' dal frame mondo al frame corpo con la
    rotazione ESATTA (inverso del quaternione COMPLETO, non solo yaw).
    Stessa identica logica di compute_projected_gravity_b.
    """
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(v_world)


def sample_target_in_room():
    """Campiona un target (pos, yaw) random dentro ROOM_MIN/ROOM_MAX, con
    margine TARGET_ROOM_MARGIN dal muro. Stessa identica logica di
    _sample_in_room() in pos_controller_env.py (box ristretto verso il
    centro della stanza, quota clampata al pavimento)."""
    center = 0.5 * (ROOM_MIN + ROOM_MAX)
    half = 0.5 * (ROOM_MAX - ROOM_MIN) * TARGET_ROOM_MARGIN
    lo = center - half
    hi = center + half
    lo[2] = torch.clamp(lo[2], min=ROOM_MIN[2])
    hi[2] = torch.maximum(hi[2], lo[2] + 1e-3)
    pos = lo + torch.rand(3) * (hi - lo)
    yaw = float(torch.empty(1).uniform_(-math.pi, math.pi))
    return pos, yaw


def leggi_comando_terminale():
    """
    Lettura NON BLOCCANTE da stdin, identica a leggi_comando_utente() nello
    script evaluate_pos_controller_continuous.py.
    """
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


class SkrlMlpPolicy(torch.nn.Module):
    """Trunk 'net_container.*' (256,128,64, ELU) + head 'policy_layer' (64->4)."""
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
        raise FileNotFoundError(
            f"Checkpoint non trovato: '{ckpt_path}'. Verifica POLICY_CKPT_PATH "
            f"o DEFAULT_CKPT_PATH e come Dockerfile.tellonode/run.sh montano/"
            f"copiano la cartella policy_pos_controller/ nel container."
        )
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


class PositionController(Node):
    def __init__(self):
        super().__init__("position_controller")

        # -- parametri da terminale (--ros-args -p nome:=valore) --
        self.declare_parameter("dof_mask_mode", "full")
        self.declare_parameter("target_mode", "variabile")
        self.declare_parameter("advance_mode", "manual")
        self.declare_parameter("num_queues", -1)

        dof_mask_mode = self.get_parameter("dof_mask_mode").get_parameter_value().string_value
        target_mode = self.get_parameter("target_mode").get_parameter_value().string_value
        advance_mode = self.get_parameter("advance_mode").get_parameter_value().string_value
        num_queues = self.get_parameter("num_queues").get_parameter_value().integer_value

        if dof_mask_mode not in DOF_MASKS:
            raise ValueError(
                f"dof_mask_mode='{dof_mask_mode}' non valido, atteso uno tra {list(DOF_MASKS.keys())}"
            )
        if target_mode not in ("singolo", "variabile"):
            raise ValueError(f"target_mode='{target_mode}' non valido, atteso 'singolo' o 'variabile'")
        if advance_mode not in ("manual", "auto"):
            raise ValueError(f"advance_mode='{advance_mode}' non valido, atteso 'manual' o 'auto'")

        self._dof_mask_mode = dof_mask_mode
        self._target_mode = target_mode
        self._advance_mode = advance_mode
        self._num_queues = num_queues  # <= 0 = infinito
        self.dof_mask = torch.tensor(DOF_MASKS[dof_mask_mode])

        self.policy, self.run_mean, self.run_var = load_skrl_policy(
            CKPT_PATH, expected_in=52, expected_out=4
        )
        self.get_logger().info(f"Policy caricata da: {CKPT_PATH}")

        # -- stato interno che nel simulatore era gestito dall'env --
        self.pos_env = torch.zeros(3)
        self.yaw = 0.0
        self.lin_vel_b = torch.zeros(3)
        self.ang_vel_b = torch.zeros(3)
        self.proj_grav_b = np.array([0.0, 0.0, -1.0])
        self.prev_pos_world = None
        self.prev_time = None

        self.prev_hl_action = torch.zeros(4)
        self.err_integral = torch.zeros(4)
        self.alpha_leaky = math.exp(-STEP_DT / INTEGRAL_TAU_S)

        # -- coda waypoint. wp_idx/hold_timer protetti da _wp_lock --
        self._wp_lock = threading.Lock()
        self.wp_pos_queue = torch.zeros(N_WAYPOINTS, 3)
        self.wp_yaw_queue = torch.zeros(N_WAYPOINTS)
        self.wp_idx = 0
        self.hold_timer = 0.0
        self.queues_completed = 0
        self._fill_random_queue_locked()  # prima coda random (target_mode)

        self.origin_mocap = torch.tensor([0.0, 0.0, 0.0])  # <-- calibrare!

        # -- stato IMU/telemetria tellopy, protetto da lock separato --
        self._tello_lock = threading.Lock()
        self.imu_received = False
        self.last_imu_wall_time = None
        self._imu_warned = False
        self.battery_pct = None
        self.tello_alt_dm = None
        self._last_mvo_vel = None

        self._mocap_rejected_count = 0

        # -- ROS I/O: Vicon (input, stato del drone) + cmd_vel (OUTPUT,
        # letto da un nodo PID separato lato drone). I target NON arrivano
        # piu' da topic: sono generati random dentro la stanza (vedi
        # _fill_random_queue_locked / target_mode). --
        self.pose_sub = self.create_subscription(
            PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.pose_received = False
        self.last_pose_wall_time = None
        self.flight_ready = False
        self._landing_started = False
        self._shutdown_requested = False  # settato dal thread terminale su 'q'

        # -- tellopy: SOLO connect/takeoff/land + IMU/telemetria, NIENTE
        # movimento (quello passa da cmd_vel_pub, letto da un nodo PID
        # separato). --
        self.drone = tellopy.Tello()
        self.drone.subscribe(self.drone.EVENT_LOG_DATA, self.tello_log_data_cb)
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.tello_flight_data_cb)

        self.timer = self.create_timer(STEP_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)

        self.get_logger().info(
            f"Nodo position_controller avviato. Vicon topic: {VICON_POSE_TOPIC} | "
            f"cmd_vel pubblicato su: {CMD_VEL_TOPIC} | Checkpoint: {CKPT_PATH}\n"
            f"Parametri: dof_mask_mode={self._dof_mask_mode} | "
            f"target_mode={self._target_mode} | advance_mode={self._advance_mode} | "
            f"num_queues={self._num_queues if self._num_queues > 0 else 'infinito'}\n"
            "Target generati random dentro ROOM_MIN/ROOM_MAX. Nel TERMINALE dove gira "
            "il nodo: INVIO = avanza al prossimo waypoint (solo advance_mode=manual), "
            "'q' = atterra e chiudi."
        )

    # ==================================================================
    # -- generazione coda TARGET random dentro la stanza (target_mode) --
    # ASSUME self._wp_lock gia' acquisito dal chiamante. --
    # ==================================================================
    def _fill_random_queue_locked(self):
        if self._target_mode == "singolo":
            pos, yaw = sample_target_in_room()
            for k in range(N_WAYPOINTS):
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
        else:  # variabile
            for k in range(N_WAYPOINTS):
                pos, yaw = sample_target_in_room()
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
        self.wp_idx = 0

    # ==================================================================
    # -- avanzamento coda (condiviso tra INVIO da terminale e auto in
    # control_loop). ASSUME self._wp_lock gia' acquisito dal chiamante.
    # Se non e' l'ultimo waypoint, avanza l'indice. Se lo e', genera una
    # nuova coda random — A MENO CHE num_queues non sia gia' stato
    # raggiunto, nel qual caso NON genera nulla e ritorna True (il
    # chiamante deve avviare atterraggio + shutdown). --
    # ==================================================================
    def _advance_waypoint_locked(self, source: str) -> bool:
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
        limite = self._num_queues if self._num_queues > 0 else "infinito"
        if self._num_queues > 0 and self.queues_completed >= self._num_queues:
            self.get_logger().info(
                f"[{source}] Coda {self.queues_completed}/{limite} completata: "
                "num_queues raggiunto, avvio atterraggio."
            )
            return True

        self._fill_random_queue_locked()
        self.err_integral[:] = 0.0
        wp = self.wp_pos_queue[0].clone()
        self.get_logger().info(
            f"[{source}] Coda {self.queues_completed}/{limite} completata: nuova coda "
            f"random generata ({self._target_mode}). Primo target: "
            f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})m"
        )
        return False

    def _land_and_shutdown(self, reason: str):
        """Avvia atterraggio + chiusura del nodo in un thread separato (non
        blocca chi chiama: sia control_loop, sul thread executor rclpy, sia
        advance_manual, sul thread terminale). Idempotente."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.get_logger().info(f"[shutdown] {reason}")

        def _do_land():
            self.land_sequence()
            if rclpy.ok():
                rclpy.shutdown()

        threading.Thread(target=_do_land, daemon=True).start()

    # ==================================================================
    # -- avanzamento MANUALE via TERMINALE (INVIO/q). Attivo SOLO se
    # advance_mode='manual'. --
    # ==================================================================
    def advance_manual(self):
        if self._advance_mode != "manual":
            self.get_logger().warn(
                f"[terminale] advance_mode='{self._advance_mode}': avanzamento manuale "
                "disattivato, INVIO ignorato."
            )
            return

        with self._wp_lock:
            should_land = self._advance_waypoint_locked("terminale")

        if should_land:
            self._land_and_shutdown("num_queues raggiunto (avanzamento da terminale)")

    def status_cb(self):
        if not self.flight_ready or not self.pose_received:
            return
        with self._wp_lock:
            w0_pos = self.wp_pos_queue[self.wp_idx].clone()
            w0_yaw = float(self.wp_yaw_queue[self.wp_idx])
            idx_now = self.wp_idx
        dist = float(torch.norm(self.pos_env - w0_pos))
        yaw_err = abs(wrap_to_pi(self.yaw - w0_yaw))
        with self._tello_lock:
            bat = self.battery_pct
        if self._advance_mode == "manual":
            mode_hint = "[INVIO=avanza, q=atterra e chiudi]"
        else:
            mode_hint = (
                f"[auto: hold={self.hold_timer:.1f}/{TARGET_HOLD_TIME_S}s "
                f"(reach<{TARGET_REACH_THRESHOLD_M}m,{math.degrees(TARGET_REACH_YAW_THRESHOLD_RAD):.0f}deg), "
                "q=atterra e chiudi]"
            )
        limite = self._num_queues if self._num_queues > 0 else "inf"
        self.get_logger().info(
            f"[status] coda {self.queues_completed + 1}/{limite} | "
            f"wp {idx_now + 1}/{N_WAYPOINTS} | dist={dist:.3f}m | "
            f"yaw_err={math.degrees(yaw_err):.1f}deg | "
            f"target=({w0_pos[0]:.2f},{w0_pos[1]:.2f},{w0_pos[2]:.2f})m | "
            f"batteria={bat}% | {mode_hint}"
        )

    # ==================================================================
    # -- CALLBACK TELLOPY: IMU e telemetria (SOLO lettura, nessun comando) --
    # ==================================================================
    def tello_log_data_cb(self, event, sender, data, **kwargs):
        imu = data.imu
        vals = (imu.gyro_x, imu.gyro_y, imu.gyro_z)
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            self.get_logger().warn("Messaggio IMU (tellopy) con NaN/Inf scartato.")
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
            self.get_logger().error(
                f"[tellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza."
            )
            self.land_sequence()

    # ==================================================================
    # -- TAKEOFF / LAND tramite tellopy (NESSUN comando di movimento qui) --
    # ==================================================================
    def takeoff_sequence(self) -> bool:
        self.get_logger().info("[tellopy] connessione al drone...")
        try:
            self.drone.connect()
            self.drone.wait_for_connection(TELLOPY_CONNECT_TIMEOUT_S)
        except Exception as e:
            self.get_logger().error(f"[tellopy] connessione fallita: {e}")
            return False

        self.get_logger().info("[tellopy] connesso. Invio takeoff...")
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
            self.get_logger().error(
                "[tellopy] Nessuna variazione di quota plausibile rilevata dopo il "
                "takeoff: ABORT, niente controllo di posizione. Verifica manualmente "
                "lo stato del drone."
            )
            return False

        self.get_logger().info(
            f"[tellopy] decollo confermato (quota>{TAKEOFF_MIN_ALT_DM}). "
            f"Attendo {POST_TAKEOFF_SETTLE_S}s di assestamento prima di attivare la policy HL..."
        )
        time.sleep(POST_TAKEOFF_SETTLE_S)

        self.flight_ready = True
        self.get_logger().info("[tellopy] controllo di posizione ATTIVATO.")
        return True

    def _publish_zero_twist(self):
        """Failsafe: pubblica un Twist nullo su cmd_vel (il nodo PID lato
        drone e' responsabile di fermare effettivamente il drone)."""
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception as e:
            self.get_logger().error(f"Errore pubblicando Twist nullo su cmd_vel: {e}")

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

        try:
            self.drone.quit()
        except Exception:
            pass
        self.get_logger().info("[tellopy] connessione chiusa.")

    # -------------------- callback Vicon --------------------
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_world = torch.tensor([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])
        q = msg.pose.orientation

        raw_vals = [p_world[0].item(), p_world[1].item(), p_world[2].item(),
                    q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            self.get_logger().warn("Pose mocap con NaN/Inf scartata.")
            return

        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            self.get_logger().warn(
                f"Quaternione mocap degenere (norma={quat_norm:.3f}), pose scartata."
            )
            return

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt_check = t - self.prev_time
            if dt_check > 1e-3:
                implied_speed = float(torch.norm(p_world - self.prev_pos_world)) / dt_check
                if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
                    self._mocap_rejected_count += 1
                    self.get_logger().warn(
                        f"Jump mocap implausibile ({implied_speed:.2f} m/s stimati), "
                        f"pose scartata (scarti consecutivi: {self._mocap_rejected_count})."
                    )
                    return

        self._mocap_rejected_count = 0

        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
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
        self.pose_received = True
        self.last_pose_wall_time = time.monotonic()

    # -------------------- ciclo di controllo @25Hz --------------------
    def control_loop(self):
        if not self.flight_ready:
            return
        if self._shutdown_requested:
            return

        if not self.pose_received:
            return
        if (time.monotonic() - self.last_pose_wall_time) > POSE_TIMEOUT_S:
            self.get_logger().warn("Pose mocap scaduta: pubblico Twist nullo su cmd_vel (failsafe).")
            self._publish_zero_twist()
            return

        with self._tello_lock:
            imu_ok = self.imu_received
            last_imu_t = self.last_imu_wall_time
        imu_stale = (not imu_ok) or (
            last_imu_t is not None and (time.monotonic() - last_imu_t) > IMU_TIMEOUT_S
        )
        if imu_stale:
            if REQUIRE_IMU:
                if not self._imu_warned:
                    self.get_logger().error(
                        "IMU tellopy scaduta/assente e REQUIRE_IMU=True: Twist nullo su cmd_vel (failsafe)."
                    )
                    self._imu_warned = True
                self._publish_zero_twist()
                return
            else:
                if not self._imu_warned:
                    self.get_logger().warn(
                        "IMU tellopy non disponibile/scaduta: procedo con ang_vel_b invariato "
                        "(REQUIRE_IMU=False)."
                    )
                    self._imu_warned = True
        else:
            self._imu_warned = False

        pos_env = self.pos_env
        yaw = self.yaw

        with self._wp_lock:
            wp_idx = self.wp_idx
            w0_pos = self.wp_pos_queue[wp_idx].clone()
            w0_yaw = float(self.wp_yaw_queue[wp_idx])

        pos_err = pos_env - w0_pos
        yaw_err_signed = wrap_to_pi(yaw - w0_yaw)

        # ==============================================================
        # -- AVANZAMENTO AUTOMATICO (advance_mode='auto'), criterio ESATTO
        # di _update_waypoint() in pos_controller_env.py: dist E yaw_err
        # (se wz e' controllabile) sotto soglia per TARGET_HOLD_TIME_S
        # CONSECUTIVI, altrimenti il timer si azzera. Se era l'ultimo
        # waypoint della coda, genera una nuova coda random invece di
        # fermarsi (volo continuo, come la rigenerazione coda in eval). --
        # ==============================================================
        if self._advance_mode == "auto":
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
                    self._land_and_shutdown("num_queues raggiunto (avanzamento automatico)")
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

        target_vel_ref = action * VEL_REF_SCALE  # [vx, vy, vz, wz] frame CORPO, m/s e rad/s

        vx = float(np.clip(float(target_vel_ref[0]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vy = float(np.clip(float(target_vel_ref[1]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        vz = float(np.clip(float(target_vel_ref[2]), -MAX_LIN_VEL_MPS, MAX_LIN_VEL_MPS))
        wz = float(np.clip(float(target_vel_ref[3]), -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS))

        # -- pubblica su cmd_vel in UNITA' FISICHE REALI (m/s, rad/s):
        # il nodo PID lato drone si aspetta valori con significato fisico
        # diretto, NON un comando stick normalizzato [-1,1]. --
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz
        twist.angular.z = wz
        self.cmd_vel_pub.publish(twist)


def terminal_input_loop(node: PositionController):
    """
    Thread daemon che legge stdin in modo NON BLOCCANTE (come
    leggi_comando_utente() nello script di eval) e traduce i comandi:
      INVIO (riga vuota)  -> node.advance_manual()
      q / quit / exit     -> avvia land_sequence() e richiede lo shutdown
    """
    invio_hint = (
        "  INVIO (riga vuota)  -> avanza al prossimo waypoint della coda attuale\n"
        if node._advance_mode == "manual"
        else "  INVIO (riga vuota)  -> IGNORATO (advance_mode='auto': avanzamento automatico)\n"
    )
    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  AVANZAMENTO WAYPOINT DA TERMINALE\n"
        f"{invio_hint}"
        "  q                   -> atterra e chiudi il nodo\n"
        "=======================================================================\n"
    )
    while rclpy.ok() and not node._shutdown_requested:
        comando = leggi_comando_terminale()
        if comando is None:
            time.sleep(TERMINAL_POLL_TIMEOUT_S)
            continue

        if comando == "":
            node.advance_manual()
        elif comando.lower() in ("q", "quit", "exit"):
            node.get_logger().info(
                "[terminale] Comando 'q' ricevuto: avvio atterraggio e chiusura del nodo."
            )
            node._shutdown_requested = True
            node.land_sequence()
            rclpy.shutdown()
            break
        else:
            print(
                f"[terminale] Comando non riconosciuto: '{comando}' "
                f"(INVIO=avanza, q=atterra e chiudi)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PositionController()

    took_off = node.takeoff_sequence()
    if not took_off:
        node.get_logger().error(
            "Takeoff fallito: chiudo il nodo senza avviare il controllo di posizione."
        )
        node.destroy_node()
        rclpy.shutdown()
        return

    input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
    input_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruzione richiesta (Ctrl+C): avvio sequenza di atterraggio.")
    finally:
        node.land_sequence()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()