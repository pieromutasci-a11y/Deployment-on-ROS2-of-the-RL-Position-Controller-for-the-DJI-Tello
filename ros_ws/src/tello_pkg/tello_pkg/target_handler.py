#!/usr/bin/env python3
"""
Nodo ROS2 'target_handler' (package tello_pkg): possiede la logica di
generazione/avanzamento del TARGET per la pipeline modulare del
controllore di posizione (Percezione / Policy / Attuazione / Target).

Non tocca MAI djitellopy: non ha nessuna connessione al drone. Riceve
solo la posa Vicon del Tello (per l'avanzamento 'auto' e per catturare
l'hover al takeoff) e la posa Vicon dell'ArUco (per 'aruco_target'), e
si abbona a /tello/flight_state (pubblicato da vel_command_handler) per
sapere quando il drone e' decollato.

MODALITA' (parametro ROS2 target_mode, modificabile A RUNTIME):
    variabile     N_WAYPOINTS target random distinti in sequenza dentro
                  la stanza (ROOM_MIN/ROOM_MAX, margine TARGET_ROOM_MARGIN).
    singolo       1 solo target random, ripetuto su tutti gli slot della
                  coda.
    custom        1 solo target FISSO, preso dai parametri ROS2
                  custom_target_x/y/z/yaw (modificabili a runtime).
                  Stessa meccanica di 'singolo' ma senza randomicita'.
    hover         il target diventa la posizione/yaw ESATTI del drone nel
                  momento in cui /tello/flight_state passa a True
                  (appena decollato). Fisso finche' non cambia modalita'.
    aruco_target  come 'singolo'/'variabile' per avanzamento e
                  num_queues (e' in QUEUE_MODES), ma il VALORE del
                  target e' la posizione Vicon CORRENTE del subject
                  ArUco (ARUCO_POSE_TOPIC, yaw fissato a 0), rinfrescata
                  ad OGNI tick (25Hz) e ripetuta sui 4 slot: resta
                  sempre "live", non si blocca al valore campionato
                  all'inizio della coda.

Il cambio di target_mode/advance_mode/num_queues/custom_target_* a
runtime (es. 'ros2 param set') e' gestito da un
add_on_set_parameters_callback: il nuovo target viene generato
IMMEDIATAMENTE al cambio, non al giro successivo. Nessuna protezione
contro salti improvvisi del target: e' una scelta esplicita, la
responsabilita' di eventuali limiti di velocita'/accelerazione resta
del nodo di attuazione (vel_command_handler) o della policy.

AVANZAMENTO (parametro advance_mode, rilevante solo per singolo/variabile/
aruco_target): l'avanzamento manuale ha DUE ingressi equivalenti (stesso
metodo advance_manual()):
    - INVIO (riga vuota) da terminale, letto dal thread stdin di QUESTO
      processo (utile lanciando il nodo da solo con 'ros2 run').
    - un messaggio std_msgs/Empty su ADVANCE_TOPIC
      (/target_handler/advance) — serve per pilotarlo da un altro
      processo, es. il nodo 'mission_console' incluso nei launch file
      (ros2 launch NON inoltra in modo affidabile lo stdin del
      terminale a piu' processi figli contemporaneamente).
    manual  avanza al prossimo waypoint della coda. Ignorato nelle
            modalita' custom/hover (non hanno una coda da avanzare).
    auto    stesso identico criterio ESATTO del training
            (_update_waypoint in pos_controller_env.py): dist E yaw_err
            sotto soglia per TARGET_HOLD_TIME_S secondi CONSECUTIVI.

num_queues: quante code (rigenerazioni di N_WAYPOINTS target, solo per
singolo/variabile) completare prima di richiedere l'atterraggio. Questo
nodo NON possiede la connessione al drone, quindi non puo' chiamare
land() da solo: al raggiungimento del limite pubblica un segnale VUOTO
su /tello/land_request, che vel_command_handler ascolta per avviare
l'atterraggio vero (stesso trattamento di batteria critica/Ctrl+C).

OUTPUT verso observation_handler:
    topic 'targets' (std_msgs/Float32MultiArray), QoS reliable +
    transient_local (un subscriber che si connette tardi riceve subito
    l'ultimo target pubblicato), pubblicato ad ogni tick del timer
    interno (STEP_DT, 25Hz) — anche quando il target non cambia, per
    semplicita' (QoS transient_local copre comunque i subscriber tardivi).
    Layout del vettore (17 elementi):
        [0]                 wp_idx corrente (float, castare a int)
        [1:13]  (12 elementi)  wp_pos_queue appiattita, N_WAYPOINTS*3,
                                riga k = [x_k, y_k, z_k]
        [13:17] (4 elementi)   wp_yaw_queue, N_WAYPOINTS yaw in radianti
"""

import math
import threading
import select
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Empty, Float32MultiArray

# ============================================================
# CONFIG — deve rispecchiare params/env.yaml del checkpoint (stessi
# valori usati in position_controller_VICON_VERSION.py).
# ============================================================
STEP_DT = 0.04                     # 25 Hz, stessa frequenza del loop di controllo HL
N_WAYPOINTS = 4                    # env.yaml: n_waypoints

ROOM_MIN = np.array([-2.0, -1.5, 0.1])
ROOM_MAX = np.array([2.0, 1.5, 2.0])
TARGET_ROOM_MARGIN = 0.8           # env.yaml: target_room_margin

TARGET_REACH_THRESHOLD_M = 0.15
TARGET_REACH_YAW_THRESHOLD_RAD = 0.20
TARGET_HOLD_TIME_S = 1.2

VICON_POSE_TOPIC = "/vicon/tello_42_boosted/tello_42_boosted"
ARUCO_POSE_TOPIC = "/vicon/aruco42/aruco42"
FLIGHT_STATE_TOPIC = "/tello/flight_state"
LAND_REQUEST_TOPIC = "/tello/land_request"
TARGETS_TOPIC = "targets"
ADVANCE_TOPIC = "/target_handler/advance"

TERMINAL_POLL_TIMEOUT_S = 0.2

VALID_TARGET_MODES = ("singolo", "variabile", "custom", "hover", "aruco_target")
VALID_ADVANCE_MODES = ("manual", "auto")
QUEUE_MODES = ("singolo", "variabile", "aruco_target")   # uniche modalita' con avanzamento/coda


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def sample_target_in_room():
    """Campiona un target (pos, yaw) random dentro ROOM_MIN/ROOM_MAX, con
    margine TARGET_ROOM_MARGIN dal muro. Stessa identica logica di
    _sample_in_room() in pos_controller_env.py."""
    center = 0.5 * (ROOM_MIN + ROOM_MAX)
    half = 0.5 * (ROOM_MAX - ROOM_MIN) * TARGET_ROOM_MARGIN
    lo = center - half
    hi = center + half
    lo[2] = max(lo[2], ROOM_MIN[2])
    hi[2] = max(hi[2], lo[2] + 1e-3)
    pos = lo + np.random.rand(3) * (hi - lo)
    yaw = float(np.random.uniform(-math.pi, math.pi))
    return pos, yaw


def leggi_comando_terminale():
    """Lettura NON BLOCCANTE da stdin."""
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


class TargetHandler(Node):
    def __init__(self):
        super().__init__("target_handler")

        self.declare_parameter("target_mode", "variabile")
        self.declare_parameter("advance_mode", "manual")
        self.declare_parameter("num_queues", -1)
        self.declare_parameter("custom_target_x", 0.0)
        self.declare_parameter("custom_target_y", 0.0)
        self.declare_parameter("custom_target_z", 1.0)
        self.declare_parameter("custom_target_yaw", 0.0)
        self.declare_parameter("enable_terminal_input", True)

        self.enable_terminal_input = self.get_parameter("enable_terminal_input").get_parameter_value().bool_value

        target_mode = self.get_parameter("target_mode").get_parameter_value().string_value
        advance_mode = self.get_parameter("advance_mode").get_parameter_value().string_value
        num_queues = self.get_parameter("num_queues").get_parameter_value().integer_value

        if target_mode not in VALID_TARGET_MODES:
            raise ValueError(f"target_mode='{target_mode}' non valido, atteso uno tra {VALID_TARGET_MODES}")
        if advance_mode not in VALID_ADVANCE_MODES:
            raise ValueError(f"advance_mode='{advance_mode}' non valido, atteso uno tra {VALID_ADVANCE_MODES}")

        self._target_mode = target_mode
        self._advance_mode = advance_mode
        self._num_queues = num_queues  # <= 0 = infinito

        # -- stato coda target, protetto da _wp_lock --
        self._wp_lock = threading.Lock()
        self.wp_pos_queue = np.zeros((N_WAYPOINTS, 3))
        self.wp_yaw_queue = np.zeros(N_WAYPOINTS)
        self.wp_idx = 0
        self.hold_timer = 0.0
        self.queues_completed = 0
        self._mission_complete = False   # num_queues raggiunto: niente piu' rigenerazioni

        # -- stato hover: catturato al takeoff, None finche' non succede --
        self._hover_target = None   # (pos: np.ndarray(3), yaw: float) oppure None

        # -- ultima posa nota del Tello (per hover/auto-advance) e dell'ArUco --
        self._pos_env = None
        self._yaw = None
        self._last_aruco_pos = None
        self._flying = False

        self._refresh_targets_locked()

        # -- ROS I/O --
        self.pose_sub = self.create_subscription(PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10)
        self.aruco_sub = self.create_subscription(PoseStamped, ARUCO_POSE_TOPIC, self.aruco_cb, 10)
        self.flight_state_sub = self.create_subscription(
            Bool, FLIGHT_STATE_TOPIC, self.flight_state_cb, 10
        )
        self.advance_sub = self.create_subscription(
            Empty, ADVANCE_TOPIC, self.advance_topic_cb, 10
        )

        targets_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.targets_pub = self.create_publisher(Float32MultiArray, TARGETS_TOPIC, targets_qos)
        self.land_request_pub = self.create_publisher(Empty, LAND_REQUEST_TOPIC, 10)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.timer = self.create_timer(STEP_DT, self.control_loop)

        self.get_logger().info(
            f"target_handler avviato. target_mode={self._target_mode} | "
            f"advance_mode={self._advance_mode} | "
            f"num_queues={self._num_queues if self._num_queues > 0 else 'infinito'}\n"
            "Nel TERMINALE dove gira questo nodo: INVIO = avanza al prossimo "
            "waypoint (solo target_mode in singolo/variabile E advance_mode=manual)."
        )

    # ==================================================================
    # -- generazione/aggiornamento coda target in base a target_mode.
    # ASSUME self._wp_lock gia' acquisito dal chiamante (tranne la prima
    # chiamata in __init__, dove nessun altro thread e' ancora attivo). --
    # ==================================================================
    def _refresh_targets_locked(self):
        mode = self._target_mode
        if mode == "singolo":
            pos, yaw = sample_target_in_room()
            for k in range(N_WAYPOINTS):
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
            self.wp_idx = 0
        elif mode == "variabile":
            for k in range(N_WAYPOINTS):
                pos, yaw = sample_target_in_room()
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
            self.wp_idx = 0
        elif mode == "custom":
            pos = np.array([
                self.get_parameter("custom_target_x").get_parameter_value().double_value,
                self.get_parameter("custom_target_y").get_parameter_value().double_value,
                self.get_parameter("custom_target_z").get_parameter_value().double_value,
            ])
            yaw = self.get_parameter("custom_target_yaw").get_parameter_value().double_value
            for k in range(N_WAYPOINTS):
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = yaw
            self.wp_idx = 0
        elif mode == "hover":
            if self._hover_target is not None:
                pos, yaw = self._hover_target
                for k in range(N_WAYPOINTS):
                    self.wp_pos_queue[k] = pos
                    self.wp_yaw_queue[k] = yaw
            # se non ancora catturato: lascia la coda com'e', verra'
            # riempita al prossimo fronte di flight_state (vedi flight_state_cb).
            self.wp_idx = 0
        elif mode == "aruco_target":
            pos = self._last_aruco_pos if self._last_aruco_pos is not None else (
                self._pos_env if self._pos_env is not None else np.zeros(3)
            )
            for k in range(N_WAYPOINTS):
                self.wp_pos_queue[k] = pos
                self.wp_yaw_queue[k] = 0.0
            self.wp_idx = 0

    def _capture_hover_target_locked(self):
        if self._pos_env is None or self._yaw is None:
            self.get_logger().warn(
                "hover: takeoff rilevato ma nessuna posa Vicon ancora ricevuta, "
                "impossibile catturare il target hover."
            )
            return
        self._hover_target = (self._pos_env.copy(), self._yaw)
        self.get_logger().info(
            f"hover: target catturato a ({self._hover_target[0][0]:.2f}, "
            f"{self._hover_target[0][1]:.2f}, {self._hover_target[0][2]:.2f})m, "
            f"yaw={math.degrees(self._hover_target[1]):.1f}deg"
        )
        if self._target_mode == "hover":
            self._refresh_targets_locked()

    # ==================================================================
    # -- avanzamento coda. Valido SOLO per target_mode in QUEUE_MODES.
    # ASSUME self._wp_lock gia' acquisito. Ritorna True se e' stato
    # raggiunto num_queues (la richiesta di atterraggio viene pubblicata
    # qui: questo nodo non puo' chiamare land() da solo). --
    # ==================================================================
    def _advance_waypoint_locked(self, source: str) -> bool:
        if self.wp_idx < N_WAYPOINTS - 1:
            self.wp_idx += 1
            wp = self.wp_pos_queue[self.wp_idx]
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
                "num_queues raggiunto, richiedo atterraggio a vel_command_handler."
            )
            self._mission_complete = True
            self.land_request_pub.publish(Empty())
            return True

        self._refresh_targets_locked()
        wp = self.wp_pos_queue[0]
        self.get_logger().info(
            f"[{source}] Coda {self.queues_completed}/{limite} completata: nuova coda "
            f"random generata ({self._target_mode}). Primo target: "
            f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})m"
        )
        return False

    def advance_manual(self, source: str = "terminale"):
        if self._mission_complete:
            return
        if self._target_mode not in QUEUE_MODES:
            self.get_logger().warn(
                f"[{source}] target_mode='{self._target_mode}' non ha una coda da "
                "avanzare, avanzamento ignorato."
            )
            return
        if self._advance_mode != "manual":
            self.get_logger().warn(
                f"[{source}] advance_mode='{self._advance_mode}': avanzamento manuale "
                "disattivato, avanzamento ignorato."
            )
            return
        with self._wp_lock:
            self._advance_waypoint_locked(source)

    def advance_topic_cb(self, msg: Empty):
        self.advance_manual(source="console")

    # -------------------- parametri a runtime --------------------
    def _on_param_change(self, params):
        from rcl_interfaces.msg import SetParametersResult

        new_target_mode = self._target_mode
        new_advance_mode = self._advance_mode
        new_num_queues = self._num_queues
        custom_changed = False

        # limiti stanza per gli assi del target custom (stessi ROOM_MIN/ROOM_MAX
        # usati per il campionamento random): custom_target_yaw non ha limiti.
        custom_axis_bounds = {
            "custom_target_x": (ROOM_MIN[0], ROOM_MAX[0]),
            "custom_target_y": (ROOM_MIN[1], ROOM_MAX[1]),
            "custom_target_z": (ROOM_MIN[2], ROOM_MAX[2]),
        }

        for p in params:
            if p.name == "target_mode":
                if p.value not in VALID_TARGET_MODES:
                    return SetParametersResult(successful=False, reason=f"target_mode non valido, atteso {VALID_TARGET_MODES}")
                new_target_mode = p.value
            elif p.name == "advance_mode":
                if p.value not in VALID_ADVANCE_MODES:
                    return SetParametersResult(successful=False, reason=f"advance_mode non valido, atteso {VALID_ADVANCE_MODES}")
                new_advance_mode = p.value
            elif p.name == "num_queues":
                new_num_queues = p.value
            elif p.name in custom_axis_bounds:
                lo, hi = custom_axis_bounds[p.name]
                if not (lo <= p.value <= hi):
                    return SetParametersResult(
                        successful=False,
                        reason=f"{p.name}={p.value} fuori dai limiti della stanza [{lo:.2f}, {hi:.2f}]",
                    )
                custom_changed = True
            elif p.name == "custom_target_yaw":
                custom_changed = True

        with self._wp_lock:
            mode_changed = new_target_mode != self._target_mode
            self._target_mode = new_target_mode
            self._advance_mode = new_advance_mode
            self._num_queues = new_num_queues
            self._mission_complete = False  # un cambio esplicito di parametri riabilita la missione

            if mode_changed and new_target_mode == "hover" and self._flying:
                self._capture_hover_target_locked()
            elif mode_changed or (custom_changed and self._target_mode == "custom"):
                self._refresh_targets_locked()

        if mode_changed:
            self.get_logger().info(f"target_mode cambiato a runtime: '{new_target_mode}'")

        return SetParametersResult(successful=True)

    # -------------------- callback Vicon (Tello) --------------------
    def pose_cb(self, msg: PoseStamped):
        from scipy.spatial.transform import Rotation as R

        p = msg.pose.position
        q = msg.pose.orientation
        self._pos_env = np.array([p.x, p.y, p.z])
        self._yaw = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2])

    # -------------------- callback Vicon (ArUco) --------------------
    def aruco_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._last_aruco_pos = np.array([p.x, p.y, p.z])

    # -------------------- callback flight_state (vel_command_handler) --------------------
    def flight_state_cb(self, msg: Bool):
        was_flying = self._flying
        self._flying = msg.data
        if self._flying and not was_flying and self._target_mode == "hover":
            with self._wp_lock:
                self._capture_hover_target_locked()
        if not self._flying and was_flying:
            # atterrato/reset: la prossima volta che decolla va ricatturato
            self._hover_target = None

    # -------------------- loop @25Hz: aruco live, auto-advance, publish --------------------
    def control_loop(self):
        with self._wp_lock:
            if self._target_mode == "aruco_target" and not self._mission_complete:
                # a differenza di singolo/variabile, qui la coda viene
                # rinfrescata ad OGNI tick con la posizione CORRENTE del
                # marker (resta sempre variabile), pur restando in
                # QUEUE_MODES: avanzamento/num_queues funzionano come per
                # le altre modalita' a coda, ma il valore del target che
                # si insegue e' sempre quello live.
                self._refresh_targets_locked()

            if (
                not self._mission_complete
                and self._target_mode in QUEUE_MODES
                and self._advance_mode == "auto"
                and self._pos_env is not None
                and self._yaw is not None
            ):
                w0_pos = self.wp_pos_queue[self.wp_idx]
                w0_yaw = float(self.wp_yaw_queue[self.wp_idx])
                dist = float(np.linalg.norm(self._pos_env - w0_pos))
                yaw_err = abs(wrap_to_pi(self._yaw - w0_yaw))
                converged = dist < TARGET_REACH_THRESHOLD_M and yaw_err < TARGET_REACH_YAW_THRESHOLD_RAD
                self.hold_timer = self.hold_timer + STEP_DT if converged else 0.0
                if self.hold_timer >= TARGET_HOLD_TIME_S:
                    self.hold_timer = 0.0
                    self._advance_waypoint_locked("auto")

            wp_idx = self.wp_idx
            wp_pos_queue = self.wp_pos_queue.copy()
            wp_yaw_queue = self.wp_yaw_queue.copy()

        msg = Float32MultiArray()
        msg.data = [float(wp_idx)] + wp_pos_queue.flatten().tolist() + wp_yaw_queue.tolist()
        self.targets_pub.publish(msg)


def terminal_input_loop(node: TargetHandler):
    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  AVANZAMENTO WAYPOINT DA TERMINALE (target_handler)\n"
        "  INVIO (riga vuota) -> avanza al prossimo waypoint (solo target_mode\n"
        "                        in singolo/variabile E advance_mode=manual)\n"
        "=======================================================================\n"
    )
    while rclpy.ok():
        comando = leggi_comando_terminale()
        if comando is None:
            time.sleep(TERMINAL_POLL_TIMEOUT_S)
            continue
        if comando == "":
            node.advance_manual()
        else:
            print(f"[terminale] Comando non riconosciuto: '{comando}' (solo INVIO e' gestito qui)")


def main(args=None):
    rclpy.init(args=args)
    node = TargetHandler()

    if node.enable_terminal_input:
        input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
        input_thread.start()
    else:
        node.get_logger().info(
            "enable_terminal_input=false: thread stdin interno disattivato "
            "(usa mission_console/il topic /target_handler/advance per l'avanzamento manuale)."
        )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
