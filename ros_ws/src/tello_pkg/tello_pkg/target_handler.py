#!/usr/bin/env python3
"""Nodo 'target_handler' (tello_pkg): genera e fa avanzare la coda di waypoint (nessuna connessione al drone).

Legge la posa Vicon del Tello e dell'ArUco e /tello/flight_state.
target_mode (parametro, modificabile a runtime): variabile (N_WAYPOINTS target random), singolo
(uno ripetuto), custom (fisso da custom_target_x/y/z/yaw), hover (posa al decollo), aruco_target
(posizione live dell'ArUco). advance_mode: manual (INVIO da terminale o Empty su
/target_handler/advance) oppure auto (criterio del training: errore sotto soglia per
TARGET_HOLD_TIME_S). Completate num_queues code pubblica /tello/land_request.

Output: 'targets' (Float32MultiArray, QoS reliable + transient_local), a ogni tick, 17 elementi:
    [0] wp_idx | [1:13] wp_pos_queue appiattita (N_WAYPOINTS x [x, y, z]) | [13:17] wp_yaw_queue
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

# Parametri dell'ambiente (come params/env.yaml del checkpoint), soglie, topic e modalita'
STEP_DT = 0.04
N_WAYPOINTS = 4

ROOM_MIN = np.array([-1.5, -1.0, 0.1])
ROOM_MAX = np.array([1.5, 1.0, 2.0])
TARGET_ROOM_MARGIN = 0.8

TARGET_REACH_THRESHOLD_M = 0.15
TARGET_REACH_YAW_THRESHOLD_RAD = 0.20
TARGET_HOLD_TIME_S = 1.2

VICON_POSE_TOPIC = "/vicon/Tello_2/Tello_2"
ARUCO_POSE_TOPIC = "/vicon/aruco42/aruco42"
ARUCO_TARGET_Z_OFFSET_M = 0.0
FLIGHT_STATE_TOPIC = "/tello/flight_state"
LAND_REQUEST_TOPIC = "/tello/land_request"
TARGETS_TOPIC = "targets"
ADVANCE_TOPIC = "/target_handler/advance"

TERMINAL_POLL_TIMEOUT_S = 0.2

VALID_TARGET_MODES = ("singolo", "variabile", "custom", "hover", "aruco_target")
VALID_ADVANCE_MODES = ("manual", "auto")
QUEUE_MODES = ("singolo", "variabile", "aruco_target")


# Utility: angolo, campionamento di un target nella stanza, lettura non bloccante dello stdin
def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def sample_target_in_room():
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
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


# Nodo: stato della coda di waypoint e I/O ROS
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
        self._num_queues = num_queues

        self._wp_lock = threading.Lock()
        self.wp_pos_queue = np.zeros((N_WAYPOINTS, 3))
        self.wp_yaw_queue = np.zeros(N_WAYPOINTS)
        self.wp_idx = 0
        self.hold_timer = 0.0
        self.queues_completed = 0
        self._mission_complete = False

        self._hover_target = None

        self._pos_env = None
        self._yaw = None
        self._last_aruco_pos = None
        self._flying = False

        self._refresh_targets_locked()

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

    # Generazione della coda in base a target_mode
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
            self.wp_idx = 0
        elif mode == "aruco_target":
            pos = self._last_aruco_pos if self._last_aruco_pos is not None else (
                self._pos_env if self._pos_env is not None else np.zeros(3)
            )
            pos = pos + np.array([0.0, 0.0, ARUCO_TARGET_Z_OFFSET_M])
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

    # Avanzamento della coda (manuale, da topic, automatico)
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

    # Cambio parametri a runtime (valida e rigenera subito il target)
    def _on_param_change(self, params):
        from rcl_interfaces.msg import SetParametersResult

        new_target_mode = self._target_mode
        new_advance_mode = self._advance_mode
        new_num_queues = self._num_queues
        custom_changed = False

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
            self._mission_complete = False

            if mode_changed and new_target_mode == "hover" and self._flying:
                self._capture_hover_target_locked()
            elif mode_changed or (custom_changed and self._target_mode == "custom"):
                self._refresh_targets_locked()

        if mode_changed:
            self.get_logger().info(f"target_mode cambiato a runtime: '{new_target_mode}'")

        return SetParametersResult(successful=True)

    # Callback: pose Vicon (Tello e ArUco) e flight_state
    def pose_cb(self, msg: PoseStamped):
        from scipy.spatial.transform import Rotation as R

        p = msg.pose.position
        q = msg.pose.orientation
        self._pos_env = np.array([p.x, p.y, p.z])
        self._yaw = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2])

    def aruco_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._last_aruco_pos = np.array([p.x, p.y, p.z])

    def flight_state_cb(self, msg: Bool):
        was_flying = self._flying
        self._flying = msg.data
        if self._flying and not was_flying and self._target_mode == "hover":
            with self._wp_lock:
                self._capture_hover_target_locked()
        if not self._flying and was_flying:
            self._hover_target = None

    # Loop periodico: ArUco live, avanzamento automatico, pubblicazione di 'targets'
    def control_loop(self):
        with self._wp_lock:
            if self._target_mode == "aruco_target" and not self._mission_complete:
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


# Thread stdin (INVIO = avanza) e main
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
