#!/usr/bin/env python3
"""
Nodo ROS2 'observation_handler' (package tello_pkg): possiede tutta la
lettura Vicon, le trasformate/quaternioni e la costruzione dello spazio
delle osservazioni (52 elementi) per la pipeline modulare del
controllore di posizione (Percezione / Policy / Attuazione / Target).

Non tocca MAI djitellopy: nessuna connessione al drone. Riceve:
  - posa Vicon del Tello (VICON_POSE_TOPIC) — input primario.
  - la coda target corrente dal topic 'targets' (pubblicato da
    target_handler, layout documentato li').
  - l'ultima azione della policy dal topic /tello/policy_action
    (pubblicato da policy_handler) — serve come 'prev_action' nell'obs,
    stesso identico ruolo che aveva self.prev_hl_action nel controllore
    monolitico. E' un piccolo anello di retroazione (A pubblica obs, B
    calcola l'azione e la ripubblica, A la rilegge) voluto: e' l'unico
    modo per portare prev_action fuori dal processo della policy.

CALCOLI (invariati rispetto a position_controller_VICON_VERSION.py):
  - quaternione Vicon -> angoli di Eulero (roll, pitch, yaw =
    as_euler("xyz")) -> derivata numerica -> velocita' angolare nel
    frame CORPO (euler_rates_to_body_rates, NON una semplice rotazione:
    serve la matrice cinematica degli angoli di Eulero).
  - velocita' lineari: differenze finite mondo -> frame corpo (rotazione
    ESATTA, quaternione completo) + filtro passa-basso VEL_FILTER_ALPHA.
  - projected_gravity_b: rotazione del vettore gravita' mondo nel frame
    corpo tramite l'inverso del quaternione.
  - sanity check mocap invariati: NaN/Inf, quaternione degenere, jump
    implausibile (posizione e velocita' angolare).

INTEGRALE D'ERRORE (integral_norm nell'obs): si azzera SOLO quando
l'INDICE del waypoint attivo (wp_idx, dal topic 'targets') cambia
valore rispetto al messaggio precedente — non quando cambia il VALORE
del target restando sullo stesso indice (rilevante per aruco_target,
che aggiorna la posizione ad ogni tick restando su wp_idx invariato:
l'integrale continua ad accumularsi come se si inseguisse lo stesso
obiettivo concettuale).

WATCHDOG VICON PERSO (due livelli, stessa filosofia del monolitico ma
senza accesso diretto al drone):
  - se la posa scade (> POSE_TIMEOUT_S): questo nodo smette di
    pubblicare osservazioni fresche. La "fame di dati" a valle
    (policy_handler prima, vel_command_handler poi) fa scattare da sola
    il watchdog di hover gia' previsto in vel_command_handler sui
    comandi di velocita' scaduti.
  - se la perdita persiste oltre POSE_LOST_LAND_TIMEOUT_S (3s
    CONSECUTIVI): pubblica anche su /tello/land_request (stesso canale
    usato da target_handler per fine-missione), cosi'
    vel_command_handler tratta una perdita Vicon prolungata come un
    atterraggio vero, non solo hover.

OUTPUT: topic 'observations' (std_msgs/Float32MultiArray, 52 elementi,
stesso ordine ESATTO del vettore obs nel controllore monolitico),
pubblicato ad ogni tick del timer interno (STEP_DT, 25Hz).
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty, Float32MultiArray
from scipy.spatial.transform import Rotation as R

# ============================================================
# CONFIG — deve rispecchiare params/env.yaml del checkpoint (stessi
# valori usati in position_controller_VICON_VERSION.py / target_handler.py).
# ============================================================
STEP_DT = 0.04                     # 25 Hz
N_WAYPOINTS = 4
WP_PREVIEW_HORIZON = 4
INTEGRAL_TAU_S = 5.0
INTEGRAL_CLAMP = 1.0
INTEGRAL_OBS_SCALE = 0.5

ROOM_MIN = np.array([-2.0, -1.5, 0.1])
ROOM_MAX = np.array([2.0, 1.5, 2.0])

DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),
}

VICON_POSE_TOPIC = "/vicon/tello_42_boosted/tello_42_boosted"
TARGETS_TOPIC = "targets"
POLICY_ACTION_TOPIC = "/tello/policy_action"
LAND_REQUEST_TOPIC = "/tello/land_request"
OBSERVATIONS_TOPIC = "observations"

POSE_TIMEOUT_S = 0.5
POSE_LOST_LAND_TIMEOUT_S = 3.0

MAX_PLAUSIBLE_SPEED_MPS = 5.0
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1

VEL_FILTER_ALPHA = 0.3

ORIGIN_MOCAP = np.array([0.0, 0.0, 0.0])   # <-- calibrare come nel monolitico


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_projected_gravity_b(qx, qy, qz, qw) -> np.ndarray:
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(np.array([0.0, 0.0, -1.0]))


def compute_lin_vel_body(v_world: np.ndarray, qx, qy, qz, qw) -> np.ndarray:
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(v_world)


def euler_rates_to_body_rates(roll_dot, pitch_dot, yaw_dot, roll, pitch) -> np.ndarray:
    """Vedi docstring identico in position_controller_VICON_VERSION.py:
    convenzione R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matrice cinematica
    degli angoli di Eulero (NON una semplice rotazione world->body)."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    wx = roll_dot - yaw_dot * sp
    wy = pitch_dot * cr + yaw_dot * sr * cp
    wz = -pitch_dot * sr + yaw_dot * cr * cp
    return np.array([wx, wy, wz])


class ObservationHandler(Node):
    def __init__(self):
        super().__init__("observation_handler")

        self.declare_parameter("dof_mask_mode", "full")
        dof_mask_mode = self.get_parameter("dof_mask_mode").get_parameter_value().string_value
        if dof_mask_mode not in DOF_MASKS:
            raise ValueError(f"dof_mask_mode='{dof_mask_mode}' non valido, atteso uno tra {list(DOF_MASKS.keys())}")
        self.dof_mask = np.array(DOF_MASKS[dof_mask_mode])

        # -- stato derivato dal Vicon (aggiornato in pose_cb) --
        self.pos_env = np.zeros(3)
        self.yaw = 0.0
        self.lin_vel_b = np.zeros(3)
        self.ang_vel_b = np.zeros(3)
        self.proj_grav_b = np.array([0.0, 0.0, -1.0])
        self.prev_pos_world = None
        self.prev_euler = None
        self.prev_time = None
        self.pose_received = False
        self.last_pose_wall_time = None
        self._mocap_rejected_count = 0
        self._ang_vel_rejected_count = 0
        self._pose_lost_since = None
        self._land_requested_for_pose_loss = False

        # -- ultima azione della policy, usata come prev_action nell'obs --
        self.prev_action = np.zeros(4)

        # -- coda target ricevuta da target_handler --
        self._targets_lock = threading.Lock()
        self.wp_idx = 0
        self._last_wp_idx = None
        self.wp_pos_queue = np.zeros((N_WAYPOINTS, 3))
        self.wp_yaw_queue = np.zeros(N_WAYPOINTS)
        self.targets_received = False

        # -- integrale d'errore leaky, azzerato solo su cambio wp_idx --
        self.err_integral = np.zeros(4)
        self.alpha_leaky = math.exp(-STEP_DT / INTEGRAL_TAU_S)

        # -- ROS I/O --
        self.pose_sub = self.create_subscription(PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10)
        self.targets_sub = self.create_subscription(Float32MultiArray, TARGETS_TOPIC, self.targets_cb, 10)
        self.policy_action_sub = self.create_subscription(
            Twist, POLICY_ACTION_TOPIC, self.policy_action_cb, 10
        )

        self.obs_pub = self.create_publisher(Float32MultiArray, OBSERVATIONS_TOPIC, 10)
        self.land_request_pub = self.create_publisher(Empty, LAND_REQUEST_TOPIC, 10)

        self.timer = self.create_timer(STEP_DT, self.control_loop)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"observation_handler avviato. Vicon topic: {VICON_POSE_TOPIC} | "
            f"dof_mask_mode={dof_mask_mode} | pubblico su '{OBSERVATIONS_TOPIC}'."
        )

    # -------------------- parametri a runtime --------------------
    def _on_param_change(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name == "dof_mask_mode":
                if p.value not in DOF_MASKS:
                    return SetParametersResult(
                        successful=False,
                        reason=f"dof_mask_mode non valido, atteso uno tra {list(DOF_MASKS.keys())}",
                    )
                self.dof_mask = np.array(DOF_MASKS[p.value])
                self.get_logger().info(f"dof_mask_mode cambiato a runtime: '{p.value}'")

        return SetParametersResult(successful=True)

    # -------------------- callback target_handler --------------------
    def targets_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) != 1 + N_WAYPOINTS * 3 + N_WAYPOINTS:
            self.get_logger().warn(f"Messaggio 'targets' con lunghezza inattesa ({len(data)}), scartato.")
            return
        wp_idx = int(round(data[0]))
        wp_pos_queue = np.array(data[1:1 + N_WAYPOINTS * 3]).reshape(N_WAYPOINTS, 3)
        wp_yaw_queue = np.array(data[1 + N_WAYPOINTS * 3:])

        with self._targets_lock:
            if self._last_wp_idx is not None and wp_idx != self._last_wp_idx:
                self.err_integral[:] = 0.0
            self._last_wp_idx = wp_idx
            self.wp_idx = wp_idx
            self.wp_pos_queue = wp_pos_queue
            self.wp_yaw_queue = wp_yaw_queue
            self.targets_received = True

    # -------------------- callback policy_handler --------------------
    def policy_action_cb(self, msg: Twist):
        self.prev_action = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])

    # -------------------- callback Vicon (Tello) --------------------
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_world = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        raw_vals = [p_world[0], p_world[1], p_world[2], q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            self.get_logger().warn("Pose mocap con NaN/Inf scartata.")
            return

        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            self.get_logger().warn(f"Quaternione mocap degenere (norma={quat_norm:.3f}), pose scartata.")
            return

        if self.prev_pos_world is not None and self.prev_time is not None:
            dt_check = t - self.prev_time
            if dt_check > 1e-3:
                implied_speed = float(np.linalg.norm(p_world - self.prev_pos_world)) / dt_check
                if implied_speed > MAX_PLAUSIBLE_SPEED_MPS:
                    self._mocap_rejected_count += 1
                    self.get_logger().warn(
                        f"Jump mocap implausibile ({implied_speed:.2f} m/s stimati), "
                        f"pose scartata (scarti consecutivi: {self._mocap_rejected_count})."
                    )
                    return
        self._mocap_rejected_count = 0

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
                    self._ang_vel_rejected_count = 0
                    self.ang_vel_b = VEL_FILTER_ALPHA * w_body_raw + (1 - VEL_FILTER_ALPHA) * self.ang_vel_b
                else:
                    self._ang_vel_rejected_count += 1
                    self.get_logger().warn(
                        f"Velocita' angolare implausibile dal mocap "
                        f"({np.max(np.abs(w_body_raw)):.1f} rad/s), campione scartato "
                        f"(scarti consecutivi: {self._ang_vel_rejected_count})."
                    )

        self.prev_pos_world = p_world
        self.prev_euler = (roll, pitch, yaw)
        self.prev_time = t
        self.pos_env = p_world - ORIGIN_MOCAP
        self.yaw = yaw
        self.pose_received = True
        self.last_pose_wall_time = time.monotonic()

    # -------------------- loop @25Hz: integrale, obs, publish --------------------
    def control_loop(self):
        if self.last_pose_wall_time is None:
            return  # Vicon mai arrivato: niente da pubblicare, nessuna escalation

        now_mono = time.monotonic()
        pose_age = now_mono - self.last_pose_wall_time
        if pose_age > POSE_TIMEOUT_S:
            if self._pose_lost_since is None:
                self._pose_lost_since = now_mono
                self.get_logger().warn("Pose Vicon scaduta: sospendo la pubblicazione delle osservazioni.")
            elif now_mono - self._pose_lost_since > POSE_LOST_LAND_TIMEOUT_S:
                if not self._land_requested_for_pose_loss:
                    self.get_logger().error(
                        f"Pose Vicon assente da oltre {POSE_LOST_LAND_TIMEOUT_S}s: richiedo atterraggio."
                    )
                    self.land_request_pub.publish(Empty())
                    self._land_requested_for_pose_loss = True
            return
        self._pose_lost_since = None
        self._land_requested_for_pose_loss = False

        if not self.targets_received:
            return  # target_handler non ha ancora pubblicato nessuna coda

        pos_env = self.pos_env
        yaw = self.yaw

        with self._targets_lock:
            wp_idx = self.wp_idx
            wp_pos_queue = self.wp_pos_queue.copy()
            wp_yaw_queue = self.wp_yaw_queue.copy()

        w0_pos = wp_pos_queue[wp_idx]
        w0_yaw = float(wp_yaw_queue[wp_idx])
        pos_err = pos_env - w0_pos
        yaw_err_signed = wrap_to_pi(yaw - w0_yaw)

        self.err_integral[:3] = np.clip(
            self.alpha_leaky * self.err_integral[:3] + pos_err * STEP_DT,
            -INTEGRAL_CLAMP, INTEGRAL_CLAMP,
        )
        self.err_integral[3] = np.clip(
            self.alpha_leaky * self.err_integral[3] + yaw_err_signed * STEP_DT,
            -INTEGRAL_CLAMP, INTEGRAL_CLAMP,
        )

        blocks = []
        e0 = wrap_to_pi(w0_yaw - yaw)
        blocks += [w0_pos - pos_env, np.array([math.sin(e0), math.cos(e0)])]
        for k in range(1, WP_PREVIEW_HORIZON):
            i_c = min(wp_idx + k, N_WAYPOINTS - 1)
            i_p = min(wp_idx + k - 1, N_WAYPOINTS - 1)
            p_c, p_p = wp_pos_queue[i_c], wp_pos_queue[i_p]
            y_c, y_p = wp_yaw_queue[i_c], wp_yaw_queue[i_p]
            dyaw = wrap_to_pi(y_c - y_p)
            blocks += [p_c - p_p, np.array([math.sin(dyaw), math.cos(dyaw)])]
        preview = np.concatenate(blocks)

        clearance = np.concatenate([ROOM_MAX - pos_env, pos_env - ROOM_MIN])
        integral_norm = self.err_integral / INTEGRAL_OBS_SCALE

        obs = np.concatenate([
            pos_env,
            np.array([math.sin(yaw), math.cos(yaw)]),
            self.lin_vel_b,
            self.ang_vel_b,
            self.proj_grav_b,
            self.prev_action,
            preview,
            clearance,
            self.dof_mask,
            integral_norm,
        ]).astype(np.float32)

        assert obs.size == 52, f"obs size={obs.size}, attesa 52"

        msg = Float32MultiArray()
        msg.data = obs.tolist()
        self.obs_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObservationHandler()
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
