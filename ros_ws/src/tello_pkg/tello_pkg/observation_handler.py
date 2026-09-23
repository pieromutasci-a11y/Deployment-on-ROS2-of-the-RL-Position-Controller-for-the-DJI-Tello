#!/usr/bin/env python3
"""Nodo 'observation_handler' (tello_pkg): costruisce il vettore di osservazione (52) dalla posa Vicon.

Sottoscrive la posa Vicon del Tello, la coda 'targets' (da target_handler), /tello/policy_action
(serve come prev_action nell'osservazione) e /tello/flight_state. Nessun accesso al drone.
Da posizione e quaternione ricava velocita' lineare e angolare nel frame corpo (differenze
finite + filtro passa-basso), gravita' proiettata, errore verso il waypoint attivo, preview dei
successivi, distanze dai muri e integrale d'errore leaky (azzerato al cambio di wp_idx).
Scarta i campioni mocap non plausibili (NaN, quaternione degenere, salti).
Watchdog Vicon: posa scaduta (POSE_TIMEOUT_S) -> smette di pubblicare (hover a valle);
persa oltre POSE_LOST_LAND_TIMEOUT_S -> pubblica anche /tello/land_request.
Output: 'observations' (Float32MultiArray, 52 elementi, stesso ordine del training).
A volo attivo registra i campioni; a fine sessione salva CSV e grafici in 'output_dir'.
"""

import csv
import math
import os
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Empty, Float32MultiArray
from scipy.spatial.transform import Rotation as R

import matplotlib
# matplotlib senza display (backend Agg)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parametri dell'ambiente (come params/env.yaml del checkpoint), topic, timeout e soglie di plausibilita' del mocap
STEP_DT = 0.005
N_WAYPOINTS = 4
WP_PREVIEW_HORIZON = 4
INTEGRAL_TAU_S = 5.0
INTEGRAL_CLAMP = 1.0
INTEGRAL_OBS_SCALE = 0.5

ROOM_MIN = np.array([-1.5, -1.0, 0.1])
ROOM_MAX = np.array([1.5, 1.0, 2.0])

# Stessi valori di vel_command_handler: servono solo a loggare la velocita' comandata, vanno tenuti allineati
VEL_REF_SCALE = np.array([1.0, 1.0, 1.0, 1.5])
RC_SCALE_PCT = np.array([0.40, 0.40, 0.40, 0.40])

# Maschere dei gradi di liberta' (feature di osservazione, non maschera sull'azione)
DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),
}

VICON_POSE_TOPIC = "/vicon/Tello_2/Tello_2"
TARGETS_TOPIC = "targets"
POLICY_ACTION_TOPIC = "/tello/policy_action"
LAND_REQUEST_TOPIC = "/tello/land_request"
FLIGHT_STATE_TOPIC = "/tello/flight_state"
OBSERVATIONS_TOPIC = "observations"

POSE_TIMEOUT_S = 0.5
POSE_LOST_LAND_TIMEOUT_S = 3.0

MAX_PLAUSIBLE_SPEED_MPS = 5.0
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1


# Radice di ros_ws: colcon esegue una copia del file da build/install, quindi si risale fino a src/tello_pkg
def _find_ros_ws_root(start_path):
    d = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "src", "tello_pkg")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(start_path, "..", "..", ".."))

# Filtro sulle velocita' e origine del frame mocap
VEL_FILTER_ALPHA = 0.3

ORIGIN_MOCAP = np.array([0.0, 0.0, 0.0])


# Trasformazioni: gravita' proiettata, velocita' lineare e angolare nel frame corpo
def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_projected_gravity_b(qx, qy, qz, qw) -> np.ndarray:
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(np.array([0.0, 0.0, -1.0]))


def compute_lin_vel_body(v_world: np.ndarray, qx, qy, qz, qw) -> np.ndarray:
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(v_world)


def euler_rates_to_body_rates(roll_dot, pitch_dot, yaw_dot, roll, pitch) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    wx = roll_dot - yaw_dot * sp
    wy = pitch_dot * cr + yaw_dot * sr * cp
    wz = -pitch_dot * sr + yaw_dot * cr * cp
    return np.array([wx, wy, wz])


# Nodo: osservazione da posa Vicon, coda target e ultima azione
class ObservationHandler(Node):
    def __init__(self):
        super().__init__("observation_handler")

        ros_ws_dir = _find_ros_ws_root(os.path.dirname(os.path.abspath(__file__)))
        default_output_dir = os.path.join(ros_ws_dir, "Results")
        self.declare_parameter("output_dir", default_output_dir)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("save_plot", True)
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        self.save_csv_flag = self.get_parameter("save_csv").get_parameter_value().bool_value
        self.save_plot_flag = self.get_parameter("save_plot").get_parameter_value().bool_value
        self.start_time = time.time()
        self._history_lock = threading.Lock()
        self.history = []
        self.flight_ready = False

        self.declare_parameter("dof_mask_mode", "full")
        dof_mask_mode = self.get_parameter("dof_mask_mode").get_parameter_value().string_value
        if dof_mask_mode not in DOF_MASKS:
            raise ValueError(f"dof_mask_mode='{dof_mask_mode}' non valido, atteso uno tra {list(DOF_MASKS.keys())}")
        self.dof_mask = np.array(DOF_MASKS[dof_mask_mode])

        self.declare_parameter("target_mode", "variabile")
        self.declare_parameter("advance_mode", "manual")
        self.target_mode = self.get_parameter("target_mode").get_parameter_value().string_value
        self.advance_mode = self.get_parameter("advance_mode").get_parameter_value().string_value

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

        self.prev_action = np.zeros(4)

        self._targets_lock = threading.Lock()
        self.wp_idx = 0
        self._last_wp_idx = None
        self.wp_pos_queue = np.zeros((N_WAYPOINTS, 3))
        self.wp_yaw_queue = np.zeros(N_WAYPOINTS)
        self.targets_received = False

        self.err_integral = np.zeros(4)
        self.alpha_leaky = math.exp(-STEP_DT / INTEGRAL_TAU_S)

        self.pose_sub = self.create_subscription(PoseStamped, VICON_POSE_TOPIC, self.pose_cb, 10)
        self.targets_sub = self.create_subscription(Float32MultiArray, TARGETS_TOPIC, self.targets_cb, 10)
        self.policy_action_sub = self.create_subscription(
            Twist, POLICY_ACTION_TOPIC, self.policy_action_cb, 10
        )
        flight_state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.flight_state_sub = self.create_subscription(
            Bool, FLIGHT_STATE_TOPIC, self.flight_state_cb, flight_state_qos
        )

        self.obs_pub = self.create_publisher(Float32MultiArray, OBSERVATIONS_TOPIC, 10)
        self.land_request_pub = self.create_publisher(Empty, LAND_REQUEST_TOPIC, 10)

        self.timer = self.create_timer(STEP_DT, self.control_loop)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"observation_handler avviato. Vicon topic: {VICON_POSE_TOPIC} | "
            f"dof_mask_mode={dof_mask_mode} | pubblico su '{OBSERVATIONS_TOPIC}'."
        )

    # Parametri a runtime
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
            elif p.name == "target_mode":
                self.target_mode = p.value
            elif p.name == "advance_mode":
                self.advance_mode = p.value

        return SetParametersResult(successful=True)

    # Callback: coda target, azione della policy, flight_state
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

    def policy_action_cb(self, msg: Twist):
        self.prev_action = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])

    def flight_state_cb(self, msg: Bool):
        was_ready = self.flight_ready
        self.flight_ready = msg.data
        if self.flight_ready and not was_ready:
            self.start_time = time.time()
            with self._history_lock:
                self.history = []
            self.get_logger().info("flight_ready=True: avvio registrazione CSV/plot (sincronizzata al decollo).")
        elif was_ready and not self.flight_ready:
            self.save_data_and_plots()
            with self._history_lock:
                self.history = []

    # Callback Vicon: validazione del campione, velocita' per differenze finite filtrate, assetto
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

    # Loop periodico: watchdog Vicon, integrale d'errore, assemblaggio e pubblicazione dell'osservazione
    def control_loop(self):
        if self.last_pose_wall_time is None:
            return

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
            return

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

        # Layout (52): pos(3) | sin,cos yaw(2) | v lineare corpo(3) | w corpo(3) | gravita' proiettata(3)
        # | prev_action(4) | preview(20) | clearance(6) | dof_mask(4) | integrale(4)
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

        # Log CSV/plot solo a volo attivo
        if not self.flight_ready:
            return

        # Velocita' comandata, solo per logging: nessun ulteriore clip (VEL_REF_SCALE porta wz fino a 1.5)
        cmd_vel_ref = np.clip(self.prev_action, -1.0, 1.0) * VEL_REF_SCALE
        cmd_vx = float(cmd_vel_ref[0]) * RC_SCALE_PCT[0]
        cmd_vy = float(cmd_vel_ref[1]) * RC_SCALE_PCT[1]
        cmd_vz = float(cmd_vel_ref[2]) * RC_SCALE_PCT[2]
        cmd_wz = float(cmd_vel_ref[3]) * RC_SCALE_PCT[3]

        with self._history_lock:
            self.history.append({
                "time": time.time() - self.start_time,
                "x": pos_env[0], "y": pos_env[1], "z": pos_env[2],
                "yaw_deg": math.degrees(yaw),
                "wp_idx": wp_idx,
                "target_x": w0_pos[0], "target_y": w0_pos[1], "target_z": w0_pos[2],
                "target_yaw_deg": math.degrees(w0_yaw),
                "lin_vel_b_x": self.lin_vel_b[0],
                "lin_vel_b_y": self.lin_vel_b[1],
                "lin_vel_b_z": self.lin_vel_b[2],
                "ang_vel_b_x": self.ang_vel_b[0],
                "ang_vel_b_y": self.ang_vel_b[1],
                "ang_vel_b_z": self.ang_vel_b[2],
                "cmd_lin_vel_b_x": cmd_vx,
                "cmd_lin_vel_b_y": cmd_vy,
                "cmd_lin_vel_b_z": cmd_vz,
                "cmd_ang_vel_b_z": cmd_wz,
                "pos_err_norm": float(np.linalg.norm(pos_err)),
            })

    # Salvataggio a fine sessione: CSV e griglia di grafici PNG in output_dir
    def save_data_and_plots(self):
        with self._history_lock:
            data = list(self.history)

        if not data:
            self.get_logger().warn("Nessun dato registrato durante la sessione, skip salvataggio.")
            return

        self.get_logger().info("Avvio salvataggio dati e generazione grafici...")
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{self.target_mode}_{self.advance_mode}"

        if self.save_csv_flag:
            csv_path = os.path.join(self.output_dir, f"observation_log_{run_tag}_{timestamp_str}.csv")
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                self.get_logger().info(f"Dati salvati in: {csv_path}")
            except Exception as e:
                self.get_logger().error(f"Errore salvataggio CSV: {e}")

        if self.save_plot_flag:
            plot_png = os.path.join(self.output_dir, f"observation_plots_{run_tag}_{timestamp_str}.png")
            try:
                t = [d["time"] for d in data]
                x = [d["x"] for d in data]
                y = [d["y"] for d in data]
                z = [d["z"] for d in data]
                tx = [d["target_x"] for d in data]
                ty = [d["target_y"] for d in data]
                tz = [d["target_z"] for d in data]

                fig = plt.figure(figsize=(16, 20))
                fig.suptitle(
                    f"Observation Handler — Pose/Target/Velocity Overview ({timestamp_str})",
                    fontsize=16, fontweight="bold",
                )

                ax1 = fig.add_subplot(3, 2, 1, projection="3d")
                ax1.plot(x, y, z, label="Drone path", color="purple")
                ax1.scatter(x[0], y[0], z[0], color="green", s=40, label="Start")
                ax1.scatter(x[-1], y[-1], z[-1], color="red", s=40, label="End")
                ax1.plot(tx, ty, tz, label="Target path", color="orange", linestyle="--", linewidth=1)
                rmin, rmax = ROOM_MIN, ROOM_MAX
                for s, e in (
                    ((rmin[0], rmin[1], rmin[2]), (rmax[0], rmin[1], rmin[2])),
                    ((rmin[0], rmax[1], rmin[2]), (rmax[0], rmax[1], rmin[2])),
                    ((rmin[0], rmin[1], rmax[2]), (rmax[0], rmin[1], rmax[2])),
                    ((rmin[0], rmax[1], rmax[2]), (rmax[0], rmax[1], rmax[2])),
                    ((rmin[0], rmin[1], rmin[2]), (rmin[0], rmax[1], rmin[2])),
                    ((rmax[0], rmin[1], rmin[2]), (rmax[0], rmax[1], rmin[2])),
                    ((rmin[0], rmin[1], rmax[2]), (rmin[0], rmax[1], rmax[2])),
                    ((rmax[0], rmin[1], rmax[2]), (rmax[0], rmax[1], rmax[2])),
                    ((rmin[0], rmin[1], rmin[2]), (rmin[0], rmin[1], rmax[2])),
                    ((rmax[0], rmin[1], rmin[2]), (rmax[0], rmin[1], rmax[2])),
                    ((rmin[0], rmax[1], rmin[2]), (rmin[0], rmax[1], rmax[2])),
                    ((rmax[0], rmax[1], rmin[2]), (rmax[0], rmax[1], rmax[2])),
                ):
                    ax1.plot(*zip(s, e), color="grey", linewidth=0.6, alpha=0.5)
                ax1.set_xlabel("X (m)")
                ax1.set_ylabel("Y (m)")
                ax1.set_zlabel("Z (m)")
                ax1.set_title("3D Trajectory (drone vs target, room bounds)")
                ax1.legend(fontsize=8)

                ax2 = fig.add_subplot(3, 2, 2)
                ax2.plot(x, y, label="Drone XY", color="purple")
                ax2.plot(tx, ty, label="Target XY", color="orange", linestyle="--")
                ax2.scatter(x[0], y[0], color="green", s=40, label="Start")
                ax2.scatter(x[-1], y[-1], color="red", s=40, label="End")
                ax2.add_patch(plt.Rectangle(
                    (rmin[0], rmin[1]), rmax[0] - rmin[0], rmax[1] - rmin[1],
                    fill=False, edgecolor="grey", linestyle=":", label="Room bounds",
                ))
                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Y (m)")
                ax2.set_title("Top-down Trajectory in Room (XY)")
                ax2.set_aspect("equal", adjustable="box")
                ax2.grid(True)
                ax2.legend(fontsize=8)

                ax3 = fig.add_subplot(3, 2, 3)
                ax3.plot(t, x, label="Drone X", color="r")
                ax3.plot(t, y, label="Drone Y", color="g")
                ax3.plot(t, z, label="Drone Z", color="b")
                ax3.plot(t, tx, label="Target X", color="r", linestyle=":")
                ax3.plot(t, ty, label="Target Y", color="g", linestyle=":")
                ax3.plot(t, tz, label="Target Z", color="b", linestyle=":")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Position (m)")
                ax3.set_title("Position: Drone vs Target")
                ax3.grid(True)
                ax3.legend(fontsize=8)

                ax4 = fig.add_subplot(3, 2, 4)
                ax4.plot(t, [d["yaw_deg"] for d in data], label="Drone yaw (deg)", color="b")
                ax4.plot(t, [d["target_yaw_deg"] for d in data], label="Target yaw (deg)", color="b", linestyle=":")
                ax4b = ax4.twinx()
                ax4b.plot(t, [d["pos_err_norm"] for d in data], label="Pos error norm (m)", color="darkred", alpha=0.7)
                ax4.set_xlabel("Time (s)")
                ax4.set_ylabel("Yaw (deg)")
                ax4b.set_ylabel("Pos error norm (m)")
                ax4.set_title("Yaw (drone vs target) & Position Error Norm")
                ax4.grid(True)
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4b.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

                ax5 = fig.add_subplot(3, 2, 5)
                ax5.plot(t, [d["lin_vel_b_x"] for d in data], label="vx misurata (m/s)", color="r")
                ax5.plot(t, [d["lin_vel_b_y"] for d in data], label="vy misurata (m/s)", color="g")
                ax5.plot(t, [d["lin_vel_b_z"] for d in data], label="vz misurata (m/s)", color="b")
                ax5.plot(t, [d["cmd_lin_vel_b_x"] for d in data], label="vx comandata", color="r", linestyle=":")
                ax5.plot(t, [d["cmd_lin_vel_b_y"] for d in data], label="vy comandata", color="g", linestyle=":")
                ax5.plot(t, [d["cmd_lin_vel_b_z"] for d in data], label="vz comandata", color="b", linestyle=":")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Linear vel body (m/s)")
                ax5.set_title("Linear Velocity (body frame): misurata vs comandata")
                ax5.grid(True)
                ax5.legend(fontsize=7)

                ax6 = fig.add_subplot(3, 2, 6)
                ax6.plot(t, [d["ang_vel_b_x"] for d in data], label="wx (rad/s)", color="darkred")
                ax6.plot(t, [d["ang_vel_b_y"] for d in data], label="wy (rad/s)", color="darkgreen")
                ax6.plot(t, [d["ang_vel_b_z"] for d in data], label="wz misurata (rad/s)", color="darkblue")
                ax6.plot(t, [d["cmd_ang_vel_b_z"] for d in data], label="wz comandata", color="darkblue", linestyle=":")
                ax6.set_xlabel("Time (s)")
                ax6.set_ylabel("Angular vel body (rad/s)")
                ax6.set_title("Angular Velocity (body frame): misurata vs comandata")
                ax6.grid(True)
                ax6.legend(fontsize=7)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(plot_png, dpi=200)
                plt.close(fig)
                self.get_logger().info(f"Grafici salvati in: {plot_png}")
            except Exception as e:
                self.get_logger().error(f"Errore durante il plot dei grafici: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObservationHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_data_and_plots()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
