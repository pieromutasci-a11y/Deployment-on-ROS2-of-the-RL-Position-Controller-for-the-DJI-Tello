#!/usr/bin/env python3
"""
Nodo ROS2 'read_sensors' (package tello_node): NESSUN comando al drone,
NESSUNA policy caricata — legge e stampa a terminale, periodicamente,
esattamente le stesse sorgenti sensoriali usate da
position_controller_VICON_VERSION.py, per verificarle PRIMA di lanciare
il controllore vero:

  - Vicon (topic --ros-args -p vicon_pose_topic:=...): posizione,
    quaternione, yaw derivato, proj_grav_b e lin_vel_b calcolati con la
    STESSA logica esatta del controllore (rotazione completa dal
    quaternione, filtro passa-basso alpha=0.3).
  - tellopy EVENT_LOG_DATA: IMU (gyro_x/y/z), MVO (vel_x/y/z).
  - tellopy EVENT_FLIGHT_DATA: batteria, quota (height), fly_mode.

Si connette al drone via tellopy SOLO per la telemetria (connect(), MAI
takeoff/land/comandi di movimento): sicuro da lanciare con il drone a
terra per controllare che tutte le fonti dati arrivino con valori
plausibili prima di fidarsi del controllore.
"""

import math
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

import tellopy

DEFAULT_VICON_POSE_TOPIC = "/vicon/tello/pose"
STATUS_PRINT_PERIOD_S = 0.5
TELLOPY_CONNECT_TIMEOUT_S = 60.0

# -- sanity check mocap, stessa soglia del controllore --
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1


def compute_projected_gravity_b(qx, qy, qz, qw):
    """Identica a compute_projected_gravity_b() in
    position_controller_VICON_VERSION.py."""
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(np.array([0.0, 0.0, -1.0]))


def compute_lin_vel_body(v_world: np.ndarray, qx, qy, qz, qw) -> np.ndarray:
    """Identica a compute_lin_vel_body() in
    position_controller_VICON_VERSION.py."""
    rot_world_to_body = R.from_quat([qx, qy, qz, qw]).inv()
    return rot_world_to_body.apply(v_world)


class SensorReader(Node):
    def __init__(self):
        super().__init__("read_sensors")

        self.declare_parameter("vicon_pose_topic", DEFAULT_VICON_POSE_TOPIC)
        vicon_pose_topic = self.get_parameter("vicon_pose_topic").get_parameter_value().string_value

        # -- stato Vicon, protetto da _vicon_lock (pose_cb gira sul thread
        # executor rclpy, status_cb sullo stesso thread ma via timer: lock
        # comunque presente per coerenza/robustezza futura) --
        self._vicon_lock = threading.Lock()
        self.pos = None
        self.yaw = None
        self.proj_grav_b = None
        self.lin_vel_b = np.zeros(3)
        self.prev_pos = None
        self.prev_time = None
        self.pose_count = 0
        self.pose_rejected_count = 0
        self.last_pose_wall_time = None

        # -- stato tellopy, protetto da _tello_lock (i callback EVENT_*
        # girano sul thread interno di tellopy, non sul thread rclpy) --
        self._tello_lock = threading.Lock()
        self.gyro = None
        self.mvo_vel = None
        self.battery_pct = None
        self.height_dm = None
        self.fly_mode = None
        self.imu_count = 0
        self.flight_data_count = 0
        self.last_imu_wall_time = None
        self.last_flight_wall_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped, vicon_pose_topic, self.pose_cb, 10
        )

        self.drone = tellopy.Tello()
        self.drone.subscribe(self.drone.EVENT_LOG_DATA, self.tello_log_data_cb)
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.tello_flight_data_cb)

        self.get_logger().info(
            "[tellopy] connessione al drone (SOLO telemetria, NESSUN comando di volo)..."
        )
        try:
            self.drone.connect()
            self.drone.wait_for_connection(TELLOPY_CONNECT_TIMEOUT_S)
            self.get_logger().info("[tellopy] connesso.")
        except Exception as e:
            self.get_logger().error(
                f"[tellopy] connessione fallita: {e} — continuo a stampare solo i dati Vicon "
                "(se disponibili)."
            )

        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)

        self.get_logger().info(
            f"Nodo read_sensors avviato. Vicon topic: {vicon_pose_topic} | "
            f"stampa ogni {STATUS_PRINT_PERIOD_S}s. Ctrl+C per uscire."
        )

    # -------------------- callback Vicon --------------------
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        raw_vals = [p[0], p[1], p[2], q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            self.pose_rejected_count += 1
            self.get_logger().warn("[vicon] Pose con NaN/Inf scartata.")
            return

        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            self.pose_rejected_count += 1
            self.get_logger().warn(
                f"[vicon] Quaternione degenere (norma={quat_norm:.3f}), pose scartata."
            )
            return

        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        proj_grav_b = compute_projected_gravity_b(q.x, q.y, q.z, q.w)

        with self._vicon_lock:
            if self.prev_pos is not None and self.prev_time is not None:
                dt = max(t - self.prev_time, 1e-3)
                v_world = (p - self.prev_pos) / dt
                v_body_raw = compute_lin_vel_body(v_world, q.x, q.y, q.z, q.w)
                alpha_f = 0.3
                self.lin_vel_b = alpha_f * v_body_raw + (1 - alpha_f) * self.lin_vel_b
            self.prev_pos = p
            self.prev_time = t
            self.pos = p
            self.yaw = yaw
            self.proj_grav_b = proj_grav_b
            self.pose_count += 1
            self.last_pose_wall_time = time.monotonic()

    # -------------------- callback tellopy --------------------
    def tello_log_data_cb(self, event, sender, data, **kwargs):
        imu = data.imu
        mvo = data.mvo
        with self._tello_lock:
            self.gyro = (imu.gyro_x, imu.gyro_y, imu.gyro_z)
            self.mvo_vel = (mvo.vel_x, mvo.vel_y, mvo.vel_z)
            self.imu_count += 1
            self.last_imu_wall_time = time.monotonic()

    def tello_flight_data_cb(self, event, sender, data, **kwargs):
        with self._tello_lock:
            self.battery_pct = getattr(data, "battery_percentage", None)
            self.height_dm = getattr(data, "height", None)
            self.fly_mode = getattr(data, "fly_mode", None)
            self.flight_data_count += 1
            self.last_flight_wall_time = time.monotonic()

    # -------------------- stampa periodica --------------------
    def status_cb(self):
        now = time.monotonic()

        with self._vicon_lock:
            pos, yaw = self.pos, self.yaw
            proj_grav_b = self.proj_grav_b
            lin_vel_b = self.lin_vel_b.copy()
            pose_count, pose_rejected = self.pose_count, self.pose_rejected_count
            pose_age = (now - self.last_pose_wall_time) if self.last_pose_wall_time else None

        with self._tello_lock:
            gyro, mvo_vel = self.gyro, self.mvo_vel
            bat, height_dm, fly_mode = self.battery_pct, self.height_dm, self.fly_mode
            imu_age = (now - self.last_imu_wall_time) if self.last_imu_wall_time else None
            flight_age = (now - self.last_flight_wall_time) if self.last_flight_wall_time else None

        if pos is not None:
            self.get_logger().info(
                f"[vicon] pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m | "
                f"yaw={math.degrees(yaw):+.1f}deg | "
                f"proj_grav_b=({proj_grav_b[0]:+.2f},{proj_grav_b[1]:+.2f},{proj_grav_b[2]:+.2f}) | "
                f"lin_vel_b=({lin_vel_b[0]:+.2f},{lin_vel_b[1]:+.2f},{lin_vel_b[2]:+.2f})m/s | "
                f"n_msg={pose_count} scartati={pose_rejected} | age={pose_age:.2f}s"
            )
        else:
            self.get_logger().warn("[vicon] NESSUN messaggio ricevuto ancora.")

        if gyro is not None:
            self.get_logger().info(
                f"[tellopy][imu] gyro=({gyro[0]:+.3f},{gyro[1]:+.3f},{gyro[2]:+.3f}) rad/s (presunto) | "
                f"mvo_vel=({mvo_vel[0]:+.3f},{mvo_vel[1]:+.3f},{mvo_vel[2]:+.3f}) | "
                f"n_msg={self.imu_count} | age={imu_age:.2f}s"
            )
        else:
            self.get_logger().warn("[tellopy][imu] Nessun EVENT_LOG_DATA ricevuto ancora.")

        if bat is not None:
            self.get_logger().info(
                f"[tellopy][flight] batteria={bat}% | height={height_dm} (decimetri, presunto) | "
                f"fly_mode={fly_mode} | n_msg={self.flight_data_count} | age={flight_age:.2f}s"
            )
        else:
            self.get_logger().warn("[tellopy][flight] Nessun EVENT_FLIGHT_DATA ricevuto ancora.")

        self.get_logger().info("-" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = SensorReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruzione richiesta (Ctrl+C): chiudo.")
    finally:
        try:
            node.drone.quit()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
