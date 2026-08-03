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

REGISTRAZIONE E PLOT DATI:
I dati ricevuti durante l'esecuzione vengono salvati in memoria e, alla
chiusura del nodo (Ctrl+C o shutdown), vengono automaticamente:
  1. Salvati in file CSV (nella cartella specificata dal parametro output_dir).
  2. Graficati e salvati in formato PNG ad alta risoluzione.

Si connette al drone via tellopy SOLO per la telemetria (connect(), MAI
takeoff/land/comandi di movimento): sicuro da lanciare con il drone a
terra per controllare che tutte le fonti dati arrivino con valori
plausibili prima di fidarsi del controllore.
"""

import csv
import math
import os
import sys
import time
import threading
from datetime import datetime
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per salvataggio figure senza server X
import matplotlib.pyplot as plt

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

        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.declare_parameter("vicon_pose_topic", DEFAULT_VICON_POSE_TOPIC)
        self.declare_parameter("output_dir", script_dir)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("save_plot", True)

        vicon_pose_topic = self.get_parameter("vicon_pose_topic").get_parameter_value().string_value
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        self.save_csv_flag = self.get_parameter("save_csv").get_parameter_value().bool_value
        self.save_plot_flag = self.get_parameter("save_plot").get_parameter_value().bool_value

        self.start_time = time.time()

        # -- registri dati per salvataggio e plot --
        self.vicon_history = []
        self.imu_history = []
        self.flight_history = []

        # -- stato Vicon, protetto da _vicon_lock --
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

        # -- stato tellopy, protetto da _tello_lock --
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

            t_rel = time.time() - self.start_time
            self.vicon_history.append({
                "time": t_rel,
                "x": p[0], "y": p[1], "z": p[2],
                "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
                "yaw_deg": math.degrees(yaw),
                "proj_grav_x": proj_grav_b[0],
                "proj_grav_y": proj_grav_b[1],
                "proj_grav_z": proj_grav_b[2],
                "lin_vel_b_x": self.lin_vel_b[0],
                "lin_vel_b_y": self.lin_vel_b[1],
                "lin_vel_b_z": self.lin_vel_b[2]
            })

    # -------------------- callback tellopy --------------------
    def tello_log_data_cb(self, event, sender, data, **kwargs):
        imu = data.imu
        mvo = data.mvo
        with self._tello_lock:
            self.gyro = (imu.gyro_x, imu.gyro_y, imu.gyro_z)
            self.mvo_vel = (mvo.vel_x, mvo.vel_y, mvo.vel_z)
            self.imu_count += 1
            self.last_imu_wall_time = time.monotonic()

            t_rel = time.time() - self.start_time
            self.imu_history.append({
                "time": t_rel,
                "gyro_x": imu.gyro_x, "gyro_y": imu.gyro_y, "gyro_z": imu.gyro_z,
                "mvo_vel_x": mvo.vel_x, "mvo_vel_y": mvo.vel_y, "mvo_vel_z": mvo.vel_z
            })

    def tello_flight_data_cb(self, event, sender, data, **kwargs):
        with self._tello_lock:
            bat = getattr(data, "battery_percentage", None)
            h_dm = getattr(data, "height", None)
            fm = getattr(data, "fly_mode", None)
            self.battery_pct = bat
            self.height_dm = h_dm
            self.fly_mode = fm
            self.flight_data_count += 1
            self.last_flight_wall_time = time.monotonic()

            t_rel = time.time() - self.start_time
            self.flight_history.append({
                "time": t_rel,
                "battery": bat,
                "height_dm": h_dm,
                "fly_mode": fm
            })

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

    # -------------------- salvataggio CSV e grafico PLOT --------------------
    def save_data_and_plots(self):
        self.get_logger().info("Avvio salvataggio dati e generazione grafici...")
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        with self._vicon_lock:
            vicon_data = list(self.vicon_history)
        with self._tello_lock:
            imu_data = list(self.imu_history)
            flight_data = list(self.flight_history)

        total_samples = len(vicon_data) + len(imu_data) + len(flight_data)
        if total_samples == 0:
            self.get_logger().warn("Nessun dato registrato durante la sessione, skip salvataggio.")
            return

        # 1. Salvataggio CSV
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

            if imu_data:
                imu_csv = os.path.join(self.output_dir, f"sensor_tello_imu_{timestamp_str}.csv")
                try:
                    with open(imu_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=imu_data[0].keys())
                        writer.writeheader()
                        writer.writerows(imu_data)
                    self.get_logger().info(f"Dati Tello IMU salvati in: {imu_csv}")
                except Exception as e:
                    self.get_logger().error(f"Errore salvataggio Tello IMU CSV: {e}")

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

        # 2. Generazione e salvataggio Grafici PLOT
        if self.save_plot_flag:
            plot_png = os.path.join(self.output_dir, f"sensor_plots_{timestamp_str}.png")
            try:
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle(f"Sensor Data Overview ({timestamp_str})", fontsize=16, fontweight='bold')

                # Subplot 1: Vicon Position (X, Y, Z) vs Time
                ax1 = fig.add_subplot(3, 2, 1)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax1.plot(t_vicon, [d["x"] for d in vicon_data], label="Pos X (m)", color="r")
                    ax1.plot(t_vicon, [d["y"] for d in vicon_data], label="Pos Y (m)", color="g")
                    ax1.plot(t_vicon, [d["z"] for d in vicon_data], label="Pos Z (m)", color="b")
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Position (m)")
                    ax1.set_title("Vicon Position")
                    ax1.grid(True)
                    ax1.legend()
                else:
                    ax1.set_title("Vicon Position (No Data)")

                # Subplot 2: Traiettoria 3D Vicon
                ax2 = fig.add_subplot(3, 2, 2, projection='3d')
                if vicon_data:
                    x_v = [d["x"] for d in vicon_data]
                    y_v = [d["y"] for d in vicon_data]
                    z_v = [d["z"] for d in vicon_data]
                    ax2.plot(x_v, y_v, z_v, label="3D Path", color="purple")
                    ax2.scatter(x_v[0], y_v[0], z_v[0], color="green", s=40, label="Start")
                    ax2.scatter(x_v[-1], y_v[-1], z_v[-1], color="red", s=40, label="End")
                    ax2.set_xlabel("X (m)")
                    ax2.set_ylabel("Y (m)")
                    ax2.set_zlabel("Z (m)")
                    ax2.set_title("Vicon 3D Trajectory")
                    ax2.legend()
                else:
                    ax2.set_title("Vicon 3D Trajectory (No Data)")

                # Subplot 3: Vicon Yaw e Gravita' Proiettata (Body)
                ax3 = fig.add_subplot(3, 2, 3)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax3.plot(t_vicon, [d["yaw_deg"] for d in vicon_data], label="Yaw (deg)", color="black")
                    ax3.plot(t_vicon, [d["proj_grav_x"] for d in vicon_data], label="Proj Grav X", color="m", linestyle="--")
                    ax3.plot(t_vicon, [d["proj_grav_y"] for d in vicon_data], label="Proj Grav Y", color="c", linestyle="--")
                    ax3.plot(t_vicon, [d["proj_grav_z"] for d in vicon_data], label="Proj Grav Z", color="y", linestyle="--")
                    ax3.set_xlabel("Time (s)")
                    ax3.set_ylabel("Yaw / Grav Vector")
                    ax3.set_title("Vicon Yaw & Projected Gravity Body")
                    ax3.grid(True)
                    ax3.legend()
                else:
                    ax3.set_title("Vicon Yaw & Grav (No Data)")

                # Subplot 4: Velocita' Lineare (Vicon Body vs Tello MVO)
                ax4 = fig.add_subplot(3, 2, 4)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax4.plot(t_vicon, [d["lin_vel_b_x"] for d in vicon_data], label="Vicon LinVel X (m/s)", color="r")
                    ax4.plot(t_vicon, [d["lin_vel_b_y"] for d in vicon_data], label="Vicon LinVel Y (m/s)", color="g")
                    ax4.plot(t_vicon, [d["lin_vel_b_z"] for d in vicon_data], label="Vicon LinVel Z (m/s)", color="b")
                if imu_data:
                    t_imu = [d["time"] for d in imu_data]
                    ax4.plot(t_imu, [d["mvo_vel_x"] for d in imu_data], label="MVO Vel X", color="r", linestyle=":")
                    ax4.plot(t_imu, [d["mvo_vel_y"] for d in imu_data], label="MVO Vel Y", color="g", linestyle=":")
                    ax4.plot(t_imu, [d["mvo_vel_z"] for d in imu_data], label="MVO Vel Z", color="b", linestyle=":")
                ax4.set_xlabel("Time (s)")
                ax4.set_ylabel("Vel (m/s)")
                ax4.set_title("Linear Velocity (Vicon vs Tello MVO)")
                ax4.grid(True)
                ax4.legend()

                # Subplot 5: Giroscopio Tellopy IMU
                ax5 = fig.add_subplot(3, 2, 5)
                if imu_data:
                    t_imu = [d["time"] for d in imu_data]
                    ax5.plot(t_imu, [d["gyro_x"] for d in imu_data], label="Gyro X", color="darkred")
                    ax5.plot(t_imu, [d["gyro_y"] for d in imu_data], label="Gyro Y", color="darkgreen")
                    ax5.plot(t_imu, [d["gyro_z"] for d in imu_data], label="Gyro Z", color="darkblue")
                    ax5.set_xlabel("Time (s)")
                    ax5.set_ylabel("Gyro (rad/s)")
                    ax5.set_title("Tellopy IMU Gyroscope")
                    ax5.grid(True)
                    ax5.legend()
                else:
                    ax5.set_title("Tellopy IMU Gyroscope (No Data)")

                # Subplot 6: Batteria e Quota Tello Flight Data
                ax6 = fig.add_subplot(3, 2, 6)
                if flight_data:
                    t_fl = [d["time"] for d in flight_data]
                    bat_vals = [d["battery"] for d in flight_data if d["battery"] is not None]
                    h_vals = [d["height_dm"] / 10.0 if d["height_dm"] is not None else None for d in flight_data]
                    
                    if any(b is not None for b in bat_vals):
                        ax6.plot(t_fl, [d["battery"] for d in flight_data], label="Battery (%)", color="orange")
                    if any(h is not None for h in h_vals):
                        ax6.plot(t_fl, h_vals, label="Height (m)", color="teal")
                    ax6.set_xlabel("Time (s)")
                    ax6.set_ylabel("Value")
                    ax6.set_title("Tello Flight Data (Battery & Height)")
                    ax6.grid(True)
                    ax6.legend()
                else:
                    ax6.set_title("Tello Flight Data (No Data)")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(plot_png, dpi=200)
                plt.close(fig)
                self.get_logger().info(f"Grafici dei sensori salvati in: {plot_png}")
            except Exception as e:
                self.get_logger().error(f"Errore durante il plot dei grafici: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SensorReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruzione richiesta (Ctrl+C): chiudo.")
    finally:
        node.save_data_and_plots()
        try:
            node.drone.quit()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

