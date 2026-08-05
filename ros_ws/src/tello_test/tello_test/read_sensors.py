#!/usr/bin/env python3
"""
Nodo ROS2 'read_sensors' (package tello_test): NESSUN comando al drone,
NESSUNA policy caricata — legge e stampa a terminale, periodicamente,
esattamente le stesse sorgenti sensoriali usate da
position_controller_VICON_VERSION.py, per verificarle PRIMA di lanciare
il controllore vero:

  - Vicon (topic --ros-args -p vicon_pose_topic:=...): posizione,
    quaternione, yaw derivato, proj_grav_b, lin_vel_b E ang_vel_b,
    calcolati con la STESSA logica esatta del controllore (rotazione
    completa dal quaternione, derivate di Eulero -> velocita' angolari
    nel corpo, filtro passa-basso VEL_FILTER_ALPHA).
  - djitellopy state (broadcast SDK ufficiale): batteria ('bat'), quota
    ('h', cm), distanza ToF ('tof', cm), velocita' riportate dal drone
    ('vgx/vgy/vgz', unita' SDK grezze).

LIBRERIA: solo djitellopy. tellopy e' stata RIMOSSA da tutto il progetto:
le due librerie non convivono sullo stesso drone (appena djitellopy entra
in modalita' SDK con 'command', il firmware smette di alimentare il flusso
di log binario "app" da cui tellopy leggeva il giroscopio — osservato
sperimentalmente proprio con questo nodo). Di conseguenza le velocita'
angolari NON vengono piu' da un giroscopio ma sono derivate dal Vicon,
esattamente come fa ora il controllore.

REGISTRAZIONE E PLOT DATI:
I dati ricevuti durante l'esecuzione vengono salvati in memoria e, alla
chiusura del nodo (Ctrl+C o shutdown), vengono automaticamente:
  1. Salvati in file CSV (nella cartella specificata dal parametro output_dir).
  2. Graficati e salvati in formato PNG ad alta risoluzione (griglia 4x2):
       1) posizione Vicon x/y/z          2) traiettoria 3D
       3) angoli di Eulero roll/pitch/yaw (la sorgente di ang_vel_b)
       4) gravita' proiettata nel corpo  5) velocita' lineari (Vicon vs SDK)
       6) ang_vel_b wx/wy/wz             7) istogramma di ang_vel_b
       8) batteria + quota (SDK vs Vicon)
     I pannelli 3-6-7 servono a giudicare la qualita' delle velocita'
     angolari derivate: a drone FERMO l'istogramma (7) deve essere una
     campana stretta attorno a 0; se e' largo, il rumore del mocap sta
     passando in osservazione e va alzato il filtro VEL_FILTER_ALPHA.

Si connette al drone via djitellopy SOLO per la telemetria (connect(),
MAI takeoff/land/comandi di movimento): sicuro da lanciare con il drone
a terra per controllare che tutte le fonti dati arrivino con valori
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

from djitellopy import Tello as DJITello

DEFAULT_VICON_POSE_TOPIC = "/vicon/tello_42_boosted/tello_42_boosted"
STATUS_PRINT_PERIOD_S = 0.5
TELEMETRY_POLL_PERIOD_S = 0.1   # frequenza di campionamento dello state djitellopy per il log

# -- sanity check mocap, stesse soglie del controllore --
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0

# -- filtro passa-basso sulle velocita' derivate dal mocap, stesso del controllore --
VEL_FILTER_ALPHA = 0.3


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


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


def euler_rates_to_body_rates(roll_dot, pitch_dot, yaw_dot, roll, pitch) -> np.ndarray:
    """Identica a euler_rates_to_body_rates() in
    position_controller_VICON_VERSION.py: converte le derivate degli
    angoli di Eulero in velocita' angolare nel frame CORPO tramite la
    matrice cinematica T (NON una semplice rotazione R^-1).
    Convenzione R = Rz(yaw) @ Ry(pitch) @ Rx(roll), come as_euler("xyz")."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    wx = roll_dot - yaw_dot * sp
    wy = pitch_dot * cr + yaw_dot * sr * cp
    wz = -pitch_dot * sr + yaw_dot * cr * cp
    return np.array([wx, wy, wz])


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
        self.flight_history = []

        # -- stato Vicon, protetto da _vicon_lock --
        self._vicon_lock = threading.Lock()
        self.pos = None
        self.yaw = None
        self.proj_grav_b = None
        self.lin_vel_b = np.zeros(3)
        self.ang_vel_b = np.zeros(3)
        self.prev_pos = None
        self.prev_euler = None
        self.prev_time = None
        self.pose_count = 0
        self.pose_rejected_count = 0
        self.ang_vel_rejected_count = 0
        self.last_pose_wall_time = None

        # -- stato telemetria drone (state djitellopy), protetto da _tello_lock --
        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.height_cm = None
        self.tof_cm = None
        self.vg_raw = None          # (vgx, vgy, vgz) — unita' SDK grezze, NON verificate
        self.flight_data_count = 0
        self.last_flight_wall_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped, vicon_pose_topic, self.pose_cb, 10
        )

        self.drone = DJITello()

        self.get_logger().info(
            "[djitellopy] connessione al drone (SOLO telemetria, NESSUN comando di volo)..."
        )
        try:
            self.drone.connect()
            self.get_logger().info("[djitellopy] connesso.")
        except Exception as e:
            self.get_logger().error(
                f"[djitellopy] connessione fallita: {e} — continuo a stampare solo i dati Vicon "
                "(se disponibili)."
            )

        self.telemetry_timer = self.create_timer(TELEMETRY_POLL_PERIOD_S, self.poll_telemetry_cb)
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

        euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
        proj_grav_b = compute_projected_gravity_b(q.x, q.y, q.z, q.w)

        with self._vicon_lock:
            if self.prev_pos is not None and self.prev_time is not None:
                dt = max(t - self.prev_time, 1e-3)

                # -- velocita' LINEARI: differenze finite mondo -> corpo --
                v_world = (p - self.prev_pos) / dt
                v_body_raw = compute_lin_vel_body(v_world, q.x, q.y, q.z, q.w)
                self.lin_vel_b = VEL_FILTER_ALPHA * v_body_raw \
                                  + (1 - VEL_FILTER_ALPHA) * self.lin_vel_b

                # -- velocita' ANGOLARI: derivate di Eulero (con wrap_to_pi
                # obbligatorio sul salto +pi/-pi) -> frame corpo via matrice
                # cinematica. Stessa identica logica del controllore. --
                if self.prev_euler is not None:
                    roll_dot = wrap_to_pi(roll - self.prev_euler[0]) / dt
                    pitch_dot = wrap_to_pi(pitch - self.prev_euler[1]) / dt
                    yaw_dot = wrap_to_pi(yaw - self.prev_euler[2]) / dt
                    w_body_raw = euler_rates_to_body_rates(
                        roll_dot, pitch_dot, yaw_dot, roll, pitch
                    )
                    if np.all(np.isfinite(w_body_raw)) and \
                            float(np.max(np.abs(w_body_raw))) < MAX_PLAUSIBLE_ANG_SPEED_RADPS:
                        self.ang_vel_b = VEL_FILTER_ALPHA * w_body_raw \
                                          + (1 - VEL_FILTER_ALPHA) * self.ang_vel_b
                    else:
                        self.ang_vel_rejected_count += 1

            self.prev_pos = p
            self.prev_euler = (roll, pitch, yaw)
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
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "yaw_deg": math.degrees(yaw),
                "proj_grav_x": proj_grav_b[0],
                "proj_grav_y": proj_grav_b[1],
                "proj_grav_z": proj_grav_b[2],
                "lin_vel_b_x": self.lin_vel_b[0],
                "lin_vel_b_y": self.lin_vel_b[1],
                "lin_vel_b_z": self.lin_vel_b[2],
                "ang_vel_b_x": self.ang_vel_b[0],
                "ang_vel_b_y": self.ang_vel_b[1],
                "ang_vel_b_z": self.ang_vel_b[2],
            })

    # -------------------- polling telemetria djitellopy --------------------
    def poll_telemetry_cb(self):
        """Campiona lo state broadcast SDK (dict aggiornato in background
        da djitellopy). Sostituisce le vecchie callback EVENT_LOG_DATA /
        EVENT_FLIGHT_DATA di tellopy."""
        try:
            state = self.drone.get_current_state()
        except Exception:
            state = {}
        if not state:
            return

        bat = state.get("bat")
        h_cm = state.get("h")
        tof_cm = state.get("tof")
        vg = (state.get("vgx"), state.get("vgy"), state.get("vgz"))

        with self._tello_lock:
            self.battery_pct = bat
            self.height_cm = h_cm
            self.tof_cm = tof_cm
            self.vg_raw = vg
            self.flight_data_count += 1
            self.last_flight_wall_time = time.monotonic()

            t_rel = time.time() - self.start_time
            self.flight_history.append({
                "time": t_rel,
                "battery": bat,
                "height_cm": h_cm,
                "tof_cm": tof_cm,
                "vgx": vg[0], "vgy": vg[1], "vgz": vg[2],
            })

    # -------------------- stampa periodica --------------------
    def status_cb(self):
        now = time.monotonic()

        with self._vicon_lock:
            pos, yaw = self.pos, self.yaw
            proj_grav_b = self.proj_grav_b
            lin_vel_b = self.lin_vel_b.copy()
            ang_vel_b = self.ang_vel_b.copy()
            pose_count, pose_rejected = self.pose_count, self.pose_rejected_count
            ang_rejected = self.ang_vel_rejected_count
            pose_age = (now - self.last_pose_wall_time) if self.last_pose_wall_time else None

        with self._tello_lock:
            bat, height_cm, tof_cm = self.battery_pct, self.height_cm, self.tof_cm
            vg = self.vg_raw
            flight_age = (now - self.last_flight_wall_time) if self.last_flight_wall_time else None

        if pos is not None:
            self.get_logger().info(
                f"[vicon] pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m | "
                f"yaw={math.degrees(yaw):+.1f}deg | "
                f"proj_grav_b=({proj_grav_b[0]:+.2f},{proj_grav_b[1]:+.2f},{proj_grav_b[2]:+.2f}) | "
                f"lin_vel_b=({lin_vel_b[0]:+.2f},{lin_vel_b[1]:+.2f},{lin_vel_b[2]:+.2f})m/s | "
                f"n_msg={pose_count} scartati={pose_rejected} | age={pose_age:.2f}s"
            )
            self.get_logger().info(
                f"[vicon][ang] ang_vel_b=({ang_vel_b[0]:+.3f},{ang_vel_b[1]:+.3f},"
                f"{ang_vel_b[2]:+.3f}) rad/s (derivata dal quaternione) | "
                f"scartati={ang_rejected}"
            )
        else:
            self.get_logger().warn("[vicon] NESSUN messaggio ricevuto ancora.")

        if bat is not None:
            vg_str = (
                f"({vg[0]},{vg[1]},{vg[2]})" if vg and all(v is not None for v in vg) else "n/d"
            )
            self.get_logger().info(
                f"[djitellopy][state] batteria={bat}% | height={height_cm}cm | "
                f"tof={tof_cm}cm | vg_raw={vg_str} (unita' SDK non verificate) | "
                f"n_msg={self.flight_data_count} | age={flight_age:.2f}s"
            )
        else:
            self.get_logger().warn("[djitellopy][state] Nessuno state ricevuto ancora.")

        self.get_logger().info("-" * 70)

    # -------------------- salvataggio CSV e grafico PLOT --------------------
    def save_data_and_plots(self):
        self.get_logger().info("Avvio salvataggio dati e generazione grafici...")
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        with self._vicon_lock:
            vicon_data = list(self.vicon_history)
        with self._tello_lock:
            flight_data = list(self.flight_history)

        total_samples = len(vicon_data) + len(flight_data)
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
                fig = plt.figure(figsize=(16, 18))
                fig.suptitle(
                    f"Sensor Data Overview ({timestamp_str}) — ang_vel_b derivata dal Vicon",
                    fontsize=16, fontweight='bold'
                )

                # Subplot 1: Vicon Position (X, Y, Z) vs Time
                ax1 = fig.add_subplot(4, 2, 1)
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
                ax2 = fig.add_subplot(4, 2, 2, projection='3d')
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

                # Subplot 3: Angoli di Eulero (le grandezze DERIVATE per ottenere ang_vel_b)
                ax3 = fig.add_subplot(4, 2, 3)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax3.plot(t_vicon, [d["roll_deg"] for d in vicon_data], label="Roll (deg)", color="r")
                    ax3.plot(t_vicon, [d["pitch_deg"] for d in vicon_data], label="Pitch (deg)", color="g")
                    ax3.plot(t_vicon, [d["yaw_deg"] for d in vicon_data], label="Yaw (deg)", color="b")
                    ax3.set_xlabel("Time (s)")
                    ax3.set_ylabel("Angle (deg)")
                    ax3.set_title("Vicon Euler Angles (source of ang_vel_b)")
                    ax3.grid(True)
                    ax3.legend()
                else:
                    ax3.set_title("Vicon Euler Angles (No Data)")

                # Subplot 4: Gravita' proiettata nel corpo
                ax4 = fig.add_subplot(4, 2, 4)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax4.plot(t_vicon, [d["proj_grav_x"] for d in vicon_data], label="Proj Grav X", color="m")
                    ax4.plot(t_vicon, [d["proj_grav_y"] for d in vicon_data], label="Proj Grav Y", color="c")
                    ax4.plot(t_vicon, [d["proj_grav_z"] for d in vicon_data], label="Proj Grav Z", color="y")
                    ax4.axhline(-1.0, color="grey", linestyle="--", linewidth=0.8,
                                label="atteso Z=-1 (drone livellato)")
                    ax4.set_xlabel("Time (s)")
                    ax4.set_ylabel("Gravity (body frame)")
                    ax4.set_title("Projected Gravity Body")
                    ax4.grid(True)
                    ax4.legend(fontsize=8)
                else:
                    ax4.set_title("Projected Gravity Body (No Data)")

                # Subplot 5: Velocita' Lineare corpo (Vicon) vs vg riportate dal drone
                ax5 = fig.add_subplot(4, 2, 5)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax5.plot(t_vicon, [d["lin_vel_b_x"] for d in vicon_data], label="Vicon LinVel X (m/s)", color="r")
                    ax5.plot(t_vicon, [d["lin_vel_b_y"] for d in vicon_data], label="Vicon LinVel Y (m/s)", color="g")
                    ax5.plot(t_vicon, [d["lin_vel_b_z"] for d in vicon_data], label="Vicon LinVel Z (m/s)", color="b")
                if flight_data and any(d.get("vgx") is not None for d in flight_data):
                    t_fl = [d["time"] for d in flight_data]
                    ax5.plot(t_fl, [d["vgx"] for d in flight_data], label="SDK vgx (raw)", color="r", linestyle=":")
                    ax5.plot(t_fl, [d["vgy"] for d in flight_data], label="SDK vgy (raw)", color="g", linestyle=":")
                    ax5.plot(t_fl, [d["vgz"] for d in flight_data], label="SDK vgz (raw)", color="b", linestyle=":")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Vel (m/s | raw SDK)")
                ax5.set_title("Linear Velocity (Vicon body vs SDK vg raw)")
                ax5.grid(True)
                ax5.legend(fontsize=8)

                # Subplot 6: VELOCITA' ANGOLARI nel corpo, derivate dal Vicon
                # (quelle che finiscono in osservazione alla policy)
                ax6 = fig.add_subplot(4, 2, 6)
                if vicon_data:
                    t_vicon = [d["time"] for d in vicon_data]
                    ax6.plot(t_vicon, [d["ang_vel_b_x"] for d in vicon_data], label="wx (roll rate)", color="darkred")
                    ax6.plot(t_vicon, [d["ang_vel_b_y"] for d in vicon_data], label="wy (pitch rate)", color="darkgreen")
                    ax6.plot(t_vicon, [d["ang_vel_b_z"] for d in vicon_data], label="wz (yaw rate)", color="darkblue")
                    ax6.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
                    ax6.set_xlabel("Time (s)")
                    ax6.set_ylabel("Ang vel (rad/s)")
                    ax6.set_title("Body Angular Velocity ang_vel_b (derived from Vicon)")
                    ax6.grid(True)
                    ax6.legend(fontsize=8)
                else:
                    ax6.set_title("Body Angular Velocity (No Data)")

                # Subplot 7: istogramma ang_vel — utile per giudicare il RUMORE
                # a drone fermo (dovrebbe essere una campana stretta su 0)
                ax7 = fig.add_subplot(4, 2, 7)
                if vicon_data:
                    for key, lab, col in (("ang_vel_b_x", "wx", "darkred"),
                                          ("ang_vel_b_y", "wy", "darkgreen"),
                                          ("ang_vel_b_z", "wz", "darkblue")):
                        vals = [d[key] for d in vicon_data]
                        ax7.hist(vals, bins=60, alpha=0.5, label=lab, color=col)
                    ax7.set_xlabel("Ang vel (rad/s)")
                    ax7.set_ylabel("Occorrenze")
                    ax7.set_title("ang_vel_b distribution (a drone fermo: stretta su 0)")
                    ax7.grid(True)
                    ax7.legend(fontsize=8)
                else:
                    ax7.set_title("ang_vel_b distribution (No Data)")

                # Subplot 8: Batteria e Quota dallo state SDK
                ax8 = fig.add_subplot(4, 2, 8)
                if flight_data:
                    t_fl = [d["time"] for d in flight_data]
                    if any(d["battery"] is not None for d in flight_data):
                        ax8.plot(t_fl, [d["battery"] for d in flight_data], label="Battery (%)", color="orange")
                    if any(d["height_cm"] is not None for d in flight_data):
                        h_vals = [d["height_cm"] / 100.0 if d["height_cm"] is not None else None
                                  for d in flight_data]
                        ax8.plot(t_fl, h_vals, label="Height SDK (m)", color="teal")
                    if vicon_data:
                        ax8.plot([d["time"] for d in vicon_data],
                                 [d["z"] for d in vicon_data],
                                 label="Height Vicon Z (m)", color="purple", linestyle="--")
                    ax8.set_xlabel("Time (s)")
                    ax8.set_ylabel("Value")
                    ax8.set_title("Tello State (Battery & Height, SDK vs Vicon)")
                    ax8.grid(True)
                    ax8.legend(fontsize=8)
                else:
                    ax8.set_title("Tello State (No Data)")

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
            node.drone.end()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

