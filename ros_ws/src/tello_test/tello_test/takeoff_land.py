#!/usr/bin/env python3
"""Nodo 'takeoff_land' (tello_test): sequenza di volo automatica con registrazione dei sensori nello stesso processo.

Un solo processo perche' djitellopy lega porte UDP fisse (8889 comandi, 8890 stato): due processi
sullo stesso host non possono parlare con lo stesso drone.
Sequenza: connessione -> takeoff -> test assi (+x, -y, +z a rc molto basso) -> hover per HOVER_TIME_S
-> land -> chiusura. Per tutta l'esecuzione registra Vicon (posizione, assetto, velocita' derivate)
e stato djitellopy, come read_sensors. Comunque termini (fine sequenza, Ctrl+C, comando l/q, batteria
sotto BATTERY_FAILSAFE_PCT) atterra per prima cosa e solo dopo salva CSV e grafici 4x2 in 'output_dir'.
Comandi da terminale opzionali: l/land atterra e chiude, q/quit/exit idem.
"""

import csv
import math
import os
import select
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

import matplotlib
# matplotlib senza display (backend Agg)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from djitellopy import Tello as DJITello

# Decollo: TAKEOFF_MIN_ALT_CM (cm) e' solo la soglia per confermare il decollo; il Tello rifiuta un takeoff troppo a ridosso di 'command' (PRE_TAKEOFF_SETTLE_S)
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_CM = 15
PRE_TAKEOFF_SETTLE_S = 2.0
POST_TAKEOFF_SETTLE_S = 3.0
BATTERY_FAILSAFE_PCT = 15

# Durata del volo tra takeoff e land automatici
HOVER_TIME_S = 5.0

# Test assi: rc molto basso (range -100..100); send_rc_control invia un solo pacchetto, va ri-inviato ogni TEST_MOVE_RC_PERIOD_S
TEST_MOVE_RC = 15
TEST_MOVE_X_S = 4.0
TEST_MOVE_Y_S = 4.0
TEST_MOVE_Z_S = 1.0
TEST_MOVE_RC_PERIOD_S = 0.1
TEST_MOVE_SETTLE_S = 0.5

# Stampa, polling telemetria e topic Vicon
STATUS_PRINT_PERIOD_S = 2.0
TELEMETRY_POLL_PERIOD_S = 0.1
TERMINAL_POLL_TIMEOUT_S = 0.2

DEFAULT_VICON_POSE_TOPIC = "/vicon/Tello_2/Tello_2"

# Sanity check mocap (oltre MAX_PLAUSIBLE_ANG_SPEED_RADPS la derivata di Eulero e' rumore) e filtro velocita'
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0

VEL_FILTER_ALPHA = 0.3


# Trasformazioni (stesso calcolo di observation_handler): gravita' proiettata, velocita' lineare e angolare nel corpo
def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_projected_gravity_b(qx, qy, qz, qw):
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


# Input da terminale (non bloccante)
def leggi_comando_terminale():
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


# Nodo: registrazione continua di sensori e Vicon, sequenza di volo automatica
class TakeoffLand(Node):
    def __init__(self):
        super().__init__("takeoff_land")

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

        self.vicon_history = []
        self.flight_history = []

        self.flying = False
        self._landing_started = False
        self._shutdown_requested = False

        self._vicon_lock = threading.Lock()
        self.pos = None
        self.yaw = None
        self.roll = None
        self.pitch = None
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

        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.tello_alt_cm = None
        self.tof_cm = None
        self.vg_raw = None
        self.flight_data_count = 0
        self.last_flight_wall_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped, vicon_pose_topic, self.pose_cb, 10
        )

        self.declare_parameter("tello_ip", "192.168.10.1")
        tello_ip = self.get_parameter("tello_ip").get_parameter_value().string_value
        self.drone = DJITello(host=tello_ip)

        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)
        self.telemetry_timer = self.create_timer(TELEMETRY_POLL_PERIOD_S, self._poll_telemetry)

        self.get_logger().info(
            "Nodo takeoff_land avviato: sequenza automatica takeoff -> "
            "test assi (+x/-y/+z, rc basso) -> "
            f"hover {HOVER_TIME_S}s -> land in corso. Vicon topic: {vicon_pose_topic}. "
            "Override manuale da terminale: l=atterra subito | q=atterra e chiudi."
        )

    def _log(self, level, msg):
        try:
            getattr(self.get_logger(), level)(msg)
        except Exception:
            pass

    # Callback Vicon: validazione del campione e velocita' derivate
    def pose_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        raw_vals = [p[0], p[1], p[2], q.x, q.y, q.z, q.w]
        if any(math.isnan(v) or math.isinf(v) for v in raw_vals):
            self.pose_rejected_count += 1
            self._log("warn", "[vicon] Pose con NaN/Inf scartata.")
            return

        quat_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not (MIN_QUAT_NORM < quat_norm < MAX_QUAT_NORM):
            self.pose_rejected_count += 1
            self._log(
                "warn",
                f"[vicon] Quaternione degenere (norma={quat_norm:.3f}), pose scartata.",
            )
            return

        euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
        proj_grav_b = compute_projected_gravity_b(q.x, q.y, q.z, q.w)

        with self._vicon_lock:
            if self.prev_pos is not None and self.prev_time is not None:
                dt = max(t - self.prev_time, 1e-3)

                v_world = (p - self.prev_pos) / dt
                v_body_raw = compute_lin_vel_body(v_world, q.x, q.y, q.z, q.w)
                self.lin_vel_b = VEL_FILTER_ALPHA * v_body_raw \
                                  + (1 - VEL_FILTER_ALPHA) * self.lin_vel_b

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
            self.roll = roll
            self.pitch = pitch
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

    # Telemetria djitellopy e stampa periodica
    def _poll_telemetry(self):
        try:
            state = self.drone.get_current_state()
        except Exception:
            state = {}

        battery = state.get("bat")
        height_cm = state.get("h")
        tof_cm = state.get("tof")
        vg = (state.get("vgx"), state.get("vgy"), state.get("vgz"))

        with self._tello_lock:
            self.battery_pct = battery
            self.tello_alt_cm = height_cm
            self.tof_cm = tof_cm
            self.vg_raw = vg
            if state:
                self.flight_data_count += 1
                self.last_flight_wall_time = time.monotonic()
                t_rel = time.time() - self.start_time
                self.flight_history.append({
                    "time": t_rel,
                    "battery": battery,
                    "height_cm": height_cm,
                    "tof_cm": tof_cm,
                    "vgx": vg[0], "vgy": vg[1], "vgz": vg[2],
                })

        if (
            battery is not None
            and battery < BATTERY_FAILSAFE_PCT
            and self.flying
            and not self._landing_started
        ):
            self._log("error", f"[djitellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza.")
            self.land()

    def status_cb(self):
        with self._tello_lock:
            bat, alt = self.battery_pct, self.tello_alt_cm
        with self._vicon_lock:
            pos, yaw = self.pos, self.yaw
            ang_vel_b = self.ang_vel_b.copy()
            pose_count = self.pose_count
        self.get_logger().info(
            f"[status] {'IN VOLO' if self.flying else 'A TERRA'} | "
            f"batteria={bat}% | quota={alt}cm"
        )
        if pos is not None:
            self.get_logger().info(
                f"[vicon] pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m | "
                f"yaw={math.degrees(yaw):+.1f}deg | "
                f"ang_vel_b=({ang_vel_b[0]:+.3f},{ang_vel_b[1]:+.3f},{ang_vel_b[2]:+.3f})rad/s | "
                f"n_msg={pose_count}"
            )
        else:
            self.get_logger().warn("[vicon] NESSUN messaggio ricevuto ancora.")

    # Sequenza di volo: decollo, test assi, atterraggio
    def takeoff(self):
        if self.flying:
            self.get_logger().warn("[takeoff] Gia' in volo, comando ignorato.")
            return

        self.get_logger().info("[djitellopy] connessione al drone...")
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] connessione fallita: {e}")
            return
        self.get_logger().info("[djitellopy] connesso.")

        self._poll_telemetry()
        with self._tello_lock:
            bat = self.battery_pct
        self.get_logger().info(
            f"[djitellopy] batteria={bat}% | assestamento {PRE_TAKEOFF_SETTLE_S}s "
            "prima del takeoff (drone fermo su superficie piana)..."
        )
        time.sleep(PRE_TAKEOFF_SETTLE_S)

        self.get_logger().info("[djitellopy] invio takeoff...")
        try:
            self.drone.takeoff()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] comando takeoff fallito: {e}")
            return

        self._landing_started = False
        self.flying = True

        deadline = time.monotonic() + TAKEOFF_CONFIRM_TIMEOUT_S
        confirmed = False
        while time.monotonic() < deadline:
            self._poll_telemetry()
            with self._tello_lock:
                alt = self.tello_alt_cm
            if alt is not None and alt > TAKEOFF_MIN_ALT_CM:
                confirmed = True
                break
            time.sleep(0.2)

        if not confirmed:
            self.get_logger().error(
                "[djitellopy] Nessuna variazione di quota plausibile rilevata dopo "
                "il takeoff: verifica manualmente lo stato del drone. Il nodo "
                "considera comunque il drone IN VOLO per sicurezza (verra' "
                "atterrato normalmente)."
            )
        else:
            self.get_logger().info(
                f"[djitellopy] decollo confermato (quota>{TAKEOFF_MIN_ALT_CM}cm). "
                f"Assestamento {POST_TAKEOFF_SETTLE_S}s..."
            )

        time.sleep(POST_TAKEOFF_SETTLE_S)
        self.get_logger().info("[djitellopy] IN VOLO.")

    def _stick_zero(self):
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            self._log("error", f"[test-assi] errore azzerando i comandi rc: {e}")

    # Ordine argomenti djitellopy: (left_right, forward_backward, up_down, yaw); '-y' = destra perche' la policy usa FLU (y positivo = sinistra)
    def test_axis_movements(self):
        if not self.flying or self._shutdown_requested:
            return

        assi = (
            ("forward_backward", "+x (avanti)", TEST_MOVE_X_S, (0, TEST_MOVE_RC, 0, 0)),
            ("left_right", "-y (destra)", TEST_MOVE_Y_S, (TEST_MOVE_RC, 0, 0, 0)),
            ("up_down", "+z (su)", TEST_MOVE_Z_S, (0, 0, TEST_MOVE_RC, 0)),
        )

        try:
            for nome_canale, descrizione, durata_s, rc_values in assi:
                if self._shutdown_requested or not self.flying:
                    break

                self._log(
                    "info",
                    f"[test-assi] {descrizione}: {nome_canale}={TEST_MOVE_RC} "
                    f"per {durata_s}s...",
                )

                deadline = time.monotonic() + durata_s
                errore = False
                while time.monotonic() < deadline and not self._shutdown_requested:
                    try:
                        self.drone.send_rc_control(*rc_values)
                    except Exception as e:
                        self._log("error", f"[test-assi] comando {nome_canale} fallito: {e}")
                        errore = True
                        break
                    time.sleep(TEST_MOVE_RC_PERIOD_S)

                self._stick_zero()
                if errore:
                    break
                time.sleep(TEST_MOVE_SETTLE_S)
        finally:
            self._stick_zero()
            self._log("info", "[test-assi] sequenza terminata, comandi rc azzerati.")

    def land(self):
        if not self.flying or self._landing_started:
            self._log("warn", "[land] Non in volo (o atterraggio gia' in corso), comando ignorato.")
            return
        self._landing_started = True

        self._log("info", "[djitellopy] invio land...")
        try:
            self.drone.land()
        except Exception as e:
            self._log("error", f"[djitellopy] comando land fallito: {e}")
            try:
                self.drone.send_command_without_return("land")
            except Exception:
                pass

        self.flying = False
        self._landing_started = False
        self._log("info", "[djitellopy] ATTERRATO.")

    # Salvataggio CSV e griglia di grafici 4x2 a fine sessione (dopo l'atterraggio)
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
                fig.suptitle(
                    f"Sensor Data Overview ({timestamp_str}) — ang_vel_b derivata dal Vicon",
                    fontsize=16, fontweight='bold'
                )

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


# Thread stdin, sequenza automatica e main
def terminal_input_loop(node: TakeoffLand):
    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  l / land     -> atterra subito e chiudi il nodo (override manuale)\n"
        "  q / quit     -> atterra (se in volo) e chiudi il nodo\n"
        "=======================================================================\n"
    )
    while rclpy.ok() and not node._shutdown_requested:
        comando = leggi_comando_terminale()
        if comando is None:
            time.sleep(TERMINAL_POLL_TIMEOUT_S)
            continue

        cmd = comando.lower()
        if cmd in ("l", "land", "q", "quit", "exit"):
            node.get_logger().info(
                f"[terminale] Comando '{cmd}' ricevuto: atterraggio (se in volo) e chiusura del nodo."
            )
            node._shutdown_requested = True
            if node.flying:
                node.land()
            rclpy.shutdown()
            break
        else:
            print(f"[terminale] Comando non riconosciuto: '{comando}' (l=atterra e chiudi, q=chiudi)")


def auto_sequence(node: TakeoffLand):
    node.takeoff()

    if node.flying and not node._shutdown_requested:
        node.test_axis_movements()

    if node.flying and not node._shutdown_requested:
        time.sleep(HOVER_TIME_S)

    if not node._shutdown_requested:
        node.land()
        node._shutdown_requested = True
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TakeoffLand()

    sigint_received = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: sigint_received.set())

    input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
    input_thread.start()

    auto_thread = threading.Thread(target=auto_sequence, args=(node,), daemon=True)
    auto_thread.start()

    try:
        while rclpy.ok() and not node._shutdown_requested and not sigint_received.is_set():
            rclpy.spin_once(node, timeout_sec=0.5)
        if sigint_received.is_set():
            node._log("warn", "Ctrl+C ricevuto: atterraggio di emergenza in corso.")
    finally:
        node._shutdown_requested = True

        try:
            if node.flying:
                node.land()
        except Exception as e:
            node._log("error", f"[land] eccezione durante l'atterraggio: {e}")

        if getattr(node.drone, "is_flying", False):
            try:
                node.drone.send_command_without_return("land")
            except Exception:
                pass

        try:
            node.save_data_and_plots()
        except Exception as e:
            node._log("error", f"Errore durante il salvataggio dati: {e}")

        try:
            node.drone.end()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
