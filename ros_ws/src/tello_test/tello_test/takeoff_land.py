#!/usr/bin/env python3
"""
Nodo ROS2 'takeoff_land' (package tello_test): FONDE in un solo processo
quello che prima erano due nodi separati (takeoff_land.py + read_sensors.py).

Motivo della fusione: entrambi usano djitellopy, che lega una porta UDP
fissa per processo (8889 comandi, 8890 stato) — due processi djitellopy
NON possono girare in parallelo sullo stesso host verso lo stesso drone
(OSError: Address already in use). Un solo processo risolve il conflitto
e permette di avere insieme comandi di volo E lettura/plot sensori.

COSA FA:
  1. Si connette al drone via djitellopy, decolla, esegue il TEST DI
     VALIDAZIONE ASSI (vedi sotto), fa hover, atterra — esattamente come
     il vecchio takeoff_land.py.
  2. IN PARALLELO, per tutta la durata dell'esecuzione (anche PRIMA del
     takeoff e DOPO il land), legge e registra:
       - Vicon (topic --ros-args -p vicon_pose_topic:=...): posizione,
         quaternione, roll/pitch/yaw, proj_grav_b, lin_vel_b E ang_vel_b
         (derivate dal quaternione, STESSA logica esatta di
         position_controller_VICON_VERSION.py — niente giroscopio).
       - djitellopy state (broadcast SDK ufficiale): batteria, quota,
         ToF, velocita' SDK grezze (vgx/vgy/vgz).
     esattamente come faceva il vecchio read_sensors.py.
  3. Alla chiusura (Ctrl+C o fine sequenza automatica), SEMPRE, prima
     salva/atterra il drone e SOLO DOPO salva CSV + grafici PNG (stessa
     griglia 4x2 di read_sensors.py) nella cartella output_dir.

LIBRERIA: solo djitellopy (tellopy rimossa da tutto il progetto: le due
non convivono sullo stesso drone, appena djitellopy entra in modalita'
SDK il firmware smette di alimentare il flusso di log binario "app" da
cui tellopy leggeva il giroscopio).

Sequenza AUTOMATICA all'avvio: connessione -> takeoff -> test assi
(+x/-y/+z, rc molto basso) -> hover per HOVER_TIME_S secondi -> land ->
chiusura del nodo (con salvataggio dati). Nessun input da terminale
richiesto per il volo; il Vicon/telemetria vengono registrati SEMPRE,
indipendentemente da cosa sta facendo il drone.

TEST DI VALIDAZIONE ASSI (dopo il takeoff, PRIMA dell'hover):
  +x (avanti)  -> send_rc_control(0, TEST_MOVE_RC, 0, 0)   per TEST_MOVE_X_S secondi
  -y (destra)  -> send_rc_control(TEST_MOVE_RC, 0, 0, 0)   per TEST_MOVE_Y_S secondi
  +z (su)      -> send_rc_control(0, 0, TEST_MOVE_RC, 0)   per TEST_MOVE_Z_S secondi
  Ordine parametri djitellopy: (left_right, forward_backward, up_down, yaw).
  Positivo = avanti/destra/su/orario (comando SDK 'rc a b c d'). -y invece
  di +y perche' la policy usa convenzione FLU (y positivo = sinistra),
  quindi "-y" fisico = "destra" = left_right positivo. send_rc_control()
  invia UN SOLO pacchetto per chiamata (a differenza di tellopy): va
  richiamato PERIODICAMENTE per tutta la durata dell'hold (vedi loop in
  test_axis_movements). Interrotto subito se arriva una richiesta di
  land/chiusura; i comandi rc vengono SEMPRE azzerati alla fine di ogni
  asse e in caso di errore.

REGISTRAZIONE E PLOT DATI (griglia 4x2, come read_sensors.py):
  1) posizione Vicon x/y/z          2) traiettoria 3D
  3) angoli di Eulero roll/pitch/yaw (la sorgente di ang_vel_b)
  4) gravita' proiettata nel corpo  5) velocita' lineari (Vicon vs SDK)
  6) ang_vel_b wx/wy/wz             7) istogramma di ang_vel_b
  8) batteria + quota (SDK vs Vicon)
  I pannelli 3-6-7 servono a giudicare la qualita' delle velocita'
  angolari derivate: a drone FERMO l'istogramma (7) deve essere una
  campana stretta attorno a 0; se e' largo, il rumore del mocap sta
  passando in osservazione e va alzato VEL_FILTER_ALPHA.

Comandi da terminale (thread stdin non bloccante, select()), opzionali,
utilizzabili SOLO come override manuale (es. atterraggio anticipato):
    l / land      -> atterra subito (se in volo) e chiude il nodo
    q / quit/exit -> atterra se in volo, poi chiude il nodo
Ctrl+C: atterra (se in volo), salva i dati, e chiude.

Failsafe: batteria critica (< BATTERY_FAILSAFE_PCT) durante il volo ->
land automatico, stessa soglia di position_controller_VICON_VERSION.py.
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
matplotlib.use('Agg')  # Backend non interattivo per salvataggio figure senza server X
import matplotlib.pyplot as plt

from djitellopy import Tello as DJITello

# -- stessi valori/soglie di position_controller_VICON_VERSION.py --
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_CM = 15            # centimetri (state 'h' djitellopy, documentato dall'SDK ufficiale). NON e' un target di altezza: e' solo la soglia per CONFERMARE che il decollo sia avvenuto (vedi nota di sicurezza in takeoff()).
PRE_TAKEOFF_SETTLE_S = 2.0         # assestamento DOPO connect() e PRIMA di takeoff(): il Tello rifiuta il takeoff ('error') se il comando arriva troppo a ridosso di 'command'.
POST_TAKEOFF_SETTLE_S = 3.0
BATTERY_FAILSAFE_PCT = 15

HOVER_TIME_S = 5.0  # durata del volo tra takeoff e land automatici

# -- test di validazione assi (vedi docstring in cima al file) --
TEST_MOVE_RC = 15         # rc MOLTO basso, range -100..100
TEST_MOVE_X_S = 4.0       # +x (avanti): forward_backward
TEST_MOVE_Y_S = 4.0       # -y (destra): left_right positivo
TEST_MOVE_Z_S = 1.0       # +z (su): up_down
TEST_MOVE_RC_PERIOD_S = 0.1   # periodo di ri-invio del comando rc durante l'hold (send_rc_control non si ripete da solo, a differenza di tellopy)
TEST_MOVE_SETTLE_S = 0.5  # pausa dopo aver azzerato i comandi rc, tra un asse e l'altro

STATUS_PRINT_PERIOD_S = 2.0
TELEMETRY_POLL_PERIOD_S = 0.1   # frequenza di campionamento dello state djitellopy per log/CSV
TERMINAL_POLL_TIMEOUT_S = 0.2

DEFAULT_VICON_POSE_TOPIC = "/vicon/tello_42_boosted/tello_42_boosted"

# -- sanity check mocap, stesse soglie del controllore --
MIN_QUAT_NORM = 0.9
MAX_QUAT_NORM = 1.1
MAX_PLAUSIBLE_ANG_SPEED_RADPS = 20.0   # oltre questo la derivata di Eulero e' rumore/glitch mocap, non moto reale

# -- filtro passa-basso sulle velocita' derivate dal mocap (lineari E angolari) --
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


def leggi_comando_terminale():
    """Lettura NON BLOCCANTE da stdin (stessa logica degli altri nodi)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


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

        # -- registri dati per salvataggio CSV e plot --
        self.vicon_history = []
        self.flight_history = []

        # -- stato volo --
        self.flying = False
        self._landing_started = False
        self._shutdown_requested = False

        # -- stato Vicon, protetto da _vicon_lock --
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

        # -- stato telemetria drone (state djitellopy), protetto da _tello_lock --
        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.tello_alt_cm = None
        self.tof_cm = None
        self.vg_raw = None          # (vgx, vgy, vgz) — unita' SDK grezze, NON verificate
        self.flight_data_count = 0
        self.last_flight_wall_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped, vicon_pose_topic, self.pose_cb, 10
        )

        # -- djitellopy: connect/takeoff/land/movimento (rc control) +
        # lettura batteria/quota dallo state broadcast SDK ufficiale. --
        self.drone = DJITello()

        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)
        self.telemetry_timer = self.create_timer(TELEMETRY_POLL_PERIOD_S, self._poll_telemetry)

        self.get_logger().info(
            "Nodo takeoff_land avviato: sequenza automatica takeoff -> "
            "test assi (+x/-y/+z, rc basso) -> "
            f"hover {HOVER_TIME_S}s -> land in corso. Vicon topic: {vicon_pose_topic}. "
            "Override manuale da terminale: l=atterra subito | q=atterra e chiudi."
        )

    def _log(self, level, msg):
        """Log best-effort: non deve MAI impedire l'invio del comando di land
        (es. se il context ROS e' gia' stato chiuso da un Ctrl+C)."""
        try:
            getattr(self.get_logger(), level)(msg)
        except Exception:
            pass

    # ==================================================================
    # -- callback Vicon: pose, orientamento, velocita' lineari/angolari --
    # ==================================================================
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

                # -- velocita' LINEARI: differenze finite mondo -> corpo --
                v_world = (p - self.prev_pos) / dt
                v_body_raw = compute_lin_vel_body(v_world, q.x, q.y, q.z, q.w)
                self.lin_vel_b = VEL_FILTER_ALPHA * v_body_raw \
                                  + (1 - VEL_FILTER_ALPHA) * self.lin_vel_b

                # -- velocita' ANGOLARI: derivate di Eulero (wrap_to_pi
                # obbligatorio sul salto +pi/-pi) -> frame corpo via
                # matrice cinematica. Stessa identica logica del
                # controllore VICON (niente giroscopio). --
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

    # ==================================================================
    # -- TELEMETRIA DRONE (state djitellopy: SOLO lettura, nessun comando) --
    # ==================================================================
    def _poll_telemetry(self):
        """Legge batteria/quota/ToF/velocita' SDK dallo state djitellopy
        (dict aggiornato in background dal broadcast SDK ufficiale),
        registra un campione per il CSV e applica il failsafe batteria
        critica. Girato da un timer indipendente da flying/control, cosi'
        la telemetria e' registrata SEMPRE (anche a terra, prima/dopo il
        volo) — stesso comportamento di read_sensors.py."""
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

    # ==================================================================
    # -- TAKEOFF / LAND / TEST ASSI tramite djitellopy --
    # ==================================================================
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

        # Assestamento PRIMA del takeoff: il Tello spesso rifiuta il
        # takeoff (risponde 'error') se il comando arriva troppo a ridosso
        # di 'command', o se non e' ancora fermo/livellato. Nel frattempo
        # logghiamo la batteria per diagnosticare rapidamente un eventuale
        # rifiuto per batteria scarica.
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

        # SICUREZZA: da qui in poi il drone e' fisicamente in volo (takeoff()
        # di djitellopy ha gia' ricevuto un "ok" dal drone, altrimenti
        # avrebbe sollevato un'eccezione sopra). Marchiamo flying=True
        # SUBITO, a prescindere dalla conferma sulla quota qui sotto, cosi'
        # che un land() (manuale/automatico/emergenza) venga SEMPRE
        # tentato. Se gatiamo flying su "confermato", una soglia sbagliata
        # o un sensore lento lascia il nodo convinto che il drone sia a
        # terra mentre non e' cosi', e nessun land() viene piu' inviato
        # (successo gia' capitato in passato).
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
        """Azzera tutti i comandi rc (hover in place)."""
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            self._log("error", f"[test-assi] errore azzerando i comandi rc: {e}")

    def test_axis_movements(self):
        """Sequenza di validazione assi (SOLO test manuale a bassa quota,
        rc molto basso): +x (avanti) -> -y (destra) -> +z (su). Vedi
        TEST DI VALIDAZIONE ASSI nel docstring in cima al file per i segni.
        send_rc_control() invia UN SOLO pacchetto per chiamata (a
        differenza di tellopy): viene richiamato ogni TEST_MOVE_RC_PERIOD_S
        per tutta la durata dell'hold, cosi' il comando non scade a meta'.
        Interrotta subito se arriva una richiesta di land/chiusura; i
        comandi rc vengono SEMPRE azzerati alla fine di ogni asse e in
        ogni caso (anche in caso di errore), per non lasciare un comando
        di movimento attivo sul drone."""
        if not self.flying or self._shutdown_requested:
            return

        # (left_right, forward_backward, up_down, yaw) — ordine djitellopy.send_rc_control
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
            # fallback "a qualunque costo": comando fire-and-forget, senza
            # attesa/retry/eccezioni, per non restare bloccati se anche i
            # retry di send_control_command falliscono.
            try:
                self.drone.send_command_without_return("land")
            except Exception:
                pass

        self.flying = False
        self._landing_started = False
        self._log("info", "[djitellopy] ATTERRATO.")

    # ==================================================================
    # -- salvataggio CSV e grafico PLOT (identico a read_sensors.py) --
    # ==================================================================
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


def terminal_input_loop(node: TakeoffLand):
    """Override manuale OPZIONALE: la sequenza takeoff/hover/land parte da
    sola in auto_sequence(); questo thread serve solo per un atterraggio
    anticipato o una chiusura forzata."""
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
    """Sequenza automatica: takeoff -> test assi -> hover HOVER_TIME_S ->
    land -> shutdown. Se un override manuale ('l'/'q' da terminale,
    Ctrl+C) ha gia' avviato la chiusura del nodo, ogni passo si limita a
    non fare nulla (takeoff/land sono gia' idempotenti/guardati da
    self.flying e self._landing_started)."""
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
    # signal_handler_options=NO: gestiamo NOI il SIGINT (rclpy di default puo'
    # "assorbire" il Ctrl+C internamente senza sollevare KeyboardInterrupt,
    # facendo si' che rclpy.spin() ritorni senza che l'atterraggio parta mai).
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

        # Atterraggio "a qualunque costo": prima la procedura normale...
        try:
            if node.flying:
                node.land()
        except Exception as e:
            node._log("error", f"[land] eccezione durante l'atterraggio: {e}")

        # ...poi, se per qualsiasi motivo il drone risulta ancora in volo
        # secondo djitellopy, comando diretto fire-and-forget, senza
        # passare per nessuna logica/logging/retry intermedio.
        if getattr(node.drone, "is_flying", False):
            try:
                node.drone.send_command_without_return("land")
            except Exception:
                pass

        # Salvataggio dati SOLO dopo che il drone e' a terra (o si e'
        # comunque fatto il possibile per farlo atterrare).
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
