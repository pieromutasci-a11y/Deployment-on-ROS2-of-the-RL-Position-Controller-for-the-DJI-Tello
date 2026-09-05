#!/usr/bin/env python3
"""
Nodo ROS2 'vel_command_handler' (package tello_pkg): UNICO nodo della
pipeline modulare (Percezione / Policy / Attuazione / Target) con una
connessione djitellopy reale al drone. Nessun altro nodo puo' averne
una in parallelo: djitellopy occupa le porte fisse 8889 (comandi) e
8890 (stato) sullo stesso host, quindi una sola istanza Tello() puo'
esistere per volta (stesso motivo per cui tello_test non puo' girare
insieme a questa pipeline).

CANCELLO DI PARTENZA: connect_sequence() (chiamata all'avvio del nodo)
si limita a connettersi al drone e leggere la telemetria iniziale — NON
decolla. Il takeoff vero parte SOLO alla ricezione di un segnale su
/tello/start_request (Empty), tipicamente pubblicato da mission_console
digitando 'start' nel terminale del launch, o dal proprio thread
stdin (comando 'start'/'s') se lanciato standalone con 'ros2 run'. Cosi'
si puo' lanciare l'intera pipeline, controllare batteria/parametri, e
decidere esplicitamente quando far decollare il drone.

RESPONSABILITA':
  - connessione/takeoff/land tramite djitellopy.
  - riceve /tello/policy_action (Twist, azione GREZZA clampata [-1,1]
    da policy_handler) e la converte in comando reale al drone:
    VEL_REF_SCALE -> *100 -> *RC_SCALE_PCT -> send_rc_control. Il
    riferimento della policy NON viene piu' clampato in m/s: e' gia' entro
    i limiti fisici per costruzione; RC_SCALE_PCT (uno per asse [vx,vy,vz,wz])
    scala PROPORZIONALMENTE il comando finale (es. 1.0 m/s di riferimento
    su vx -> RC=100 a piena autorita', con RC_SCALE_PCT[0]=0.4 diventa
    RC=40). NESSUN mascheramento dof_mask qui: la
    policy ha gia' imparato a non generare vy in uniciclo (dof_mask e'
    SOLO una feature di osservazione, mai un hard mask sull'azione:
    vedi memoria feedback_dof_mask_observation_only).
  - legge la telemetria (batteria/quota) dallo state djitellopy, la
    pubblica su /tello/flight_state, applica il failsafe batteria.
  - ascolta /tello/land_request (Empty, pubblicato da target_handler a
    fine missione E da observation_handler su perdita Vicon prolungata)
    e lo tratta come un trigger di atterraggio equivalente a Ctrl+C.
  - watchdog: se /tello/policy_action non arriva da oltre
    ACTION_TIMEOUT_S, forza hover (stick a zero) finche' non riprende.
  - Ctrl+C-deve-SEMPRE-atterrare: stesso pattern robusto gia' usato in
    takeoff_land.py/position_controller_VICON_VERSION.py
    (signal_handler_options=NO + handler SIGINT custom + fallback
    incondizionato in finally).
"""

import math
import time
import select
import signal
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Empty

from djitellopy import Tello as DJITello

# ============================================================
# CONFIG — stessi valori usati in position_controller_VICON_VERSION.py
# ============================================================
VEL_REF_SCALE = np.array([1.0, 1.0, 1.0, 1.5])   # [vx,vy,vz,wz]: target_lin_vel_*_scale/target_yaw_vel_scale

# NON e' un clamp fisico in m/s: il riferimento della policy (dopo
# VEL_REF_SCALE) e' per costruzione gia' entro i limiti fisici, quindi non
# va tagliato. RC_SCALE_PCT scala PROPORZIONALMENTE il comando RC finale
# (dopo la conversione riferimento -> percentuale stick in _send_vel_command):
# un riferimento di 1.0 m/s produrrebbe RC=100 a piena autorita', con uno
# scaler di 0.4 diventa RC=40. Un valore INDIPENDENTE per asse [vx,vy,vz,wz]
# (stesso ordine di VEL_REF_SCALE), cosi' si puo' es. tenere vz/wz piu'
# prudenti di vx/vy senza toccare gli altri assi.
RC_SCALE_PCT = np.array([0.40, 0.40, 0.40, 0.40])   # [vx,vy,vz,wz]

ACTION_TIMEOUT_S = 0.2             # 5 cicli @25Hz: oltre questo, hover forzato

PRE_TAKEOFF_SETTLE_S = 2.0
POST_TAKEOFF_SETTLE_S = 3.0
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_CM = 15

TELEMETRY_POLL_PERIOD_S = 1.0
TELLO_LOST_LAND_TIMEOUT_S = 3.0
BATTERY_FAILSAFE_PCT = 15

STATUS_PRINT_PERIOD_S = 1.0
TERMINAL_POLL_TIMEOUT_S = 0.2

POLICY_ACTION_TOPIC = "/tello/policy_action"
FLIGHT_STATE_TOPIC = "/tello/flight_state"
LAND_REQUEST_TOPIC = "/tello/land_request"
START_REQUEST_TOPIC = "/tello/start_request"


def leggi_comando_terminale():
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


class VelCommandHandler(Node):
    def __init__(self):
        super().__init__("vel_command_handler")

        self.declare_parameter("enable_terminal_input", True)
        self.enable_terminal_input = self.get_parameter("enable_terminal_input").get_parameter_value().bool_value

        self.flight_ready = False
        self._landing_started = False
        self._shutdown_requested = False
        self._takeoff_in_progress = False
        self.connected = False

        self._action_lock = threading.Lock()
        self.last_action = np.zeros(4)
        self.last_action_wall_time = None

        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.tello_alt_cm = None
        self._last_state_snapshot = None
        self.last_state_change_time = None
        self._state_warned = False

        # IP del drone: default 192.168.10.1 (Tello in AP mode, PC connesso
        # alla sua rete WiFi). Override via --ros-args -p tello_ip:=... se il
        # drone e' invece in station mode (join di una rete esistente), dove
        # l'IP e' assegnato dal router e non e' quello di fabbrica.
        self.declare_parameter("tello_ip", "192.168.16.196")
        tello_ip = self.get_parameter("tello_ip").get_parameter_value().string_value
        self.drone = DJITello(host=tello_ip)

        self.action_sub = self.create_subscription(
            Twist, POLICY_ACTION_TOPIC, self.policy_action_cb, 10
        )
        self.land_request_sub = self.create_subscription(
            Empty, LAND_REQUEST_TOPIC, self.land_request_cb, 10
        )
        self.start_request_sub = self.create_subscription(
            Empty, START_REQUEST_TOPIC, self.start_request_cb, 10
        )

        flight_state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.flight_state_pub = self.create_publisher(Bool, FLIGHT_STATE_TOPIC, flight_state_qos)

        self.watchdog_timer = self.create_timer(ACTION_TIMEOUT_S, self.watchdog_cb)
        self.telemetry_timer = self.create_timer(TELEMETRY_POLL_PERIOD_S, self.telemetry_cb)
        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)

        self.get_logger().info(
            f"vel_command_handler avviato. Sottoscritto a '{POLICY_ACTION_TOPIC}' e "
            f"'{LAND_REQUEST_TOPIC}', pubblico '{FLIGHT_STATE_TOPIC}'."
        )

    def _publish_flight_state(self):
        msg = Bool()
        msg.data = self.flight_ready
        self.flight_state_pub.publish(msg)

    # -------------------- ricezione azione dalla policy --------------------
    def policy_action_cb(self, msg: Twist):
        with self._action_lock:
            self.last_action = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])
            self.last_action_wall_time = time.monotonic()

        if not self.flight_ready or self._shutdown_requested:
            return

        action = np.clip(self.last_action, -1.0, 1.0)
        target_vel_ref = action * VEL_REF_SCALE  # [vx,vy,vz,wz] frame corpo, m/s e rad/s
        vx, vy, vz, wz = (float(v) for v in target_vel_ref)
        self._send_vel_command(vx, vy, vz, wz)

    # -------------------- watchdog azione scaduta --------------------
    def watchdog_cb(self):
        if not self.flight_ready or self._shutdown_requested:
            return
        with self._action_lock:
            age = (
                time.monotonic() - self.last_action_wall_time
                if self.last_action_wall_time is not None
                else float("inf")
            )
        if age > ACTION_TIMEOUT_S:
            self._send_stick_zero()

    # -------------------- land_request (target_handler / observation_handler) --------------------
    def land_request_cb(self, msg: Empty):
        self.get_logger().warn("[land_request] Richiesta di atterraggio ricevuta da un nodo a valle.")
        threading.Thread(
            target=self.emergency_land, args=("land_request esterno",), daemon=True
        ).start()

    # -------------------- start_request: cancello di partenza --------------------
    def start_request_cb(self, msg: Empty):
        if not self.connected:
            self.get_logger().warn("[start_request] Drone non ancora connesso, richiesta ignorata.")
            return
        if self.flight_ready or self._takeoff_in_progress:
            self.get_logger().warn("[start_request] Takeoff gia' in corso o drone gia' in volo, richiesta ignorata.")
            return
        if self._shutdown_requested:
            return
        self._takeoff_in_progress = True
        threading.Thread(target=self._do_takeoff, daemon=True).start()

    def _do_takeoff(self):
        try:
            took_off = self.takeoff_sequence()
            if not took_off:
                self.get_logger().error("[start_request] Takeoff fallito. Puoi ritentare inviando un altro start_request.")
        finally:
            self._takeoff_in_progress = False

    # -------------------- TELEMETRIA (batteria/quota) --------------------
    def telemetry_cb(self):
        self._poll_djitellopy_state()

        now_mono = time.monotonic()
        state_age = (
            (now_mono - self.last_state_change_time)
            if self.last_state_change_time is not None
            else float("inf")
        )
        if state_age > TELLO_LOST_LAND_TIMEOUT_S and self.flight_ready and not self._landing_started:
            if not self._state_warned:
                self.get_logger().error(
                    f"Telemetria drone (state djitellopy) ferma da {state_age:.1f}s: "
                    "nessun dato batteria/quota, avvio atterraggio."
                )
                self._state_warned = True
            threading.Thread(
                target=self.emergency_land,
                args=(f"telemetria drone assente da oltre {TELLO_LOST_LAND_TIMEOUT_S}s",),
                daemon=True,
            ).start()
        elif state_age <= TELLO_LOST_LAND_TIMEOUT_S:
            self._state_warned = False

    def _poll_djitellopy_state(self):
        try:
            state = self.drone.get_current_state()
        except Exception:
            state = {}

        now_mono = time.monotonic()
        snapshot = tuple(sorted(state.items())) if state else None
        if snapshot is not None and snapshot != self._last_state_snapshot:
            self._last_state_snapshot = snapshot
            self.last_state_change_time = now_mono

        battery = state.get("bat")
        height_cm = state.get("h")
        with self._tello_lock:
            self.battery_pct = battery
            self.tello_alt_cm = height_cm

        if battery is not None and battery < BATTERY_FAILSAFE_PCT and self.flight_ready and not self._landing_started:
            self.get_logger().error(f"[djitellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza.")
            threading.Thread(
                target=self.emergency_land, args=(f"batteria critica ({battery}%)",), daemon=True
            ).start()

    def status_cb(self):
        with self._tello_lock:
            bat = self.battery_pct
            alt = self.tello_alt_cm
        with self._action_lock:
            age = (
                time.monotonic() - self.last_action_wall_time
                if self.last_action_wall_time is not None
                else None
            )
        age_str = f"{age:.2f}s" if age is not None else "mai ricevuta"
        self.get_logger().info(
            f"[status] flight_ready={self.flight_ready} | batteria={bat}% | quota={alt}cm | "
            f"ultima policy_action={age_str}"
        )

    # ==================================================================
    # -- CONNESSIONE (senza takeoff): chiamata all'avvio del nodo --
    # ==================================================================
    def connect_sequence(self) -> bool:
        self.get_logger().info("[djitellopy] connessione al drone (comandi/rc)...")
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] connessione fallita: {e}")
            return False

        self.connected = True
        self._poll_djitellopy_state()
        with self._tello_lock:
            bat = self.battery_pct
        self.get_logger().info(
            f"[djitellopy] connesso. batteria={bat}% | in attesa di start_request per il takeoff "
            "(digita 'start' in mission_console, o 'start'/'s' nel terminale di questo nodo)."
        )
        return True

    # ==================================================================
    # -- TAKEOFF / LAND tramite djitellopy. takeoff_sequence() e' chiamata
    # SOLO da start_request_cb (cancello di partenza), mai in automatico. --
    # ==================================================================
    def takeoff_sequence(self) -> bool:
        self.get_logger().info(
            f"[djitellopy] start_request ricevuto: assestamento {PRE_TAKEOFF_SETTLE_S}s prima del takeoff..."
        )
        time.sleep(PRE_TAKEOFF_SETTLE_S)

        self.get_logger().info("[djitellopy] invio takeoff...")
        try:
            self.drone.takeoff()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] comando takeoff fallito: {e}")
            return False

        # SICUREZZA: il drone e' fisicamente in volo da qui in poi (takeoff()
        # ha gia' ricevuto un "ok"). flight_ready va marcato SUBITO, PRIMA
        # della conferma quota sotto: se la conferma fallisse, il drone
        # resterebbe comunque considerato in volo e verra' SEMPRE atterrato.
        self._landing_started = False
        self.flight_ready = True
        self._publish_flight_state()

        deadline = time.monotonic() + TAKEOFF_CONFIRM_TIMEOUT_S
        confirmed = False
        while time.monotonic() < deadline:
            self._poll_djitellopy_state()
            with self._tello_lock:
                alt = self.tello_alt_cm
            if alt is not None and alt > TAKEOFF_MIN_ALT_CM:
                confirmed = True
                break
            time.sleep(0.2)

        if not confirmed:
            self.get_logger().error(
                "[djitellopy] Nessuna variazione di quota plausibile rilevata dopo il takeoff: "
                "verifica manualmente lo stato del drone. Il nodo considera comunque il drone "
                "IN VOLO per sicurezza (verra' atterrato normalmente)."
            )
        else:
            self.get_logger().info(
                f"[djitellopy] decollo confermato (quota>{TAKEOFF_MIN_ALT_CM}cm). "
                f"Attendo {POST_TAKEOFF_SETTLE_S}s di assestamento prima di attivare il controllo..."
            )

        time.sleep(POST_TAKEOFF_SETTLE_S)
        self.get_logger().info("[djitellopy] controllo di posizione ATTIVATO.")
        return True

    def _send_stick_zero(self):
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            self.get_logger().error(f"Errore azzerando i comandi rc via djitellopy: {e}")

    def _send_vel_command(self, vx, vy, vz, wz):
        """Converte [vx,vy,vz,wz] (m/s, rad/s, frame corpo FLU) in valori rc
        -100..100 (NON una conversione cm/s calibrata: send_rc_control
        accetta solo deflessione stick -100..100, senza corrispondenza
        fisica dichiarata dall'SDK).
        La SATURAZIONE vera avviene una volta sola, a monte, in
        policy_action_cb: action clampata [-1,1] * VEL_REF_SCALE, quindi
        [vx,vy,vz,wz] sono gia' bloccati entro i massimi fisici [1,1,1,1.5].
        Qui NON si risatura piu': si normalizza ogni asse rispetto al
        proprio massimo (VEL_REF_SCALE) cosi' che "al valore massimo
        fisico" corrisponda sempre "100% di stick" prima della percentuale
        — altrimenti wz (max 1.5) verrebbe ritagliato scorrettamente a 1.0
        da un clip(-1,1) fisso. RC_SCALE_PCT scala PROPORZIONALMENTE il
        comando finale, con un fattore INDIPENDENTE per asse (es. vx al suo
        massimo -> RC=100 a piena autorita', con RC_SCALE_PCT[0]=0.4
        diventa RC=40). Scelta esplicita dell'utente, DA VALIDARE IN VOLO."""
        forward_backward = int(round(np.clip(vx / VEL_REF_SCALE[0], -1.0, 1.0) * 100 * RC_SCALE_PCT[0]))
        left_right = int(round(np.clip(-vy / VEL_REF_SCALE[1], -1.0, 1.0) * 100 * RC_SCALE_PCT[1]))
        up_down = int(round(np.clip(vz / VEL_REF_SCALE[2], -1.0, 1.0) * 100 * RC_SCALE_PCT[2]))
        yaw = int(round(np.clip(-wz / VEL_REF_SCALE[3], -1.0, 1.0) * 100 * RC_SCALE_PCT[3]))
        try:
            self.drone.send_rc_control(left_right, forward_backward, up_down, yaw)
        except Exception as e:
            self.get_logger().error(f"Errore inviando comando di movimento via djitellopy: {e}")

    def land_sequence(self):
        if self._landing_started:
            return
        self._landing_started = True

        self.flight_ready = False
        self._publish_flight_state()
        self._send_stick_zero()
        time.sleep(0.2)

        self.get_logger().info("[djitellopy] invio land...")
        try:
            self.drone.land()
        except Exception as e:
            self.get_logger().error(f"[djitellopy] comando land fallito: {e}")
            try:
                self.drone.send_command_without_return("land")
            except Exception:
                pass

        try:
            self.drone.end()
        except Exception:
            pass
        self.get_logger().info("[djitellopy] connessione chiusa.")

    def emergency_land(self, reason: str = "richiesta manuale"):
        self.get_logger().error(f"[EMERGENZA] Atterraggio forzato: {reason}")
        self.flight_ready = False
        self.land_sequence()


def terminal_input_loop(node: VelCommandHandler):
    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  vel_command_handler — TERMINALE\n"
        "  start / s        -> decolla (cancello di partenza, solo se connesso\n"
        "                      e non gia' in volo)\n"
        "  q / quit / exit   -> atterra e chiudi il nodo\n"
        "=======================================================================\n"
    )
    while rclpy.ok() and not node._shutdown_requested:
        comando = leggi_comando_terminale()
        if comando is None:
            time.sleep(TERMINAL_POLL_TIMEOUT_S)
            continue
        if comando.lower() in ("q", "quit", "exit"):
            node.get_logger().info("[terminale] Comando 'q' ricevuto: avvio atterraggio e chiusura del nodo.")
            node._shutdown_requested = True
            node.land_sequence()
            rclpy.shutdown()
            break
        elif comando.lower() in ("start", "s"):
            node.start_request_cb(Empty())
        elif comando != "":
            print(f"[terminale] Comando non riconosciuto: '{comando}' ('start'/'s'=decolla, 'q'=atterra)")


def main(args=None):
    # signal_handler_options=NO: gestiamo NOI il SIGINT (rclpy di default puo'
    # "assorbire" il Ctrl+C internamente senza sollevare KeyboardInterrupt,
    # facendo si' che rclpy.spin() ritorni senza che l'atterraggio parta mai).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VelCommandHandler()

    sigint_received = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: sigint_received.set())

    connected = node.connect_sequence()
    if not connected:
        node.get_logger().error("Connessione al drone fallita: chiudo il nodo.")
        node.destroy_node()
        rclpy.shutdown()
        return

    if node.enable_terminal_input:
        input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
        input_thread.start()
    else:
        node.get_logger().info(
            "enable_terminal_input=false: thread stdin interno disattivato "
            "(usa mission_console/i topic /tello/start_request e /tello/land_request)."
        )

    try:
        while rclpy.ok() and not node._shutdown_requested and not sigint_received.is_set():
            rclpy.spin_once(node, timeout_sec=0.5)
        if sigint_received.is_set():
            node.get_logger().warn("Ctrl+C ricevuto: atterraggio di emergenza in corso.")
    finally:
        node._shutdown_requested = True

        try:
            node.land_sequence()
        except Exception as e:
            node.get_logger().error(f"[land] eccezione durante l'atterraggio: {e}")

        if getattr(node.drone, "is_flying", False):
            try:
                node.drone.send_command_without_return("land")
            except Exception:
                pass

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
