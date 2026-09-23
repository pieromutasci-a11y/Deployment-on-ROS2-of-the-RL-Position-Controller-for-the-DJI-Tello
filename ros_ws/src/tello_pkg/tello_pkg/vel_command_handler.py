#!/usr/bin/env python3
"""Nodo 'vel_command_handler' (tello_pkg): attuazione, unico nodo con connessione djitellopy al drone.

Converte /tello/policy_action (Twist, azione clampata [-1, 1]) in comandi RC:
VEL_REF_SCALE -> x100 -> RC_SCALE_PCT -> send_rc_control (nessun dof_mask sull'azione).
Cancello di partenza: all'avvio si connette e legge la telemetria, ma decolla solo su
/tello/start_request. Pubblica /tello/flight_state; atterra su /tello/land_request, batteria
sotto soglia o Ctrl+C. Watchdog: senza azioni per ACTION_TIMEOUT_S forza l'hover.
Una sola istanza djitellopy per host (porte 8889/8890): non convive con tello_test.
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

# Scala dell'azione [-1, 1] in riferimento fisico, ordine [vx, vy, vz, wz]
VEL_REF_SCALE = np.array([1.0, 1.0, 1.0, 1.5])

# Scala proporzionale del comando RC finale, un valore per asse [vx, vy, vz, wz] (1.0 = piena autorita')
RC_SCALE_PCT = np.array([0.40, 0.40, 0.40, 0.40])

# Watchdog, decollo, telemetria e failsafe batteria
ACTION_TIMEOUT_S = 0.2

PRE_TAKEOFF_SETTLE_S = 2.0
POST_TAKEOFF_SETTLE_S = 3.0
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_CM = 15

TELEMETRY_POLL_PERIOD_S = 1.0
TELLO_LOST_LAND_TIMEOUT_S = 3.0
BATTERY_FAILSAFE_PCT = 15

STATUS_PRINT_PERIOD_S = 1.0
TERMINAL_POLL_TIMEOUT_S = 0.2

# Topic
POLICY_ACTION_TOPIC = "/tello/policy_action"
FLIGHT_STATE_TOPIC = "/tello/flight_state"
LAND_REQUEST_TOPIC = "/tello/land_request"
START_REQUEST_TOPIC = "/tello/start_request"


# Input da terminale (non bloccante)
def leggi_comando_terminale():
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            riga = sys.stdin.readline()
        except Exception:
            return None
        return riga.strip()
    return None


# Nodo: connessione al drone, sottoscrizioni e timer
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

        # IP del drone via parametro tello_ip (Tello in AP mode: 192.168.10.1)
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

    # Azione dalla policy e watchdog di azione scaduta (hover forzato)
    def policy_action_cb(self, msg: Twist):
        with self._action_lock:
            self.last_action = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])
            self.last_action_wall_time = time.monotonic()

        if not self.flight_ready or self._shutdown_requested:
            return

        action = np.clip(self.last_action, -1.0, 1.0)
        target_vel_ref = action * VEL_REF_SCALE
        vx, vy, vz, wz = (float(v) for v in target_vel_ref)
        self._send_vel_command(vx, vy, vz, wz)

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

    # Richiesta di atterraggio e cancello di partenza (start_request)
    def land_request_cb(self, msg: Empty):
        self.get_logger().warn("[land_request] Richiesta di atterraggio ricevuta da un nodo a valle.")
        threading.Thread(
            target=self.emergency_land, args=("land_request esterno",), daemon=True
        ).start()

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

    # Telemetria (batteria, quota) e failsafe
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

    # Sequenze di volo: connessione, decollo, comandi RC, atterraggio
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


# Thread stdin e main (SIGINT custom: atterra sempre)
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
