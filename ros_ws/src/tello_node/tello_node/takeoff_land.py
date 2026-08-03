#!/usr/bin/env python3
"""
Nodo ROS2 'takeoff_land' (package tello_node): fa ESCLUSIVAMENTE
takeoff/land del drone Tello via tellopy. NESSUNA policy, NESSUN Vicon,
NESSUN cmd_vel, NESSUN comando di movimento (set_pitch/roll/throttle/yaw)
— serve solo a verificare che decollo/atterraggio funzionino sull'hardware
reale prima di provare il controllore di posizione vero e proprio.

Comandi da terminale (thread stdin non bloccante, select()):
    t / takeoff   -> decolla (se non gia' in volo)
    l / land      -> atterra (se in volo)
    q / quit/exit -> atterra se in volo, poi chiude il nodo
Ctrl+C: atterra (se in volo) e chiude.

Failsafe: batteria critica (< BATTERY_FAILSAFE_PCT) durante il volo ->
land automatico, stessa soglia di position_controller_VICON_VERSION.py.
"""

import time
import select
import sys
import threading
import rclpy
from rclpy.node import Node

import tellopy

# -- stessi valori/soglie di position_controller_VICON_VERSION.py --
TELLOPY_CONNECT_TIMEOUT_S = 60.0
TAKEOFF_CONFIRM_TIMEOUT_S = 8.0
TAKEOFF_MIN_ALT_DM = 3            # <-- unita' presunte (decimetri), VERIFICARE
POST_TAKEOFF_SETTLE_S = 3.0
TELLOPY_LAND_WAIT_S = 5.0
BATTERY_FAILSAFE_PCT = 15

STATUS_PRINT_PERIOD_S = 2.0
TERMINAL_POLL_TIMEOUT_S = 0.2


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

        self._tello_lock = threading.Lock()
        self.battery_pct = None
        self.tello_alt_dm = None

        self.flying = False
        self._landing_started = False
        self._shutdown_requested = False

        self.drone = tellopy.Tello()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.tello_flight_data_cb)

        self.get_logger().info("[tellopy] connessione al drone...")
        try:
            self.drone.connect()
            self.drone.wait_for_connection(TELLOPY_CONNECT_TIMEOUT_S)
            self.get_logger().info("[tellopy] connesso.")
        except Exception as e:
            self.get_logger().error(f"[tellopy] connessione fallita: {e}")
            raise

        self.status_timer = self.create_timer(STATUS_PRINT_PERIOD_S, self.status_cb)

        self.get_logger().info(
            "Nodo takeoff_land avviato. Comandi da terminale: "
            "t=decolla | l=atterra | q=atterra e chiudi."
        )

    def status_cb(self):
        with self._tello_lock:
            bat, alt = self.battery_pct, self.tello_alt_dm
        self.get_logger().info(
            f"[status] {'IN VOLO' if self.flying else 'A TERRA'} | "
            f"batteria={bat}% | quota={alt} (decimetri, presunto)"
        )

    def tello_flight_data_cb(self, event, sender, data, **kwargs):
        battery = getattr(data, "battery_percentage", None)
        height = getattr(data, "height", None)
        with self._tello_lock:
            self.battery_pct = battery
            self.tello_alt_dm = height

        if (
            battery is not None
            and battery < BATTERY_FAILSAFE_PCT
            and self.flying
            and not self._landing_started
        ):
            self.get_logger().error(
                f"[tellopy] BATTERIA CRITICA ({battery}%): avvio LAND di emergenza."
            )
            self.land()

    def takeoff(self):
        if self.flying:
            self.get_logger().warn("[takeoff] Gia' in volo, comando ignorato.")
            return

        self.get_logger().info("[tellopy] invio takeoff...")
        try:
            self.drone.takeoff()
        except Exception as e:
            self.get_logger().error(f"[tellopy] comando takeoff fallito: {e}")
            return

        deadline = time.monotonic() + TAKEOFF_CONFIRM_TIMEOUT_S
        confirmed = False
        while time.monotonic() < deadline:
            with self._tello_lock:
                alt = self.tello_alt_dm
            if alt is not None and alt > TAKEOFF_MIN_ALT_DM:
                confirmed = True
                break
            time.sleep(0.2)

        if not confirmed:
            self.get_logger().error(
                "[tellopy] Nessuna variazione di quota plausibile rilevata dopo il "
                "takeoff: verifica manualmente lo stato del drone."
            )
            return

        self.get_logger().info(
            f"[tellopy] decollo confermato (quota>{TAKEOFF_MIN_ALT_DM}). "
            f"Assestamento {POST_TAKEOFF_SETTLE_S}s..."
        )
        time.sleep(POST_TAKEOFF_SETTLE_S)

        self._landing_started = False
        self.flying = True
        self.get_logger().info("[tellopy] IN VOLO.")

    def land(self):
        if not self.flying or self._landing_started:
            self.get_logger().warn("[land] Non in volo (o atterraggio gia' in corso), comando ignorato.")
            return
        self._landing_started = True

        self.get_logger().info("[tellopy] invio land...")
        try:
            self.drone.land()
        except Exception as e:
            self.get_logger().error(f"[tellopy] comando land fallito: {e}")

        time.sleep(TELLOPY_LAND_WAIT_S)

        self.flying = False
        self._landing_started = False
        self.get_logger().info("[tellopy] ATTERRATO.")


def terminal_input_loop(node: TakeoffLand):
    node.get_logger().info(
        "\n"
        "=======================================================================\n"
        "  t / takeoff  -> decolla\n"
        "  l / land     -> atterra\n"
        "  q / quit     -> atterra (se in volo) e chiudi il nodo\n"
        "=======================================================================\n"
    )
    while rclpy.ok() and not node._shutdown_requested:
        comando = leggi_comando_terminale()
        if comando is None:
            time.sleep(TERMINAL_POLL_TIMEOUT_S)
            continue

        cmd = comando.lower()
        if cmd in ("t", "takeoff"):
            node.takeoff()
        elif cmd in ("l", "land"):
            node.land()
        elif cmd in ("q", "quit", "exit"):
            node.get_logger().info(
                "[terminale] Comando 'q' ricevuto: atterraggio (se in volo) e chiusura del nodo."
            )
            node._shutdown_requested = True
            if node.flying:
                node.land()
            rclpy.shutdown()
            break
        else:
            print(f"[terminale] Comando non riconosciuto: '{comando}' (t=decolla, l=atterra, q=chiudi)")


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffLand()

    input_thread = threading.Thread(target=terminal_input_loop, args=(node,), daemon=True)
    input_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruzione richiesta (Ctrl+C): avvio atterraggio (se in volo).")
    finally:
        if node.flying:
            node.land()
        try:
            node.drone.quit()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
