#!/usr/bin/env python3
"""Nodo 'mission_console' (tello_pkg): console interattiva della pipeline modulare.

Va lanciato a parte ('ros2 run tello_pkg mission_console', secondo terminale):
'ros2 launch' non inoltra lo stdin ai processi figli.
All'avvio un wizard imposta via set_parameters target_mode, advance_mode,
dof_mask_mode, num_queues (e le coordinate se 'custom'); INVIO vuoto = non
modificare. Poi: INVIO = avanza waypoint, l/land = atterra, start/s = decollo.
"""

import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Empty

from tello_pkg.target_handler import VALID_TARGET_MODES, VALID_ADVANCE_MODES, ROOM_MIN, ROOM_MAX
from tello_pkg.observation_handler import DOF_MASKS

# Topic, servizi e timeout
ADVANCE_TOPIC = "/target_handler/advance"
LAND_REQUEST_TOPIC = "/tello/land_request"
START_REQUEST_TOPIC = "/tello/start_request"

TARGET_HANDLER_SET_PARAMS = "/target_handler/set_parameters"
OBSERVATION_HANDLER_SET_PARAMS = "/observation_handler/set_parameters"

SERVICE_WAIT_TIMEOUT_S = 5.0
SERVICE_CALL_TIMEOUT_S = 3.0

DOF_MASK_MODES = tuple(DOF_MASKS.keys())


# Nodo: publisher dei comandi e client set_parameters verso gli altri nodi
class MissionConsole(Node):
    def __init__(self):
        super().__init__("mission_console")

        self.advance_pub = self.create_publisher(Empty, ADVANCE_TOPIC, 10)
        self.land_request_pub = self.create_publisher(Empty, LAND_REQUEST_TOPIC, 10)
        self.start_request_pub = self.create_publisher(Empty, START_REQUEST_TOPIC, 10)

        self.target_handler_client = self.create_client(SetParameters, TARGET_HANDLER_SET_PARAMS)
        self.observation_handler_client = self.create_client(SetParameters, OBSERVATION_HANDLER_SET_PARAMS)

    def advance(self):
        self.advance_pub.publish(Empty())
        self.get_logger().info("[console] INVIO -> /target_handler/advance")

    def request_land(self):
        self.land_request_pub.publish(Empty())
        self.get_logger().warn("[console] land -> /tello/land_request")

    def request_start(self):
        self.start_request_pub.publish(Empty())
        self.get_logger().info("[console] start -> /tello/start_request")

    def set_remote_param(self, client, node_label: str, name: str, value) -> bool:
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_S):
            print(f"  [!] {node_label} non raggiungibile (servizio set_parameters assente), '{name}' NON impostato.")
            return False

        param = Parameter(name, value=value).to_parameter_msg()
        request = SetParameters.Request(parameters=[param])
        future = client.call_async(request)

        deadline = time.monotonic() + SERVICE_CALL_TIMEOUT_S
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)

        if not future.done():
            print(f"  [!] Timeout impostando '{name}' su {node_label}.")
            return False

        result = future.result().results[0]
        if not result.successful:
            print(f"  [!] {node_label} ha rifiutato '{name}={value}': {result.reason}")
            return False

        print(f"  [ok] {node_label}: {name} = {value}")
        return True


# Input da terminale (il prompt e' una riga completa: sotto ros2 launch un prompt senza newline non compare)
def _input_flush(prompt: str) -> str:
    print(prompt, flush=True)
    return sys.stdin.readline().strip()


def ask(prompt: str, choices=None):
    while True:
        raw = _input_flush(prompt)
        if raw == "":
            return None
        if choices is not None and raw not in choices:
            print(f"  Valore non valido, atteso uno tra {choices}. Riprova (INVIO = non modificare).")
            continue
        return raw


def ask_float(prompt: str):
    while True:
        raw = _input_flush(prompt)
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("  Valore non numerico, riprova (INVIO = non modificare).")


def ask_float_bounded(prompt: str, lo: float, hi: float):
    while True:
        raw = _input_flush(prompt)
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  Valore non numerico, riprova (INVIO = non modificare).")
            continue
        if not (lo <= value <= hi):
            print(f"  Valore fuori dai limiti della stanza [{lo:.2f}, {hi:.2f}], riprova (INVIO = non modificare).")
            continue
        return value


# Wizard di configurazione iniziale
def run_setup_wizard(node: MissionConsole):
    print(
        "\n"
        "=======================================================================\n"
        "  mission_console — wizard di configurazione\n"
        "  Per ogni domanda: scrivi un valore e premi INVIO per impostarlo,\n"
        "  oppure premi solo INVIO per lasciare il valore attuale invariato.\n"
        "=======================================================================\n"
    )

    target_mode = ask(f"target_mode {VALID_TARGET_MODES}: ", choices=VALID_TARGET_MODES)
    if target_mode is not None:
        node.set_remote_param(node.target_handler_client, "target_handler", "target_mode", target_mode)

    advance_mode = ask(f"advance_mode {VALID_ADVANCE_MODES}: ", choices=VALID_ADVANCE_MODES)
    if advance_mode is not None:
        node.set_remote_param(node.target_handler_client, "target_handler", "advance_mode", advance_mode)

    dof_mask_mode = ask(f"dof_mask_mode {DOF_MASK_MODES}: ", choices=DOF_MASK_MODES)
    if dof_mask_mode is not None:
        node.set_remote_param(node.observation_handler_client, "observation_handler", "dof_mask_mode", dof_mask_mode)

    if target_mode == "custom":
        print(
            "target_mode='custom': imposta le coordinate del target fisso.\n"
            f"  limiti stanza: x in [{ROOM_MIN[0]:.2f}, {ROOM_MAX[0]:.2f}]m, "
            f"y in [{ROOM_MIN[1]:.2f}, {ROOM_MAX[1]:.2f}]m, "
            f"z in [{ROOM_MIN[2]:.2f}, {ROOM_MAX[2]:.2f}]m"
        )
        x = ask_float_bounded("  custom_target_x [m]: ", ROOM_MIN[0], ROOM_MAX[0])
        if x is not None:
            node.set_remote_param(node.target_handler_client, "target_handler", "custom_target_x", x)
        y = ask_float_bounded("  custom_target_y [m]: ", ROOM_MIN[1], ROOM_MAX[1])
        if y is not None:
            node.set_remote_param(node.target_handler_client, "target_handler", "custom_target_y", y)
        z = ask_float_bounded("  custom_target_z [m]: ", ROOM_MIN[2], ROOM_MAX[2])
        if z is not None:
            node.set_remote_param(node.target_handler_client, "target_handler", "custom_target_z", z)
        yaw = ask_float("  custom_target_yaw [rad]: ")
        if yaw is not None:
            node.set_remote_param(node.target_handler_client, "target_handler", "custom_target_yaw", yaw)

    num_queues_raw = ask("num_queues [intero, <=0 = infinito]: ")
    if num_queues_raw is not None:
        try:
            num_queues = int(num_queues_raw)
            node.set_remote_param(node.target_handler_client, "target_handler", "num_queues", num_queues)
        except ValueError:
            print(f"  [!] '{num_queues_raw}' non e' un intero valido, num_queues NON modificato.")

    print(
        "\n"
        "Configurazione completata.\n"
        "start -> premi INVIO per avviare (decollo se vel_command_handler e' nel launch)\n"
    )
    _input_flush("start: ")
    node.request_start()

    print(
        "\n"
        "=======================================================================\n"
        "  mission_console — comandi disponibili durante l'esecuzione\n"
        "  INVIO (riga vuota) -> avanza al prossimo waypoint\n"
        "  l / land            -> richiede l'atterraggio\n"
        "  start / s           -> ripete il segnale di decollo (se serve)\n"
        "  Ctrl+C              -> chiude il launch (atterra sempre)\n"
        "=======================================================================\n"
    )


# Main: spin in un thread, wizard, poi loop dei comandi
def _spin_until_shutdown(node: MissionConsole):
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MissionConsole()

    spin_thread = threading.Thread(target=_spin_until_shutdown, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_setup_wizard(node)
        while rclpy.ok():
            comando = input().strip()
            if comando == "":
                node.advance()
            elif comando.lower() in ("l", "land"):
                node.request_land()
            elif comando.lower() in ("start", "s"):
                node.request_start()
            else:
                print(f"[console] Comando non riconosciuto: '{comando}' (INVIO=avanza, l/land=atterra, start/s=decolla)")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # Ordine obbligato: shutdown, join dello spin thread, solo poi destroy_node (altrimenti crash rcl)
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
