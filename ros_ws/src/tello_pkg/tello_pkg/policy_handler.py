#!/usr/bin/env python3
"""Nodo 'policy_handler' (tello_pkg): inferenza della policy di posizione (Isaac Lab, skrl PPO).

Sottoscrive 'observations' (52 float da observation_handler), fa il forward pass
sull'ultima osservazione e pubblica l'azione clampata in [-1, 1] su /tello/policy_action.
Nessun accesso al drone: scaling fisico e invio RC sono di vel_command_handler.
Se l'osservazione e' piu' vecchia di OBS_TIMEOUT_S smette di pubblicare, cosi' il watchdog di
vel_command_handler scatta invece di vedere azioni "fresche" calcolate su dati congelati.
"""

import math
import os
import threading
import time

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# Costanti, topic e percorso del checkpoint (sovrascrivibile con POLICY_CKPT_PATH)
STEP_DT = 0.04  # 25 Hz, come la frequenza di decisione in Isaac Lab
OBS_SIZE = 52
ACTION_SIZE = 4

OBSERVATIONS_TOPIC = "observations"
POLICY_ACTION_TOPIC = "/tello/policy_action"

OBS_TIMEOUT_S = 0.2

DEFAULT_CKPT_PATH = (
    "/ros_workspace/src/tello_pkg/policy_pos_controller/2026-07-31_13-02-43_ppo_torch/"
    "checkpoints/best_agent.pt"
)
CKPT_PATH = os.environ.get("POLICY_CKPT_PATH", DEFAULT_CKPT_PATH)


# Rete della policy e caricamento del checkpoint skrl (pesi e normalizzazione)
class SkrlMlpPolicy(torch.nn.Module):
    def __init__(self, dims, act=torch.nn.ELU):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(dims):
            layers.append(torch.nn.Linear(a, b))
            if i < len(dims) - 1:
                layers.append(act())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_skrl_policy(ckpt_path, expected_in, expected_out, device="cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: '{ckpt_path}'. Verifica POLICY_CKPT_PATH "
            f"o DEFAULT_CKPT_PATH e come Dockerfile.tellonode/run.sh montano/"
            f"copiano la cartella policy_pos_controller/ nel container."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["policy"]
    running_mean = ckpt["state_preprocessor"]["running_mean"].to(device).float()
    running_var = ckpt["state_preprocessor"]["running_variance"].to(device).float()

    trunk_keys = sorted(
        (k for k in sd if k.startswith("net_container.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    head_key = "policy_layer.weight" if "policy_layer.weight" in sd else "mean_layer.weight"
    head_bias_key = head_key.replace("weight", "bias")

    trunk_dims = [tuple(sd[k].shape[::-1]) for k in trunk_keys]
    head_out, head_in = sd[head_key].shape
    layer_dims = trunk_dims + [(head_in, head_out)]

    in_dim, out_dim = layer_dims[0][0], layer_dims[-1][1]
    assert in_dim == expected_in, f"input_dim={in_dim}, atteso {expected_in}"
    assert out_dim == expected_out, f"output_dim={out_dim}, atteso {expected_out}"

    model = SkrlMlpPolicy(layer_dims).to(device)
    new_sd = {}
    lin_idx = [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]
    for orig, dest in zip(trunk_keys, lin_idx[:-1]):
        base = orig.rsplit(".", 1)[0]
        new_sd[f"net.{dest}.weight"] = sd[f"{base}.weight"]
        new_sd[f"net.{dest}.bias"] = sd[f"{base}.bias"]
    new_sd[f"net.{lin_idx[-1]}.weight"] = sd[head_key]
    new_sd[f"net.{lin_idx[-1]}.bias"] = sd[head_bias_key]
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, running_mean, running_var


# Nodo: inferenza periodica sull'ultima osservazione, con watchdog di osservazione scaduta
class PolicyHandler(Node):
    def __init__(self):
        super().__init__("policy_handler")

        self.policy, self.run_mean, self.run_var = load_skrl_policy(
            CKPT_PATH, expected_in=OBS_SIZE, expected_out=ACTION_SIZE
        )
        self.get_logger().info(f"Policy caricata da: {CKPT_PATH}")

        self._obs_lock = threading.Lock()
        self.latest_obs = None
        self.last_obs_wall_time = None

        self.obs_sub = self.create_subscription(
            Float32MultiArray, OBSERVATIONS_TOPIC, self.obs_cb, 10
        )
        self.action_pub = self.create_publisher(Twist, POLICY_ACTION_TOPIC, 10)

        self.timer = self.create_timer(STEP_DT, self.control_loop)

        self.get_logger().info(
            f"policy_handler avviato. Sottoscritto a '{OBSERVATIONS_TOPIC}', "
            f"pubblico su '{POLICY_ACTION_TOPIC}'."
        )

    def obs_cb(self, msg: Float32MultiArray):
        if len(msg.data) != OBS_SIZE:
            self.get_logger().warn(
                f"Messaggio 'observations' con lunghezza inattesa ({len(msg.data)}, attesa {OBS_SIZE}), scartato."
            )
            return
        with self._obs_lock:
            self.latest_obs = torch.tensor(msg.data, dtype=torch.float32)
            self.last_obs_wall_time = time.monotonic()

    def control_loop(self):
        with self._obs_lock:
            obs = self.latest_obs
            obs_age = (
                time.monotonic() - self.last_obs_wall_time
                if self.last_obs_wall_time is not None
                else float("inf")
            )

        if obs is None or obs_age > OBS_TIMEOUT_S:
            return

        obs_n = (obs - self.run_mean) / torch.sqrt(self.run_var + 1e-8)
        with torch.no_grad():
            action = self.policy(obs_n.unsqueeze(0)).squeeze(0)
        action = action.clamp(-1.0, 1.0)

        msg = Twist()
        msg.linear.x = float(action[0])
        msg.linear.y = float(action[1])
        msg.linear.z = float(action[2])
        msg.angular.z = float(action[3])
        self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
