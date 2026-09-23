#!/usr/bin/env python3
"""Lancia la pipeline modulare con interfaccia web: target_handler, observation_handler, policy_handler
(da tello_pkg, invariati) + vel_command_handler_web (tello_pkg_web: unica connessione al drone + dashboard).

Non serve mission_console: parametri, custom target, start/land e avanzamento passano dal browser
su http://<host>:8080/. Il nodo si connette al drone all'avvio ma decolla solo dal tasto
"Avvia algoritmo".

Esempio: ros2 launch tello_pkg_web web_pipeline.launch.py target_mode:=custom dof_mask_mode:=uniciclo
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_mode = LaunchConfiguration("target_mode")
    advance_mode = LaunchConfiguration("advance_mode")
    num_queues = LaunchConfiguration("num_queues")
    dof_mask_mode = LaunchConfiguration("dof_mask_mode")
    save_csv = LaunchConfiguration("save_csv")
    save_plot = LaunchConfiguration("save_plot")

    return LaunchDescription([
        # Argomenti di lancio: valori iniziali dei parametri dei nodi
        DeclareLaunchArgument("target_mode", default_value="variabile",
                               description="singolo | variabile | custom | hover | aruco_target"),
        DeclareLaunchArgument("advance_mode", default_value="manual",
                               description="manual | auto"),
        DeclareLaunchArgument("num_queues", default_value="-1",
                               description="<=0 = infinito"),
        DeclareLaunchArgument("dof_mask_mode", default_value="full",
                               description="full | uniciclo"),
        DeclareLaunchArgument("save_csv", default_value="true"),
        DeclareLaunchArgument("save_plot", default_value="true"),

        # Nodi della pipeline
        Node(
            package="tello_pkg",
            executable="target_handler",
            name="target_handler",
            output="screen",
            parameters=[{
                "target_mode": target_mode,
                "advance_mode": advance_mode,
                "num_queues": num_queues,
                "enable_terminal_input": False,
            }],
        ),
        Node(
            package="tello_pkg",
            executable="observation_handler",
            name="observation_handler",
            output="screen",
            parameters=[{
                "dof_mask_mode": dof_mask_mode,
            }],
        ),
        Node(
            package="tello_pkg",
            executable="policy_handler",
            name="policy_handler",
            output="screen",
        ),
        Node(
            package="tello_pkg_web",
            executable="vel_command_handler_web",
            name="vel_command_handler_web",
            output="screen",
            parameters=[{
                "enable_terminal_input": False,
                "save_csv": save_csv,
                "save_plot": save_plot,
            }],
        ),
    ])
