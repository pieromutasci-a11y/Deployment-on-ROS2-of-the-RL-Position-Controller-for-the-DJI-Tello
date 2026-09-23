#!/usr/bin/env python3
"""Lancia solo i nodi senza djitellopy: target_handler, observation_handler, policy_handler.

Nessuna connessione al drone: utile per testare in sicurezza generazione target, osservazioni e
inferenza. mission_console va lanciato a parte, in un secondo terminale ('ros2 run tello_pkg
mission_console'), perche' 'ros2 launch' non inoltra lo stdin; per lo stesso motivo target_handler
parte con enable_terminal_input=false. Gli argomenti impostano i valori iniziali dei parametri.

Esempio: ros2 launch tello_pkg no_motors.launch.py target_mode:=hover advance_mode:=auto
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
    ])
