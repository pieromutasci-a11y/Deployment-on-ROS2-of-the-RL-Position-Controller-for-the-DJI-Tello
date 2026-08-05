#!/usr/bin/env python3
"""
Lancia SOLO i nodi che non toccano djitellopy: target_handler,
observation_handler, policy_handler. NESSUNA connessione al drone,
NESSUN motore: utile per testare in sicurezza generazione target,
calcolo osservazioni e inferenza della policy senza far volare (ne'
anche solo armare) il Tello.

mission_console NON e' incluso in questo launch: 'ros2 launch' NON
inoltra lo stdin del terminale ai processi figli (limite noto di ROS2,
verificato empiricamente — nessun input digitato arriva ai nodi lanciati
cosi', nemmeno ad un unico processo senza contesa). Per il controllo
interattivo (wizard di configurazione, avanzamento manuale waypoint),
apri un SECONDO terminale nello stesso container e lancia:

    ros2 run tello_pkg mission_console

target_handler viene lanciato con enable_terminal_input=false: il suo
thread stdin interno non servirebbe comunque a nulla sotto 'ros2
launch' (stesso limite), quindi resta spento per evitare log/thread
inutili — usa sempre mission_console dal secondo terminale.

Gli argomenti da riga di comando impostano i valori INIZIALI dei
parametri (utile per lanci non interattivi/automatizzati); mission_console,
lanciato a parte, puo' comunque cambiarli a runtime col suo wizard:

    ros2 launch tello_pkg no_motors.launch.py target_mode:=hover advance_mode:=auto
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
        DeclareLaunchArgument("target_mode", default_value="variabile",
                               description="singolo | variabile | custom | hover | aruco_target"),
        DeclareLaunchArgument("advance_mode", default_value="manual",
                               description="manual | auto"),
        DeclareLaunchArgument("num_queues", default_value="-1",
                               description="<=0 = infinito"),
        DeclareLaunchArgument("dof_mask_mode", default_value="full",
                               description="full | uniciclo"),

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
