#!/usr/bin/env python3
"""
Lancia TUTTA la pipeline modulare: target_handler, observation_handler,
policy_handler, vel_command_handler (UNICA connessione djitellopy al
drone).

CANCELLO DI PARTENZA: vel_command_handler si CONNETTE al drone appena
parte (per leggere batteria/stato), ma NON decolla da solo — aspetta un
comando esplicito 'start'.

mission_console NON e' incluso in questo launch: 'ros2 launch' NON
inoltra lo stdin del terminale ai processi figli (limite noto di ROS2,
verificato empiricamente — nessun input digitato arriva ai nodi lanciati
cosi', nemmeno ad un unico processo senza contesa). Per il controllo
interattivo (wizard di configurazione, decollo, avanzamento manuale
waypoint, atterraggio), apri un SECONDO terminale nello stesso
container e lancia:

    ros2 run tello_pkg mission_console

Da li' funzionano tutti i comandi:
    start / s   -> il drone decolla (/tello/start_request)
    INVIO       -> avanza waypoint (/target_handler/advance)
    l / land    -> atterraggio pulito (/tello/land_request)
    Ctrl+C (in QUESTO terminale, quello del launch) -> propaga SIGINT a
                   tutti i processi figli; vel_command_handler ha il suo
                   handler robusto che atterra SEMPRE, anche se non e'
                   mai decollato (land_sequence() e' innocua a terra).

target_handler e vel_command_handler vengono lanciati con
enable_terminal_input=false: il loro thread stdin interno non
servirebbe comunque a nulla sotto 'ros2 launch' (stesso limite),
resta spento per evitare log/thread inutili.

Esempio:
    ros2 launch tello_pkg full_pipeline.launch.py target_mode:=hover advance_mode:=auto
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
        Node(
            package="tello_pkg",
            executable="vel_command_handler",
            name="vel_command_handler",
            output="screen",
            parameters=[{
                "enable_terminal_input": False,
            }],
        ),
    ])
