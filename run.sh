#!/bin/bash
# Avvia il container Docker con ros_ws mappato in /ros_workspace

IMAGE_NAME="ros2_tello_nn:humble"
CONTAINER_NAME="ros2_tello_node"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ROS_WS="$SCRIPT_DIR/ros_ws"

mkdir -p "$HOST_ROS_WS"

# Inoltro X11 per le GUI
xhost +local:root 2>/dev/null || true

echo "Avvio container Docker '$CONTAINER_NAME'..."
echo "Mappatura Volume Host: $HOST_ROS_WS -> Container: /ros_workspace"

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --net=host \
    --ipc=host \
    --privileged \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOST_ROS_WS:/ros_workspace" \
    "$IMAGE_NAME"
