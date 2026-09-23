#!/bin/bash
# Build dell'immagine Docker

IMAGE_NAME="ros2_tello_nn:humble"
DOCKERFILE="Dockerfile.tellonode"

echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .
