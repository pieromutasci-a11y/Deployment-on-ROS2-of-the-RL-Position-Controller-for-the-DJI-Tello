#!/bin/bash
# Apre una nuova shell nel container attivo

CONTAINER_NAME="ros2_tello_node"

if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Connessione al container attivo: $CONTAINER_NAME..."
    docker exec -it "$CONTAINER_NAME" bash
else
    echo "Errore: Il container '$CONTAINER_NAME' non è in esecuzione."
    echo "Avvialo prima eseguendo ./run.sh"
    exit 1
fi
