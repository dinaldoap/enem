#!/bin/bash

DEVCONTAINER=$(docker ps --all | grep 'vsc-enem' | awk '{print $1}')
docker stop "${DEVCONTAINER}"
docker rm "${DEVCONTAINER}"
docker volume rm 'enem_vscode-server'
docker build --file=.devcontainer/devcontainer.dockerfile .
exit 0