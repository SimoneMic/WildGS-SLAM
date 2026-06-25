#!/bin/bash
NAME=wild_gs_slam
TAG=latest

sudo xhost +
sudo docker run \
     --network=host --privileged \
     -it \
     --rm \
     --gpus all \
     --ipc=host \
     --ulimit memlock=-1 \
     --ulimit stack=67108864 \
     -e ROS_DOMAIN_ID=55 \
     -e DISPLAY=unix${DISPLAY} \
     --device /dev/dri/card0:/dev/dri/card0 \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v /home/ergocub/rosbags:/home/user1/rosbags \
     ${NAME}:${TAG} bash
