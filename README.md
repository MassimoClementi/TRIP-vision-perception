# TRIP-vision-perception
Tracked Robotic Intelligent Platform - Vision Perception

The `stable` branch always represents a snapshot of the most recent tested working version. Relevant commits are also tagged and set as release. The `main` branch instead contains the work-in-progress developments.

## Introduction

This repository contains the code for acquiring images and performing subsequent elaborations. The architecture is structured as clients-to-server and the communication uses RESTful APIs. 

## Configuration

### Configure local Python environment (Conda)

Change directory to source
```
cd src/
```

Create conda environment with Python 3.8.8
```
conda create -n trip-vision-perception python=3.8.8
```

Activate the environment
```
conda activate trip-vision-perception
```

Install all required Python3 packages
```
pip3 install -r requirements.txt
```

Install torch and torchvision with CUDA support. For istance, for CUDA 11.8 use the following command
```
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```


## Run server

If not already activated, activate the conda environment
```
conda activate trip-vision-perception
```

Then run the server by using the following command
```
python3 Server.py
```

For the dockerized version, use the following command
```
docker compose up --build
```

## Run client

Make sure that the conda environment is properly selected
```
conda activate trip-vision-perception
```

Then run the client by using the following command
```
python3 Client.py
```

## Cleanup

If docker is used, it is possible to clean the docker cache content by using the following command:
```
docker system prune -a
```
