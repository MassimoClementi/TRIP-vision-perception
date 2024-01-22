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


# Run processing server

Make sure that Docker Engine is installed and running.

Then run the server by using the following command
```
docker compose up --build
```


# Run client

If not already activated, activate the conda environment
```
conda activate trip-vision-perception
```

Then run client using Python3
```
python3 Client.py
```
