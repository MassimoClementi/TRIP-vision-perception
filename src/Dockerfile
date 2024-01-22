# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.8.8
FROM python:${PYTHON_VERSION}-slim as base

LABEL maintainer="Massimo Clementi <massimo_clementi@icloud.com"
LABEL repository="TRIP-vision-perception"
LABEL version="0.1"
LABEL description="Tracked Robotic Intelligent Platform - Vision Perception"

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Install curl for debugging purposes
RUN apt-get update && apt-get -y install curl

# Install the frameworks needed to run opencv properly
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Install all necessary packages
# COPY ./requirements.txt .
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 5000

# Run the application.
CMD python3 Server.py
