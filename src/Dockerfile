# syntax=docker/dockerfile:1

FROM ubuntu:latest

LABEL maintainer="Massimo Clementi <massimo_clementi@icloud.com"
LABEL repository="TRIP-vision-perception"
LABEL version="0.1"
LABEL description="Tracked Robotic Intelligent Platform - Vision Perception"

# Update packages and install Python3
RUN apt update
RUN apt install python3 python3-pip -y

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

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
    python3 -m pip install -r requirements.txt
    
# Install PyTorch with ROCm support
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6

# Install the frameworks needed to run opencv properly
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
WORKDIR /app
COPY . .

# Expose the port that the application listens on.
EXPOSE 5000

# Run the application.
CMD python3 Server.py
