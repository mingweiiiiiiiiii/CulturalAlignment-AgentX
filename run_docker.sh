#!/bin/bash

# Build the Docker image
docker build -t cultural-alignment-server .

# Run the Docker container
docker run cultural-alignment-server
