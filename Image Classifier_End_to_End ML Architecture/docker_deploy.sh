#!/bin/bash

# Stop on errors
set -e

echo "Building Docker image..."
docker build -t cifar10-classifier .

echo "Running Docker container..."
docker run -p 5000:5000 cifar10-classifier

echo "Container is running! Access the API at http://localhost:5000"