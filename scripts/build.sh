#!/bin/bash

# Check if the service name is set
if [ -z "${SERVICE_NAME}" ]; then
    echo "Error: Environment variable SERVICE_NAME is not set."
    exit 1
fi

# Get the current git commit SHA
COMMIT_SHA=$(git rev-parse --short HEAD)

# Check if git rev-parse was successful
if [ $? -ne 0 ]; then
    echo "Error: Unable to get the current commit SHA."
    exit 1
fi

# Get uuid lowercase
UUID=$(uuidgen | tr '[:upper:]' '[:lower:]')

# Combine the service name with the commit SHA for the image tag
IMAGE_TAG="${UUID}:${COMMIT_SHA}"

# Save the image tag to a file
echo "$IMAGE_TAG" > ./newest_image_tag.txt

# Build the Docker image
docker buildx build --platform=linux/amd64 -t "$IMAGE_TAG" .
# Check if docker build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
fi

echo "Docker image built successfully: $IMAGE_TAG"
