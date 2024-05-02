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
IMAGE_TAG=$(cat ./newest_image_tag.txt)

# Construct the full image name for GCP Artifact Registry
FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_TAG}"

# Deploy the image to Cloud Run
gcloud run deploy "$SERVICE_NAME" \
--image "$FULL_IMAGE_NAME" \
--region "$REGION" \
--platform managed \
--service-account "$SERVICE_ACCOUNT_EMAIL" \
--project "$PROJECT_ID" \
--port 8080 \
--set-env-vars PROJECT_ID="$PROJECT_ID" \


# Check if gcloud run deploy was successful
if [ $? -ne 0 ]; then
    echo "Error: Cloud Run deployment failed."
    exit 1
fi

echo "Cloud Run service deployed successfully: $SERVICE_NAME"
