#!/bin/bash
# ©2025, Ovais Quraishi
# NOTE: assumes that config_vals.txt, with all settings assigned
#		exists

# Define a function for each operation
build_docker() {
  cp setup.config.template setup.config
  sed -i "s|PSQLHOST|$PSQLHOST|" setup.config
  sed -i "s|PSQLDB|$PSQLDB|" setup.config
  sed -i "s|PSQLUSER|$PSQLUSER|" setup.config
  sed -i "s|PSQLPASS|$PSQLPASS|" setup.config
  sed -i "s|OLLAMA_HOST_URL|$OLLAMA_HOST_URL|" setup.config
  sed -i "s|OLLAMA_LLM|$OLLAMA_LLM|" setup.config
  sed -i "s|SEMVER|$SEMVER|" setup.config
  sed -i "s|OTLP_URL|$OTLP_URL|" setup.config
  ./build_docker.py
}

push_image() {

  echo "docker tag ${SERVICE_NAME}:${SEMVER} ${DOCKER_HOST_URI}/${SERVICE_NAME}:${SEMVER}"
  docker tag "${SERVICE_NAME}:${SEMVER}" "${DOCKER_HOST_URI}/${SERVICE_NAME}:${SEMVER}"
  echo "docker push ${DOCKER_HOST_URI}/${SERVICE_NAME}:${SEMVER}"
  docker push "${DOCKER_HOST_URI}/${SERVICE_NAME}:${SEMVER}"
}

apply_kubernetes() {

  #cp deployment.yaml.orig deployment.yaml
  sed -i "s|SEMVER|$SEMVER|" deployment.yaml
  sed -i "s|DOCKER_HOST_URI|$DOCKER_HOST_URI|" deployment.yaml
  kubectl -n ollamagpt apply -f deployment.yaml
  kubectl -n ollamagpt apply -f service.yaml
}

# Read config
source config_vals.txt

# Call the functions with the version as an argument
build_docker $SEMVER
push_image $SEMVER $SERVICE_NAME $DOCKER_HOST_URI
apply_kubernetes $SEMVER
git checkout Dockerfile
