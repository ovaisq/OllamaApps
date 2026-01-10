#!/bin/bash
# ©2025, Ovais Quraishi
# NOTE: assumes that config_vals.txt, with all settings assigned
#		exists

# Define a function for each operation
build_docker() {
  ./build_docker.py
}

push_image() {

  echo "docker tag ${SERVICE_NAME}:${SEMVER} ${DOCKER_HOST_URI}:5000/${SERVICE_NAME}:${SEMVER}"
  docker tag "${SERVICE_NAME}:${SEMVER}" "${DOCKER_HOST_URI}:5000/${SERVICE_NAME}:${SEMVER}"
  echo "docker push ${DOCKER_HOST_URI}:5000/${SERVICE_NAME}:${SEMVER}"
  docker push "${DOCKER_HOST_URI}:5000/${SERVICE_NAME}:${SEMVER}"
}

apply_kubernetes() {

  cp deployment.yaml.orig deployment.yaml
  sed -i "s|SEMVER|$SEMVER|" deployment.yaml
  sed -i "s|DOCKER_HOST_URI|$DOCKER_HOST_URI|" deployment.yaml
  kubectl -n cicd apply -f deployment.yaml
}

cp deployment.yaml.orig deployment.yaml
# Read config
source config_vals.txt
export DOCKER_HOST=ssh://ovais@jenkins-node-1

# Call the functions with the version as an argument
build_docker $SEMVER
push_image $SEMVER $SERVICE_NAME $DOCKER_HOST_URI:5000
apply_kubernetes $SEMVER
#git checkout Dockerfile
