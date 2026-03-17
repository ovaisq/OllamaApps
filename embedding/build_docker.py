#!/usr/bin/env python3
"""Build docker image
    less BASH more Python

    ©2025, Ovais Quraishi
"""

import configparser
import docker
from pathlib import Path

from config import get_config as get_setup_config

CONFIG_FILE = 'setup.config'


def get_config():
    """Returns configuration dict from setup.config"""
    try:
        config = get_setup_config()
        return {option: config.get(section, option) 
                for section in config.sections() 
                for option in config.options(section)}
    except FileNotFoundError:
        print(f'{CONFIG_FILE} file not found. Assuming ENV VARS are set up using some other method')
        return {}

def get_ver():
    with open('ver.txt', 'r') as file:
        content = file.read()
    return content.strip()

def build_docker_container(dockerfile_path, image_name, tag="latest", build_args=None):
    """Build docker container
    """

    client = docker.from_env()
    try:
        print(f"Building Docker image {image_name}:{tag} from {dockerfile_path}...")
        _, logs = client.images.build(
            path=dockerfile_path,
            tag=f"{image_name}:{tag}",
            rm=True,
            buildargs=build_args,
            quiet=True
        )

        for log in logs:
            if 'stream' in log:
                print(log['stream'].strip())

        print(f"Docker image {image_name}:{tag} built successfully!")
        get_this_image = client.images.get(f"{image_name}:{tag}")
        #get_this_image.tag(f"{image_name}:latest")

    except docker.errors.BuildError as e:
        print(f"Failed to build Docker image {image_name}:{tag}: {e}")

    except docker.errors.APIError as e:
        print(f"Docker API error while building image {image_name}:{tag}: {e}")

if __name__ == "__main__":
    dockerfile_path = str(Path().absolute())
    image_name = "summarize-web-pages"
    tag = get_ver()

    build_docker_container(dockerfile_path, image_name, tag, get_config())
