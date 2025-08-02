#!/usr/bin/env python3
"""Build docker image
    less BASH more Python

    ©2025, Ovais Quraishi
"""

import configparser
import docker
from pathlib import Path

# override option transformation to preserve case
class CaseSensitiveConfigParser(configparser.RawConfigParser):
    def optionxform(self, optionstr):
            return optionstr  

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
    image_name = "markdown-document-chatbot"
    tag = get_ver()

    build_docker_container(dockerfile_path, image_name, tag)
