# Use Debian Bookworm as the base image ©2024, Ovais Quraishi
FROM python:3.12-bookworm

WORKDIR /app

# Debian 12 thing
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"

# Copy necessary files to /app directory
COPY ver.txt /app/
COPY config.py /app/
COPY websearch.py /app/
COPY ui.py /app/
COPY setup.config /app/
COPY requirements.txt /app/
COPY pricing.json /app/
RUN apt -y install curl
RUN apt-get update
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Service config
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run
CMD ["/app/ui.py"]
