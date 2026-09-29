# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container at /app
# (.dockerignore keeps secrets/*.py config files and local dev cruft out)
COPY . /app

# The *_config.py.template files contain no secrets (they only read
# os.getenv(...) with safe defaults) -- bake them in as the real config
# modules the app imports; actual values are supplied via env vars/env_file
# at `docker run`/compose time, never baked into the image.
RUN cp pgv_config.py.template pgv_config.py \
    && cp chroma_config.py.template chroma_config.py \
    && cp gdrive_config.py.template gdrive_config.py

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

# Expose the port the app listens on
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

# Run the script when the container launches
CMD ["python", "/app/pgv_chatty.py"]
