# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies required for the script
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port that Gradio will run on
EXPOSE 7860

# Run the script when the container launches
CMD ["python", "/app/pgv_chatty.py"]
