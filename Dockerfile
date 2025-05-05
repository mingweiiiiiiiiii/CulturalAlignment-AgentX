# Use the official Python image from the Docker Hub
FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -sSL https://ollama.com/download.sh | sh

# Pull the mxbai-embed-large model
RUN ollama pull mxbai-embed-large

# Copy the rest of the project files to the container
COPY . .

# Command to run the main script
CMD ["python", "main.py"]

# Set the image name
LABEL image_name="cultural-alignment-server"
