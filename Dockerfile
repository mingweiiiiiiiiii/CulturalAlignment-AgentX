# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Build arg for external Ollama service host
ARG OLLAMA_HOST=ollama-gpu:11434
ENV OLLAMA_HOST=${OLLAMA_HOST}

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files to the container
COPY . .

# Ollama service provided by external container 'ollama-gpu'

# Serve API endpoints via Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Set the image name
LABEL image_name="cultural-alignment-server"
