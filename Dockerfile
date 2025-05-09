# Use the official Python image from the Docker Hub
FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for Ollama
RUN apt-get update && apt-get install -y curl ca-certificates tar bash && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI via official installer with pip fallback, then verify
RUN curl -fsSL https://ollama.com/install.sh | bash \
    || pip install ollama && \
    # Ensure binary is executable and on PATH
    chmod +x /usr/local/bin/ollama && \
    ln -sf /usr/local/bin/ollama /usr/bin/ollama && \
    ollama --version

# Pulling the embedding model is deferred to container runtime when the Ollama API is running

# Copy the rest of the project files to the container
COPY . .

# Add custom entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the image name
LABEL image_name="cultural-alignment-server"
