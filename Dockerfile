# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the script is executable
RUN chmod +x /app/run_docker.sh

# Expose the port the app runs on (if any, for example, if it's a web service)
# EXPOSE 8000

# Command to run the application
CMD ["/app/run_docker.sh"]
