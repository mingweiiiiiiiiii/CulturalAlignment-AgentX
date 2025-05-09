# filepath: run_docker.ps1
# PowerShell script to build and run the Docker Compose setup

docker-compose build
docker-compose up -d --remove-orphans --force-recreate --build
