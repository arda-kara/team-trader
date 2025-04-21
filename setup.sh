#!/bin/bash

# Setup script for the AI-Augmented Full-Stack Algorithmic Trading Pipeline
# This script sets up the environment and starts the pipeline

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file with your actual credentials and API keys."
    echo "Then run this script again."
    exit 0
fi

# Create necessary directories
mkdir -p data logs

# Build and start the containers
echo "Building and starting the trading pipeline..."
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo "AI-Augmented Full-Stack Algorithmic Trading Pipeline is now running."
echo "Dashboard is available at http://localhost:8000"
echo "API documentation is available at:"
echo "- Data Ingestion: http://localhost:8001/docs"
echo "- Semantic Signal: http://localhost:8002/docs"
echo "- Strategy Generator: http://localhost:8003/docs"
echo "- Execution Engine: http://localhost:8004/docs"
echo "- Risk Management: http://localhost:8005/docs"
echo "- Agentic Oversight: http://localhost:8006/docs"
echo ""
echo "To stop the pipeline, run: docker-compose down"
