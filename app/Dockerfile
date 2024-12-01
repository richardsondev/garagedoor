# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update && apt install curl -y && apt upgrade -y

# Copy the rest of the application files
COPY ./scripts/monitor.py ./monitor.py
COPY ./scripts/config.json ./config.json
COPY ./model/garage_door_classifier.h5 ./model/garage_door_classifier.h5

# Test the Python script to verify it runs correctly
RUN python monitor.py --test

# Suppress TensorFlow warnings (optional)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Add a health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://127.0.0.1:8080 || exit 1

# Run the monitoring script
CMD ["python", "monitor.py"]
