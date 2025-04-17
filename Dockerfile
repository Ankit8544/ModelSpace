# Use official Python image as base
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the content of the local project directory to the container
COPY . /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y ffmpeg

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]

