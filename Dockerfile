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

# Expose the Flask app port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
