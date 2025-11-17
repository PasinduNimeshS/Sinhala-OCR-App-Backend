# Use Python slim image with CPU support
FROM python:3.10.6-slim

# Install system deps for opencv/torch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /app (same as project root)
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port
EXPOSE 8080

# CORRECT: app/main.py inside the app/ folder
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]