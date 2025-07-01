# Use official Python runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install build tools and Postgres client libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose nothing (ETL runs on schedule or manually)
# Define the default command to run your CLI menu
CMD ["python", "main.py"]
