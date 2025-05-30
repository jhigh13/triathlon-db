# Use official Python runtime
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose nothing (ETL runs on schedule or manually)
# Define the default command to run your CLI menu
CMD ["python", "main.py"]
