# Use a lightweight Python base
FROM python:3.11-slim

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code (optional; for dev mount, you don't need this)
COPY . .

# Default command: bash (so you can run scripts inside)
CMD ["/bin/bash"]
