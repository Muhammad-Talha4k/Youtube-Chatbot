# -------------------------------
# Base image
# -------------------------------
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory (must match your local project folder name)
WORKDIR /app

# Copy only necessary files first to leverage Docker's layer caching
COPY requirements.txt .

# Install system dependencies (for ffmpeg, git, and C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files after dependencies are installed
COPY . .

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# Expose the Streamlit default port
EXPOSE 8501

# Start your Streamlit app
CMD ["streamlit", "run", "yt_app.py"]
