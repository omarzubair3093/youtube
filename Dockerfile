# Dockerfile for the YouTube Analyzer Streamlit App

# 1. Start with a specific Python base image.
# This gives us a clean Linux environment with Python pre-installed.
FROM python:3.11-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Install system dependencies (like ffmpeg).
# This is the crucial step. We run it as the root user during the image build.
RUN apt-get update && apt-get install -y ffmpeg && \
    # Clean up the apt cache to keep the image size small
    rm -rf /var/lib/apt/lists/*

# 4. Copy your application files into the container.
COPY requirements.txt .
COPY app.py .

# 5. Install Python packages.
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expose the port that Streamlit will run on.
# This is for documentation and can help with local testing.
EXPOSE 8501

# 7. Define the command to run when the container starts.
# This is the same start command we discussed before.
# Note: We don't need to specify the port here, as Render handles that.
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
