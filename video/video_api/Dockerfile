# Base image with Python and OpenCV support
FROM python:3.13

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libv4l-dev \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /selfie

# Copy your Python script
COPY face_match.py face_embedding.py cap_self.py /video_api/


# Install Python dependencies
RUN pip install opencv-python

# Default command to run the script
CMD ["python", "capture_selfie.py"]
