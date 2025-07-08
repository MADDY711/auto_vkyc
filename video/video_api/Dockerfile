# Use Python base image
FROM python:3.13

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libv4l-dev \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy contents of video_api/ to /app in the container
COPY . .

# Install required Python packages (optional: include requirements.txt if needed)
RUN pip install opencv-python

# Run your script
CMD ["python", "cap_self.py"]
