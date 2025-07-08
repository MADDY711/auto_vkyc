FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk2.0-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    dlib \
    opencv-python \
    imutils \
    scipy

WORKDIR /app
COPY . /app

#RUN wget -nc http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
#    bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

# Run the script
CMD ["python", "spoof_detection.py"]
