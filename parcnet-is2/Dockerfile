# Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
      libavformat-dev \
      libavcodec-dev \
      libavdevice-dev \
      libavfilter-dev \
      libopus-dev \
      libsrtp2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

CMD ["bash", "-lc", "uvicorn service_webrtc:app --host 0.0.0.0 --port ${PORT:-8081}"]
