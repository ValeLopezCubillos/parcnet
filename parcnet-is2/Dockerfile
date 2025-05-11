FROM python:3.10-slim

WORKDIR /app

# 1) Instala system-libs antes de pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      libatlas-base-dev \
      libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Copia solo requirements.txt y haz pip install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copia el resto del código
COPY . .

# 4) Exponer y arrancar
EXPOSE  $PORT
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn webSocket:app --host 0.0.0.0 --port ${PORT:-8081}"]

