# Dockerfile

FROM python:3.10-slim

# Evitar warnings de locales y optimizar pip
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias de sistema (libsndfile para soundfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

EXPOSE 8081

# Usamos uvicorn con el extra "standard" para soportar WebSockets
CMD ["uvicorn", "webSocket:app", "--host", "0.0.0.0", "--port", "8081"]

