FROM python:3.9-slim

WORKDIR /app

# Instalar ffmpeg y dependencias
RUN apt-get update && apt-get install -y ffmpeg gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar el código al contenedor
COPY . /app

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Puerto que usa Uvicorn
EXPOSE 8081

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8081"]


