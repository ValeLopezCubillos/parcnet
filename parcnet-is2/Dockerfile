# Dockerfile
FROM pytorch/pytorch:2.0.1-cpu

WORKDIR /app

# Copiamos sólo requirements.txt e instalamos FastAPI y Uvicorn
COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Escuchamos en el PORT que Render nos dé
EXPOSE $PORT
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn webSocket:app --host 0.0.0.0 --port ${PORT:-8081} --log-level debug"]


