# Dockerfile
# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Sólo instalamos FastAPI y Uvicorn (todo lo demás ya viene preinstalado)
COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT
ENTRYPOINT ["sh","-c"]
CMD ["uvicorn webSocket:app --host 0.0.0.0 --port ${PORT:-8081} --log-level debug"]

