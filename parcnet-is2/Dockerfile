# Dockerfile
# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Forzamos salida sin buffer
ENV PYTHONUNBUFFERED=1
# Valor por defecto para PORT (Render lo sobreescribe a 10000)
ENV PORT=8081

EXPOSE $PORT

# ENTRYPOINT Shell para que ${PORT} se expanda
ENTRYPOINT ["sh","-c"]
CMD ["uvicorn webSocket:app --host 0.0.0.0 --port ${PORT} --log-level debug"]
