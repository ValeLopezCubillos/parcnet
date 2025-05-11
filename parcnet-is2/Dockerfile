# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE $PORT

# Usamos shell form para que $PORT se expanda automáticamente
CMD uvicorn webSocket:app --host 0.0.0.0 --port $PORT --log-level debug
