#syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# exponemos el puerto (no es obligatorio para Render, pero sí es buena práctica)
EXPOSE  $PORT

# entrypoint shell para que $PORT se expanda
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn webSocket:app --host 0.0.0.0 --port ${PORT:-8081}"]

