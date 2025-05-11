# Dockerfile

# 1. Imagen base
FROM python:3.10-slim

# 2. Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Directorio de trabajo
WORKDIR /app

# 4. Copiar y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código
COPY . .

# 6. Exponer el puerto que usa Uvicorn
EXPOSE 8081

# 7. Comando por defecto
CMD ["uvicorn", "webSocket:app", "--host", "0.0.0.0", "--port", "8081"]


