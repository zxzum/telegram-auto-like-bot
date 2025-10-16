FROM python:3.11-slim

WORKDIR /app

# Создаём директорию для сессий
RUN mkdir -p /app/sessions

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота
COPY main.py .

# Запускаем бота
CMD ["python", "main.py"]