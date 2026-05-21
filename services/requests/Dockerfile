# Сервис нагрузки: лёгкий, шлёт HTTP-запросы. Зависимостей минимум.
FROM python:3.11-slim

WORKDIR /requests

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /requests

# По умолчанию шлём в сервис ml_service внутри docker-compose сети.
# Можно переопределить при docker run -e ML_SERVICE_URL=...
ENV ML_SERVICE_URL=http://ml_service:8000

CMD ["python", "requests_sender.py"]

# === Команды для сборки и запуска ===
# docker build -t telco_churn_requests:1 .
# docker run -e ML_SERVICE_URL=http://host.docker.internal:8000 telco_churn_requests:1