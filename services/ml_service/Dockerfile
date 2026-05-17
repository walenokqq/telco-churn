# slim весит ~150МБ против ~1ГБ у полного образа
FROM python:3.11-slim

WORKDIR /ml_service

# сначала зависимости - слой закешируется и не переустановится при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /ml_service

# реальный файл модели монтируется при docker run -v
VOLUME /models

ENV MODEL_PATH=/models/model.pkl

EXPOSE 8000

# host=0.0.0.0 обязателен, иначе uvicorn слушает только внутри контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# === сборка и запуск ===
# docker build -t telco_churn_service:1 .
# docker run -p 8000:8000 -v $(pwd)/../models:/models telco_churn_service:1