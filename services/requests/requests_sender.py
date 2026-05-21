"""
Генератор нагрузки - шлёт случайные POST-запросы в ml_service.
~10% запросов намеренно битые, чтобы в метриках были 4xx.
"""
import os
import random
import time
import logging
from typing import Any

import requests


# внутри compose-сети сервис доступен по имени, не по localhost
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_service:8000")
PREDICT_ENDPOINT = f"{ML_SERVICE_URL}/api/prediction"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


GENDER_OPTS = ["Male", "Female"]
YES_NO = ["Yes", "No"]
YES_NO_NOSERVICE = ["Yes", "No", "No phone service"]
YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]
INTERNET = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def generate_random_features() -> dict[str, Any]:
    """Генерирует случайный валидный набор признаков клиента."""
    return {
        "customerID": f"{random.randint(1000, 9999)}-{random.choice(['VHVEG', 'GNVDE', 'QPYBK'])}",
        "gender": random.choice(GENDER_OPTS),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(YES_NO),
        "Dependents": random.choice(YES_NO),
        "tenure": random.randint(0, 72),
        "PhoneService": random.choice(YES_NO),
        "MultipleLines": random.choice(YES_NO_NOSERVICE),
        "InternetService": random.choice(INTERNET),
        "OnlineSecurity": random.choice(YES_NO_NOINTERNET),
        "OnlineBackup": random.choice(YES_NO_NOINTERNET),
        "DeviceProtection": random.choice(YES_NO_NOINTERNET),
        "TechSupport": random.choice(YES_NO_NOINTERNET),
        "StreamingTV": random.choice(YES_NO_NOINTERNET),
        "StreamingMovies": random.choice(YES_NO_NOINTERNET),
        "Contract": random.choice(CONTRACT),
        "PaperlessBilling": random.choice(YES_NO),
        "PaymentMethod": random.choice(PAYMENT),
        "MonthlyCharges": round(random.uniform(18.0, 120.0), 2),
        "TotalCharges": round(random.uniform(18.0, 8500.0), 2),
    }


def send_one_request(item_id: int, broken: bool = False) -> None:
    """Отправляет один POST-запрос. broken=True - убирает случайное поле, чтобы получить 422."""
    features = generate_random_features()

    if broken:
        # Убираем случайное обязательное поле -> Pydantic вернёт 422
        key_to_remove = random.choice(list(features.keys()))
        del features[key_to_remove]

    url = f"{PREDICT_ENDPOINT}/{item_id}"

    try:
        response = requests.post(url, json=features, timeout=5)
        logger.info(
            f"POST {url} → {response.status_code} | "
            f"body: {response.text[:100]}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")


def main() -> None:
    logger.info(f"Starting requests sender, target: {ML_SERVICE_URL}")
    item_id = 0
    while True:
        item_id += 1
        broken = random.random() < 0.1
        send_one_request(item_id, broken=broken)

        sleep_time = random.uniform(0, 5)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()