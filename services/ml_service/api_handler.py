"""
Обработчик предсказаний - загрузка модели и predict.

Вся ML-логика здесь, FastAPI в main.py только роутит запросы.
"""
import os
import pickle
from typing import Any

import pandas as pd


# в docker монтируется volume /models, локально можно переопределить через env
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")


class FastAPIHandler:
    """Загружает модель и делает предсказания."""

    # порядок и имена должны совпадать с X_train из research.ipynb
    REQUIRED_FEATURES = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Загружает sklearn Pipeline из pickle."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Модель не найдена по пути: {self.model_path}. "
                f"Сначала запусти services/models/get_model.py"
            )
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Модель успешно загружена из {self.model_path}")
        return model

    def _validate_features(self, features: dict) -> None:
        """Проверяет, что во входе есть все нужные признаки."""
        missing = set(self.REQUIRED_FEATURES) - set(features.keys())
        if missing:
            raise ValueError(f"Не хватает признаков: {missing}")

    def predict(self, item_id: int, features: dict) -> dict:
        """Возвращает вероятность оттока для одного клиента."""
        self._validate_features(features)

        # пайплайн ожидает дф с именованными колонками
        X = pd.DataFrame([features])[self.REQUIRED_FEATURES]

        # берём вероятность класса 1 (отток)
        proba = self.model.predict_proba(X)[0, 1]

        return {
            "item_id": item_id,
            "predict": float(proba),
        }