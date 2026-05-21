"""
Точка входа FastAPI-приложения с метриками + записью в БД.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from api_handler import FastAPIHandler
from db import save_prediction


class ChurnFeatures(BaseModel):
    """Признаки клиента для предсказания оттока."""

    customerID: str = Field(..., examples=["7590-VHVEG"])
    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: float = Field(..., examples=[845.5])


app = FastAPI(
    title="Telco Churn Prediction Service",
    description="ML-сервис предсказания оттока клиентов телеком-компании",
    version="3.0.0",  # ← бампаем версию (ЛР4 итерация с БД)
)

handler = FastAPIHandler()

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

predictions_histogram = Histogram(
    name="telco_churn_predictions",
    documentation="Распределение вероятностей оттока, выданных моделью",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


@app.get("/")
def root() -> dict:
    return {"Hello": "World"}


@app.post("/api/prediction/{item_id}")
def predict(item_id: int, features: ChurnFeatures) -> dict:
    """
    Возвращает предсказание оттока для клиента.
    Дополнительно: пишет метрику в Prometheus + сохраняет запрос/ответ в БД.
    """
    try:
        features_dict = features.model_dump()
        result = handler.predict(item_id=item_id, features=features_dict)

        # Метрика для Prometheus
        predictions_histogram.observe(result["predict"])

        # Сохранение в БД (синхронно, но не критично — ошибки БД не валят сервис)
        save_prediction(features_dict, result["predict"])

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")