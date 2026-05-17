"""
FastAPI-приложение для предсказания оттока.

Запуск локально:
    cd services/ml_service
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

После запуска:
    http://localhost:8000/        - health check
    http://localhost:8000/docs    - Swagger UI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api_handler import FastAPIHandler


# схема входного запроса - FastAPI сам провалидирует типы
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
    version="1.0.0",
)

# создаём один раз при старте, модель живёт всё время работы приложения
handler = FastAPIHandler()


@app.get("/")
def root() -> dict:
    """Health check."""
    return {"Hello": "World"}


@app.post("/api/prediction/{item_id}")
def predict(item_id: int, features: ChurnFeatures) -> dict:
    """Предсказание оттока для клиента по item_id."""
    try:
        result = handler.predict(item_id=item_id, features=features.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")