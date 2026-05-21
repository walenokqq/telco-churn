"""
Подключение к PostgreSQL и сохранение предсказаний.

Параметры подключения берутся из env (задаются в docker-compose).
"""
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# === Параметры подключения из env ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "telco_churn")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# pool_pre_ping - пингует соединение перед использованием, помогает после перезапуска БД
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


class Client(Base):
    """Соответствует таблице clients."""
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False, default=datetime.utcnow)
    customer_id = Column(String, nullable=False)
    tenure = Column(Integer)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    contract = Column(String)
    internet_service = Column(String)
    payment_method = Column(String)


class Prediction(Base):
    """Соответствует таблице predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False, default=datetime.utcnow)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    client_ts = Column(DateTime, nullable=False)
    predict = Column(Float, nullable=False)


@contextmanager
def get_session():
    """Открывает сессию, коммитит при успехе, роллбэкает при ошибке."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_prediction(features: dict, predict_value: float) -> Optional[int]:
    """Сохраняет запрос и предсказание в БД. При ошибке возвращает None, сервис не падает."""
    try:
        with get_session() as session:
            # 1. Создаём запись клиента
            client = Client(
                customer_id=features["customerID"],
                tenure=features["tenure"],
                monthly_charges=features["MonthlyCharges"],
                total_charges=features["TotalCharges"],
                contract=features["Contract"],
                internet_service=features["InternetService"],
                payment_method=features["PaymentMethod"],
            )
            session.add(client)
            session.flush()  # нужен чтобы получить client.id до commit

            # запись предсказания со ссылкой на client.id
            prediction = Prediction(
                client_id=client.id,
                client_ts=client.ts,
                predict=predict_value,
            )
            session.add(prediction)

            return client.id
    except Exception as e:
        print(f"[DB ERROR] Не удалось сохранить предсказание: {e}")
        return None