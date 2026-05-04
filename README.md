# Telco Customer Churn — End-to-End ML Project

Pet-проект по прогнозированию оттока клиентов телеком-компании (бинарная классификация). EDA → обучение моделей с MLflow → сервис на FastAPI в Docker → мониторинг через Prometheus + Grafana.

## Стек

Python 3.11, pandas, numpy, scikit-learn, lightgbm, MLflow, FastAPI, Docker, Docker Compose, Prometheus, Grafana, PostgreSQL, SHAP, Optuna.

## Структура проекта

- `data/` — исходный и очищенный датасет (не коммитятся)
- `eda/` — ноутбук с разведочным анализом и графики
- `requirements.txt` — зависимости проекта

## Запуск

Установка и активация виртуального окружения:

```bash
python3.11 -m venv .venv_my_proj
source .venv_my_proj/bin/activate
pip install -r requirements.txt
```

## Результаты EDA

- Датасет содержит 7043 клиента и 21 признак. Целевая переменная — Churn (Yes/No), 27% оттока.
- Очистка: колонка TotalCharges приведена из строки в float, 11 битых значений (клиенты с tenure=0) заполнены нулями.
- Самые сильные предикторы оттока: тип контракта, тип интернета, способ оплаты, защитные допуслуги, срок обслуживания (tenure).
- Шумовые признаки: gender, PhoneService.
- Портрет клиента группы риска: новичок с дорогим тарифом fiber optic, на помесячном контракте, без допуслуг защиты, оплачивает electronic check.