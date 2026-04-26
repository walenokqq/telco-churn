# Telco Customer Churn — End-to-End ML Project

Учебный pet-проект по прогнозированию оттока клиентов телеком-компании (бинарная классификация). Цикл из 4 лабораторных работ: EDA → обучение моделей с MLflow → сервис на FastAPI в Docker → мониторинг через Prometheus + Grafana.

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

TODO: дополняется по мере выполнения лабораторных работ.
