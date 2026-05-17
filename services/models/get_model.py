"""
Выгружает Production-модель из MLflow и сохраняет как pickle.

Сервис не должен зависеть от MLflow в рантайме -
модель просто лежит рядом как файл.
"""
import os
import pickle
import mlflow

# адрес MLflow - должен совпадать с тем, что в start_mlflow.sh
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# run_id Production-модели из research.ipynb
RUN_ID = "57444a75c2c840238e40d2dde43af954"

# второй аргумент mlflow.sklearn.log_model в ноутбуке
ARTIFACT_PATH = "model"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
    print(f"Загружаю модель из MLflow: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель сохранена в: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()