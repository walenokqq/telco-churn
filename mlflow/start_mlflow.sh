#!/bin/bash
mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5000