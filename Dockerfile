# Early-warning API: ONNX model + FastAPI. Only the serving dependencies go in the image.
FROM python:3.13-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/logagent/__init__.py src/logagent/features.py src/logagent/service.py ./logagent/
COPY models/hdfs_lgbm.onnx models/serving_config.json ./models/
ENV MODEL_DIR=/app/models

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "logagent.service:app", "--host", "0.0.0.0", "--port", "8000"]
