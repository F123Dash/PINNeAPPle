# Physics AI Model Gateway -- build from the repository root:
#   docker build -f apps/model_gateway/Dockerfile -t model-gateway .
#   docker run -p 8114:8114 -v models_data:/data model-gateway
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir ".[surrogate]" -r apps/model_gateway/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/model_gateway:/srv/apps/_shared
WORKDIR /srv/apps/model_gateway
EXPOSE 8114
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8114/health')"
# one worker: the run log and the model cache live in one process; a 1 GB container holds torch and a few models
CMD ["uvicorn", "model_gateway.api:app", "--host", "0.0.0.0", "--port", "8114", "--workers", "1"]
