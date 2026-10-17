# Inverse Model Studio -- build from the repository root:
#   docker build -f apps/inverse_studio/Dockerfile -t inverse-studio .
#   docker run -p 8104:8104 -v invs_data:/data inverse-studio
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/inverse_studio/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/inverse_studio:/srv/apps/_shared
WORKDIR /srv/apps/inverse_studio
EXPOSE 8104
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8104/health')"
# Jobs, usage and API keys live in SQLite under /data (mount a volume); the queue is safe with several workers.
CMD ["uvicorn", "inverse_studio.api:app", "--host", "0.0.0.0", "--port", "8104", "--workers", "2"]
