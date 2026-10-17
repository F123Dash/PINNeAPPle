# Model Calibration Lab -- build from the repository root:
#   docker build -f apps/calibration_lab/Dockerfile -t calibration-lab .
#   docker run -p 8101:8101 -v calib_data:/data calibration-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/calibration_lab/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/calibration_lab:/srv/apps/_shared
WORKDIR /srv/apps/calibration_lab
EXPOSE 8101
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8101/health')"
# Jobs, usage and API keys live in SQLite under /data (mount a volume); the queue is safe with several workers.
CMD ["uvicorn", "calibration_lab.api:app", "--host", "0.0.0.0", "--port", "8101", "--workers", "2"]
