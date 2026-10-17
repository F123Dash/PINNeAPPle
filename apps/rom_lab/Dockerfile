# Reduced Order Model Lab -- build from the repository root:
#   docker build -f apps/rom_lab/Dockerfile -t rom-lab .
#   docker run -p 8102:8102 -v rom_data:/data rom-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/rom_lab/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/rom_lab:/srv/apps/_shared
WORKDIR /srv/apps/rom_lab
EXPOSE 8102
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8102/health')"
# Studies run in the shared job queue (ROM_MAX_JOBS at once); state (jobs, uploads, usage) lives in /data.
CMD ["uvicorn", "rom_lab.api:app", "--host", "0.0.0.0", "--port", "8102", "--workers", "1"]
