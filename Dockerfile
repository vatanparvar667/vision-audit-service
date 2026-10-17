# GPU image: PyTorch 2.5.1 + CUDA 12.1 (same versions the service was developed and evaluated with)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models \
    STORAGE_DIR=/data

WORKDIR /srv
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
# downloads the weights into $MODELS_DIR on the first start (mount it as a volume), then serves offline
CMD ["bash", "scripts/start.sh"]
