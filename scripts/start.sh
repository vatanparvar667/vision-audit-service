#!/usr/bin/env bash
# Container entrypoint: fetch model weights once (needs internet only on the very first start), then serve.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-./models}"
if [ ! -f "$MODELS_DIR/Qwen3-VL-4B-Instruct/config.json" ] || [ ! -f "$MODELS_DIR/rtdetr_v2_r50vd/config.json" ] \
   || [ ! -f "$MODELS_DIR/owlv2-base-patch16-ensemble/config.json" ]; then
  echo "[start] downloading model weights into $MODELS_DIR"
  HF_HUB_OFFLINE=0 python scripts/download_models.py "$MODELS_DIR"
fi
if [ ! -f "$MODELS_DIR/Qwen3-VL-4B-Instruct-bnb4/config.json" ]; then
  echo "[start] saving a pre-quantized 4-bit copy of the VLM (lower RAM peak on every later start)"
  python scripts/quantize_vlm.py "$MODELS_DIR" || echo "[start] quantization skipped, the original weights will be used"
fi

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
