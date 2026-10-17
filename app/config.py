"""Service settings, read from environment variables (optionally a .env file in the repo root)."""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
# models are loaded from local folders only: the service must work without internet access
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    storage_dir: Path = field(default_factory=lambda: _env_path("STORAGE_DIR", ROOT / "storage"))
    models_dir: Path = field(default_factory=lambda: _env_path("MODELS_DIR", ROOT / "models"))
    device: str = os.environ.get("DEVICE", "cuda")
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "500"))
    # Keep uploaded videos after processing (UI needs them only client-side).
    keep_videos: bool = os.environ.get("KEEP_VIDEOS", "0") == "1"
    use_vlm: bool = os.environ.get("USE_VLM", "1") == "1"


def _api_keys() -> tuple[str, ...]:
    raw = os.environ.get("API_KEYS", "").strip()
    if raw:
        return tuple(k.strip() for k in raw.split(",") if k.strip())
    # No key configured: generate one per process so the API is never open by accident.
    key = secrets.token_urlsafe(24)
    print(f"[config] API_KEYS not set, generated temporary key: {key}", flush=True)
    return (key,)


settings = Settings(api_keys=_api_keys())
settings.storage_dir.mkdir(parents=True, exist_ok=True)
