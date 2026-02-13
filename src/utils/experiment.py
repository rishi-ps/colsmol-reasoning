import json
import random
import os
from datetime import datetime

import torch


def create_run_name(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def hf_offline_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}


def ensure_hf_repo_cached(repo_id: str, required_files: list[str]) -> None:
    if not hf_offline_enabled():
        return
    from huggingface_hub import try_to_load_from_cache

    missing = []
    for filename in required_files:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached is None:
            missing.append(filename)
    if missing:
        raise RuntimeError(
            f"Offline mode is enabled but required files are not cached for `{repo_id}`: {missing}"
        )
