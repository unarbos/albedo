"""config_validation.storage — model repo download backends (HF primary, Hippius option).

Transfer acceleration uses Xet (``HF_XET_HIGH_PERFORMANCE``); set here before huggingface_hub
is first imported. ``s3`` holds Hippius-S3 fingerprint publishing (separate from downloads).
"""
import os

# Xet is the live fast-transfer path on huggingface_hub>=1.0 ('hf_transfer' is inert there).
# Must be set before huggingface_hub is first imported, so the env is read at the right time.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from config_validation.storage.dispatch import (  # noqa: E402
    cache_dir,
    download_config,
    download_full,
    list_files,
    revision_resolves,
)

__all__ = ["cache_dir", "download_config", "download_full", "list_files", "revision_resolves"]
