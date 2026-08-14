from __future__ import annotations

import urllib.request

from config import Config


def download_run(cfg: Config, run_id: str, files: dict[str, tuple[str, int]]) -> None:
    if not cfg.s3_public:
        raise RuntimeError("DATASET_CREATOR_S3_BASE env var not set (see .env.example)")
    run_dir = cfg.run_dir(run_id)
    run_dir.mkdir(exist_ok=True)
    for name, (uri, size) in files.items():
        dest = run_dir / name
        if dest.exists() and dest.stat().st_size == size:
            continue
        url = cfg.s3_public + uri.removeprefix("s3://albedo/")
        last_err = None
        for _ in range(3):
            try:
                tmp = dest.with_suffix(dest.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                if tmp.stat().st_size != size:
                    raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
                tmp.rename(dest)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err:
            raise RuntimeError(f"download failed for {url}: {last_err}")
