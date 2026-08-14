from __future__ import annotations

from loguru import logger

from miner import commit, upload, validate

STEPS = [
    ("validate_local", "Validate (local model)"),
    ("upload", "Upload model"),
    ("check_model", "Check uploaded repo"),
    ("registered", "Hotkey registered on subnet"),
    ("commit", "Commit on-chain"),
]


def _why(res: dict) -> str:
    fails = [f"{k}: {v['reason']}" for k, v in res.items() if not v["ok"]]
    return "; ".join(fails) if fails else "ok"


def _cli_confirm(preview_text: str) -> bool:
    print(preview_text)
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def run(
    *,
    path: str,
    namespace: str,
    name: str,
    coldkey: str,
    hotkey: str,
    netuid: int,
    network: str,
    on_step=None,
    log=None,
    confirm=None,
    assume_yes: bool = False,
    skip_commit: bool = False,
):
    on_step = on_step or (lambda *a: None)
    log = log or (lambda m: None)
    logger.info(f"publish pipeline: {namespace}/{name} on netuid {netuid} ({network})")

    logger.info("step 1/5 — validate local model")
    on_step("validate_local", "running", "")
    ok, res = validate.validate_local(path)
    for k, v in res.items():
        log(f"{k}: {'OK' if v['ok'] else v['reason']}")
    on_step("validate_local", "ok" if ok else "fail", _why(res))
    if not ok:
        return False, None

    logger.info("step 2/5 — upload model")
    repo = upload.make_repo(namespace, name)
    on_step("upload", "running", repo)
    ref = upload.upload_model(path, repo)
    log(f"uploaded {ref.immutable_ref}")
    on_step("upload", "ok", ref.immutable_ref)

    logger.info("step 3/5 — check uploaded repo")
    on_step("check_model", "running", "")
    ok, res = validate.validate_remote(ref.repo, ref.digest)
    for k, v in res.items():
        log(f"{k}: {'OK' if v['ok'] else v['reason']}")
    on_step("check_model", "ok" if ok else "fail", _why(res))
    if not ok:
        return False, ref

    logger.info("step 4/5 — check hotkey registration")
    on_step("registered", "running", "")
    ss58, reg = commit.registration_check(coldkey, hotkey, netuid, network)
    on_step(
        "registered", "ok" if reg else "fail", f"{ss58} {'registered' if reg else 'NOT registered'}"
    )
    if not reg:
        log(f"hotkey {ss58} is not registered on netuid {netuid}")
        return False, ref

    if skip_commit:
        log("skip-commit: reveal = " + commit.build_reveal(ref))
        return True, ref

    logger.info("step 5/5 — commit on-chain")
    on_step("commit", "running", "")
    text = commit.preview(
        ref, ss58=ss58, coldkey=coldkey, hotkey=hotkey, netuid=netuid, network=network
    )
    proceed = assume_yes or (confirm(text) if confirm else _cli_confirm(text))
    if not proceed:
        on_step("commit", "fail", "aborted by user")
        log("aborted — nothing committed")
        return False, ref
    result = commit.submit(ref, coldkey=coldkey, hotkey=hotkey, netuid=netuid, network=network)
    ok = getattr(result, "success", True)
    on_step(
        "commit",
        "ok" if ok else "fail",
        "on-chain" if ok else str(getattr(result, "message", result)),
    )
    return ok, ref
