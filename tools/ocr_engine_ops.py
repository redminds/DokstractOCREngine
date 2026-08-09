"""OCR Engine test operator — runs pytest inside the OCR Engine container."""
import json
import subprocess
import sys
import time

DOCKER_COMPOSE_FILE = "docker/docker-compose.yml"
PROJECT_NAME = "dokstract-ocr-engine-local"
SERVICE_NAME = "ocr-engine"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kwargs.get("timeout", 120))


def _compose(service: str, *args: str) -> list[str]:
    return [
        "docker", "compose",
        "-p", PROJECT_NAME,
        "-f", DOCKER_COMPOSE_FILE,
        "exec", "-T", service,
        *args,
    ]


def cmd_freshness() -> dict:
    """Prove OCR Engine source freshness by comparing host vs container fingerprint."""
    import hashlib, os
    from pathlib import Path

    # Host fingerprint: hash all app/*.py files (excluding pycache, tests)
    # Must match Dockerfile RUN command algorithm exactly
    # Use os-independent path matching: normalize to forward slashes
    # Read binary and normalize CRLF→LF for cross-platform consistency
    app_dir = Path("app")
    host_files = sorted(
        p for p in app_dir.rglob("*.py")
        if "__pycache__" not in str(p)
        and "/tests/" not in str(p).replace("\\", "/")
    )
    host_content_bytes = b"".join(
        open(str(f), "rb").read().replace(b"\r\n", b"\n") for f in host_files
    )
    host_fp = hashlib.sha256(host_content_bytes).hexdigest()

    # Container fingerprint: read /app/.source_fingerprint
    result = _run(_compose(SERVICE_NAME, "cat", "/app/.source_fingerprint"))
    container_fp = result.stdout.strip()

    # Image metadata
    result2 = _run(["docker", "inspect", f"{PROJECT_NAME}-{SERVICE_NAME}-1",
                    "--format", "{{.Image}}"])
    image_id = result2.stdout.strip()

    result3 = _run(["docker", "image", "inspect", image_id,
                    "--format", "{{index .Config.Labels \"com.dokstract.ocr-engine.source-fingerprint-file\"}}"])
    label_state = "present" if result3.stdout.strip() else "absent"

    return {
        "host_source_fingerprint": host_fp,
        "container_source_fingerprint": container_fp,
        "image_id": image_id,
        "fingerprint_label": label_state,
        "match": host_fp == container_fp if container_fp else False,
    }


def cmd_test() -> dict:
    """Run pytest in the OCR Engine container."""
    t0 = time.perf_counter()
    result = _run(_compose(
        SERVICE_NAME,
        "python", "-m", "pytest", "tests/",
        "-v", "--tb=short",
        "-p", "no:warnings",
    ), timeout=600)
    elapsed = time.perf_counter() - t0

    parsed = {"ok": result.returncode == 0, "duration_s": round(elapsed, 1)}
    # Parse test summary
    stdout = result.stdout
    for line in stdout.split("\n") + result.stderr.split("\n"):
        if "passed" in line and ("failed" in line or "=" in line):
            parsed["summary"] = line.strip()
            break

    if not parsed["ok"]:
        parsed["errors"] = result.stderr.split("\n")[-20:]

    return parsed


def cmd_test_focused() -> dict:
    """Run focused OCR Engine tests."""
    t0 = time.perf_counter()
    result = _run(_compose(
        SERVICE_NAME,
        "python", "-m", "pytest", "tests/",
        "-v", "--tb=short",
        "-p", "no:warnings",
        "-q",
    ), timeout=300)
    elapsed = time.perf_counter() - t0
    parsed = {"ok": result.returncode == 0, "duration_s": round(elapsed, 1)}
    for line in result.stdout.split("\n"):
        if "passed" in line:
            parsed["summary"] = line.strip()
            break
    return parsed


def cmd_status() -> dict:
    """Check OCR Engine container status."""
    result = _run(["docker", "ps", "--filter", f"name={PROJECT_NAME}",
                   "--format", "json"])
    lines = [l for l in result.stdout.split("\n") if l.strip()]
    containers = [json.loads(l) for l in lines] if lines else []
    return {
        "containers": len(containers),
        "healthy": all(c.get("Status", "").find("healthy") >= 0 or c.get("Status", "").find("Up") >= 0
                       for c in containers),
        "details": [{"name": c.get("Names", ""), "status": c.get("Status", "")} for c in containers],
    }


COMMANDS = {
    "freshness": cmd_freshness,
    "test": cmd_test,
    "test-focused": cmd_test_focused,
    "status": cmd_status,
}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd not in COMMANDS:
        print(json.dumps({"error": f"Unknown command: {cmd}", "available": list(COMMANDS.keys())}))
        sys.exit(1)
    try:
        result = COMMANDS[cmd]()
        print(json.dumps(result, indent=2))
        if not result.get("ok", True):
            sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "command": cmd}))
        sys.exit(1)
