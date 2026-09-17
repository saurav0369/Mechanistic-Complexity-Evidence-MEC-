#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reproduction" / "reacher" / "frozen"
MANIFEST = FROZEN / "MEC_PRE_HIDDEN_AUTHORITY_MANIFEST_R2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> list[str]:
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["runtime_files_sha256"]
    failures: list[str] = []

    for name, expected_hash in expected.items():
        path = FROZEN / name
        if not path.exists():
            failures.append(f"missing: {name}")
            continue
        actual = sha256(path)
        if actual != expected_hash:
            failures.append(f"hash mismatch: {name}: {actual} != {expected_hash}")

    return failures


def main() -> int:
    failures = verify()
    if failures:
        print("frozen artifact verification: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("frozen artifact verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
