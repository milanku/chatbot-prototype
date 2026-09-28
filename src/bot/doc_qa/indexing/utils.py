import hashlib
import json
from pathlib import Path


def hash_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def calculate_dir_fingerprint(*, dir_path: Path) -> str:
    payload = [
        {
            "path": file_path.relative_to(dir_path).as_posix(),
            "hash": hash_file(file_path),
        }
        for file_path in sorted(dir_path.rglob("*"))
        if file_path.is_file()
    ]

    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
