import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def calculate_dir_fingerprint(*, dir_path: Path) -> str:
    payload: dict[str, Any] = {
        "files": [
            {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime,
                "hash": hash_file(file_path)
            }
            for file_path in sorted(dir_path.glob("**/*")) if file_path.is_file()
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()