from dataclasses import dataclass


@dataclass(frozen=True)
class DocReference:
    file_name: str
    heading_path: list[str]