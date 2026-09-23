"""Deterministic provenance helpers for raw VisDrone diagnostics."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def raw_inputs_sha256(data_root: Path, image_paths: Sequence[Path]) -> str:
    """Hash ordered relative names and bytes for every image/annotation pair."""
    root = data_root.resolve()
    digest = hashlib.sha256()
    for image in sorted((Path(path).resolve() for path in image_paths), key=lambda path: path.name):
        annotation = root / "annotations" / f"{image.stem}.txt"
        for kind, path in (("image", image), ("annotation", annotation)):
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            digest.update(kind.encode("ascii") + b"\0")
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()
