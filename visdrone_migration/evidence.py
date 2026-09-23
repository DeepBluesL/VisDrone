"""Evidence identities that prevent mixing experiments across changing inputs."""
import importlib.metadata
import hashlib
from pathlib import Path
import platform
import yaml


DEPENDENCY_DISTRIBUTIONS = ("torch", "torchvision", "ultralytics", "numpy", "Pillow")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted((root / "visdrone_migration").glob("*.py"))
    files += [root / "scripts" / "train.py"]
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_fingerprint():
    """Return the exact runtime versions that affect training numerics and APIs."""
    return {
        "python": platform.python_version(),
        "packages": {
            distribution: importlib.metadata.version(distribution)
            for distribution in DEPENDENCY_DISTRIBUTIONS
        },
    }


def data_fingerprint(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    result = {}
    for split in ("train", "val"):
        location = base / config[split]
        if location.is_dir():
            images = sorted(location.glob("*.jpg"))
        else:
            images = [location.parent / line.strip() for line in location.read_text().splitlines() if line.strip()]
        digest = hashlib.sha256()
        for image in images:
            parts = list(image.resolve().parts)
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            digest.update(image.name.encode())
            digest.update(file_sha256(image).encode())
            digest.update(label.read_bytes() if label.exists() else b"<background>")
        result[split] = {"images": len(images), "image_and_label_sha256": digest.hexdigest()}
    return result
