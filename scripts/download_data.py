#!/usr/bin/env python3
"""Download and verify the original VisDrone2019-DET archives.

The downloader is dependency-free, resumes interrupted HTTP downloads, validates
ZIP CRCs, extracts safely, checks split image/annotation counts, and records a
machine-readable provenance manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
OFFICIAL_REPOSITORY = "https://github.com/VisDrone/VisDrone-Dataset"
ULTRALYTICS_CONFIG = (
    "https://github.com/ultralytics/ultralytics/blob/main/"
    "ultralytics/cfg/datasets/VisDrone.yaml"
)


@dataclass(frozen=True)
class Split:
    name: str
    folder: str
    expected_images: int
    google_drive_id: str
    expected_archive_bytes: int

    @property
    def filename(self) -> str:
        return f"{self.folder}.zip"

    @property
    def urls(self) -> tuple[str, str]:
        # The first URL is the source linked by the official VisDrone repository;
        # the second is the mirror used by Ultralytics' maintained dataset config.
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={self.google_drive_id}&export=download&confirm=t",
            f"{ASSETS_BASE}/{self.filename}",
        )


SPLITS = {
    "train": Split(
        "train", "VisDrone2019-DET-train", 6471,
        "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn", 1549875511,
    ),
    "val": Split(
        "val", "VisDrone2019-DET-val", 548,
        "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59", 81638851,
    ),
    "test-dev": Split(
        "test-dev", "VisDrone2019-DET-test-dev", 1610,
        "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V", 311251787,
    ),
}


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url: str, offset: int, timeout: int):
    headers = {"User-Agent": "VisDrone-reproducible-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def download_resumable(
    url: str,
    destination: Path,
    retries: int = 8,
    timeout: int = 60,
) -> Path:
    """Download *url* to *destination*, preserving a .part file for resumption."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        print(f"[download] using existing {destination.name} ({human_bytes(destination.stat().st_size)})")
        return destination

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with _request(url, offset, timeout) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range")
                if offset and status != 206 and not content_range:
                    # The server ignored Range. Restart without mixing two files.
                    mode = "wb"
                    offset = 0
                else:
                    mode = "ab" if offset else "wb"
                remaining = int(response.headers.get("Content-Length", "0") or 0)
                total = offset + remaining if remaining else 0
                print(
                    f"[download] {destination.name}: resuming at {human_bytes(offset)}"
                    + (f" / {human_bytes(total)}" if total else ""),
                    flush=True,
                )
                started = time.monotonic()
                last_report = started
                downloaded = offset
                with part.open(mode) as output:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        output.write(block)
                        downloaded += len(block)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            elapsed = max(now - started, 0.001)
                            rate = (downloaded - offset) / elapsed
                            pct = f" ({downloaded / total:.1%})" if total else ""
                            print(
                                f"[download] {destination.name}: {human_bytes(downloaded)}{pct}, "
                                f"{human_bytes(int(rate))}/s",
                                flush=True,
                            )
                            last_report = now
            part.replace(destination)
            print(f"[download] completed {destination.name}: {human_bytes(destination.stat().st_size)}")
            return destination
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}") from exc
            delay = min(2 ** (attempt - 1), 30)
            print(f"[download] attempt {attempt}/{retries} failed: {exc}; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _download_range(
    url: str,
    part: Path,
    start: int,
    end: int,
    total: int,
    retries: int,
    timeout: int,
) -> None:
    expected_size = end - start + 1
    part.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == expected_size:
            return
        if have > expected_size:
            invalid = part.with_suffix(part.suffix + f".invalid-{int(time.time())}")
            part.replace(invalid)
            have = 0
        offset = start + have
        headers = {
            "User-Agent": "VisDrone-reproducible-downloader/1.0",
            "Range": f"bytes={offset}-{end}",
        }
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                required_prefix = f"bytes {offset}-{end}/{total}"
                if status != 206 or content_range != required_prefix:
                    raise RuntimeError(
                        f"server rejected byte range {offset}-{end}: "
                        f"status={status}, Content-Range={content_range!r}"
                    )
                with part.open("ab") as output:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        output.write(block)
            if part.stat().st_size == expected_size:
                return
            raise RuntimeError(
                f"short range: expected {expected_size} bytes, got {part.stat().st_size}"
            )
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"range {start}-{end} failed after {retries} attempts") from exc
            delay = min(2 ** (attempt - 1), 20)
            print(
                f"[download] range {start}-{end} attempt {attempt}/{retries} failed: "
                f"{exc}; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def verify_range_source(url: str, total: int, timeout: int = 30) -> None:
    """Fail fast on quota/error pages before scheduling all archive chunks."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VisDrone-reproducible-downloader/1.0",
            "Range": "bytes=0-0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")
        if status != 206 or content_range != f"bytes 0-0/{total}":
            raise RuntimeError(
                f"source does not serve expected byte ranges: "
                f"status={status}, Content-Range={content_range!r}"
            )


def download_segmented(
    url: str,
    destination: Path,
    total: int,
    connections: int = 8,
    retries: int = 8,
    timeout: int = 60,
) -> Path:
    """Download an exact-size file with independently resumable HTTP ranges."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == total:
        print(f"[download] using existing {destination.name} ({human_bytes(total)})")
        return destination
    if destination.exists():
        invalid = destination.with_suffix(destination.suffix + f".invalid-{int(time.time())}")
        destination.replace(invalid)
        print(f"[download] preserved wrong-size archive as {invalid.name}", file=sys.stderr)

    parts_dir = destination.parent / (destination.name + ".parts")
    # Small fixed chunks survive short-lived/proxied connections and remain
    # reusable when the caller changes the concurrency level.
    segment_size = 8 * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, total, segment_size)):
        end = min(start + segment_size - 1, total - 1)
        ranges.append((index, start, end))
    def part_path(start: int, end: int) -> Path:
        # Range boundaries in the name keep resumes correct if --connections changes.
        return parts_dir / f"part-{start:012d}-{end:012d}"
    already = sum(
        min(part_path(start, end).stat().st_size, end - start + 1)
        if part_path(start, end).exists() else 0
        for index, start, end in ranges
    )
    print(
        f"[download] {destination.name}: {len(ranges)} ranges, "
        f"resuming at {human_bytes(already)} / {human_bytes(total)}",
        flush=True,
    )
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(connections, len(ranges))) as pool:
        futures = {
            pool.submit(
                _download_range,
                url,
                part_path(start, end),
                start,
                end,
                total,
                retries,
                timeout,
            ): (index, start, end)
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index, start, end = futures[future]
            future.result()
            elapsed = max(time.monotonic() - started, 0.001)
            current = sum(
                part_path(range_start, range_end).stat().st_size
                if part_path(range_start, range_end).exists() else 0
                for _, range_start, range_end in ranges
            )
            rate = max(current - already, 0) / elapsed
            print(
                f"[download] {destination.name}: range {index + 1}/{len(ranges)} complete; "
                f"{human_bytes(current)} / {human_bytes(total)} at {human_bytes(int(rate))}/s",
                flush=True,
            )

    assembling = destination.with_suffix(destination.suffix + ".assembling")
    with assembling.open("wb") as output:
        for index, start, end in ranges:
            part = part_path(start, end)
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"wrong size for range part: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if assembling.stat().st_size != total:
        raise RuntimeError(f"assembled file has wrong size: {assembling.stat().st_size} != {total}")
    assembling.replace(destination)
    print(f"[download] completed {destination.name}: {human_bytes(total)}", flush=True)
    return destination


def verify_zip(path: Path) -> int:
    """Validate central directory and every entry CRC; return member count."""
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"CRC failure in {path.name}: {bad}")
        return len(archive.infolist())


def _safe_member_path(root: Path, member: str) -> Path:
    # ZIP member names always use '/'. Resolve to reject traversal and absolutes.
    target = (root / Path(*member.split("/"))).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"unsafe ZIP path: {member!r}") from exc
    return target


def extract_resumable(path: Path, raw_dir: Path, expected_folder: str) -> Path:
    """Safely extract members, atomically replacing only incomplete member files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for index, info in enumerate(archive.infolist(), start=1):
            target = _safe_member_path(raw_dir, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            temporary = target.with_name(target.name + ".extracting")
            with archive.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"incomplete extracted member: {info.filename}")
            temporary.replace(target)
            if index % 500 == 0:
                print(f"[extract] {path.name}: {index}/{len(archive.infolist())} members", flush=True)
    extracted = raw_dir / expected_folder
    if not extracted.is_dir():
        raise RuntimeError(f"archive did not create expected folder: {extracted}")
    return extracted


def count_files(folder: Path, extensions: Iterable[str]) -> int:
    suffixes = {ext.lower() for ext in extensions}
    return sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def verify_extracted(folder: Path, expected: int) -> dict[str, int | bool]:
    images_dir = folder / "images"
    annotations_dir = folder / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise RuntimeError(f"missing images or annotations directory under {folder}")
    image_count = count_files(images_dir, (".jpg", ".jpeg", ".png"))
    annotation_count = count_files(annotations_dir, (".txt",))
    if image_count != expected:
        raise RuntimeError(f"{folder.name}: expected {expected} images, found {image_count}")
    if annotation_count != expected:
        raise RuntimeError(f"{folder.name}: expected {expected} annotations, found {annotation_count}")
    return {
        "images": image_count,
        "annotations": annotation_count,
        "matches_expected": True,
    }


def process_split(
    split: Split,
    archives_dir: Path,
    raw_dir: Path,
    extract: bool,
    connections: int,
) -> dict:
    errors = []
    archive_path = archives_dir / split.filename
    source_url = split.urls[0]
    for source_url in split.urls:
        request_url = source_url
        try:
            verify_range_source(request_url, split.expected_archive_bytes)
            archive_path = download_segmented(
                request_url,
                archive_path,
                split.expected_archive_bytes,
                connections=connections,
                retries=4,
            )
            members = verify_zip(archive_path)
            break
        except (RuntimeError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
            print(f"[download] switching mirror for {split.filename}", file=sys.stderr)
    else:
        raise RuntimeError("; ".join(errors))
    print(f"[verify] {archive_path.name}: ZIP CRC OK ({members} members)", flush=True)
    counts = None
    extracted_path = None
    if extract:
        extracted_path = extract_resumable(archive_path, raw_dir, split.folder)
        counts = verify_extracted(extracted_path, split.expected_images)
        print(
            f"[verify] {split.name}: {counts['images']} images, "
            f"{counts['annotations']} annotations",
            flush=True,
        )
    return {
        "split": split.name,
        "url": source_url,
        "alternate_urls": [url for url in split.urls if url != source_url],
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "expected_archive_bytes": split.expected_archive_bytes,
        "archive_sha256": sha256_file(archive_path),
        "zip_crc_ok": True,
        "zip_members": members,
        "expected_images": split.expected_images,
        "extracted_to": str(extracted_path.resolve()) if extracted_path else None,
        "counts": counts,
    }


def write_manifest(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {item["split"]: item for item in results}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            for item in previous.get("archives", []):
                merged.setdefault(item["split"], item)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    payload = {
        "dataset": "VisDrone2019-DET",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "official_repository": OFFICIAL_REPOSITORY,
            "mirror_configuration": ULTRALYTICS_CONFIG,
            "archive_base_url": ASSETS_BASE,
            "note": "Ultralytics release assets mirror the three archives linked by the official repository.",
        },
        "expected_total_images": sum(item.expected_images for item in SPLITS.values()),
        "archives": sorted(merged.values(), key=lambda item: item["split"]),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Dataset directory (default: VisDrone/data).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
        help="Splits to download (default: all three).",
    )
    parser.add_argument("--workers", type=int, default=3, help="Concurrent archive downloads (default: 3).")
    parser.add_argument(
        "--connections",
        type=int,
        default=8,
        help="Resumable HTTP ranges per archive (default: 8).",
    )
    parser.add_argument("--no-extract", action="store_true", help="Validate archives without extracting them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.connections < 1:
        raise SystemExit("--workers and --connections must be at least 1")
    data_dir = args.data_dir.resolve()
    archives_dir = data_dir / "archives"
    raw_dir = data_dir / "raw"
    selected = [SPLITS[name] for name in args.splits]
    print(f"Dataset directory: {data_dir}")
    print("Archives are retained in data/archives for provenance and future verification.")
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as pool:
        futures = {
            pool.submit(
                process_split,
                split,
                archives_dir,
                raw_dir,
                not args.no_extract,
                args.connections,
            ): split
            for split in selected
        }
        for future in concurrent.futures.as_completed(futures):
            split = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"ERROR: {split.name}: {exc}", file=sys.stderr)
                return 1
    manifest = data_dir / "source_provenance.json"
    write_manifest(manifest, results)
    print(f"Wrote provenance manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
