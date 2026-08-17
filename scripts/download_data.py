from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class DownloadError(RuntimeError):
    """Raised when a dataset archive is unsafe or cannot be verified."""


class BinaryResponse(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> BinaryResponse: ...

    def __exit__(self, *args: object) -> None: ...


@dataclass(frozen=True)
class VocArchive:
    url: str
    filename: str
    md5: str


ARCHIVES = (
    VocArchive(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "VOCtrainval_06-Nov-2007.tar",
        "c52e279531787c972589f7e41ab4ae64",
    ),
    VocArchive(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "VOCtest_06-Nov-2007.tar",
        "b6e924de25625d8de591ea690078ad9f",
    ),
)


def download_archive(
    archive: VocArchive,
    destination: Path,
    *,
    opener: Callable[[str], BinaryResponse] = urllib.request.urlopen,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / archive.filename
    partial = output.with_suffix(output.suffix + ".part")
    if output.is_file() and _md5(output) == archive.md5:
        return output
    partial.unlink(missing_ok=True)
    digest = hashlib.md5()  # noqa: S324 - required by the published VOC checksum
    try:
        with opener(archive.url) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != archive.md5:
            raise DownloadError(
                f"checksum mismatch for {archive.filename}: expected {archive.md5}, got {digest.hexdigest()}"
            )
        partial.replace(output)
        return output
    finally:
        partial.unlink(missing_ok=True)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    try:
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            for member in members:
                member_path = Path(member.name)
                target = (destination / member_path).resolve()
                if (
                    member_path.is_absolute()
                    or not target.is_relative_to(destination_root)
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise DownloadError(f"unsafe tar member: {member.name}")
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise DownloadError(f"cannot read tar member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as exc:
        raise DownloadError(f"cannot extract {archive_path}: {exc}") from exc


def download_voc2007(data_dir: Path) -> tuple[Path, ...]:
    archives_dir = data_dir / "archives"
    downloaded = tuple(download_archive(archive, archives_dir) for archive in ARCHIVES)
    for path in downloaded:
        safe_extract_tar(path, data_dir)
    return downloaded


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required by the published VOC checksum
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify Pascal VOC 2007")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in download_voc2007(args.data_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
