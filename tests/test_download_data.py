from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from scripts.download_data import DownloadError, VocArchive, download_archive, safe_extract_tar


class BytesResponse(io.BytesIO):
    def __enter__(self) -> BytesResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_checksum_mismatch_leaves_no_archive(tmp_path: Path) -> None:
    archive = VocArchive("https://example.test/voc.tar", "voc.tar", "0" * 32)

    with pytest.raises(DownloadError, match="checksum"):
        download_archive(archive, tmp_path, opener=lambda _: BytesResponse(b"not the expected archive"))

    assert not (tmp_path / "voc.tar").exists()
    assert not (tmp_path / "voc.tar.part").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_unsafe_tar_member_is_rejected(tmp_path: Path, kind: str) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../escape.txt" if kind == "traversal" else "link")
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
            archive.addfile(member)
        else:
            payload = b"escape"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(DownloadError, match="unsafe tar member"):
        safe_extract_tar(archive_path, tmp_path / "extract")

    assert not (tmp_path / "escape.txt").exists()
