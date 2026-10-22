"""Bosch 데이터 압축 해제 및 검증.

외부 zip(bosch-production-line-performance.zip) -> 내부 zip 7개 -> CSV 7개.
각 단계의 파일 크기와 SHA-256을 JSON으로 기록한다. 원본 zip은 삭제하지 않는다.
사용: python -I scripts/extract_bosch.py DATA/raw/bosch
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main(root: Path) -> None:
    outer = root / "bosch-production-line-performance.zip"
    manifest = {"outer_zip": {"name": outer.name, "bytes": outer.stat().st_size, "sha256": sha256(outer)},
                "inner_zips": [], "csv": []}

    with zipfile.ZipFile(outer) as z:
        names = z.namelist()
        print("outer zip 내용:", names, flush=True)
        z.extractall(root)

    for name in sorted(names):
        inner = root / name
        manifest["inner_zips"].append({"name": name, "bytes": inner.stat().st_size, "sha256": sha256(inner)})
        with zipfile.ZipFile(inner) as z:
            for info in z.infolist():
                print(f"{name} -> {info.filename} ({info.file_size:,} bytes)", flush=True)
                z.extract(info, root)
                csv = root / info.filename
                with csv.open("rb") as f:
                    header = f.readline().decode("utf-8", "replace").rstrip("\n")
                manifest["csv"].append({
                    "name": info.filename,
                    "bytes": csv.stat().st_size,
                    "sha256": sha256(csv),
                    "n_columns": header.count(",") + 1,
                    "header_head": header[:120],
                })

    out = root / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("완료:", out, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
