"""Command-line interface for the ENEM EDA workflow."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import tempfile
import traceback
import zipfile
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from .version import __version__

DEFAULT_URL = "https://download.inep.gov.br/microdados/microdados_enem_2025.zip"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="enem",
        description="Download, extract, and analyze ENEM microdata.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_URL,
        help="Remote URL or local path to an ENEM ZIP archive.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the extracted files should be written.",
    )
    parser.add_argument(
        "--report",
        default="eda_report.txt",
        help="Path to the generated EDA report.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory used for the download cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive instead of using the cache.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


@lru_cache(maxsize=8)
def _download_to_cache(url: str, cache_dir: str) -> Path:
    """Download the dataset into a cache directory and return the cached
    archive path."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(url).path).name or "enem.zip"
    destination = cache_path / archive_name

    if destination.exists():
        return destination

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        source_path = Path(url).expanduser()
        if source_path.exists():
            temp_path.write_bytes(source_path.read_bytes())
        else:
            with urlopen(url) as response:
                temp_path.write_bytes(response.read())
        shutil.move(str(temp_path), str(destination))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return destination


def download_dataset(
    source: str,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download the archive to the cache and copy it to the requested output
    directory."""
    output_path = Path(output_dir or "data")
    output_path.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_dir or Path(tempfile.gettempdir()) / "enem-cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(source).path).name or "enem.zip"
    cached_archive = _download_to_cache(source, str(cache_root))

    target_archive = output_path / archive_name
    if force_download or not target_archive.exists():
        shutil.copy2(cached_archive, target_archive)

    return target_archive


def extract_archive(archive_path: Path, output_dir: str | Path) -> Path:
    """Extract the first CSV-like file from the archive and return its path."""
    extract_dir = Path(output_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    for candidate in sorted(extract_dir.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".csv", ".txt", ".tsv"}:
            return candidate

    raise FileNotFoundError(
        "No supported data file was found in the extracted archive."
    )


def _read_csv_rows(csv_path: str | Path) -> list[list[str]]:
    """Read a CSV file using a robust decoding fallback for common
    encodings."""
    data_path = Path(csv_path)
    raw_bytes = data_path.read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    handle = io.StringIO(text, newline="")
    sample = text[:4096]
    handle.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        reader = csv.reader(handle, dialect=dialect)
    except csv.Error:
        reader = csv.reader(handle, delimiter=",")

    return [row for row in reader if row]


def generate_eda_report(
    csv_path: str | Path, output_path: str | Path | None = None
) -> str:
    """Generate a simple text-based exploratory data analysis report from a CSV
    file."""
    rows = _read_csv_rows(csv_path)

    if not rows:
        raise ValueError("The input file does not contain any rows.")

    header = rows[0]
    data_rows = rows[1:]

    lines = [
        "EDA Report",
        f"Rows: {len(data_rows)}",
        f"Columns: {len(header)}",
        "Column summary:",
    ]

    for index, column_name in enumerate(header):
        values = [row[index] if index < len(row) else "" for row in data_rows]
        non_empty = [value for value in values if str(value).strip()]
        missing = len(values) - len(non_empty)
        lines.append(f"- {column_name}: missing={missing}, non-empty={len(non_empty)}")

    report_text = "\n".join(lines) + "\n"

    if output_path is not None:
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report_text, encoding="utf-8")

    return report_text


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    try:
        archive_path = download_dataset(
            source=args.source,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
        data_path = extract_archive(archive_path, args.output_dir)
        generate_eda_report(data_path, args.report)
    except Exception as exc:  # pragma: no cover - defensive CLI handling
        print(f"Unable to generate EDA report: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
