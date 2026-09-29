"""Download Ist-Daten monthly archives and the service point list."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

ARCHIVE_URL = (
    "https://archive.opentransportdata.swiss/istdaten/{year}/ist-daten-v2-{year}-{month:02d}.zip"
)

SERVICE_POINT_URL = (
    "https://data.opentransportdata.swiss/dataset/39e5f264-257a-4f3f-bccc-5322c37058c5"
    "/resource/0032bb30-5dd1-4f6e-9f3a-352c5179707e/download/actual-date-swiss-service-point.csv"
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# v2 archives start here. An earlier window would mix two schemas.
V2_START = (2025, 7)

# The data portal rejects the default urllib agent with a 403.
HEADERS = {"User-Agent": "van-hs26-project/1.0 (FHNW student project)"}


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) between two 'YYYY-MM' strings."""
    sy, sm = (int(p) for p in start.split("-"))
    ey, em = (int(p) for p in end.split("-"))
    if (sy, sm) < V2_START:
        raise ValueError(f"v2 archives start at {V2_START[0]}-{V2_START[1]:02d}, got {start}")
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def remote_size(url: str) -> int | None:
    """Content-Length of a URL, or None if the server does not say."""
    try:
        with urlopen(Request(url, method="HEAD", headers=HEADERS), timeout=60) as r:
            value = r.headers.get("Content-Length")
        return int(value) if value else None
    except OSError:
        return None


def download(url: str, target: Path) -> None:
    """Fetch url to target unless a complete copy is already there."""
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    if target.exists() and expected and target.stat().st_size == expected:
        print(f"have   {target.name}")
        return

    print(f"get    {target.name}" + (f"  ({expected / 1e9:.2f} GB)" if expected else ""))
    tmp = target.with_suffix(target.suffix + ".part")
    show_progress = expected and sys.stdout.isatty()
    with urlopen(Request(url, headers=HEADERS), timeout=600) as response, tmp.open("wb") as fh:
        done = 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if show_progress:
                print(f"\r       {100 * done / expected:5.1f}%", end="", flush=True)
    if show_progress:
        print()
    tmp.rename(target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-09", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2026-08", help="last month, YYYY-MM")
    ap.add_argument("--skip-service-points", action="store_true")
    args = ap.parse_args()

    failures = []
    if not args.skip_service_points:
        try:
            download(SERVICE_POINT_URL, RAW / "servicepoints" / Path(SERVICE_POINT_URL).name)
        except OSError as err:
            failures.append(("service points", err))

    for year, month in month_range(args.start, args.end):
        if (year, month) >= (date.today().year, date.today().month):
            print(f"skip   {year}-{month:02d}, month not complete")
            continue
        url = ARCHIVE_URL.format(year=year, month=month)
        try:
            download(url, RAW / "istdaten" / Path(url).name)
        except OSError as err:
            # One bad month should not cost the eleven that follow it.
            print(f"fail   {Path(url).name}: {err}")
            failures.append((Path(url).name, err))

    for name, err in failures:
        print(f"failed: {name}: {err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
