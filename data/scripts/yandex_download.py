#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from tqdm import tqdm

API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def resolve_download_url(public_link: str) -> str:
    params = {"public_key": public_link}
    resp = requests.get(API_URL, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to resolve download URL (status {resp.status_code}): {resp.text}"
        )
    data = resp.json()
    href = data.get("href")
    if not href:
        raise RuntimeError(f"No 'href' in response: {data}")
    return href


def download_file(download_url: str, dst_path: Path, chunk_size: int = 1 << 20) -> None:
    with requests.get(download_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        desc = dst_path.name
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        )

        with dst_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))

        progress.close()


def yadisk_download(public_link: str, dst_dir: Path) -> None:
    dst_dir = Path(dst_dir)
    download_url = resolve_download_url(public_link)
    parsed = urlparse(download_url)
    qs = parse_qs(parsed.query)
    filename = qs["filename"][0]
    filename = unquote(filename)
    dst_path = dst_dir / filename
    download_file(download_url, dst_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    PUBLIC_LINK = "https://disk.360.yandex.ru/d/6LNWs_woE4JWeA"
    DST_DIR = project_root / "data" / "zip"
    yadisk_download(PUBLIC_LINK, DST_DIR)
