"""
Fetches all chess.com games for a given username and saves raw JSON per month
to data/raw/. Already-downloaded months are skipped so re-runs are safe.
"""

import json
import time
import pathlib
import urllib.request
import urllib.error

USERNAME = "ozengnr"
OUTPUT_DIR = pathlib.Path("data/raw")
HEADERS = {"User-Agent": f"chess-archive-fetcher/1.0 (github: {USERNAME})"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def archive_path(archive_url: str) -> pathlib.Path:
    # archive_url ends in /YYYY/MM
    parts = archive_url.rstrip("/").split("/")
    year, month = parts[-2], parts[-1]
    return OUTPUT_DIR / f"{year}_{month}.json"


def fetch_archives(username: str) -> list[str]:
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    data = get(url)
    return data.get("archives", [])


def fetch_month(archive_url: str) -> dict:
    return get(f"{archive_url}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching archive list for {USERNAME}...")
    archives = fetch_archives(USERNAME)
    print(f"  Found {len(archives)} monthly archives")

    skipped = 0
    downloaded = 0

    for url in archives:
        path = archive_path(url)
        if path.exists():
            skipped += 1
            continue

        parts = url.rstrip("/").split("/")
        year, month = parts[-2], parts[-1]
        print(f"  Downloading {year}-{month}...", end=" ", flush=True)

        try:
            data = fetch_month(url)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            n = len(data.get("games", []))
            print(f"{n} games saved to {path}")
            downloaded += 1
            time.sleep(0.5)  # be polite to the API
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} — skipping")

    print(f"\nDone. Downloaded: {downloaded}, skipped (already cached): {skipped}")


if __name__ == "__main__":
    main()
