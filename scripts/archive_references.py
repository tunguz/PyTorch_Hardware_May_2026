#!/usr/bin/env python3
"""Archive numbered Markdown footnote references into a local folder."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REF_RE = re.compile(r"^\[\^(\d+)\]:\s*(.*?)\s+(https?://\S+)\s*$", re.MULTILINE)
TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def slugify(text: str, limit: int = 72) -> str:
    text = re.sub(r"https?://", "", text.lower())
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:limit].strip("-") or "reference")


def extension_for(content_type: str, url: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        return ".html"
    if media_type.startswith("text/"):
        if url.endswith(".md") or "markdown" in media_type:
            return ".md"
        return ".txt"
    guessed = mimetypes.guess_extension(media_type)
    return guessed or ".bin"


def html_title(body: bytes) -> str:
    match = TITLE_RE.search(body[:300_000])
    if not match:
        return ""
    title = match.group(1)
    title = re.sub(rb"\s+", b" ", title).strip()
    return html.unescape(title.decode("utf-8", errors="replace"))


def fetch(url: str, retries: int = 2) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36 reference-archive"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Encoding": "identity",
    }

    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read()
                return {
                    "ok": 200 <= response.status < 400 and len(body) > 0,
                    "status": response.status,
                    "final_url": response.geturl(),
                    "content_type": response.headers.get("Content-Type", ""),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "title": html_title(body),
                    "body": body,
                    "error": "",
                }
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return {
                "ok": False,
                "status": exc.code,
                "final_url": exc.geturl(),
                "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest() if body else "",
                "title": html_title(body),
                "body": body,
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - archive script should report every URL failure.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    return {
        "ok": False,
        "status": None,
        "final_url": "",
        "content_type": "",
        "bytes": 0,
        "sha256": "",
        "title": "",
        "body": b"",
        "error": last_error,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: archive_references.py REPORT.md OUTPUT_DIR", file=sys.stderr)
        return 2

    report_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    text = report_path.read_text(encoding="utf-8")
    refs = [
        {"number": int(number), "label": label.strip(), "url": url.strip()}
        for number, label, url in REF_RE.findall(text)
    ]
    refs.sort(key=lambda item: item["number"])

    rows: list[dict[str, Any]] = []
    for ref in refs:
        result = fetch(ref["url"])
        ext = extension_for(result["content_type"], result["final_url"] or ref["url"])
        filename = f"ref-{ref['number']:02d}-{slugify(ref['label'])}{ext}"
        path = output_dir / filename
        path.write_bytes(result.pop("body"))
        row = {
            **ref,
            **result,
            "file": filename,
        }
        rows.append(row)
        status = row["status"] if row["status"] is not None else "ERR"
        print(f"{ref['number']:02d} {status} {row['bytes']} {ref['url']}")

    manifest_json = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    summary_md = output_dir / "README.md"

    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "number",
        "label",
        "url",
        "ok",
        "status",
        "final_url",
        "content_type",
        "bytes",
        "sha256",
        "title",
        "file",
        "error",
    ]
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["ok"])
    failures = [row for row in rows if not row["ok"]]
    duplicate_urls = len(rows) - len({row["url"] for row in rows})
    lines = [
        "# Reference Archive",
        "",
        f"- Report: `{report_path.name}`",
        f"- Total numbered references: {len(rows)}",
        f"- Successful fetches: {ok_count}",
        f"- Failed fetches: {len(failures)}",
        f"- Duplicate URL entries preserved: {duplicate_urls}",
        "",
        "## Failed Fetches",
        "",
    ]
    if failures:
        for row in failures:
            lines.append(f"- [^{row['number']}]: status={row['status']} error={row['error']} url={row['url']}")
    else:
        lines.append("None.")
    lines.extend(["", "See `manifest.csv` and `manifest.json` for the full audit."])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
