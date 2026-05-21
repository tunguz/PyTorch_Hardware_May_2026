#!/usr/bin/env python3
"""Archive numbered Markdown footnote references into a local folder."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import mimetypes
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any


REF_RE = re.compile(r"^\[\^(\d+)\]:\s*(.*?)\s+(https?://\S+)\s*$", re.MULTILINE)
TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HTML_REDIRECT_RE = re.compile(
    rb"""(?:url=|href=|location\.replace\()["']?([^"'>)]+)""",
    re.IGNORECASE,
)


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


def html_redirect_target(body: bytes, base_url: str) -> str:
    if len(body) > 20_000:
        return ""
    if b"Redirecting" not in body[:5_000] and b"location.replace" not in body[:5_000]:
        return ""
    match = HTML_REDIRECT_RE.search(body[:10_000])
    if not match:
        return ""
    target = html.unescape(match.group(1).decode("utf-8", errors="replace").strip())
    if not target or target.startswith("javascript:"):
        return ""
    return urllib.parse.urljoin(base_url, target)


def fetch(url: str, retries: int = 2) -> dict[str, Any]:
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 reference-archive"
    )

    for attempt in range(retries + 1):
        with tempfile.NamedTemporaryFile(delete=False) as body_file:
            body_path = Path(body_file.name)
        curl = [
            "curl",
            "--location",
            "--silent",
            "--show-error",
            "--compressed",
            "--max-time",
            "75",
            "--retry",
            "1",
            "--user-agent",
            user_agent,
            "--output",
            str(body_path),
            "--write-out",
            "\n%{http_code}\n%{url_effective}\n%{content_type}\n",
            url,
        ]
        completed = subprocess.run(curl, text=True, capture_output=True, check=False)
        body = body_path.read_bytes() if body_path.exists() else b""
        body_path.unlink(missing_ok=True)

        write_out = completed.stdout.splitlines()
        status_text = write_out[-3] if len(write_out) >= 3 else ""
        final_url = write_out[-2] if len(write_out) >= 2 else ""
        content_type = write_out[-1] if len(write_out) >= 1 else ""
        try:
            status = int(status_text)
        except ValueError:
            status = None

        target = html_redirect_target(body, final_url or url)
        if target and target != url:
            return fetch(target, retries=retries)

        ok = completed.returncode == 0 and status is not None and 200 <= status < 400 and len(body) > 0
        if ok or attempt == retries:
            return {
                "ok": ok,
                "status": status,
                "final_url": final_url,
                "content_type": content_type,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest() if body else "",
                "title": html_title(body),
                "body": body,
                "error": completed.stderr.strip(),
            }

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
    for old_file in output_dir.glob("ref-*"):
        old_file.unlink()

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
