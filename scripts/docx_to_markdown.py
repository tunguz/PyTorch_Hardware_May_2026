#!/usr/bin/env python3
"""Extract the canonical Word report into the Markdown source used by builds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DOCX_IN = ROOT / "State of PyTorch Hardware Acceleration May 2026.docx"
MD_OUT = ROOT / "deep-research-report.md"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W, "a": A, "r": R}


def wrap_markdown(text: str, marker: str) -> str:
    if not text or not text.strip():
        return text
    leading = re.match(r"^\s*", text).group(0)
    trailing = re.search(r"\s*$", text).group(0)
    core = text[len(leading) : len(text) - len(trailing) if trailing else len(text)]
    if not core:
        return text
    return f"{leading}{marker}{core}{marker}{trailing}"


def is_enabled(element: etree._Element | None) -> bool:
    if element is None:
        return False
    value = element.get(f"{{{W}}}val")
    return value not in {"0", "false", "False"}


def run_text(run: etree._Element, image_counter: list[int]) -> str:
    parts: list[str] = []
    for child in run:
        name = etree.QName(child).localname
        if name == "t":
            parts.append(child.text or "")
        elif name == "tab":
            parts.append(" ")
        elif name == "br":
            parts.append(" ")
        elif name == "drawing":
            image_counter[0] += 1
            parts.append(f"{{{{IMAGE:{image_counter[0]}}}}}")
    return "".join(parts)


def render_runs(parent: etree._Element, image_counter: list[int], convert_citations: bool = True) -> str:
    rendered: list[str] = []
    for child in parent:
        name = etree.QName(child).localname
        if name == "r":
            text = run_text(child, image_counter)
            if not text:
                continue
            rpr = child.find("w:rPr", NS)
            fonts = rpr.find("w:rFonts", NS) if rpr is not None else None
            font_names = " ".join(fonts.attrib.values()).lower() if fonts is not None else ""
            is_code = "courier" in font_names or "consolas" in font_names or "menlo" in font_names
            if convert_citations:
                text = re.sub(r"\[(\d+)\]", lambda match: f"[^{match.group(1)}]", text)
            if is_code and "{{IMAGE:" not in text:
                text = wrap_markdown(text.replace("`", "\\`"), "`")
            elif rpr is not None:
                if is_enabled(rpr.find("w:b", NS)):
                    text = wrap_markdown(text, "**")
                if is_enabled(rpr.find("w:i", NS)):
                    text = wrap_markdown(text, "*")
            rendered.append(text)
        elif name == "hyperlink":
            rendered.append(render_runs(child, image_counter, convert_citations=convert_citations))
        elif name in {"proofErr", "bookmarkStart", "bookmarkEnd"}:
            continue
        else:
            nested = render_runs(child, image_counter, convert_citations=convert_citations)
            if nested:
                rendered.append(nested)
    text = "".join(rendered)
    text = re.sub(r"(\[\^\d+\])(?=\[\^\d+\])", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def paragraph_style(paragraph: etree._Element) -> str:
    style = paragraph.find("w:pPr/w:pStyle", NS)
    return style.get(f"{{{W}}}val") if style is not None else ""


def paragraph_text(paragraph: etree._Element, image_counter: list[int], convert_citations: bool = True) -> str:
    return render_runs(paragraph, image_counter, convert_citations=convert_citations)


def table_to_markdown(table: etree._Element, image_counter: list[int]) -> str:
    rows: list[list[str]] = []
    for tr in table.findall("w:tr", NS):
        row: list[str] = []
        for tc in tr.findall("w:tc", NS):
            cell_parts: list[str] = []
            for p in tc.findall("w:p", NS):
                text = paragraph_text(p, image_counter)
                if text:
                    cell_parts.append(text)
            cell = " ".join(cell_parts)
            cell = cell.replace("|", "\\|")
            row.append(cell)
        if row:
            rows.append(row)
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def is_citation_only(text: str) -> bool:
    return bool(re.fullmatch(r"(?:\[\^\d+\]\s*)+", text.strip()))


def should_merge(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if is_citation_only(current):
        return True
    prev_plain = re.sub(r"[*_`]", "", previous).rstrip()
    current_plain = re.sub(r"[*_`]", "", current).lstrip()
    if not re.search(r'[.!?:;\]")”’]$', prev_plain):
        return True
    if re.search(r"\b(?:and|or|using|with|for|to|of|like|as|is|plus|includes?)$", prev_plain, re.I):
        return True
    if current_plain and current_plain[0].islower():
        return True
    return False


def merge_paragraphs(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for paragraph in paragraphs:
        if merged and should_merge(merged[-1], paragraph):
            joined = f"{merged[-1].rstrip()} {paragraph.lstrip()}"
            joined = re.sub(r"\*\*([^*]+)\*\* \*\*([^*]+)\*\*", r"**\1 \2**", joined)
            merged[-1] = joined
        else:
            merged.append(paragraph)
    return merged


def clean_reference(text: str) -> str:
    text = re.sub(r"^(?:\*\*)?\[\d+\](?:\*\*)?\s*", "", text.strip())
    text = re.sub(r"(?<!\s)(https?://)", r" \1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_markdown(docx_path: Path) -> str:
    with ZipFile(docx_path) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))

    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("DOCX has no document body")

    blocks: list[str] = []
    pending_paragraphs: list[str] = []
    references: list[str] = []
    image_counter = [0]
    title_seen = False
    skip_front_matter = 0
    in_references = False

    def flush_paragraphs() -> None:
        nonlocal pending_paragraphs
        if pending_paragraphs:
            blocks.extend(merge_paragraphs(pending_paragraphs))
            pending_paragraphs = []

    for child in body:
        name = etree.QName(child).localname
        if name == "sectPr":
            continue
        if name == "tbl":
            if in_references:
                continue
            flush_paragraphs()
            table_md = table_to_markdown(child, image_counter)
            if table_md:
                blocks.append(table_md)
            continue
        if name != "p":
            continue

        style = paragraph_style(child)
        text = paragraph_text(child, image_counter, convert_citations=not in_references)
        if text == "{{IMAGE:1}}":
            continue
        if text == "{{IMAGE:2}}":
            flush_paragraphs()
            blocks.append(
                "![Two-axis map of PyTorch hardware backends by PyTorch-native ergonomics "
                "and datacenter scalability](backend-position-map.png)"
            )
            continue
        if not text:
            continue

        if not title_seen:
            blocks.append(f"# {text}")
            title_seen = True
            skip_front_matter = 2
            continue
        if skip_front_matter:
            skip_front_matter -= 1
            continue

        if style == "Heading1":
            flush_paragraphs()
            if text == "Citation":
                break
            blocks.append(f"## {text}")
            in_references = text == "References"
            continue

        if in_references:
            references.append(clean_reference(text))
        else:
            pending_paragraphs.append(text)

    flush_paragraphs()

    if references:
        blocks.append("\n".join(f"[^{idx}]: {ref}" for idx, ref in enumerate(references, start=1)))

    return "\n\n".join(blocks).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, default=DOCX_IN)
    parser.add_argument("--out", type=Path, default=MD_OUT)
    args = parser.parse_args()

    args.out.write_text(extract_markdown(args.docx), encoding="utf-8")


if __name__ == "__main__":
    main()
