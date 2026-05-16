#!/usr/bin/env python3
"""Build the May 2026 PyTorch hardware report artifacts."""

from __future__ import annotations

import html
import re
import shutil
import textwrap
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deep-research-report.md"
HTML_OUT = ROOT / "index.html"
README_OUT = ROOT / "README.md"
PDF_OUT = ROOT / "State of PyTorch Hardware Acceleration May 2026.pdf"
INFOGRAPHIC_OUT = ROOT / "infographic.jpeg"
INFOGRAPHIC_SOURCE = ROOT / "assets" / "pytorch-hardware-2026-infographic.jpg"

TITLE = "State of PyTorch Hardware Acceleration: May 2026"
SHORT_TITLE = "PyTorch Hardware 2026"
SUBTITLE = (
    "A comparative technical analysis of NVIDIA CUDA, AMD ROCm, "
    "Google TPU/XLA, and Apple Silicon MPS"
)
AUTHOR = "Bojan Tunguz"
REPORT_URL = "https://tunguz.github.io/PyTorch_Hardware_May_2026/"
REPO_URL = "https://github.com/tunguz/PyTorch_Hardware_May_2026"
OLD_REPORT_URL = "https://tunguz.github.io/PyTorch_Hardware_2025/"
OLD_REPO_URL = "https://github.com/tunguz/PyTorch_Hardware_2025"
REPORT_VERSION = "0.2"
REPORT_DATE = "May 15, 2026"

RATINGS = {
    "NVIDIA CUDA": {
        "Maturity": "High",
        "torch.compile": "High",
        "Kernels": "High",
        "Debugging": "High",
        "Cost Efficiency": "Medium",
    },
    "AMD ROCm": {
        "Maturity": "Medium",
        "torch.compile": "Medium",
        "Kernels": "Medium",
        "Debugging": "Medium",
        "Cost Efficiency": "Medium",
    },
    "Google TPU": {
        "Maturity": "Medium",
        "torch.compile": "Medium",
        "Kernels": "Medium",
        "Debugging": "Medium",
        "Cost Efficiency": "High",
    },
    "Apple Silicon MPS": {
        "Maturity": "Medium",
        "torch.compile": "Low",
        "Kernels": "Low",
        "Debugging": "Medium",
        "Cost Efficiency": "High",
    },
}

SCORES = {"Low": 1, "Medium": 2, "High": 3}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def clean_source(text: str) -> str:
    replacements = {
        "\u2014": " - ",
        "\u2013": " - ",
        "\u2011": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() + "\n"


def split_sections(md_text: str) -> tuple[str, list[tuple[str, str, str]]]:
    match = re.search(r"^## ", md_text, flags=re.M)
    intro = md_text[: match.start()].strip() if match else md_text.strip()
    rest = md_text[match.start() :] if match else ""
    sections: list[tuple[str, str, str]] = []
    for part in re.split(r"(?m)^## ", rest):
        if not part.strip():
            continue
        title, _, body = part.partition("\n")
        sections.append((title.strip(), slugify(title), body.strip()))
    return intro, sections


def add_heading_ids(html_text: str, sections: list[tuple[str, str, str]]) -> str:
    for title, slug, _ in sections:
        escaped_title = html.escape(title)
        html_text = html_text.replace(f"<h2>{escaped_title}</h2>", f'<h2 id="{slug}">{escaped_title}</h2>')
    return html_text


def rating_badge(value: str) -> str:
    css = {"High": "rating-high", "Medium": "rating-medium", "Low": "rating-low"}[value]
    return f'<span class="rating {css}">{value}</span>'


def matrix_table_html() -> str:
    features = ["Maturity", "torch.compile", "Kernels", "Debugging", "Cost Efficiency"]
    columns = list(RATINGS)
    rows = []
    for feature in features:
        cells = "\n".join(f"<td>{rating_badge(RATINGS[col][feature])}</td>" for col in columns)
        rows.append(f"<tr><th>{feature}</th>{cells}</tr>")
    headers = "\n".join(f"<th>{col}</th>" for col in columns)
    return f"""
    <div class="overflow-x-auto">
      <table class="matrix-table">
        <thead><tr><th>Evaluation Area</th>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_infographic() -> None:
    if not INFOGRAPHIC_SOURCE.exists():
        raise FileNotFoundError(f"Missing source infographic: {INFOGRAPHIC_SOURCE}")
    shutil.copyfile(INFOGRAPHIC_SOURCE, INFOGRAPHIC_OUT)


def build_html(md_text: str) -> None:
    intro, sections = split_sections(md_text)
    intro_body = intro.split("\n", 1)[1].strip() if intro.startswith("# ") else intro
    report_html = markdown.markdown(
        intro_body + "\n\n" + "\n\n".join(f"## {t}\n\n{b}" for t, _, b in sections),
        extensions=["tables", "fenced_code", "footnotes"],
    )
    report_html = add_heading_ids(report_html, sections)
    nav = "\n".join(f'<li><a href="#{slug}">{title}</a></li>' for title, slug, _ in sections)
    platforms = list(RATINGS)
    scores = {feature: [SCORES[RATINGS[p][feature]] for p in platforms] for feature in ["Maturity", "torch.compile", "Kernels", "Debugging", "Cost Efficiency"]}

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(TITLE)}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    html {{ scroll-behavior: smooth; }}
    body {{ font-family: Inter, sans-serif; background: #f8fafc; color: #0f172a; }}
    code, pre {{ font-family: "JetBrains Mono", monospace; }}
    .sticky-sidebar {{ position: sticky; top: 6rem; max-height: calc(100vh - 7rem); overflow-y: auto; }}
    .sidebar-link a {{ display: block; padding: .35rem 0 .35rem 1rem; color: #475569; border-left: 1px solid #cbd5e1; font-size: .9rem; }}
    .sidebar-link a:hover {{ color: #4f46e5; border-left: 2px solid #4f46e5; }}
    .report-body {{ font-size: 1rem; line-height: 1.76; color: #334155; }}
    .report-body h2 {{ scroll-margin-top: 6rem; margin-top: 3.5rem; margin-bottom: 1rem; font-size: 1.55rem; line-height: 1.2; font-weight: 800; color: #0f172a; }}
    .report-body p {{ margin: 1rem 0; }}
    .report-body strong {{ color: #0f172a; font-weight: 700; }}
    .report-body code {{ background: #eef2ff; color: #3730a3; padding: .12rem .3rem; border-radius: .25rem; font-size: .9em; }}
    .report-body sup a {{ color: #4f46e5; text-decoration: none; font-weight: 700; }}
    .report-body .footnote {{ margin-top: 2rem; border-top: 1px solid #e2e8f0; padding-top: 1rem; font-size: .88rem; }}
    .report-body .footnote ol {{ padding-left: 1.25rem; }}
    .report-body .footnote li {{ margin: .45rem 0; }}
    .report-body table, .matrix-table {{ width: 100%; border-collapse: collapse; margin: 1.2rem 0; font-size: .92rem; }}
    .report-body th, .report-body td, .matrix-table th, .matrix-table td {{ border: 1px solid #e2e8f0; padding: .85rem; vertical-align: top; }}
    .report-body th, .matrix-table th {{ background: #f1f5f9; color: #0f172a; font-weight: 700; text-align: left; }}
    .report-body tr:nth-child(even), .matrix-table tr:nth-child(even) {{ background: #fbfdff; }}
    .rating {{ display: inline-flex; min-width: 4.8rem; justify-content: center; border-radius: 999px; padding: .25rem .7rem; font-size: .78rem; font-weight: 700; border: 1px solid; }}
    .rating-high {{ background: #dcfce7; color: #166534; border-color: #86efac; }}
    .rating-medium {{ background: #fef3c7; color: #92400e; border-color: #fbbf24; }}
    .rating-low {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}
    @media print {{
      nav, aside, .no-print {{ display: none !important; }}
      body {{ background: white; }}
      main {{ max-width: none !important; }}
      .shadow-sm {{ box-shadow: none !important; }}
    }}
  </style>
</head>
<body>
  <nav class="fixed w-full bg-white/90 backdrop-blur-sm border-b border-slate-200 z-50 top-0">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <i class="fa-solid fa-microchip text-indigo-600 text-2xl"></i>
          <span class="font-bold text-xl tracking-tight text-slate-800">{SHORT_TITLE}</span>
        </div>
        <div class="hidden md:flex space-x-8 text-sm font-medium text-slate-600">
          <a href="#scorecard" class="hover:text-indigo-600 transition">Decision Matrix</a>
          <a href="#deep-dive-on-the-torch-compile-landscape" class="hover:text-indigo-600 transition">Compilation</a>
          <a href="#the-flashattention-test" class="hover:text-indigo-600 transition">Kernels</a>
          <a href="#final-recommendation" class="hover:text-indigo-600 transition">Recommendation</a>
        </div>
      </div>
    </div>
  </nav>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-14">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      <aside class="hidden lg:block lg:col-span-3">
        <div class="sticky-sidebar pr-4">
          <h5 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Contents</h5>
          <ul class="space-y-1 sidebar-link">{nav}</ul>
        </div>
      </aside>
      <main class="lg:col-span-9 space-y-10">
        <header class="border-b border-slate-200 pb-8">
          <p class="text-sm font-mono text-indigo-700 mb-3">Version {REPORT_VERSION} - {REPORT_DATE}</p>
          <h1 class="text-4xl md:text-5xl font-extrabold text-slate-950 tracking-tight mb-4">{TITLE}</h1>
          <p class="text-xl text-slate-600 font-light">{SUBTITLE}</p>
          <p class="text-md text-slate-500 mt-4 font-medium">By {AUTHOR}</p>
        </header>
        <section class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          <img src="infographic.jpeg" alt="May 2026 PyTorch hardware backend scorecard" class="w-full">
        </section>
        <section id="scorecard" class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h2 class="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2"><i class="fa-solid fa-table-cells text-indigo-500"></i> Executive Decision Matrix</h2>
          <p class="text-slate-600 mb-6">The ratings below preserve the practical scoring lens from the 2025 report while updating the findings to the May 15, 2026 source report. They measure how often a senior engineer can stay inside normal PyTorch habits without backend-specific surprises.</p>
          {matrix_table_html()}
        </section>
        <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h2 class="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2"><i class="fa-solid fa-chart-line text-indigo-500"></i> Backend Profile</h2>
          <canvas id="backendChart" height="130"></canvas>
        </section>
        <article class="report-body bg-white rounded-xl shadow-sm border border-slate-200 p-6 md:p-9">
          {report_html}
          <h2 id="citation">Citation</h2>
          <p>This May 2026 report uses the same citation convention as the prior report and cites that prior work as its predecessor:</p>
          <p>Tunguz, B. (2025). <em>State of PyTorch Hardware Acceleration 2025</em>. GitHub Pages. <a href="{OLD_REPORT_URL}">{OLD_REPORT_URL}</a></p>
          <pre><code>@misc{{tunguz2025pytorchhardware,
  author = {{Tunguz, Bojan}},
  title = {{State of PyTorch Hardware Acceleration 2025}},
  year = {{2025}},
  howpublished = {{\\url{{{OLD_REPORT_URL}}}}},
  note = {{GitHub repository: \\url{{{OLD_REPO_URL}}}}}
}}</code></pre>
        </article>
      </main>
    </div>
  </div>
  <script>
    const labels = {platforms!r};
    const datasets = {[
        {"label": feature, "data": values}
        for feature, values in scores.items()
    ]!r};
    const palette = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b"];
    new Chart(document.getElementById("backendChart"), {{
      type: "radar",
      data: {{
        labels,
        datasets: datasets.map((dataset, index) => ({{
          ...dataset,
          borderColor: palette[index],
          backgroundColor: palette[index] + "22",
          pointBackgroundColor: palette[index],
          borderWidth: 2
        }}))
      }},
      options: {{
        responsive: true,
        scales: {{ r: {{ min: 0, max: 3, ticks: {{ stepSize: 1 }} }} }},
        plugins: {{ legend: {{ position: "bottom" }} }}
      }}
    }});
  </script>
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")


def pdf_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def make_para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(pdf_inline(text), style)


def build_pdf(md_text: str) -> None:
    _, sections = split_sections(md_text)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, alignment=TA_CENTER, spaceAfter=12)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=24)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#0f172a"))
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=7, textColor=colors.HexColor("#1f2937"))
    small_style = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.3, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=6)

    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=TITLE,
        author=AUTHOR,
    )
    story = [
        ReportImage(str(INFOGRAPHIC_OUT), width=6.3 * inch, height=9.45 * inch),
        PageBreak(),
        Paragraph(TITLE, title_style),
        Paragraph(SUBTITLE, subtitle_style),
        Paragraph(f"{AUTHOR} - Version {REPORT_VERSION} - {REPORT_DATE}", subtitle_style),
        Spacer(1, 0.08 * inch),
    ]

    for title, _, body in sections:
        story.append(Paragraph(html.escape(title), h2_style))
        table_lines: list[str] = []
        paragraph: list[str] = []
        for line in body.splitlines():
            if line.startswith("|"):
                if paragraph:
                    story.append(make_para(" ".join(paragraph), body_style))
                    paragraph = []
                table_lines.append(line)
                continue
            if table_lines and not line.startswith("|"):
                append_markdown_table(story, table_lines, small_style)
                table_lines = []
            if re.match(r"\[\^\d+\]:", line.strip()):
                if paragraph:
                    story.append(make_para(" ".join(paragraph), body_style))
                    paragraph = []
                story.append(make_para(line.strip(), small_style))
                continue
            if not line.strip():
                if paragraph:
                    story.append(make_para(" ".join(paragraph), body_style))
                    paragraph = []
                continue
            paragraph.append(line.strip())
        if paragraph:
            story.append(make_para(" ".join(paragraph), body_style))
        if table_lines:
            append_markdown_table(story, table_lines, small_style)

    story.append(PageBreak())
    story.append(Paragraph("Citation", h2_style))
    story.append(make_para("This report cites the previous edition using the same citation convention used by the older GitHub repository.", body_style))
    story.append(Paragraph(f"Tunguz, B. (2025). <i>State of PyTorch Hardware Acceleration 2025</i>. GitHub Pages.<br/>{OLD_REPORT_URL}", body_style))
    story.append(make_para("The matching BibTeX entry is included in README.md and in the interactive HTML report.", small_style))

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(0.65 * inch, 0.35 * inch, SHORT_TITLE)
        canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def append_markdown_table(story: list, lines: list[str], style: ParagraphStyle) -> None:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append([Paragraph(pdf_inline(cell), style) for cell in cells])
    if not rows:
        return
    col_count = len(rows[0])
    table = Table(rows, colWidths=[7.2 * inch / col_count] * col_count, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.1 * inch))


def build_readme() -> None:
    readme = f"""# {TITLE}

A comparative technical analysis of the PyTorch hardware acceleration landscape as
of {REPORT_DATE}, covering NVIDIA CUDA, AMD ROCm, Google TPU/XLA, and Apple
Silicon MPS.

Version: {REPORT_VERSION}

## Read the Report

- Interactive report: [tunguz.github.io/PyTorch_Hardware_May_2026]({REPORT_URL})
- PDF version: [State of PyTorch Hardware Acceleration May 2026.pdf](State%20of%20PyTorch%20Hardware%20Acceleration%20May%202026.pdf)
- Source research brief: [deep-research-report.md](deep-research-report.md)
- Infographic: [infographic.jpeg](infographic.jpeg)
- Source infographic: [assets/pytorch-hardware-2026-infographic.jpg](assets/pytorch-hardware-2026-infographic.jpg)

## Repository Contents

- `index.html` - interactive GitHub Pages report
- `State of PyTorch Hardware Acceleration May 2026.pdf` - static PDF report
- `deep-research-report.md` - source research brief used for the report
- `infographic.jpeg` - visual summary
- `assets/pytorch-hardware-2026-infographic.jpg` - source infographic image
- `README.md` - project overview, license, and citation information

## Citation

If you use this work, please cite it as:

Tunguz, B. (2026). *State of PyTorch Hardware Acceleration: May 2026*. GitHub
Pages. {REPORT_URL}

BibTeX:

```bibtex
@misc{{tunguz2026pytorchhardware,
  author = {{Tunguz, Bojan}},
  title = {{State of PyTorch Hardware Acceleration: May 2026}},
  year = {{2026}},
  howpublished = {{\\url{{{REPORT_URL}}}}},
  note = {{GitHub repository: \\url{{{REPO_URL}}}}}
}}
```

This report builds on and cites the previous edition:

Tunguz, B. (2025). *State of PyTorch Hardware Acceleration 2025*. GitHub Pages.
{OLD_REPORT_URL}

```bibtex
@misc{{tunguz2025pytorchhardware,
  author = {{Tunguz, Bojan}},
  title = {{State of PyTorch Hardware Acceleration 2025}},
  year = {{2025}},
  howpublished = {{\\url{{{OLD_REPORT_URL}}}}},
  note = {{GitHub repository: \\url{{{OLD_REPO_URL}}}}}
}}
```

## License

This repository is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

You may share and adapt this work, including for commercial purposes, as long as
you provide appropriate attribution, link to the license, and indicate whether
changes were made.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
"""
    README_OUT.write_text(readme, encoding="utf-8")


def main() -> None:
    source = clean_source(SOURCE.read_text(encoding="utf-8"))
    build_infographic()
    build_html(source)
    build_pdf(source)
    build_readme()


if __name__ == "__main__":
    main()
