"""Render docs/report.md to docs/report.pdf (Markdown → HTML → headless Chrome).

python docs/build_pdf.py
"""

import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).parent
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
STYLE = """
@page { size: A4; margin: 18mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8.5pt;
                         color: #777; } }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1d2433; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; margin-top: 22pt; border-bottom: 1px solid #c8ccd4; padding-bottom: 3pt;
     break-before: page; }
h2:first-of-type { break-before: auto; }
h3 { font-size: 12pt; margin-top: 16pt; }
h2, h3 { break-after: avoid; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c8ccd4; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f7; padding: 0 2pt; }
pre { background: #f2f4f7; padding: 6pt 8pt; font-size: 7.5pt; white-space: pre;
      break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 120mm; display: block; margin: 8pt auto 2pt;
      border: 1px solid #e3e6eb; }
.caption { font-size: 9pt; color: #555; text-align: center; break-before: avoid; }
.toc { margin: 10pt 0 4pt; }
.toc ul { list-style: none; padding-left: 0; margin: 2pt 0; }
.toc a { color: inherit; text-decoration: none; }
"""


def main() -> None:
    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2-2"}}
    )
    body = converter.convert((DOCS / "report.md").read_text(encoding="utf-8"))
    body = body.replace("<p><em>Figure:", '<p class="caption"><em>Figure:')
    # Contents after the title block.
    first_section = body.index("<h2")
    contents = f"<p><strong>Contents</strong></p>{converter.toc}"
    body = body[:first_section] + contents + body[first_section:]
    html = DOCS / "report.html"
    html.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><style>{STYLE}</style></head>'
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    pdf = DOCS / "report.pdf"
    subprocess.run(
        [
            str(CHROME),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    html.unlink()
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
