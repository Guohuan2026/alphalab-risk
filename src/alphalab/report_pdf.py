"""Render docs/REPORT.md to a print-ready PDF with headless Google Chrome.

No Python dependency is added: the Markdown and the Mermaid charts are rendered in the
browser by pinned library versions (marked, mermaid) loaded from jsDelivr, and Chrome's
``--print-to-pdf`` writes A4 pages. Needs Chrome and an internet connection for the two
scripts. Chrome is taken from ``ALPHALAB_CHROME`` or the macOS default path. The page marks
itself rendered only after both libraries finish; the PDF is written only if a first pass
(``--dump-dom``) finds that marker, so a failed CDN load raises instead of producing an empty
or chart-less PDF. The Markdown is trusted project text: ``marked`` does not sanitise HTML.
Run: ``python -m alphalab.report_pdf [REPORT.md] [OUT.pdf]``.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MARKED = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
CHROME = os.environ.get(
    "ALPHALAB_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
RENDERED = 'data-rendered="ok"'


class RenderError(RuntimeError):
    """The report page did not finish rendering (scripts not loaded or a chart failed)."""


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.42;
       color: #111; max-width: 100%; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.6pt;
        break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #eee; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #f4f4f4;
             border-left: 3px solid #666; font-size: 9.6pt; }
code { font-family: "Menlo", monospace; font-size: 8.8pt; }
.mermaid { break-inside: avoid; margin: 8pt auto; text-align: center; }
hr { border: 0; border-top: 1px solid #bbb; margin: 12pt 0; }
"""


def build_html(markdown_text: str, title: str = "AlphaLab-Risk research report") -> str:
    """A self-rendering HTML page: the Markdown is embedded as JSON (never as raw HTML),
    converted by marked; ```mermaid blocks become Mermaid diagrams."""
    payload = json.dumps(markdown_text).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style>
<script src="{MARKED}"></script><script src="{MERMAID}"></script></head>
<body><main id="doc"></main>
<script>
const md = {payload};
document.getElementById("doc").innerHTML = marked.parse(md, {{gfm: true}});
document.querySelectorAll("code.language-mermaid").forEach(c => {{
  const d = document.createElement("div"); d.className = "mermaid"; d.textContent = c.textContent;
  c.parentElement.replaceWith(d);
}});
mermaid.initialize({{startOnLoad: false, theme: "neutral"}});
mermaid.run({{querySelector: ".mermaid"}}).then(() => {{
  const pending = document.querySelector(".mermaid:not([data-processed])");
  if (!pending) document.body.dataset.rendered = "ok";
}});
</script></body></html>"""


def _chrome(chrome: str, budget_ms: int, *args: str) -> subprocess.CompletedProcess:
    base = [chrome, "--headless=new", "--disable-gpu", f"--virtual-time-budget={budget_ms}"]
    return subprocess.run([*base, *args], check=True, capture_output=True, text=True)


def render(md_path: Path, out_pdf: Path, chrome: str = CHROME, budget_ms: int = 20000) -> Path:
    page = build_html(md_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(page, encoding="utf-8")
        dom = _chrome(chrome, budget_ms, "--dump-dom", src.as_uri()).stdout
        if RENDERED not in dom:
            raise RenderError(
                "report page did not finish rendering (no internet for the pinned scripts, "
                "or a Mermaid chart failed); no PDF written"
            )
        out_pdf.unlink(missing_ok=True)
        _chrome(
            chrome,
            budget_ms,
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf.resolve()}",
            src.as_uri(),
        )
    if not out_pdf.exists() or out_pdf.read_bytes()[:5] != b"%PDF-":
        raise RenderError(f"Chrome did not write a valid PDF to {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    args = sys.argv[1:]
    md = Path(args[0]) if args else Path("docs/REPORT.md")
    out = Path(args[1]) if len(args) > 1 else Path("AlphaLab-Risk_Report.pdf")
    print(render(md, out))
