"""The PDF page builder embeds the report safely and pins its browser libraries."""

import json
import re

import pytest

from alphalab import report_pdf as R


def test_markdown_is_embedded_as_json_not_raw_html():
    md = "# Title\n\n<script>alert(1)</script> and </script> inside text\n"
    page = R.build_html(md)
    assert "<script>alert(1)</script>" not in page  # never injected as markup
    blob = re.search(r"const md = (.*);\n", page).group(1)
    assert json.loads(blob.replace("<\\/", "</")) == md


def test_library_versions_are_pinned():
    for url in (R.MARKED, R.MERMAID):
        assert re.search(r"@\d+\.\d+\.\d+/", url), url


def test_mermaid_blocks_are_converted_and_page_is_a4():
    page = R.build_html("```mermaid\nxychart-beta\n```\n")
    assert "language-mermaid" in page and "mermaid.run" in page
    assert "size: A4" in page


def test_render_refuses_an_unrendered_page(tmp_path, monkeypatch):
    calls = []

    def fake(chrome, budget, *args):
        calls.append(args)
        return R.subprocess.CompletedProcess(args, 0, stdout="<body>no marker</body>", stderr="")

    monkeypatch.setattr(R, "_chrome", fake)
    md = tmp_path / "r.md"
    md.write_text("# x\n")
    out = tmp_path / "out.pdf"
    with pytest.raises(R.RenderError):
        R.render(md, out)
    assert len(calls) == 1 and not out.exists()  # never printed


def test_render_checks_the_written_file(tmp_path, monkeypatch):
    def fake(chrome, budget, *args):
        if any(a.startswith("--print-to-pdf=") for a in args):
            out.write_bytes(b"not a pdf")
        return R.subprocess.CompletedProcess(args, 0, stdout=f"<body {R.RENDERED}>", stderr="")

    monkeypatch.setattr(R, "_chrome", fake)
    md = tmp_path / "r.md"
    md.write_text("# x\n")
    out = tmp_path / "out.pdf"
    with pytest.raises(R.RenderError):
        R.render(md, out)
