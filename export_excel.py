"""Build one Excel workbook per LLM from outputs/<provider>/*.json.

  outputs/Gemini_Extraction.xlsx
  outputs/Claude_Extraction.xlsx

Sheet "Extracted Features": one row per record, exact schema column order.
Sheet "Papers": one row per PDF (model, #records, summary, error).
"""
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from schema import COLUMNS

HERE = Path(__file__).resolve().parent
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _sheet(ws, header, rows, widths=None):
    ws.append(header)
    for c in ws[1]:
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for r in rows:
        ws.append([v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False) for v in r])
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    for i, h in enumerate(header, 1):
        w = (widths or {}).get(h) or min(45, max(12, len(h) + 2))
        ws.column_dimensions[get_column_letter(i)].width = w


def build_workbook(columns: list, records: list[dict], papers: list[dict]) -> bytes:
    """Workbook for the web app: selected feature columns + a per-paper sheet."""
    import io
    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted Features"
    _sheet(ws, columns, [[r.get(c) for c in columns] for r in records],
           {"Title": 50, "Authors": 40, "Notes": 80})
    p_cols = ["Paper_ID", "Title", "Category", "Found_In_Sheets", "Text_Source", "Extracted_By", "Source", "Provider", "Model", "Num_Records", "Seconds", "Paper_Summary", "Status",
              "Reason"]
    _sheet(wb.create_sheet("Papers"), p_cols, [[p.get(c) for c in p_cols] for p in papers],
           {"Paper_ID": 40, "Source": 50, "Paper_Summary": 90, "Status": 60, "Reason": 60})
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export(provider: str) -> Path:
    src = HERE / "outputs" / provider
    records, papers = [], []
    for f in sorted(src.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        records += [[rec.get(c) for c in COLUMNS] for rec in d.get("records", [])]
        papers.append([d.get("paper_id"), d.get("file"), d.get("model"), len(d.get("records", [])),
                       d.get("context_chars"), d.get("full_text_chars"), d.get("seconds"),
                       d.get("paper_summary"), d.get("error")])

    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted Features"
    _sheet(ws, COLUMNS, records, {"Title": 50, "Authors": 40, "Notes": 80})
    _sheet(wb.create_sheet("Papers"),
           ["Paper_ID", "File", "Model", "Num_Records", "Context_Chars", "Full_Text_Chars",
            "Seconds", "Paper_Summary", "Error"],
           papers, {"Paper_ID": 40, "File": 50, "Paper_Summary": 90, "Error": 40})
    out = HERE / "outputs" / f"{provider.capitalize()}_Extraction.xlsx"
    wb.save(out)
    print(f"[{provider}] Excel: {out} ({len(papers)} papers, {len(records)} rows)")
    return out


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["gemini", "claude"]):
        export(p)
