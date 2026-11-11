"""Extract spintronics features from every PDF -> one JSON per PDF per LLM.

Usage:
  python run_extraction.py                        # both LLMs, all PDFs
  python run_extraction.py --provider gemini      # only Gemini
  python run_extraction.py --provider claude --limit 5
  python run_extraction.py --only 2308.13408v2    # single paper (file stem)
  python run_extraction.py --force                # redo already-done papers

Outputs (inside this folder):
  parsed/<stem>.json            the parsed, trimmed context sent to the LLMs
  outputs/gemini/<stem>.json    Gemini extraction
  outputs/claude/<stem>.json    Claude extraction
  outputs/<provider>_all.csv    flattened table (one row per record) with the exact columns
"""
import argparse
import csv
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from llm_clients import build_prompt, make_client  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402
from schema import COLUMNS, META_FIELDS, NUMERIC_FIELDS, normalize_device_type  # noqa: E402

DEFAULT_PDF_DIR = HERE.parent / "1 Spintronics"


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # take the first complete JSON object; ignores text (or a second object) after it
        return json.JSONDecoder().raw_decode(text[text.find("{"):])[0]


def to_number(v):
    if v is None or isinstance(v, (int, float)):
        return v
    s = str(v).strip().replace(",", "").replace("~", "").replace("≈", "")
    m = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        return float(m.group(0))
    m = re.match(r"^([-+]?\d*\.?\d+)\s*[x×]\s*10\^?\s*([-+]?\d+)", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return None


def normalize(records: list, paper_id: str, columns: list | None = None,
              numeric: set | None = None) -> list:
    """Clean LLM records. columns/numeric default to the full schema; the web
    app passes the user's selected (possibly custom) features instead."""
    columns = columns or COLUMNS
    numeric = (NUMERIC_FIELDS if numeric is None else numeric) & set(columns)
    out = []
    for rec in records or []:
        row = {c: rec.get(c) for c in columns}
        row["Paper_ID"] = paper_id
        for c in numeric:
            if isinstance(row[c], str):
                n = to_number(row[c])
                if n is None and row[c]:
                    row["Notes"] = f"{row.get('Notes') or ''} [{c} raw: {row[c]}]".strip()
                row[c] = n
        if row.get("Year") is not None:
            row["Year"] = int(row["Year"])
        for c in columns:
            if isinstance(row[c], str) and row[c].strip().lower() in ("", "null", "none", "n/a", "na", "not reported"):
                row[c] = None
        if "Device_Type" in columns:
            raw_type = row["Device_Type"]
            row["Device_Type"] = normalize_device_type(raw_type)
            if row["Device_Type"] == "other" and raw_type and "Notes" in row:
                row["Notes"] = f"{row.get('Notes') or ''} [Device_Type raw: {raw_type}]".strip()
        if "Missing_Value_Flag" in columns:
            row["Missing_Value_Flag"] = sum(1 for c in columns if c not in META_FIELDS and row[c] is None)
        out.append(row)
    return out


def process(pdf: Path, client, parsed_dir: Path, out_dir: Path, force: bool) -> str:
    out_path = out_dir / f"{pdf.stem}.json"
    if out_path.exists() and not force:
        return f"skip   {pdf.name}"
    parsed_path = parsed_dir / f"{pdf.stem}.json"
    if parsed_path.exists():
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    else:
        parsed = parse_pdf(pdf)
        parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    raw = client.generate(build_prompt(parsed))
    try:
        data = parse_json(raw)
        err = None
    except Exception as e:
        data, err = {}, f"JSON parse failed: {e}"
    result = {
        "paper_id": pdf.stem,
        "file": pdf.name,
        "provider": client.name,
        "model": client.model,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "context_chars": parsed["context_chars"],
        "full_text_chars": parsed["full_chars"],
        "paper_summary": data.get("paper_summary"),
        "records": normalize(data.get("records"), pdf.stem),
        "error": err,
    }
    if err:
        result["raw_response"] = raw
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"{'ERR ' if err else 'ok '}   {pdf.name}  ({len(result['records'])} records, {result['seconds']}s)"


def write_csv(out_dir: Path, provider: str):
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8")).get("records", [])
    csv_path = out_dir.parent / f"{provider}_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[{provider}] CSV: {csv_path} ({len(rows)} rows)")
    from export_excel import export
    export(provider)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    ap.add_argument("--provider", choices=["gemini", "claude", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", help="comma-separated PDF stems")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--claude-workers", type=int, default=4, help="parallel claude calls")
    ap.add_argument("--gemini-workers", type=int, default=5,
                    help="parallel gemini calls (request starts still spaced by GEMINI_MIN_INTERVAL)")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        pdfs = [p for p in pdfs if p.stem in wanted]
    if args.limit:
        pdfs = pdfs[: args.limit]
    print(f"{len(pdfs)} PDFs from {args.pdf_dir}")

    parsed_dir = HERE / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    # parse once up front (shared by both providers)
    for p in pdfs:
        pp = parsed_dir / f"{p.stem}.json"
        if not pp.exists():
            try:
                pp.write_text(json.dumps(parse_pdf(p), indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                print(f"parse fail {p.name}: {e}")

    providers = ["gemini", "claude"] if args.provider == "both" else [args.provider]

    def run_provider(name):
        client = make_client(name)
        out_dir = HERE / "outputs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        workers = args.claude_workers if name == "claude" else args.gemini_workers

        def job(p):
            try:
                return process(p, client, parsed_dir, out_dir, args.force)
            except Exception as e:
                (HERE / "outputs" / f"{name}_errors.log").open("a", encoding="utf-8").write(
                    f"{p.name}\n{traceback.format_exc()}\n")
                return f"FAIL  {p.name}: {str(e)[:200]}"

        with ThreadPoolExecutor(workers) as ex:
            for i, msg in enumerate(ex.map(job, pdfs), 1):
                print(f"[{name} {i}/{len(pdfs)}] {msg}", flush=True)
        write_csv(out_dir, name)

    # the two providers run side by side
    with ThreadPoolExecutor(len(providers)) as ex:
        list(ex.map(run_provider, providers))


if __name__ == "__main__":
    sys.exit(main())
