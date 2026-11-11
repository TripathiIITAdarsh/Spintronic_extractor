"""Live-demo helper: extract a single PDF and write the JSON next to it.

Runs the same pipeline as run_extraction.py but ONLY on the PDFs inside
--pdf-dir, writes each result to <pdf-dir>/<stem>.<provider>.json, and does
NOT touch outputs/ or regenerate the aggregated CSV/Excel.

Usage:
  python demo_single.py                                   # all PDFs in ./demo_pdf, gemini
  python demo_single.py --pdf-dir demo_pdf --provider gemini
  python demo_single.py --pdf-dir demo_pdf --provider claude
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

from llm_clients import build_prompt, make_client  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402
from run_extraction import normalize, parse_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(HERE / "demo_pdf"))
    ap.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return 1

    client = make_client(args.provider)
    print(f"{len(pdfs)} PDF(s) from {pdf_dir} -> {args.provider}")

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] parsing {pdf.name} ...", flush=True)
        parsed = parse_pdf(pdf)
        print(f"           context {parsed['context_chars']} chars; calling {args.provider} ...", flush=True)

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

        out = pdf_dir / f"{pdf.stem}.{args.provider}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        tag = "ERR " if err else "ok  "
        print(f"           {tag} {len(result['records'])} record(s) in {result['seconds']}s -> {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
