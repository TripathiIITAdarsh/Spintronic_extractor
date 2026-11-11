"""Device-type & material extraction for every paper-search export (one Excel per category).

For each Excel in Agent_paper_list/exports the category is read from its Summary sheet
("Query" row). Papers from all sheets are merged; a paper listed in several sheets or
several categories is extracted ONCE and its result reused everywhere.

Per paper:
  * open-access PDF found  -> relevant excerpts of the full text  (Text_Source = "Full text (PDF)")
  * otherwise              -> title + abstract from the sheet, 10 papers per call
                              (Text_Source = "Abstract only" / "Title only")

Results are cached in <out>/_cache, so a run stopped by a usage limit continues where it
left off when started again.

Usage:
  python extract_categories.py                       # all categories, local Claude Code (subscription)
  python extract_categories.py --provider gemini     # same with the Gemini API key from .env
  python extract_categories.py --limit 5             # only 5 new papers (test)
  python extract_categories.py --report-only         # rebuild the Excel files from the cache
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

import local_backends  # noqa: E402
import sources  # noqa: E402
from export_excel import _sheet  # noqa: E402
from llm_clients import build_abstract_prompt, build_prompt, make_client  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402
from run_extraction import normalize, parse_json  # noqa: E402
from schema import ADDITIONAL_FIELDS, DEVICE_MATERIAL_FEATURES, FIELDS, NUMERIC_FIELDS  # noqa: E402

FEATURES = DEVICE_MATERIAL_FEATURES
LLM_FIELDS = {k: FIELDS.get(k) or ADDITIONAL_FIELDS[k] for k in FEATURES}
OUT_COLUMNS = FEATURES + ["DOI", "Text_Source", "Extracted_By", "Found_In_Sheets"]
LOCAL_LABEL = "Local (rules)"        # label of old cache entries without an "extracted_by" field
DEFAULT_EXPORTS = HERE.parent / "Agent_paper_list" / "exports"
DEFAULT_OUT = HERE.parent / "Category_Features"
PDF_BUDGET_CHARS = 14000      # device type & materials need less context than the full schema
ABSTRACT_BATCH = 10
PARSE_LOCK = threading.Lock()  # PyMuPDF is not thread-safe
STOP = threading.Event()       # set when the usage limit is hit


def log(msg: str):
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


# ------------------------------------------------------------------ load categories
def load_categories(exports: Path) -> list[dict]:
    cats = []
    for f in sorted(exports.glob("*.xls*")):
        if f.name.startswith("~$"):            # Excel lock file of an open workbook
            continue
        data = f.read_bytes()
        cat = sources.table_category(data, f.name)
        refs = sources.refs_from_table(data, f.name)
        cats.append({"file": f.name, "category": cat, "name": sources.category_label(cat), "refs": refs})
    seen = Counter(c["name"] for c in cats)
    for c in cats:                            # two exports of the same query -> keep both, distinguish by date
        if seen[c["name"]] > 1:
            c["name"] += " (" + c["file"][:10] + ")"
    return cats


def merge_papers(cats: list[dict]) -> list[dict]:
    """One entry per unique paper across all categories, with {category: [sheets]}."""
    papers, index = [], {}
    for c in cats:
        for r in c["refs"]:
            keys = sources.ref_keys(r)
            g = next((index[k] for k in keys if k in index), None)
            if g is None:
                g = {"ref": {**r, "urls": list(r["urls"]), "sheets": []}, "cats": {}}
                papers.append(g)
            else:
                ref = g["ref"]
                ref["doi"] = ref["doi"] or r["doi"]
                ref["title"] = ref["title"] or r["title"]
                ref["year"] = ref.get("year") or r.get("year")
                if len(r.get("abstract") or "") > len(ref.get("abstract") or ""):
                    ref["abstract"] = r["abstract"]
                ref["urls"] += [u for u in r["urls"] if u not in ref["urls"]]
            sheets = g["cats"].setdefault(c["name"], [])
            sheets += [s for s in r.get("sheets", []) if s not in sheets]
            for k in keys | sources.ref_keys(g["ref"]):
                index[k] = g
    return papers


# ------------------------------------------------------------------ cache
class Cache:
    def __init__(self, out: Path):
        self.dir = out / "_cache"
        self.pdf_dir = out / "_pdfs"
        for d in (self.dir / "papers", self.dir / "resolve", self.pdf_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.index = json.loads(self.index_path.read_text("utf-8")) if self.index_path.exists() else {}
        self.lock = threading.Lock()

    def pid(self, ref: dict) -> str:
        keys = sorted(sources.ref_keys(ref))
        pid = next((self.index[k] for k in keys if k in self.index), None)
        if pid is None:
            primary = next((k for p in ("doi:", "arxiv:", "title:", "url:") for k in keys if k.startswith(p)), keys[0])
            pid = hashlib.sha1(primary.encode()).hexdigest()[:12]
        for k in keys:
            self.index[k] = pid
        return pid

    def save_index(self):
        self.index_path.write_text(json.dumps(self.index, indent=1), "utf-8")

    def get(self, kind: str, pid: str):
        p = self.dir / kind / f"{pid}.json"
        try:
            return json.loads(p.read_text("utf-8")) if p.exists() else None
        except (ValueError, OSError):          # half-written file from an interrupted run: redo it
            return None

    def put(self, kind: str, pid: str, obj: dict):
        p = self.dir / kind / f"{pid}.json"
        tmp = p.with_suffix(".tmp")            # write then rename, so a crash never leaves a broken file
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(p)


# ------------------------------------------------------------------ extraction
def is_usage_limit(err: str) -> bool:
    low = err.lower()
    return ("usage limit" in low or "limit reached" in low or "retries exhausted" in low
            or "quota exhausted for all models" in low or "rate_limit" in low)


def finish_records(records: list, paper: dict) -> list:
    ref = paper["ref"]
    for r in records:                          # the sheet's title / year are authoritative
        r["Title"] = ref.get("title") or r.get("Title")
        r["Year"] = ref.get("year") or r.get("Year")
    return records


def extract_pdf(client, paper: dict, pdf_path: Path) -> dict:
    with PARSE_LOCK:
        parsed = parse_pdf(pdf_path, budget_chars=PDF_BUDGET_CHARS)
    if parsed["context_chars"] < 1500:          # scanned / image-only PDF: not enough text
        raise ValueError("PDF has almost no extractable text")
    t0 = time.time()
    raw = client.generate(build_prompt(parsed, LLM_FIELDS))
    data = parse_json(raw)
    recs = finish_records(normalize(data.get("records"), paper["pid"], columns=FEATURES, numeric=NUMERIC_FIELDS), paper)
    return {"records": recs, "text_source": "Full text (PDF)", "summary": data.get("paper_summary"),
            "seconds": round(time.time() - t0, 1)}


def extract_abstracts(client, batch: list[dict]) -> dict[str, dict]:
    items = [{"id": f"P{i + 1}", "title": p["ref"]["title"], "abstract": p["ref"].get("abstract")}
             for i, p in enumerate(batch)]
    raw = client.generate(build_abstract_prompt(items, LLM_FIELDS))
    data = parse_json(raw)
    by_id = {str(e.get("id")): e for e in data.get("papers", []) if isinstance(e, dict)}
    out = {}
    for item, p in zip(items, batch):
        e = by_id.get(item["id"])
        if e is None:                          # model skipped it: retried next run
            continue
        recs = finish_records(normalize(e.get("records"), p["pid"], columns=FEATURES, numeric=NUMERIC_FIELDS), p)
        src = "Abstract only" if (p["ref"].get("abstract") or "").strip() else "Title only"
        out[p["pid"]] = {"records": recs, "text_source": src}
    return out


def extract_local(p: dict, cache) -> dict:
    """Rules-only answer (no API): same text the API would have seen, values only if written in it."""
    rs = cache.get("resolve", p["pid"]) or {}
    text, src = "", None
    if rs.get("pdf"):
        ctx_file = cache.dir / "context" / f"{p['pid']}.txt"   # parsed excerpt, shared with eval_local.py
        try:
            if ctx_file.exists():
                text = ctx_file.read_text("utf-8")
            else:
                with PARSE_LOCK:
                    text = parse_pdf(cache.pdf_dir / rs["pdf"], budget_chars=PDF_BUDGET_CHARS)["context"]
                ctx_file.parent.mkdir(exist_ok=True)
                ctx_file.write_text(text, "utf-8")
        except Exception:
            text = ""
        if len(text) >= 1500:
            src = "Full text (PDF)"
        else:
            text = ""
    if not text:
        text = p["ref"].get("abstract") or ""
        src = "Abstract only" if text.strip() else "Title only"
    backend = local_backends.get()               # chosen with LOCAL_BACKEND / LOCAL_MODEL in .env
    recs = backend.extract(p["ref"].get("title"), text, LLM_FIELDS)
    recs = normalize(recs, p["pid"], columns=FEATURES, numeric=NUMERIC_FIELDS)
    return {"records": finish_records(recs, p), "text_source": src, "method": "local", "extracted_by": backend.label}


def extracted_by(res: dict) -> str:
    if res.get("extracted_by"):
        return res["extracted_by"]
    return LOCAL_LABEL if res.get("method") == "local" else f"Claude ({res.get('model') or 'sonnet'})"


# ------------------------------------------------------------------ outputs
def write_outputs(cats: list[dict], papers: list[dict], cache: Cache, out: Path):
    from openpyxl import Workbook
    all_rows = []
    for c in cats:
        rows, paper_rows = [], []
        members = [p for p in papers if c["name"] in p["cats"]]
        for p in members:
            ref, res, rs = p["ref"], cache.get("papers", p["pid"]), cache.get("resolve", p["pid"]) or {}
            base = {"Title": ref.get("title"), "Year": ref.get("year"), "DOI": ref.get("doi"),
                    "Found_In_Sheets": ", ".join(p["cats"][c["name"]])}
            if res is None:
                status, src, recs, by = "not extracted yet (run again)", "not extracted", [], None
            else:
                status, src, recs, by = "ok", res["text_source"], res["records"], extracted_by(res)
            for r in (recs or [{}]):           # every paper appears at least once
                row = {k: r.get(k) for k in FEATURES}
                row["Title"] = row["Title"] or base["Title"]
                row["Year"] = row["Year"] or base["Year"]
                row.update(DOI=base["DOI"], Found_In_Sheets=base["Found_In_Sheets"], Text_Source=src,
                           Extracted_By=by)
                rows.append(row)
                all_rows.append({"Category": c["name"], **row})
            paper_rows.append({**base, "Paper_ID": p["pid"], "Text_Source": src, "Extracted_By": by,
                               "Num_Records": len(recs), "PDF": rs.get("msg"), "Status": status,
                               "Other_Categories": "; ".join(n for n in p["cats"] if n != c["name"])})

        wb = Workbook()
        ws = wb.active
        ws.title = "Features"
        _sheet(ws, OUT_COLUMNS, [[r.get(k) for k in OUT_COLUMNS] for r in rows],
               {"Title": 60, "Main_Material": 30, "DOI": 30, "Found_In_Sheets": 25})
        dt = Counter(r.get("Device_Type") or "(not stated)" for r in rows)
        _sheet(wb.create_sheet("Device types"), ["Device_Type", "Rows"], dt.most_common())
        mm = Counter(m.strip() for r in rows for m in (r.get("Main_Material") or "").split(";") if m.strip())
        _sheet(wb.create_sheet("Main materials"), ["Main_Material", "Rows"], mm.most_common())
        pcols = ["Paper_ID", "Title", "Year", "DOI", "Found_In_Sheets", "Text_Source", "Extracted_By",
                 "Num_Records", "PDF", "Other_Categories", "Status"]
        _sheet(wb.create_sheet("Papers"), pcols, [[r.get(k) for k in pcols] for r in paper_rows],
               {"Title": 60, "PDF": 60, "Other_Categories": 50, "Extracted_By": 25})
        src_count = Counter(r["Text_Source"] for r in paper_rows)
        by_count = Counter(r["Extracted_By"] or "not extracted" for r in paper_rows)
        info = [["Category (search query)", c["category"]], ["Source export", c["file"]],
                ["Papers (unique)", len(members)], ["Rows (devices / materials)", len(rows)],
                *[[f"Papers from: {k}", v] for k, v in src_count.most_common()],
                *[[f"Extracted by: {k}", v] for k, v in by_count.most_common()],
                ["Note", "'Local (...)' rows were filled by the local backend (LOCAL_BACKEND in .env) because the "
                         "API was unavailable; they only keep values written in the text and are re-done by the "
                         "API on the next run."],
                ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")]]
        _sheet(wb.create_sheet("Summary", 0), ["Item", "Value"], info, {"Item": 30, "Value": 90})
        wb.active = 1
        path = out / f"{c['name']}.xlsx"
        try:
            wb.save(path)
        except PermissionError:                # file open in Excel
            path = out / f"{c['name']} (new).xlsx"
            wb.save(path)
        (out / "json").mkdir(exist_ok=True)
        (out / "json" / f"{c['name']}.json").write_text(json.dumps(
            {"category": c["category"], "source_export": c["file"], "papers": paper_rows, "rows": rows},
            indent=2, ensure_ascii=False, default=str), "utf-8")
        log(f"wrote {path.name}: {len(members)} papers, {len(rows)} rows  {dict(by_count)}")

    wb = Workbook()
    _sheet(wb.active, ["Category"] + OUT_COLUMNS, [[r.get(k) for k in ["Category"] + OUT_COLUMNS] for r in all_rows],
           {"Category": 40, "Title": 60, "Main_Material": 30, "DOI": 30})
    wb.active.title = "All categories"
    try:
        wb.save(out / "_All_categories.xlsx")
    except PermissionError:
        wb.save(out / "_All_categories (new).xlsx")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default=str(DEFAULT_EXPORTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude",
                    help="claude = local Claude Code (subscription); gemini = API key from .env")
    ap.add_argument("--model", default=None, help="claude: sonnet/opus/haiku; gemini: model list")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="extract at most N new papers (testing)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-fallback", action="store_true",
                    help="leave papers empty when the API fails instead of using the local rules")
    ap.add_argument("--local-only", action="store_true", help="no API at all: local rules for every paper")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out)
    cats = load_categories(Path(args.exports))
    papers = merge_papers(cats)
    for p in papers:
        p["pid"] = cache.pid(p["ref"])
    cache.save_index()
    log(f"{len(cats)} categories, {sum(len(c['refs']) for c in cats)} category entries -> {len(papers)} unique papers")
    for c in cats:
        log(f"   {c['name']}  ({len(c['refs'])} papers)  <- {c['file']}")

    client, model_name = None, None
    if not (args.local_only or args.report_only):
        try:
            if args.provider == "claude":
                client = make_client("claude", mode="cli", model=args.model or "sonnet")
                model_name = f"Claude ({client.model})"
            else:
                client = make_client("gemini", model=args.model)
                model_name = "Gemini"
        except Exception as e:
            log(f"API not available ({e}); using the local rules for every paper")

    current_local = local_backends.get().label

    def needs_work(p):
        """Never done; or done locally and the API is available now; or (--local-only) done by a
        different local backend than the one configured now (e.g. rules -> a better model)."""
        res = cache.get("papers", p["pid"])
        if res is None:
            return True
        if res.get("method") != "local":
            return False
        return client is not None or (args.local_only and extracted_by(res) != current_local)

    todo = [p for p in papers if needs_work(p)]
    if args.limit:
        todo = todo[: args.limit]
    log(f"{len(papers) - len(todo)} already extracted (cache); {len(todo)} to do now")

    if todo and not args.report_only and client is not None:
        # 1) find open-access PDFs (cached)
        need = [p for p in todo if cache.get("resolve", p["pid"]) is None]
        if need:
            log(f"looking for open-access PDFs of {len(need)} papers ...")
            with ThreadPoolExecutor(6) as ex:
                futs = {ex.submit(sources.resolve_ref, p["ref"]): p for p in need}
                for i, fut in enumerate(as_completed(futs), 1):
                    p = futs[fut]
                    try:
                        data, msg = fut.result()
                    except Exception as e:
                        data, msg = None, f"lookup error: {e}"
                    rec = {"pdf": None, "msg": msg}
                    if data:
                        (cache.pdf_dir / f"{p['pid']}.pdf").write_bytes(data)
                        rec["pdf"] = f"{p['pid']}.pdf"
                    cache.put("resolve", p["pid"], rec)
                    if i % 25 == 0 or i == len(need):
                        log(f"   PDFs checked {i}/{len(need)}")
        with_pdf = [p for p in todo if (cache.get("resolve", p["pid"]) or {}).get("pdf")]
        no_pdf = [p for p in todo if p not in with_pdf]
        log(f"{len(with_pdf)} papers with full-text PDF, {len(no_pdf)} from title/abstract")

        # 2) full-text papers, one call each
        def run_pdf(p):
            if STOP.is_set():
                return p, None, "skipped (stopped)"
            try:
                return p, extract_pdf(client, p, cache.pdf_dir / f"{p['pid']}.pdf"), None
            except Exception as e:
                return p, None, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run_pdf, p) for p in with_pdf]
            for i, fut in enumerate(as_completed(futs), 1):
                p, res, err = fut.result()
                if res:
                    cache.put("papers", p["pid"], {**res, "pid": p["pid"], "title": p["ref"]["title"],
                                                   "doi": p["ref"]["doi"], "model": client.model, "method": "llm",
                                                   "extracted_by": model_name,
                                                   "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    log(f"[pdf {i}/{len(with_pdf)}] ok  {len(res['records'])} rec  {p['ref']['title'][:70]}")
                elif err and not err.startswith("skipped"):
                    log(f"[pdf {i}/{len(with_pdf)}] FAIL {p['ref']['title'][:60]}: {err[:200]}")
                    if is_usage_limit(err):
                        STOP.set()
                    elif "pdf" in err.lower() or "json" in err.lower():
                        no_pdf.append(p)       # unreadable PDF / bad reply -> fall back to the abstract

        # 3) title + abstract papers, ABSTRACT_BATCH per call
        no_pdf = [p for p in no_pdf if (p["ref"].get("title") or p["ref"].get("abstract"))]
        batches = [no_pdf[i:i + ABSTRACT_BATCH] for i in range(0, len(no_pdf), ABSTRACT_BATCH)]

        def run_batch(b):
            if STOP.is_set():
                return b, None, "skipped (stopped)"
            try:
                return b, extract_abstracts(client, b), None
            except Exception as e:
                return b, None, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run_batch, b) for b in batches]
            for i, fut in enumerate(as_completed(futs), 1):
                b, res, err = fut.result()
                if res is not None:
                    for p in b:
                        if p["pid"] in res:
                            cache.put("papers", p["pid"], {**res[p["pid"]], "pid": p["pid"], "title": p["ref"]["title"],
                                                           "doi": p["ref"]["doi"], "model": client.model,
                                                           "method": "llm", "extracted_by": model_name,
                                                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    log(f"[abstracts {i}/{len(batches)}] ok  {len(res)}/{len(b)} papers")
                elif err and not err.startswith("skipped"):
                    log(f"[abstracts {i}/{len(batches)}] FAIL: {err[:200]}")
                    if is_usage_limit(err):
                        STOP.set()

        if STOP.is_set():
            log("API usage limit reached; the API part stops here (run again later to continue with the API).")

    # 4) local fallback for every paper the API could not do (or for all, with --local-only)
    if todo and not args.report_only and not args.no_fallback:
        rest = [p for p in todo if cache.get("papers", p["pid"]) is None]
        if rest:
            log(f"local fallback (rules, no API) for {len(rest)} papers ...")
            for i, p in enumerate(rest, 1):
                try:
                    cache.put("papers", p["pid"], {**extract_local(p, cache), "pid": p["pid"],
                                                   "title": p["ref"]["title"], "doi": p["ref"]["doi"],
                                                   "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                except Exception as e:
                    log(f"[local] FAIL {p['ref']['title'][:60]}: {type(e).__name__}: {e}")
                if i % 50 == 0 or i == len(rest):
                    log(f"   local fallback {i}/{len(rest)}")

    write_outputs(cats, papers, cache, out)
    done = [cache.get("papers", p["pid"]) for p in papers]
    by = Counter(extracted_by(r) if r else "not extracted" for r in done)
    log(f"done. {dict(by)}. Output: {out}")
    n_local = sum(n for k, n in by.items() if local_backends.is_local(k) or k == "Local fallback (rules)")
    if n_local:
        log(f"{n_local} papers have local answers; run again when the API is available to upgrade them.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted; progress is cached")
