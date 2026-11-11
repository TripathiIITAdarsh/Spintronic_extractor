"""Compare the local fallback extractor with Claude's results on the category papers.

For every paper Claude has extracted (Category_Features/_cache), run local_extractor on the
same text Claude saw (PDF excerpt or title + abstract) and report, per feature:
  recall    = local found the same value / Claude found a value
  precision = local agrees with Claude / local gave a value (where Claude also gave one)
Usage:  python eval_local.py
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
import local_extractor  # noqa: E402
import sources  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402

OUT = HERE.parent / "Category_Features"
FEATS = ["Device_Type", "Main_Material", "Free_Layer_Material", "Tunnel_Barrier_Material",
         "Reference_Layer_Material", "Heavy_Metal_SOT_Layer", "Substrate"]


def norm(v) -> set[str]:
    parts = re.split(r"[;,]| and ", str(v or ""))
    out = set()
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p).lower()
        p = re.sub(r"[^a-z0-9]", "", p.replace("h-bn", "hbn"))
        if p:
            out.add(p)
    return out


def agree(a, b) -> bool:
    A, B = norm(a), norm(b)
    return any(x == y or (min(len(x), len(y)) >= 3 and (x in y or y in x)) for x in A for y in B)


def main():
    refs = []
    for f in glob.glob(str(HERE.parent / "Agent_paper_list" / "exports" / "*.xls*")):
        if "~$" not in f:
            refs += sources.refs_from_table(open(f, "rb").read(), f)
    refs = sources.dedupe(refs)
    index = json.loads((OUT / "_cache" / "index.json").read_text("utf-8"))
    by_pid = {}
    for r in refs:
        pid = next((index[k] for k in sources.ref_keys(r) if k in index), None)
        if pid:
            by_pid[pid] = r

    stats = defaultdict(lambda: {"claude": 0, "local": 0, "both": 0, "match": 0})
    examples = defaultdict(list)
    n = 0
    for f in sorted((OUT / "_cache" / "papers").glob("*.json")):
        res = json.loads(f.read_text("utf-8"))
        pid, ref = res["pid"], by_pid.get(res["pid"])
        if not ref or not res["records"]:
            continue
        if res["text_source"].startswith("Full text") and (OUT / "_pdfs" / f"{pid}.pdf").exists():
            ctx_file = OUT / "_cache" / "context" / f"{pid}.txt"      # parsed once, reused by later runs
            if ctx_file.exists():
                text = ctx_file.read_text("utf-8")
            else:
                text = parse_pdf(OUT / "_pdfs" / f"{pid}.pdf", budget_chars=14000)["context"]
                ctx_file.parent.mkdir(exist_ok=True)
                ctx_file.write_text(text, "utf-8")
        else:
            text = ref.get("abstract") or ""
        loc = local_extractor.extract(ref.get("title"), text)
        n += 1
        for k in FEATS:
            cv = [r.get(k) for r in res["records"] if r.get(k)]
            s = stats[k]
            s["claude"] += bool(cv)
            s["local"] += bool(loc[k])
            if cv and loc[k]:
                s["both"] += 1
                ok = any(agree(loc[k], c) for c in cv)
                s["match"] += ok
                if not ok and len(examples[k]) < 4:
                    examples[k].append((ref["title"][:55], loc[k], cv[0]))
        if n % 25 == 0:
            print(f"  {n} papers ...", flush=True)

    print(f"\n{n} papers compared (local rules vs Claude)\n")
    print(f"{'feature':26s} {'Claude filled':>13s} {'local filled':>12s} {'recall':>7s} {'precision':>9s}")
    for k in FEATS:
        s = stats[k]
        rec = s["match"] / s["claude"] if s["claude"] else 0
        prec = s["match"] / s["both"] if s["both"] else 0
        print(f"{k:26s} {s['claude']:13d} {s['local']:12d} {rec:7.0%} {prec:9.0%}")
    print("\nexamples of disagreement (title | local | Claude):")
    for k, ex in examples.items():
        for t, lv, cv in ex:
            print(f"  {k:24s} {t:55s} | {lv} | {cv}")


if __name__ == "__main__":
    main()
