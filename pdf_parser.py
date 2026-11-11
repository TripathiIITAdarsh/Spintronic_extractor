"""PDF -> compact, relevance-ranked context for LLM extraction.

Instead of sending the whole paper, we:
  1. extract text page by page with PyMuPDF (plus tables when detectable),
  2. pull metadata (title, DOI, year, journal) from PDF metadata + first page,
  3. drop the References / Acknowledgements tail,
  4. always keep the abstract, figure/table captions and tables,
  5. score every remaining paragraph by spintronics keyword + number/unit
     density and keep the best ones (in original order) up to a char budget.
"""
import re
from pathlib import Path

import pymupdf as fitz

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")

TAIL_HEADINGS = re.compile(
    r"^\s*(references|bibliography|acknowledg(e)?ments?|conflicts? of interest|"
    r"declaration of competing interest|author contributions|data availability)\s*:?\s*$",
    re.I | re.M,
)
CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?|table)\s*\d+", re.I)

# keyword -> weight; covers every family of columns in schema.FIELDS
KEYWORDS = {
    # transport
    "tmr": 3, "magnetoresistance": 3, "ra product": 3, "resistance-area": 3,
    "parallel state": 2, "antiparallel": 2, "conductance": 1, "v_half": 2, "v1/2": 2,
    "switching current": 3, "critical current": 3, "current density": 3,
    "switching voltage": 3, "a/cm": 3, "ma/cm": 3, "bias": 1,
    # stack
    "cofeb": 3, "mgo": 3, "free layer": 3, "reference layer": 3, "pinned layer": 3,
    "tunnel barrier": 3, "seed layer": 2, "buffer layer": 2, "capping": 2, "cap layer": 2,
    "synthetic antiferro": 3, "saf": 2, "ru spacer": 2, "insertion": 2, "dusting": 2,
    "heavy metal": 3, "/ta/": 1, "pt/": 1, "stack": 2, "multilayer": 1, "nm)": 1,
    "junction": 2, "pillar": 2, "diameter": 2, "mtj": 3, "spin valve": 2,
    # magnetic
    "saturation magnetization": 3, "m_s": 1, "ms =": 2, "emu/cm": 3, "emu/cc": 3,
    "coercivity": 3, "coercive": 3, "anisotropy field": 3, "h_k": 2, "hk": 1,
    "exchange bias": 3, "damping": 3, "gilbert": 3, "spin polarization": 3,
    "curie": 3, "t_c": 1, "thermal stability": 3, "δ": 1, "perpendicular": 2,
    "pma": 2, "in-plane": 1, "anisotropy": 2,
    # SOT
    "spin hall": 3, "spin-orbit torque": 3, "sot": 2, "damping-like": 3,
    "field-like": 3, "field-free": 3, "efficiency": 2,
    # fab / measurement
    "sputter": 3, "magnetron": 3, "mbe": 2, "molecular beam": 3, "pulsed laser": 2,
    "exfoliat": 2, "cvd": 1, "anneal": 3, "substrate": 2, "deposited": 2,
    "room temperature": 1, "measured at": 2, " k ": 1, "°c": 2,
    # DFT
    "dft": 3, "density functional": 3, "pbe": 3, "lda": 2, "hse": 2, "gga": 2,
    "hubbard": 3, "dft+u": 3, "spin-orbit coupling": 2, "soc": 1, "magnetic moment": 3,
    "μb": 3, "mu_b": 2, "formation energy": 3, "half-metal": 3, "lattice constant": 3,
    "vasp": 2, "quantum espresso": 2, "monte carlo": 1, "å": 2,
}
UNIT_RE = re.compile(
    r"\d\s*(nm|Å|%|Ω|ohm|kΩ|MΩ|µm|μm|um|Oe|kOe|mT|T\b|emu|A/cm|A ?cm|A/m|mA|µA|μA|uA|"
    r"mV|V\b|K\b|°C|eV|meV|μB|µB|GHz)",
)


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)       # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _paragraphs(text: str):
    """Split page text into paragraph-ish blocks of reasonable size."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    for b in blocks:
        b = re.sub(r"\s*\n\s*", " ", b)
        if len(b) > 1800:  # long blocks -> sentence windows
            sents = re.split(r"(?<=[.!?])\s+", b)
            cur = ""
            for s in sents:
                if len(cur) + len(s) > 1200 and cur:
                    out.append(cur.strip())
                    cur = ""
                cur += s + " "
            if cur.strip():
                out.append(cur.strip())
        elif len(b) > 25:
            out.append(b)
    return out


def _score(par: str, extra: dict | None = None) -> float:
    low = par.lower()
    kw = sum(w for k, w in KEYWORDS.items() if k in low)
    if extra:
        kw += sum(w for k, w in extra.items() if k in low)
    units = len(UNIT_RE.findall(par))
    return kw + 1.5 * min(units, 12)


def _metadata(doc, first_text: str) -> dict:
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    if len(title) < 15 or title.lower().endswith((".pdf", ".doc")) or "untitled" in title.lower():
        # fallback: largest-font line on page 1
        title = ""
        try:
            spans = []
            for b in doc[0].get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    t = " ".join(s["text"] for s in l["spans"]).strip()
                    if len(t) > 15 and l["spans"]:
                        spans.append((max(s["size"] for s in l["spans"]), t))
            if spans:
                top = max(s for s, _ in spans)
                title = " ".join(t for s, t in spans if abs(s - top) < 0.6)[:300]
        except Exception:
            pass
    doi = None
    for src in (meta.get("subject") or "", first_text, meta.get("keywords") or ""):
        m = DOI_RE.search(src)
        if m:
            doi = m.group(1).rstrip(".)]")
            break
    years = YEAR_RE.findall((meta.get("subject") or "") + " " + first_text[:4000])
    return {
        "pdf_title": title or None,
        "pdf_author": (meta.get("author") or "").strip() or None,
        "pdf_subject": (meta.get("subject") or "").strip() or None,
        "doi_guess": doi,
        "year_candidates": sorted(set(years))[-5:],
    }


def _tables(page) -> list[str]:
    out = []
    try:
        for t in page.find_tables().tables:
            rows = t.extract()
            if not rows or len(rows) < 2:
                continue
            cells = [(c or "").strip() for r in rows for c in r if (c or "").strip()]
            if len(cells) < 4:
                continue
            avg_len = sum(map(len, cells)) / len(cells)
            numeric = sum(bool(re.search(r"\d", c)) for c in cells) / len(cells)
            if avg_len > 45 or numeric < 0.3:  # prose mis-detected as a table
                continue
            md = "\n".join(" | ".join((c or "").replace("\n", " ").strip() for c in r) for r in rows)
            if len(md) > 40:
                out.append(md[:3000])
    except Exception:
        pass
    return out


def parse_pdf(path: str | Path, budget_chars: int = 28000, extra_keywords: list[str] | None = None) -> dict:
    """extra_keywords: terms (e.g. from user-added features) that should also
    pull paragraphs into the context."""
    path = Path(path)
    extra = {k.lower(): 3 for k in (extra_keywords or []) if len(k) >= 3}
    doc = fitz.open(path)
    pages = [_clean(p.get_text("text")) for p in doc]
    full = "\n\n".join(pages)

    # cut References / Acknowledgements tail (only if it occurs in latter half)
    body = full
    for m in TAIL_HEADINGS.finditer(full):
        if m.start() > 0.45 * len(full):
            body = full[: m.start()]
            break

    meta = _metadata(doc, pages[0] if pages else "")

    tables = []
    for i, p in enumerate(doc):
        if i > 40:
            break
        tables += [f"[Table p{i+1}]\n{t}" for t in _tables(p)]

    paras = _paragraphs(body)
    abstract = []
    head = body[:6000]
    m = re.search(r"abstract[:.\s]*(.{200,2500}?)(\n\s*\n|1\.?\s*introduction|keywords)", head, re.I | re.S)
    if m:
        abstract.append(re.sub(r"\s+", " ", m.group(1)).strip())
    else:  # first few substantial paragraphs
        abstract = [p for p in paras[:8] if len(p) > 300][:2]

    captions = [p for p in paras if CAPTION_RE.match(p)]
    rest = [(i, p) for i, p in enumerate(paras) if p not in captions and p not in abstract]

    budget = budget_chars
    sections = []

    def take(label, items, cap):
        nonlocal budget
        buf, used = [], 0
        for t in items:
            t = t[:1500]
            if used + len(t) > cap or budget - len(t) < 0:
                continue
            buf.append(t)
            used += len(t)
            budget -= len(t)
        if buf:
            sections.append(f"## {label}\n" + "\n\n".join(buf))

    take("FIRST PAGE HEADER", [re.sub(r"\s+", " ", pages[0][:1500])] if pages else [], 1500)
    take("ABSTRACT", abstract, 3000)
    take("TABLES", tables, 6000)
    take("FIGURE/TABLE CAPTIONS", captions, 4000)

    ranked = sorted(rest, key=lambda ip: _score(ip[1], extra), reverse=True)
    chosen, used = [], 0
    for i, p in ranked:
        if _score(p, extra) < 3:
            break
        if used + len(p) > budget:
            continue
        chosen.append((i, p))
        used += len(p)
    chosen.sort()  # restore reading order
    if chosen:
        sections.append("## RELEVANT BODY PARAGRAPHS (ranked, in reading order)\n" +
                        "\n\n".join(p for _, p in chosen))

    context = "\n\n".join(sections)
    return {
        "paper_id": path.stem,
        "file": path.name,
        "n_pages": len(doc),
        "full_chars": len(full),
        "context_chars": len(context),
        "metadata": meta,
        "context": context,
    }


if __name__ == "__main__":
    import json, sys
    r = parse_pdf(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "context"}, indent=2, ensure_ascii=False))
    print(r["context"][:4000])
