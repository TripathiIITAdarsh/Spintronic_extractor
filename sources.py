"""Turn web-app uploads into a folder of PDFs.

Supported inputs:
  * .pdf            -> saved as is
  * .zip            -> every .pdf inside (any depth) is extracted
  * .xlsx/.xls/.csv -> one paper per row; DOI, URL and Title columns are used
                       when present (any cell with a DOI/URL otherwise). Each
                       paper is resolved to a PDF by trying, in order:
        arXiv id (from URL or DOI)  -> arxiv.org/pdf/<id>
        DOI                         -> open-access PDFs listed by OpenAlex and Europe PMC
        Europe PMC link (MED id)    -> PubMed Central open-access PDF
        direct .pdf URL / any URL   -> downloaded, or <meta citation_pdf_url> followed
        DOI landing page            -> <meta citation_pdf_url> followed
        no DOI, only a title        -> OpenAlex title search (exact title match only)
    Paywalled papers can't be downloaded; they are reported so the user can
    upload those PDFs manually.
"""
import difflib
import io
import re
import zipfile
from pathlib import Path

import requests

MAX_PDF_BYTES = 80 * 1024 * 1024          # per PDF
MAX_ZIP_TOTAL_BYTES = 2 * 1024 ** 3       # guard against zip bombs
UA = {"User-Agent": "Mozilla/5.0 (compatible; SpintronicsExtractor/1.0)"}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+", re.I)
URL_RE = re.compile(r"https?://[^\s\"<>]+", re.I)
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?\d)(?:v\d+)?(?:\.pdf)?(?:$|[?#\s])", re.I)
ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.([\w.\-]+)", re.I)
EPMC_MED_RE = re.compile(r"europepmc\.org/(?:abstract|article)/MED/(\d+)", re.I)
EPMC_PMC_RE = re.compile(r"(PMC\d+)", re.I)
META_PDF_RE = re.compile(
    r"<meta[^>]+name=[\"']citation_pdf_url[\"'][^>]+content=[\"']([^\"']+)[\"']"
    r"|<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+name=[\"']citation_pdf_url[\"']", re.I)


# ------------------------------------------------------------------ files
def safe_name(name: str, taken: set) -> str:
    base = re.sub(r"\.pdf$", "", Path(name).name, flags=re.I)   # only strip a real .pdf suffix
    stem = re.sub(r"[^\w\-. ]+", "_", base).strip(" ._")[:120] or "paper"
    cand, i = f"{stem}.pdf", 2
    while cand.lower() in taken:
        cand, i = f"{stem}_{i}.pdf", i + 1
    taken.add(cand.lower())
    return cand


def save_pdf_bytes(data: bytes, name: str, out_dir: Path, taken: set) -> Path:
    p = out_dir / safe_name(name, taken)
    p.write_bytes(data)
    return p


def pdfs_from_zip(data: bytes, out_dir: Path, taken: set) -> tuple[list[Path], list[str]]:
    saved, notes, total = [], [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                continue
            if "__MACOSX" in info.filename:
                continue
            total += info.file_size
            if info.file_size > MAX_PDF_BYTES or total > MAX_ZIP_TOTAL_BYTES:
                notes.append(f"skipped (too large): {info.filename}")
                continue
            # only the base name is used, so paths inside the zip can't escape out_dir
            saved.append(save_pdf_bytes(zf.read(info), Path(info.filename).name, out_dir, taken))
    return saved, notes


# ------------------------------------------------------------------ tables of papers
def _read_sheets(data: bytes, filename: str) -> dict:
    import pandas as pd
    if filename.lower().endswith(".csv"):
        return {"(csv)": pd.read_csv(io.BytesIO(data), dtype=str, header=None)}
    return pd.read_excel(io.BytesIO(data), dtype=str, header=None, sheet_name=None)


def _header_map(df) -> tuple[int | None, dict]:
    """Find a header row (within the first 10) naming DOI / URL / Title columns."""
    for i in range(min(10, len(df))):
        cells = [str(c).strip().lower() for c in df.iloc[i].fillna("")]
        cols = {}
        for j, c in enumerate(cells):
            if c == "doi" and "doi" not in cols:
                cols["doi"] = j
            elif c in ("title", "paper title") and "title" not in cols:
                cols["title"] = j
            elif c in ("abstract", "summary") and "abstract" not in cols:
                cols["abstract"] = j
            elif c in ("year", "publication year", "pub year") and "year" not in cols:
                cols["year"] = j
            elif c in ("url", "link", "links", "pdf", "pdf url", "pdf link"):
                cols.setdefault("urls", []).append(j)
        if "doi" in cols or "urls" in cols:
            return i, cols
    return None, {}


def _clean_doi(s: str) -> str | None:
    m = DOI_RE.search(s or "")
    return m.group(0).rstrip(".)]").lower() if m else None


def _rows_to_refs(df, sheet: str = "") -> list[dict]:
    df = df.fillna("")
    hdr, cols = _header_map(df)
    refs = []
    for _, row in df.iloc[(hdr + 1) if hdr is not None else 0:].iterrows():
        cells = [str(c) for c in row]
        text = " ".join(cells)
        doi = _clean_doi(cells[cols["doi"]]) if "doi" in cols else None
        urls = []
        for j in cols.get("urls", []):
            urls += URL_RE.findall(cells[j])
        urls += [u for u in URL_RE.findall(text) if u not in urls]
        urls = [u.rstrip(".)]") for u in urls]
        doi = doi or next((d for d in (_clean_doi(u) for u in urls) if d), None) or _clean_doi(text)
        title = cells[cols["title"]].strip() if "title" in cols else ""
        abstract = cells[cols["abstract"]].strip() if "abstract" in cols else ""
        ym = re.search(r"\b(19|20)\d{2}\b", cells[cols["year"]]) if "year" in cols else None
        if doi or urls:
            refs.append({"doi": doi, "urls": urls, "title": title, "sheets": [sheet] if sheet else [],
                         "abstract": abstract, "year": int(ym.group(0)) if ym else None})
    return refs


def table_category(data: bytes, filename: str) -> str:
    """Category of a paper-search export: the 'Query' row of its Summary sheet, else the file name."""
    try:
        for name, df in _read_sheets(data, filename).items():
            if name.strip().lower() != "summary":
                continue
            for row in df.fillna("").itertuples(index=False):
                cells = [str(c).strip() for c in row]
                if cells and cells[0].lower() in ("query", "search query", "category") and len(cells) > 1 and cells[1]:
                    return cells[1]
    except Exception:
        pass
    stem = Path(filename).stem
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d+__", "", stem)          # drop the export timestamp prefix


def category_label(cat: str) -> str:
    """Readable category name that is also a valid file name:
    '"magnetic tunnel junction" + "oxides"' -> 'MTJ + oxides'."""
    name = cat.replace('"', "").replace("'", "")
    name = re.sub(r"(?i)\bmagnetic tunnel junctions?\b", "MTJ", name)
    name = re.sub(r"[\\/:*?<>|]+", " ", name)
    return re.sub(r"\s+", " ", name).strip(" .")[:120] or "category"


# ------------------------------------------------------------------ same paper?
def norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def arxiv_id(s: str) -> str | None:
    m = ARXIV_RE.search(s or "") or ARXIV_DOI_RE.search(s or "")
    return m.group(1).lower() if m else None


def ref_keys(ref: dict) -> set[str]:
    """Identifiers of a paper; two rows sharing any of them are the same paper."""
    keys = set()
    if ref.get("doi"):
        keys.add("doi:" + ref["doi"].lower())
    for s in (ref.get("urls") or []) + [ref.get("doi") or ""]:
        a = arxiv_id(s)
        if a:
            keys.add("arxiv:" + a)
    t = norm_text(ref.get("title"))
    if len(t) >= 20:                    # short / generic titles are not trusted for matching
        keys.add("title:" + t)
    if not keys and ref.get("urls"):
        keys.add("url:" + ref["urls"][0].lower())
    return keys


def dedupe(refs: list[dict]) -> list[dict]:
    """Merge rows that are the same paper (shared DOI, arXiv id or title), across sheets.
    The merged paper keeps every link, the first DOI/title seen, and all sheets it appeared in."""
    out, index = [], {}
    for r in refs:
        keys = ref_keys(r)
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            hit = {"doi": r.get("doi"), "urls": list(r.get("urls") or []), "title": r.get("title") or "",
                   "sheets": list(r.get("sheets") or []), "abstract": r.get("abstract") or "",
                   "year": r.get("year"), "categories": list(r.get("categories") or [])}
            out.append(hit)
        else:
            hit["categories"] += [c for c in (r.get("categories") or []) if c not in hit["categories"]]
            hit["doi"] = hit["doi"] or r.get("doi")
            hit["title"] = hit["title"] or r.get("title") or ""
            hit["year"] = hit.get("year") or r.get("year")
            if len(r.get("abstract") or "") > len(hit.get("abstract") or ""):   # keep the fullest abstract
                hit["abstract"] = r["abstract"]
            hit["urls"] += [u for u in (r.get("urls") or []) if u not in hit["urls"]]
            hit["sheets"] += [s for s in (r.get("sheets") or []) if s not in hit["sheets"]]
        for k in keys | ref_keys(hit):
            index[k] = hit
    return out


def table_sheets(data: bytes, filename: str) -> tuple[dict[str, int], int]:
    """{sheet name: number of papers} for sheets that contain papers, and the number of
    unique papers across all of them."""
    per_sheet = {name: _rows_to_refs(df, name) for name, df in _read_sheets(data, filename).items()}
    counts = {k: len(dedupe(v)) for k, v in per_sheet.items() if v}
    return counts, len(dedupe([r for v in per_sheet.values() for r in v]))


def refs_from_table(data: bytes, filename: str, sheets: list[str] | None = None) -> list[dict]:
    """Unique papers listed in the given sheets (all sheets when None)."""
    frames = _read_sheets(data, filename)
    names = [n for n in frames if sheets is None or n in sheets]
    return dedupe([r for n in names for r in _rows_to_refs(frames[n], n)])


def ref_from_text(line: str) -> dict:
    """A pasted line: DOI and/or URL."""
    urls = [u.rstrip(".)]") for u in URL_RE.findall(line)]
    return {"doi": _clean_doi(line), "urls": urls, "title": "", "sheets": [], "abstract": "", "year": None}


def pdf_fingerprint(path: Path) -> dict:
    """DOI and first-page text of an uploaded PDF, to match it against listed papers.
    Not thread-safe (PyMuPDF): call from one thread."""
    import pymupdf
    try:
        doc = pymupdf.open(path)
        text = " ".join(doc[i].get_text() for i in range(min(2, len(doc))))
        meta = doc.metadata or {}
    except Exception:
        return {"doi": None, "arxiv": None, "text": ""}
    raw = " ".join([meta.get("subject") or "", meta.get("keywords") or "", text])
    m = re.search(r"arxiv[:\s]*(\d{4}\.\d{4,5})", raw, re.I)
    return {"doi": _clean_doi(raw), "arxiv": m.group(1).lower() if m else None, "text": norm_text(text)}


def ref_matches_pdf(ref: dict, fp: dict) -> bool:
    if ref.get("doi") and fp["doi"] and ref["doi"].lower() == fp["doi"]:
        return True
    if fp["arxiv"] and any(arxiv_id(s) == fp["arxiv"] for s in (ref.get("urls") or []) + [ref.get("doi") or ""]):
        return True
    words = [w for w in norm_text(ref.get("title")).split() if len(w) > 3]
    if len(words) >= 4 and fp["text"]:
        page = set(fp["text"].split())
        return sum(w in page for w in words) / len(words) >= 0.9
    return False


# ------------------------------------------------------------------ downloading
def _get_pdf(url: str) -> bytes | None:
    """Download url if it is a complete PDF; one iterator so no bytes are lost."""
    try:
        with requests.get(url, headers=UA, timeout=(15, 60), stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                return None
            buf = bytearray()
            for chunk in r.iter_content(1 << 16):
                buf += chunk
                if len(buf) >= 5 and not buf.startswith(b"%PDF"):
                    return None
                if len(buf) > MAX_PDF_BYTES:
                    return None
            expected = r.headers.get("content-length")
            if expected and expected.isdigit() and r.headers.get("content-encoding") in (None, "", "identity") \
                    and len(buf) < int(expected):
                return None                                  # connection dropped mid-download
        if len(buf) < 1024 or b"%%EOF" not in buf[-4096:]:
            return None                                      # truncated / not a real PDF
        return bytes(buf)
    except requests.RequestException:
        return None


def _get_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=UA, timeout=(15, 45), allow_redirects=True)
        if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
            return r.text[:2_000_000]
    except requests.RequestException:
        pass
    return None


def _json(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, headers=UA, params=params, timeout=(15, 30))
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def _openalex_work_pdfs(w: dict) -> list[str]:
    urls = []
    for loc in [w.get("best_oa_location") or {}] + (w.get("locations") or []):
        if loc.get("pdf_url"):
            urls.append(loc["pdf_url"])
    if (w.get("open_access") or {}).get("oa_url"):
        urls.append(w["open_access"]["oa_url"])
    return urls


def _openalex_by_doi(doi: str) -> list[str]:
    w = _json(f"https://api.openalex.org/works/https://doi.org/{doi}")
    return _openalex_work_pdfs(w) if w else []


def _openalex_by_title(title: str) -> tuple[list[str], str | None]:
    """Only accepts a near-exact title match, so a different paper is never picked."""
    if len(title) < 15:
        return [], None
    res = _json("https://api.openalex.org/works", {"search": title, "per-page": 3})
    norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
    for w in (res or {}).get("results", []):
        if difflib.SequenceMatcher(None, norm(w.get("title")), norm(title)).ratio() > 0.93:
            doi = _clean_doi(w.get("doi") or "")
            return _openalex_work_pdfs(w), doi
    return [], None


def _europepmc_pdfs(query: str) -> list[str]:
    res = _json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                {"query": query, "resultType": "core", "format": "json", "pageSize": 1})
    items = ((res or {}).get("resultList") or {}).get("result") or []
    if not items:
        return []
    it, urls = items[0], []
    for u in ((it.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
        if u.get("documentStyle") == "pdf" and u.get("availability", "").lower() in ("open access", "free"):
            urls.append(u["url"])
    if it.get("pmcid") and (it.get("isOpenAccess") == "Y" or it.get("inEPMC") == "Y"):
        urls.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={it['pmcid']}&blobtype=pdf")
    return urls


def resolve_ref(ref: dict) -> tuple[bytes | None, str]:
    """Returns (pdf_bytes or None, status message)."""
    doi, urls, title = ref.get("doi"), ref.get("urls") or [], ref.get("title") or ""
    candidates = []
    for s in urls + ([doi] if doi else []):
        m = ARXIV_RE.search(s) or ARXIV_DOI_RE.search(s)
        if m:
            candidates.append(f"https://arxiv.org/pdf/{m.group(1)}")
    if doi:
        candidates += _openalex_by_doi(doi)
        candidates += _europepmc_pdfs(f'DOI:"{doi}"')
    for u in urls:
        m = EPMC_MED_RE.search(u)
        if m:
            candidates += _europepmc_pdfs(f"EXT_ID:{m.group(1)} AND SRC:MED")
        elif EPMC_PMC_RE.search(u) and "europepmc" in u.lower():
            pmc = EPMC_PMC_RE.search(u).group(1).upper()
            candidates.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc}&blobtype=pdf")
    candidates += urls
    if doi:
        candidates.append(f"https://doi.org/{doi}")
    elif title:
        found, found_doi = _openalex_by_title(title)
        candidates += found
        if found_doi:
            candidates.append(f"https://doi.org/{found_doi}")

    blocked = []
    for url in dict.fromkeys(candidates):
        data = _get_pdf(url)
        if data:
            return data, f"downloaded from {url}"
        html = _get_html(url)
        if html:
            mm = META_PDF_RE.search(html)
            pdf_url = mm and (mm.group(1) or mm.group(2))
            if pdf_url:
                data = _get_pdf(requests.compat.urljoin(url, pdf_url))
                if data:
                    return data, f"downloaded from {pdf_url}"
                blocked.append(pdf_url)
    why = "no open-access copy found (likely paywalled)"
    if blocked:
        why = "the publisher has a PDF link but blocked the automatic download"
    return None, f"{why}; upload the PDF manually"


def resolve_link(link: str) -> tuple[bytes | None, str]:
    return resolve_ref(ref_from_text(link))


def ref_name(ref: dict) -> str:
    """Readable, unique-ish file name for a paper."""
    for s in (ref.get("urls") or []) + [ref.get("doi") or ""]:
        m = ARXIV_RE.search(s) or ARXIV_DOI_RE.search(s)
        if m:
            return "arXiv_" + m.group(1).replace("/", "_")
    if ref.get("doi"):
        return ref["doi"].replace("/", "_")
    if ref.get("title"):
        return re.sub(r"\W+", "_", ref["title"])[:80]
    return re.sub(r"^https?://", "", ref["urls"][0])[:100]


def ref_label(ref: dict) -> str:
    return (f"https://doi.org/{ref['doi']}" if ref.get("doi") else (ref.get("urls") or [""])[0])


def link_to_name(link: str) -> str:
    return ref_name(ref_from_text(link))
