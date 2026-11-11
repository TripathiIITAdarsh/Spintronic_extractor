"""Spintronics paper extractor: web interface.

Upload PDFs, ZIP folders of PDFs, or an Excel/CSV of paper links (DOI / arXiv /
URL), pick the features you want (or add new ones) and download an Excel file.

Run locally:   streamlit run app.py
"""
import hashlib
import hmac
import io
import json
import os
import re
import tempfile
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


def _load_secrets():
    """On Streamlit Cloud / HF Spaces keys come from st.secrets instead of .env."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


_load_secrets()

import local_backends  # noqa: E402
import sources  # noqa: E402
from export_excel import build_workbook  # noqa: E402
from llm_clients import build_abstract_prompt, build_prompt, make_client  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402
from run_extraction import normalize, parse_json  # noqa: E402
from schema import ADDITIONAL_FIELDS, DEVICE_MATERIAL_FEATURES, FIELDS, GROUPS, NUMERIC_FIELDS  # noqa: E402

CUSTOM_PATH = HERE / "custom_features.json"
DEFAULT_SELECTION = list(DEVICE_MATERIAL_FEATURES)      # "Device & materials" preset
ABSTRACT_BATCH = 10
EXTRA_QA = ["Notes", "Extraction_Confidence"]
PARSE_LOCK = threading.Lock()  # PyMuPDF is not thread-safe

st.set_page_config(page_title="Spintronics Paper Extractor", page_icon="🧲", layout="wide")


# ------------------------------------------------------------------ access
def require_password():
    pw = os.environ.get("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    st.title("Spintronics Paper Extractor")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, pw):
        st.session_state.authed = True
        st.rerun()
    if entered:
        st.error("Wrong password")
    st.stop()


require_password()


# ------------------------------------------------------------------ features
def load_custom() -> list[dict]:
    try:
        return json.loads(CUSTOM_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_custom(items: list[dict]):
    CUSTOM_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


if "custom" not in st.session_state:
    st.session_state.custom = load_custom()
if "selected" not in st.session_state:
    st.session_state.selected = list(DEFAULT_SELECTION)


def catalog() -> tuple[dict, set]:
    """All selectable features {name: description for the LLM} + numeric ones."""
    cat = {k: v for k, v in FIELDS.items() if k not in ("Paper_ID", "Missing_Value_Flag")}
    cat.update(ADDITIONAL_FIELDS)
    numeric = set(NUMERIC_FIELDS)
    for f in st.session_state.custom:
        desc = f.get("description") or f["name"].replace("_", " ")
        if f["type"] == "Number":
            desc += " -- plain number" + (f" in {f['unit']} (convert to this unit)" if f.get("unit") else "")
            numeric.add(f["name"])
        elif f["type"] == "Yes/No":
            desc += " -- Y | N"
        cat[f["name"]] = desc
    return cat, numeric


def add_group(names):
    sel = st.session_state.selected
    st.session_state.selected = sel + [n for n in names if n not in sel and n in catalog()[0]]


def set_selection(names):
    st.session_state.selected = list(names)


def add_feature():
    ss = st.session_state
    name = re.sub(r"\W+", "_", ss.nf_name.strip()).strip("_")
    if not name:
        ss.nf_msg = ("error", "Give the feature a name.")
        return
    if name in catalog()[0] or name == "Paper_ID":
        ss.nf_msg = ("error", f"'{name}' already exists.")
        return
    item = {"name": name, "description": ss.nf_desc.strip(), "unit": ss.nf_unit.strip(), "type": ss.nf_type}
    ss.custom = ss.custom + [item]
    save_custom(ss.custom)
    ss.selected = ss.selected + [name]
    ss.nf_name = ss.nf_desc = ss.nf_unit = ""
    ss.nf_msg = ("success", f"Added '{name}' and selected it.")


def delete_features():
    ss = st.session_state
    gone = set(ss.nf_delete)
    ss.custom = [f for f in ss.custom if f["name"] not in gone]
    save_custom(ss.custom)
    ss.selected = [s for s in ss.selected if s not in gone]
    ss.nf_delete = []


# ------------------------------------------------------------------ errors
KEY_HELP = ("Create a new key at https://aistudio.google.com/apikey, put it in the `.env` file as "
            "`GEMINI_API_KEY=...` (or in the host's secrets), then click **Re-check key** in the sidebar.")


def explain_error(raw: str) -> str:
    """Plain-language reason for a failure; the raw error is kept alongside it."""
    low = raw.lower()
    if "api key not valid" in low or "unauthenticated" in low or " 401" in low:
        return "The Gemini API key is invalid or was deleted. " + KEY_HELP
    if "permission_denied" in low or " 403" in low:
        return "The Gemini API key is not allowed to use the Gemini API (403). " + KEY_HELP
    if "quota exhausted for all models" in low or "perday" in low:
        return ("The free daily Gemini quota is used up for every model in the list. "
                "Try again tomorrow, or enable billing on the Gemini key.")
    if "no model answered" in low:
        parts = []
        for model, err in re.findall(r"(gemini[\w.\-]+): (.*?)(?=; gemini|\)\s*$|$)", raw):
            e = err.lower()
            what = ("was overloaded" if e.startswith("http 50") or "unavailable" in e else
                    "hit its quota / rate limit" if e.startswith("http 429") else
                    "did not reply in time" if "timeout" in e else err[:60])
            parts.append(f"{model} {what}")
        return ("No Gemini model could answer this paper in time (" + "; ".join(parts) + "). "
                "Run this paper again later.")
    if "high demand" in low or "unavailable" in low or "http 50" in low:
        return ("Google's Gemini servers were overloaded ('high demand', HTTP 503) and no model in the list "
                "answered in time. This is on Google's side and usually passes within minutes. Try again shortly.")
    if "resource_exhausted" in low or "http 429" in low:
        return ("Gemini rejected requests for being too fast (per-minute limit). "
                "Lower 'Papers processed in parallel' and try again.")
    if "pdf parsing failed" in low:
        return "This PDF could not be read. It may be damaged, password-protected, or a scanned image."
    if "not valid json" in low:
        return ("The model's reply was not valid JSON, usually because a very long reply got cut off. "
                "Try again, or select fewer features.")
    if "timed out" in low or "timeout" in low:
        return "Gemini took too long to reply. Try again."
    if "connectionerror" in low or "max retries exceeded" in low or "name resolution" in low:
        return "Could not reach the Gemini API. Check the internet connection."
    return raw[:300]


def is_fatal(raw: str) -> bool:
    """Errors that will hit every remaining paper too: stop instead of repeating them."""
    low = raw.lower()
    return any(s in low for s in ("api key not valid", "unauthenticated", "permission_denied",
                                  "quota exhausted for all models"))


@st.cache_data(ttl=300, show_spinner=False)
def check_gemini_key(key: str) -> tuple[bool, str]:
    """Validates the key by listing models; this does not use any generation quota."""
    if not key:
        return False, "No Gemini API key is configured. " + KEY_HELP
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         headers={"x-goog-api-key": key}, params={"pageSize": 1}, timeout=20)
    except requests.RequestException as e:
        return False, explain_error(f"ConnectionError: {e}")
    if r.status_code == 200:
        return True, "Gemini API key OK"
    return False, explain_error(f"Gemini HTTP {r.status_code}: {r.text}")


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Model")
gemini_key = os.environ.get("GEMINI_API_KEY", "")
key_ok, key_msg = check_gemini_key(gemini_key)
st.sidebar.write("Extraction uses **Google Gemini** with the key configured on the server.")
if key_ok:
    st.sidebar.success(key_msg, icon="✅")
else:
    st.sidebar.error(key_msg, icon="🔑")
    if st.sidebar.button("Re-check key"):
        check_gemini_key.clear()
        load_dotenv(HERE / ".env", override=True)
        st.rerun()

st.sidebar.caption(f"Fallback when the API fails: **{local_backends.get().label}** "
                   "(set LOCAL_BACKEND / LOCAL_MODEL in .env)")
workers = st.sidebar.slider("Papers processed in parallel", 1, 6, 3,
                            help="Gemini's free tier limits requests per minute; lower this if you see rate-limit errors.")
st.sidebar.caption("Text is parsed locally; only the relevant excerpts of each paper (≈28k characters, "
                   "references removed) are sent to the model.")


# ------------------------------------------------------------------ main page
st.title("🧲 Spintronics Paper Extractor")
st.caption("Upload papers, choose the features you want, and download them as an Excel sheet.")

st.subheader("1 · Papers")
uploads = st.file_uploader(
    "PDF files, ZIP folders containing PDFs, or an Excel/CSV file with paper links (DOI, arXiv or URL)",
    type=["pdf", "zip", "xlsx", "xls", "csv"], accept_multiple_files=True)
pasted = st.text_area("…or paste links / DOIs, one per line", height=80,
                      placeholder="10.1038/s41467-018-04953-8\nhttps://arxiv.org/abs/2308.13408")
st.caption("Links work for open-access papers (arXiv, open-access journals, PubMed Central). Paywalled "
           "papers can't be downloaded automatically. They are listed in the results so you can upload those PDFs.")


@st.cache_data(show_spinner=False, max_entries=20)
def sheet_info(data: bytes, name: str):
    return sources.table_sheets(data, name)


@st.cache_data(show_spinner=False, max_entries=50)
def table_refs(data: bytes, name: str, sheets: tuple[str, ...]):
    return sources.refs_from_table(data, name, list(sheets))


@st.cache_data(show_spinner=False, max_entries=50)
def table_category(data: bytes, name: str) -> str:
    return sources.category_label(sources.table_category(data, name))


sheet_choice: dict[str, list[str]] = {}   # table file -> sheets to read
for f in uploads or []:
    if not f.name.lower().endswith((".xlsx", ".xls", ".csv")):
        continue
    try:
        counts, unique_total = sheet_info(f.getvalue(), f.name)
    except Exception as e:
        st.error(f"Could not read **{f.name}**: {e}")
        continue
    if not counts:
        st.warning(f"No DOIs or links found in **{f.name}**.")
        continue
    if len(counts) == 1:
        sheet_choice[f.name] = list(counts)
        st.caption(f"📄 **{f.name}**: {unique_total} papers found.")
        continue
    picked = st.multiselect(f"Sheets of **{f.name}** to include", list(counts), default=list(counts),
                            key=f"sheets::{f.name}", format_func=lambda s, c=counts: f"{s} ({c[s]})")
    sheet_choice[f.name] = picked
    if picked == list(counts):
        st.caption(f"📄 {sum(counts.values())} rows across {len(counts)} sheets → **{unique_total} unique papers**. "
                   "A paper listed in several sheets is processed once (matched by DOI, arXiv ID or title).")
    elif picked:
        n = len(table_refs(f.getvalue(), f.name, tuple(picked)))
        st.caption(f"📄 {len(picked)} sheet(s) selected → **{n} unique papers**.")

st.subheader("2 · Features to extract")
cat, numeric = catalog()
st.session_state.selected = [s for s in st.session_state.selected if s in cat]

st.write("Quick add a group:")
group_cols = st.columns(4)
for i, (g, names) in enumerate(GROUPS.items()):
    group_cols[i % 4].button(f"+ {g}", on_click=add_group, args=(names,), width="stretch")
custom_names = [f["name"] for f in st.session_state.custom]
c1, c2, c3 = st.columns(3)
c1.button("Select all", on_click=set_selection, args=(list(cat),), width="stretch")
c2.button("Clear", on_click=set_selection, args=([],), width="stretch")
if custom_names:
    c3.button("+ My custom features", on_click=add_group, args=(custom_names,), width="stretch")

st.multiselect("Selected features (type to search)", options=list(cat), key="selected",
               format_func=lambda n: f"★ {n}" if n in custom_names else n)
include_qa = st.checkbox("Also add Notes and Extraction_Confidence columns (recommended)", value=True)
use_abstracts = st.checkbox(
    "For listed papers without a downloadable PDF, extract from the title + abstract in the Excel sheet",
    value=True, help="Paywalled papers then still get Device_Type / Main_Material etc. from their abstract. "
                     "These rows are marked 'Abstract only' in the Text_Source column.")
use_fallback = st.checkbox(
    f"If the API fails, use the local extractor instead: {local_backends.get().label}", value=True,
    help="Used when Gemini is unavailable (quota used up, overloaded, invalid key, garbled reply). Choose the "
         "local extractor with LOCAL_BACKEND / LOCAL_MODEL in .env (keyword rules by default, or any Ollama / "
         "OpenAI-compatible model, or your own Python model). Its values are kept only if they are written in "
         "the paper text. Such rows show the local extractor in the Extracted_By column.")

with st.expander("➕ Add a new feature"):
    with st.form("new_feature", clear_on_submit=False):
        a, b = st.columns([2, 1])
        a.text_input("Feature name", key="nf_name", placeholder="e.g. Magnon_Frequency_GHz")
        b.selectbox("Type", ["Number", "Text", "Yes/No"], key="nf_type")
        a.text_input("What should be extracted?", key="nf_desc",
                     placeholder="e.g. Ferromagnetic resonance / magnon frequency")
        b.text_input("Unit (for numbers)", key="nf_unit", placeholder="GHz")
        st.form_submit_button("Add feature", on_click=add_feature)
    if st.session_state.get("nf_msg"):
        kind, msg = st.session_state.pop("nf_msg")
        getattr(st, kind)(msg)
    if custom_names:
        st.multiselect("Remove custom features", custom_names, key="nf_delete")
        st.button("Delete selected", on_click=delete_features)

with st.expander(f"What each selected feature means ({len(st.session_state.selected)})"):
    st.dataframe(pd.DataFrame([{"Feature": n, "Meaning / unit": cat[n]} for n in st.session_state.selected]),
                 hide_index=True, width="stretch")


# ------------------------------------------------------------------ run
def _merge_info(i: dict, ref: dict):
    """Add what a listed paper knows (title, year, sheets, categories) to a PDF's info."""
    i["title"] = i["title"] or ref.get("title")
    i["year"] = i.get("year") or ref.get("year")
    for k in ("sheets", "categories"):
        i[k] += [s for s in (ref.get(k) or []) if s not in i[k]]


def collect_pdfs(work: Path):
    """Every input becomes at most one PDF per paper.
    -> [(pdf_path, source_label)], [failed paper rows], {pdf_path: info}, [notes],
       [listed papers without PDF to extract from their abstract]"""
    taken, items, failed, refs, notes, abstract_refs = set(), [], [], [], [], []
    info: dict[Path, dict] = {}
    by_hash: dict[str, Path] = {}       # identical files (same PDF uploaded twice, or two links -> same PDF)

    def add_pdf(data: bytes, name: str, label: str, ref: dict | None = None) -> bool:
        h = hashlib.sha1(data).hexdigest()
        if h in by_hash:
            first = by_hash[h]
            _merge_info(info[first], ref or {})
            notes.append(f"DUP   {label}: same file as {first.name}, processed once")
            return False
        path = sources.save_pdf_bytes(data, name, work, taken)
        by_hash[h] = path
        items.append((path, label))
        info[path] = {"title": None, "year": None, "sheets": [], "categories": []}
        _merge_info(info[path], ref or {})
        return True

    for f in uploads or []:
        n, data = f.name.lower(), f.getvalue()
        if n.endswith(".pdf"):
            add_pdf(data, f.name, f.name)
        elif n.endswith(".zip"):
            try:
                paths, zip_notes = sources.pdfs_from_zip(data, Path(tempfile.mkdtemp(dir=work)), set())
            except Exception as e:
                failed.append({"Paper_ID": f.name, "Source": f.name, "Status": f"bad zip: {e}"})
                continue
            for p in paths:
                add_pdf(p.read_bytes(), p.name, f"{f.name} › {p.name}")
            failed += [{"Paper_ID": f.name, "Source": f.name, "Status": m} for m in zip_notes]
            if not paths:
                failed.append({"Paper_ID": f.name, "Source": f.name, "Status": "no PDFs inside zip"})
        elif f.name in sheet_choice:
            try:
                cat_name = table_category(data, f.name)
                refs += [{**r, "categories": [cat_name]} for r in table_refs(data, f.name, tuple(sheet_choice[f.name]))]
            except Exception as e:
                failed.append({"Paper_ID": f.name, "Source": f.name, "Status": f"could not read sheet: {e}"})
    refs += [sources.ref_from_text(l) for l in (pasted or "").splitlines() if l.strip()]
    refs = sources.dedupe([r for r in refs if r["doi"] or r["urls"]])

    # a listed paper whose PDF was uploaded is not downloaded again
    if refs and items:
        prints = {p: sources.pdf_fingerprint(p) for p, _ in items}
        remaining = []
        for r in refs:
            match = next((p for p, fp in prints.items() if sources.ref_matches_pdf(r, fp)), None)
            if match is None:
                remaining.append(r)
                continue
            _merge_info(info[match], r)
            notes.append(f"MATCH {sources.ref_label(r)}: using uploaded {match.name}")
        refs = remaining

    if refs:
        bar = st.progress(0.0, text=f"Finding PDFs for {len(refs)} linked papers…")
        with ThreadPoolExecutor(6) as ex:
            futs = {ex.submit(sources.resolve_ref, r): r for r in refs}
            for i, fut in enumerate(as_completed(futs), 1):
                ref = futs[fut]
                data, msg = fut.result()
                label = sources.ref_label(ref)
                if data:
                    add_pdf(data, sources.ref_name(ref), label, ref)
                elif use_abstracts and (ref.get("abstract") or ref.get("title")):
                    abstract_refs.append(ref)
                    notes.append(f"ABSTR {label}: {msg.split(';')[0]}; using the title + abstract from the sheet")
                else:
                    failed.append({"Paper_ID": sources.ref_name(ref), "Title": ref.get("title"),
                                   "Found_In_Sheets": ", ".join(ref.get("sheets", [])),
                                   "Category": "; ".join(ref.get("categories", [])),
                                   "_categories": ref.get("categories", []), "Source": label, "Status": msg})
                bar.progress(i / len(refs), text=f"Finding PDFs for linked papers… {i}/{len(refs)}")
        bar.empty()
    return items, failed, info, notes, abstract_refs


def extract_abstract_batch(client, refs: list[dict], fields: dict, columns: list, numeric: set) -> list:
    """Title + abstract of up to ABSTRACT_BATCH papers in one call -> records per paper (None if skipped)."""
    items = [{"id": f"P{i + 1}", "title": r.get("title"), "abstract": r.get("abstract")} for i, r in enumerate(refs)]
    raw = client.generate(build_abstract_prompt(items, fields))
    try:
        data = parse_json(raw)
    except Exception as e:
        raise RuntimeError(f"Model reply was not valid JSON ({e}); reply started with: {raw[:200]!r}") from e
    by_id = {str(e.get("id")): e for e in data.get("papers", []) if isinstance(e, dict)}
    return [normalize(by_id[it["id"]].get("records"), sources.ref_name(r), columns=columns, numeric=numeric)
            if it["id"] in by_id else None for it, r in zip(items, refs)]


def call_api_or_fallback(make_request, fallback: bool, api_state: dict):
    """Run an API request; on failure (or when the API is already known to be down) return
    (None, error) so the caller uses the local rules -- or re-raise when fallback is off."""
    if api_state.get("down"):
        if not fallback:
            raise RuntimeError(api_state["down"])
        return None, api_state["down"]
    try:
        return make_request(), None
    except Exception as e:
        err = " ".join(f"{type(e).__name__}: {e}".split())
        if is_fatal(err):                       # bad key / all quotas used: stop calling the API
            api_state["down"] = err
        if not fallback:
            raise
        return None, err


def extract_one(client, path: Path, source: str, fields: dict, columns: list, numeric: set,
                extra_kw: list, fallback: bool, api_state: dict) -> tuple[dict, list, dict]:
    t0 = time.time()
    try:
        with PARSE_LOCK:
            parsed = parse_pdf(path, extra_keywords=extra_kw)
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {type(e).__name__}: {e}") from e

    def request():
        raw = client.generate(build_prompt(parsed, fields))
        try:
            return parse_json(raw)
        except Exception as e:
            raise RuntimeError(f"Model reply was not valid JSON ({e}); reply started with: {raw[:200]!r}") from e

    data, err = call_api_or_fallback(request, fallback, api_state)
    by = f"Gemini ({client.model})" if client and err is None else local_backends.get().label
    if err is not None:
        data = {"records": local_backends.get().extract(parsed["metadata"].get("pdf_title"), parsed["context"], fields)}
    records = normalize(data.get("records"), path.stem, columns=columns, numeric=numeric)
    paper = {"Paper_ID": path.stem, "Source": source, "Provider": "gemini", "Model": client.model if client else None,
             "Num_Records": len(records), "Seconds": round(time.time() - t0, 1),
             "Paper_Summary": data.get("paper_summary"), "Status": "ok", "Extracted_By": by,
             "Reason": f"API failed, {by} used: {explain_error(err)}" if err else None}
    js = {"paper_id": path.stem, "source": source, "extracted_by": by, "api_error": err,
          "context_chars": parsed["context_chars"], "paper_summary": data.get("paper_summary"),
          "records": records}
    return paper, records, js


st.subheader("3 · Extract")
selected = list(st.session_state.selected)
have_input = bool(uploads) or bool((pasted or "").strip())
if not key_ok and use_fallback:
    st.warning(f"**Gemini is unavailable:** {key_msg}\n\nPapers will be extracted with the local extractor "
               f"({local_backends.get().label}). Its values are kept only when they are written in the paper.",
               icon="🔑")
elif not key_ok:
    st.error(f"**Extraction is disabled:** {key_msg}", icon="🔑")
go = st.button("🚀 Extract features", type="primary",
               disabled=not (have_input and selected and (key_ok or use_fallback)))
if not selected:
    st.info("Select at least one feature.")

if go:
    run_log: list[str] = []

    def log_line(msg: str):
        run_log.append(f"{datetime.now():%H:%M:%S}  {msg}")

    api_state: dict = {}                        # {"down": reason} once the API is known to be unusable
    client = None
    if key_ok:
        try:
            client = make_client("gemini", api_key=gemini_key,
                                 model=os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL"))
            client.on_event = lambda msg: log_line(f"INFO  {msg}")  # model switches, overloads, quota
        except Exception as e:
            api_state["down"] = str(e)
    else:
        api_state["down"] = key_msg
    if client is None and not use_fallback:
        st.error(f"Could not start Gemini: {explain_error(api_state.get('down', ''))}")
        st.stop()
    if api_state.get("down"):
        log_line(f"INFO  API unavailable ({api_state['down'][:150]}); using {local_backends.get().label}")

    llm_cols = selected + [c for c in EXTRA_QA if include_qa and c not in selected]
    fields = {c: cat[c] for c in llm_cols}
    columns = ["Paper_ID"] + llm_cols
    extra_kw = []
    for f in st.session_state.custom:
        if f["name"] in selected:
            extra_kw.append(f["name"].replace("_", " ").lower())
            extra_kw += [w for w in re.findall(r"[a-zA-Z]{5,}", f"{f['name']} {f.get('description', '')}")]

    work = Path(tempfile.mkdtemp(prefix="spx_"))
    items, papers, info, notes, abstract_refs = collect_pdfs(work)
    for n in notes:
        log_line(n)
    for p in papers:  # inputs that never became a PDF (bad zip, paywalled link, ...)
        p["Reason"] = explain_error(p["Status"])
        log_line(f"SKIP  {p['Source']}: {p['Status']}")
    # provenance columns: which category / sheet listed a paper, and what text its values come from
    metas = list(info.values()) + abstract_refs + papers
    has_cats = any(m.get("categories") or m.get("_categories") for m in metas)
    has_sheets = any(m.get("sheets") or m.get("Found_In_Sheets") for m in metas)
    out_columns = (["Paper_ID"] + (["Category"] if has_cats else []) + (["Found_In_Sheets"] if has_sheets else [])
                   + llm_cols + ["Text_Source", "Extracted_By"])

    def finish(recs: list, meta: dict, text_source: str, by: str) -> list:
        """Provenance, plus the sheet's title / year (authoritative), on every record of one paper."""
        for r in recs:
            r.update(Category="; ".join(meta.get("categories") or []), _categories=list(meta.get("categories") or []),
                     Found_In_Sheets=", ".join(meta.get("sheets") or []), Text_Source=text_source, Extracted_By=by)
            if "Title" in r and meta.get("title"):
                r["Title"] = meta["title"]
            if "Year" in r and meta.get("year"):
                r["Year"] = meta["year"]
        return recs

    def paper_meta(paper: dict, meta: dict, text_source: str | None) -> dict:
        paper.update(Title=meta.get("title") or paper.get("Title"), Text_Source=text_source,
                     Found_In_Sheets=", ".join(meta.get("sheets") or []),
                     Category="; ".join(meta.get("categories") or []), _categories=list(meta.get("categories") or []))
        return paper

    n_total = len(items) + len(abstract_refs)
    log_line(f"{len(items)} PDF(s) + {len(abstract_refs)} abstract-only paper(s) to process with {workers} in "
             f"parallel; {len(fields)} features; "
             + (f"Gemini models: {', '.join(client.models)}" if client else f"{local_backends.get().label} only"))
    records, jsons = [], {}
    stop_reason = None
    if not n_total:
        st.error("No papers to process.")
    else:
        bar = st.progress(0.0, text=f"Extracting from {n_total} papers…")
        live = st.empty()
        done = 0

        def tick():
            bar.progress(min(1.0, done / n_total), text=f"Extracting… {done}/{n_total} papers")
            live.code("\n".join(run_log[-15:]), language=None)

        # 1) papers with a PDF: full text, one call each
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(extract_one, client, p, src, fields, columns, numeric, extra_kw,
                              use_fallback, api_state): (p, src) for p, src in items}
            for fut in as_completed(futs):
                p, src = futs[fut]
                base = {"Paper_ID": p.stem, "Source": src, "Provider": "gemini", "Num_Records": 0}
                text_source = None
                if fut.cancelled():
                    paper = {**base, "Status": f"skipped: run stopped ({stop_reason})",
                             "Reason": "Not processed because the run was stopped by the error above."}
                    log_line(f"SKIP  {p.name}: run stopped")
                else:
                    try:
                        paper, recs, js = fut.result()
                        text_source = "Full text (PDF)"
                        records += finish(recs, info[p], text_source, paper["Extracted_By"])
                        js.update(title=info[p]["title"], found_in_sheets=info[p]["sheets"],
                                  categories=info[p]["categories"], text_source=text_source)
                        jsons[p.stem] = js
                        if js.get("api_error"):
                            log_line(f"LOCAL {p.name}: API failed ({js['api_error'][:150]}), used {paper['Extracted_By']}")
                        else:
                            log_line(f"OK    {p.name}: {len(recs)} record(s), {paper['Seconds']} s, {paper['Model']}")
                    except Exception as e:
                        raw = " ".join(f"{type(e).__name__}: {e}".split())
                        paper = {**base, "Model": getattr(client, "model", None),
                                 "Status": f"failed: {raw[:500]}", "Reason": explain_error(raw)}
                        log_line(f"FAIL  {p.name}: {raw[:400]}")
                        if is_fatal(raw) and stop_reason is None and not use_fallback:
                            stop_reason = explain_error(raw)
                            log_line(f"STOP  this error affects every paper, cancelling the rest: {stop_reason}")
                            for f in futs:
                                f.cancel()
                papers.append(paper_meta(paper, info[p], text_source))
                done += 1
                tick()

        # 2) listed papers without a PDF: title + abstract from the sheet, ABSTRACT_BATCH per call
        batches = [abstract_refs[i:i + ABSTRACT_BATCH] for i in range(0, len(abstract_refs), ABSTRACT_BATCH)]

        def local_records(ref):
            return normalize(local_backends.get().extract(ref.get("title"), ref.get("abstract") or "", fields),
                             sources.ref_name(ref), columns=columns, numeric=numeric)

        def run_abstract_batch(b):
            res, err = call_api_or_fallback(lambda: extract_abstract_batch(client, b, fields, columns, numeric),
                                            use_fallback, api_state)
            if err is not None:
                return [local_records(r) for r in b], err, [local_backends.get().label] * len(b)
            by = f"Gemini ({client.model})"
            # papers the model left out of its reply also get the local rules
            return ([x if x is not None else (local_records(r) if use_fallback else None) for x, r in zip(res, b)],
                    None, [by if x is not None else local_backends.get().label for x in res])

        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(run_abstract_batch, b): b for b in (batches if stop_reason is None else [])}
            for fut in as_completed(futs):
                b = futs[fut]
                err = None
                try:
                    results, api_err, bys = fut.result()
                    if api_err:
                        log_line(f"LOCAL abstract batch of {len(b)} papers: API failed ({api_err[:150]}), "
                                 f"used {local_backends.get().label}")
                except Exception as e:
                    results, bys = [None] * len(b), [None] * len(b)
                    api_err, err = None, " ".join(f"{type(e).__name__}: {e}".split())
                    log_line(f"FAIL  abstract batch of {len(b)} papers: {err[:400]}")
                for ref, recs, by in zip(b, results, bys):
                    pid = sources.ref_name(ref)
                    src = "Abstract only" if (ref.get("abstract") or "").strip() else "Title only"
                    base = {"Paper_ID": pid, "Source": sources.ref_label(ref), "Provider": "gemini",
                            "Model": getattr(client, "model", None), "Extracted_By": by}
                    if recs is None:
                        why = err or "the model left this paper out of its reply; run it again"
                        papers.append(paper_meta({**base, "Num_Records": 0, "Status": f"failed: {why[:500]}",
                                                  "Reason": explain_error(why)}, ref, None))
                        continue
                    records += finish(recs, ref, src, by)
                    jsons[pid] = {"paper_id": pid, "source": base["Source"], "title": ref.get("title"),
                                  "categories": ref.get("categories"), "found_in_sheets": ref.get("sheets"),
                                  "text_source": src, "extracted_by": by, "records": recs}
                    note = f"API failed, {by} used: {explain_error(api_err)}" if local_backends.is_local(by) and api_err else None
                    papers.append(paper_meta({**base, "Num_Records": len(recs), "Status": "ok", "Reason": note}, ref, src))
                    if not local_backends.is_local(by):
                        log_line(f"OK    {pid}: {len(recs)} record(s) from {src.lower()}")
                if err and is_fatal(err) and stop_reason is None and not use_fallback:
                    stop_reason = explain_error(err)
                    log_line(f"STOP  this error affects every paper, cancelling the rest: {stop_reason}")
                    for f in futs:
                        f.cancel()
                done += len(b)
                tick()
        if stop_reason is not None:     # abstract papers never started because the run stopped
            started = {sources.ref_name(r) for b in batches for r in b if any(p["Paper_ID"] == sources.ref_name(r) for p in papers)}
            for r in abstract_refs:
                if sources.ref_name(r) not in started:
                    papers.append(paper_meta({"Paper_ID": sources.ref_name(r), "Source": sources.ref_label(r),
                                              "Num_Records": 0, "Status": f"skipped: run stopped ({stop_reason})",
                                              "Reason": "Not processed because the run was stopped by the error above."},
                                             r, None))
        bar.empty()
        live.empty()
    log_line("finished")

    cats_seen = list(dict.fromkeys(c for m in records + papers for c in m.get("_categories", [])))
    st.session_state.result = {
        "columns": out_columns, "records": records, "papers": papers, "jsons": jsons, "log": run_log,
        "duplicates": sum(1 for n in notes if n.startswith(("DUP", "MATCH"))), "categories": cats_seen,
        "provider": "gemini", "stamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }


# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res:
    st.subheader("Results")
    ok = sum(1 for p in res["papers"] if p.get("Status") == "ok")
    m1, m2, m3 = st.columns(3)
    m1.metric("Papers extracted", ok)
    m2.metric("Rows (devices / materials)", len(res["records"]))
    m3.metric("Failed / skipped", len(res["papers"]) - ok)
    by_src = Counter(p.get("Text_Source") for p in res["papers"] if p.get("Status") == "ok")
    if by_src.get("Abstract only") or by_src.get("Title only"):
        st.caption("📑 " + " · ".join(f"{n} from {s.lower()}" for s, n in by_src.most_common() if s) +
                   ". Abstract-only rows have fewer details (no PDF was available); see the Text_Source column.")
    n_local = sum(1 for p in res["papers"] if local_backends.is_local(p.get("Extracted_By")))
    if n_local:
        st.warning(f"🛟 {n_local} paper(s) were filled by the **local extractor** because the API failed "
                   "(see Extracted_By and the run log). These rows only keep values written in the text."
                   " Run them again when the API is available for full answers.")
    if res.get("duplicates"):
        st.caption(f"♻️ {res['duplicates']} duplicate(s) were processed only once: the same file uploaded twice, "
                   "or a listed paper whose PDF you uploaded. See the run log for details.")

    failed = [p for p in res["papers"] if p.get("Status") != "ok"]
    if failed:
        # group identical reasons so one bad key doesn't print the same message 50 times
        reasons: dict[str, list[str]] = {}
        for p in failed:
            reasons.setdefault(p.get("Reason") or p.get("Status", ""), []).append(p.get("Paper_ID") or p.get("Source"))
        st.error("**Why papers failed**\n\n" + "\n\n".join(
            f"- **{len(ids)} paper(s):** {reason}" for reason, ids in reasons.items()), icon="⚠️")
        st.dataframe(pd.DataFrame([{"Paper": p.get("Paper_ID"), "Title": p.get("Title"),
                                    "Sheets": p.get("Found_In_Sheets"), "Source": p.get("Source"),
                                    "Reason": p.get("Reason"), "Technical details": p.get("Status")}
                                   for p in failed]),
                     width="stretch", hide_index=True)

    with st.expander("Run log", expanded=bool(failed)):
        log_text = "\n".join(res.get("log", []))
        st.code(log_text or "(empty)", language=None)
        st.download_button("Download log", log_text, f"extraction_log_{res['stamp']}.txt", "text/plain")

    if not res["records"]:
        st.stop()

    xlsx = build_workbook(res["columns"], res["records"], res["papers"])
    jbuf = io.BytesIO()
    with zipfile.ZipFile(jbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, js in res["jsons"].items():
            zf.writestr(f"{stem}.json", json.dumps(js, indent=2, ensure_ascii=False))
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download Excel", xlsx, f"extraction_{res['provider']}_{res['stamp']}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", width="stretch")
    d2.download_button("⬇️ Download JSON (one file per paper)", jbuf.getvalue(),
                       f"extraction_{res['provider']}_{res['stamp']}_json.zip", "application/zip",
                       width="stretch")
    if res.get("categories"):
        # one workbook per category (a paper listed in several categories appears in each of them)
        cbuf = io.BytesIO()
        cat_cols = [c for c in res["columns"] if c != "Category"]
        with zipfile.ZipFile(cbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for cname in res["categories"]:
                zf.writestr(f"{sources.category_label(cname)}.xlsx", build_workbook(
                    cat_cols, [r for r in res["records"] if cname in r.get("_categories", [])],
                    [p for p in res["papers"] if cname in p.get("_categories", [])]))
        st.download_button(f"⬇️ Download one Excel per category ({len(res['categories'])} categories, ZIP)",
                           cbuf.getvalue(), f"extraction_by_category_{res['stamp']}.zip", "application/zip",
                           width="stretch")

    st.dataframe(pd.DataFrame(res["records"], columns=res["columns"]), width="stretch", hide_index=True)
    with st.expander("Per-paper status"):
        st.dataframe(pd.DataFrame(res["papers"]).drop(columns=["Reason"], errors="ignore"),
                     width="stretch", hide_index=True)
