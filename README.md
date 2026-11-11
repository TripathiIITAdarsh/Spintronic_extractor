# Spintronics feature extraction

Extracts the 63 columns in `schema.py` from every PDF in `../1 Spintronics`. Each PDF is run through two LLMs, and each LLM writes its own JSON file.

## Pipeline
1. `pdf_parser.py` uses PyMuPDF to parse the PDF. It drops References and Acknowledgements, keeps the first-page header, the abstract, numeric tables and the figure/table captions, and then adds body paragraphs ranked by spintronics keywords and how many numbers with units they contain. The whole paper is never sent: the context is capped at about 28k characters. The parsed context is cached in `parsed/<stem>.json`.
2. `llm_clients.py` holds the prompt and the two clients:
   - **Gemini** uses the REST API with `GEMINI_API_KEY` from `.env`. The model is `gemini-3.8-flash`. Calls are spaced 7 s apart to stay within the free-tier rate limit.
   - **Claude** runs Claude Code in headless mode (`claude -p`), which bills to your subscription. If `ANTHROPIC_API_KEY` is set, it uses the Anthropic API instead.
3. `run_extraction.py` normalizes the results: it converts numbers, computes `Missing_Value_Flag` and sets `Paper_ID` to the file stem.

A paper can contain several devices or materials, so each JSON has a `records` list with one row per device. In the CSV, each record becomes one row.

## Run
```
python run_extraction.py                      # both LLMs, all PDFs (skips ones already done)
python run_extraction.py --provider gemini
python run_extraction.py --only 2308.13408v2 --force
```

## Outputs
- `outputs/gemini/<stem>.json`, `outputs/claude/<stem>.json`
- `outputs/gemini_all.csv`, `outputs/claude_all.csv`: the exact column order you requested
- `outputs/<provider>_errors.log`: papers that failed (rerun to retry)

## Web app (`app.py`)
```
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```
1. **Papers**: upload any mix of PDFs, ZIP folders containing PDFs (nested folders are fine), or an Excel/CSV file with paper links. You can also paste DOIs or URLs. Links are resolved to PDFs through arXiv, an open-access copy found via OpenAlex, or the publisher page's `citation_pdf_url`. Paywalled papers can't be downloaded automatically. They are listed as failed so you can upload those PDFs yourself. Every sheet of a multi-sheet Excel file is read by default, and you can untick sheets in the app. A paper is processed only once, even when it is listed in several sheets: rows count as the same paper when they share a DOI, an arXiv ID or a title. If you upload a PDF (or a ZIP of PDFs) for a paper that is also in the sheet, the uploaded file is used and nothing is downloaded. This is how to include paywalled papers. The output has a `Found_In_Sheets` column.
2. **Features**: pick features from the dropdown, or add a whole group with the "+ Transport", "+ Magnetic", … buttons. Use **Add a new feature** to define a new feature (name, type, unit, description). It is saved to `custom_features.json` and marked with ★ in the dropdown.
3. **Extract**: you get an Excel file with the selected columns plus a per-paper status sheet, and a ZIP containing one JSON file per paper.

## Hosting
The app is a standard Streamlit app. Free hosts that work:
- **Streamlit Community Cloud** (share.streamlit.io): push this folder to a GitHub repo, then choose New app → `app.py`.
- **Hugging Face Spaces**: create a Space with the Streamlit SDK and upload these files.
- **Render** (render.com): push this folder to GitHub, then choose New → Blueprint and pick the repo. `render.yaml` sets the build and start commands and asks for `GEMINI_API_KEY`, `APP_PASSWORD` . The free instance sleeps after about 15 min without visitors, so the first load after that takes about a minute. It also has only 512 MB of RAM and a small CPU share, so upload papers in batches of about 20–30 and expect parsing to be slower than on your laptop.

Put the keys in the host's **Secrets** settings (see `.streamlit/secrets.toml.example`), not in the repo. `.env` is git-ignored.

Things to know before hosting:
- The web app uses Gemini only, with the key from `.env` or the host's secrets. There is no key field in the UI. The sidebar shows whether the key works.
- Gemini's free tier allows about 20 requests per day per model. For real use, enable billing on the Gemini key.
- Set `APP_PASSWORD`. Otherwise anyone with the link can spend your API quota.
- On free hosts, `custom_features.json` is reset when the app restarts.

## Device type & materials per category (`extract_categories.py`)
This extracts ma'am's feature set from every paper-search export in `../Agent_paper_list/exports` and writes **one Excel per category** to `../Category_Features/`. The features are Title, Year, Device_Type, Main_Material, Free_Layer_Material, Tunnel_Barrier_Material, Reference_Layer_Material, Heavy_Metal_SOT_Layer and Substrate.
```
python extract_categories.py                    # local Claude Code (subscription)
python extract_categories.py --provider gemini  # Gemini API key from .env
python extract_categories.py --report-only      # rebuild the Excel files from the cache
```
- **Category name:** the "Query" row of each export's Summary sheet, e.g. `MTJ DFT NEGF Heusler and half heusler compound.xlsx`.
- **Each paper is extracted once,** even if it is listed in several sheets or several categories. The result is reused in every category file that lists it.
- **Text source:** full text when a free PDF exists. Otherwise the title and abstract from the sheet are used, 10 papers per call. The `Text_Source` column says which.
- **Device_Type** is always one of: STT-MTJ, SOT-MTJ, VCMA-MTJ, vdW MTJ, MTJ (other), spin valve, spin-FET, thin film, bulk, nanoparticle, simulation/theory, review, other.
- **Resumable:** results are cached in `_cache/`. If a usage limit stops the run, run the same command again to continue.
- **Same in the web app:** the "Device & materials" preset (the default selection) does the same thing. Upload the export Excel files, and paywalled papers fall back to their abstract. A "one Excel per category" ZIP is offered.

## Local extractor: swap in a better model without changing code
When the API fails (quota, usage limit, overload, bad key), papers are filled by a **local extractor**. Choose it in `.env`:
```
LOCAL_BACKEND=rules                          # default: keyword rules, no model
LOCAL_BACKEND=ollama                         # any Ollama model
LOCAL_MODEL=qwen3:14b
LOCAL_BACKEND=openai                         # any OpenAI-compatible server (LM Studio, vLLM, llama.cpp,
LOCAL_MODEL=<model>                          #   a lab GPU server, Colab + tunnel)
LOCAL_URL=http://host:port
LOCAL_BACKEND=python:my_model:extract        # your own trained model (copy local_model_template.py)
```
- **Optional settings:** `LOCAL_MAX_CHARS` (default 12000), `LOCAL_NUM_CTX` (Ollama, default 8192), `LOCAL_TIMEOUT`, `LOCAL_API_KEY`.
- **Safety net for every backend:** material values that are not written in the paper are dropped, and Device_Type is mapped onto the fixed list. If the backend errors or returns nothing usable, the keyword rules answer instead.
- **Output columns:** rows show the backend in `Extracted_By`, e.g. `Local (ollama: qwen3:14b)`.
- **Upgrading earlier rows:** `python extract_categories.py --local-only` redoes rows from a *different* local backend with the one configured now. A normal run with the API redoes every local row.
- **Measuring a backend against Claude:** `python eval_local.py` measures the rules. For a model backend, set `LOCAL_BACKEND` and compare its output files with Claude's.
