"""Template for plugging YOUR OWN model in as the local extractor.

1. Copy this file (e.g. to my_model.py) and fill in `extract`.
2. In .env set:   LOCAL_BACKEND=python:my_model:extract
3. Restart the app / rerun extract_categories.py. Nothing else changes.

Whatever you return goes through the same safety net as every local backend: material values
that are not written in the paper text are dropped, and Device_Type is mapped onto the fixed list.
"""


def extract(title: str, text: str, fields: dict) -> list[dict]:
    """
    title  : paper title
    text   : the paper text the API would have seen (PDF excerpt, or the abstract)
    fields : {feature name: description}, e.g. {"Device_Type": "...", "Main_Material": "...", ...}

    Return one dict per device / material system in the paper, keys = feature names
    (missing keys or None = not found). Example:
        [{"Device_Type": "SOT-MTJ", "Main_Material": "CoFeB", "Tunnel_Barrier_Material": "MgO"}]
    """
    # --- example: a fine-tuned Hugging Face token-classification model ---------------------
    # from transformers import AutoTokenizer, AutoModelForTokenClassification
    # (load once at module level, run it on `text`, map entity labels -> feature names)
    raise NotImplementedError("fill in your model here")
