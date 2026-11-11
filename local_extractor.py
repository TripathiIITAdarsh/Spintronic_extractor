"""Local, deterministic fallback for the "Device & materials" features (no API, no model).

Used when the LLM API fails (quota, usage limit, overload, bad key). It only reports
what is literally written in the text, so it never invents a value, but it leaves more
fields empty than an LLM and cannot reason about roles that are only implied.

  Device_Type               keyword rules on title + text
  Tunnel_Barrier_Material   known barrier near "barrier", or the barrier inside a stack like Fe/MgO/Fe
  Free / Reference layer    material next to "free layer" / "pinned layer" ..., or both electrodes of a
                            symmetric stack (Fe/MgO/Fe -> Fe, Fe)
  Heavy_Metal_SOT_Layer     known heavy metal, only when the text is about spin-orbit torque / spin Hall
  Substrate                 known substrate after "on ... substrate" / "grown on"
  Main_Material             chemical formulas in the title, else the most frequent ones in the text
"""
import re
from collections import Counter

from schema import normalize_device_type

ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br
Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta
W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu""".split())

BARRIERS = ["MgO", "Al2O3", "AlOx", "AlO", "h-BN", "hBN", "BN", "MgAl2O4", "SrTiO3", "BaTiO3", "PbTiO3",
            "EuO", "EuS", "CrI3", "CrBr3", "MoS2", "WS2", "WSe2", "MoSe2", "HfO2", "TiO2", "ZnO", "Ga2O3",
            "SiO2", "graphene", "In2Se3", "CuInP2S6", "BiFeO3", "La2O3", "NiO", "CoFe2O4", "NiFe2O4",
            "GaAs", "Si", "Ge", "vacuum", "vdW gap"]
HEAVY_METALS = ["beta-W", "β-W", "Pt", "Ta", "W", "Hf", "Ir", "Pd", "Au0.25Pt0.75", "PtMn", "IrMn", "Bi2Se3",
                "Bi2Te3", "BiSb", "Sb2Te3", "(Bi,Sb)2Te3", "WTe2", "MoTe2", "PtTe2", "TaS2", "NbSe2", "Mo"]
SUBSTRATES = ["Si/SiO2", "SiO2/Si", "Si(001)", "Si(100)", "Si", "MgO(001)", "MgO(100)", "MgO", "sapphire",
              "Al2O3(0001)", "SrTiO3(001)", "SrTiO3", "LaAlO3", "GaAs(001)", "GaAs", "glass", "quartz",
              "GGG", "Gd3Ga5O12", "SiC", "mica", "PMN-PT", "KTaO3", "DyScO3", "NdGaO3"]
SINGLE_METALS = {"Fe", "Co", "Ni", "Cr", "Mn", "Pt", "Ta", "W", "Ru", "Cu", "Au", "Ag", "Pd", "Ir", "Hf", "Gd", "Tb"}
COMMON_ALIASES = {"permalloy": "NiFe", "py": "NiFe", "h-bn": "hBN", "hexagonal boron nitride": "hBN"}

FORMULA = re.compile(r"(?<![A-Za-z0-9])((?:\(?[A-Z][a-z]?(?:,\s?[A-Z][a-z]?)*(?:\d+(?:\.\d+)?|[xyδ])?\)?"
                     r"(?:\d+(?:\.\d+)?)?(?:[-–][xyδ\d.]+)?){1,8})(?![A-Za-z0-9])")
NOT_MAIN = {"sio2", "si", "si3n4", "sin", "al2o3", "h2o", "o2", "n2", "co2", "ar"}   # substrates, gases, ...
STACK = re.compile(r"(?<![\w/])((?:[A-Za-z][\w().,\-]*\s*/\s*){2,}[A-Za-z][\w().,\-]*)")
ROLE_WORDS = {
    "Free_Layer_Material": r"free (?:magnetic )?layer|storage layer|recording layer|soft layer|switching layer",
    "Reference_Layer_Material": r"reference layer|pinned layer|fixed layer|hard layer|polarizer",
    "Tunnel_Barrier_Material": r"tunnel(?:ing)? barrier|barrier layer|insulating barrier|spacer|barrier",
    "Substrate": r"substrates?|grown on|deposited on|epitaxially on",
}


def _compact(text: str) -> str:
    """'Fe 4 GeTe 2' (PDF subscripts) -> 'Fe4GeTe2'."""
    t = text.replace("–", "-").replace("−", "-")
    for _ in range(3):
        t = re.sub(r"([A-Z][a-z]?) (\d+(?:\.\d+)?)(?=[ A-Z(/,.;)-]|$)", r"\1\2", t)
        # 'Fe4 GeTe2' -> 'Fe4GeTe2': a subscripted piece followed by the rest of the formula
        t = re.sub(r"([A-Z][a-z]?\d+(?:\.\d+)?) (?=(?:[A-Z][a-z]?){1,3}\d)", r"\1", t)
    return t


def _is_formula(tok: str, allow_single=False) -> bool:
    syms = re.findall(r"[A-Z][a-z]?", tok)
    if not syms or any(s not in ELEMENTS for s in syms):
        return False
    if "".join(syms) != re.sub(r"[^A-Za-z]", "", tok).replace("x", "").replace("y", ""):
        return False
    if len(syms) == 1:
        return allow_single and syms[0] in SINGLE_METALS and not re.search(r"\d", tok)
    # all-caps runs like SOC, IV, NO are words, not formulas (unless they carry digits)
    return bool(re.search(r"[a-z\d]", tok))


def _tidy(tok: str) -> str:
    tok = tok.rstrip("-–.")
    while tok.count("(") != tok.count(")") and tok and tok[0] in "()" or (tok.endswith(")") and tok.count(")") > tok.count("(")):
        tok = tok[1:] if tok[0] in "()" and tok.count("(") != tok.count(")") else tok[:-1]
    return tok


def formulas(text: str, allow_single=False) -> list[str]:
    return [_tidy(m.group(1)) for m in FORMULA.finditer(text) if _is_formula(m.group(1), allow_single)]


def _in_list(tok: str, known: list[str]) -> str | None:
    t = re.sub(r"\s+", "", tok).lower()
    for k in known:
        if t == k.lower() or t.startswith(k.lower() + "("):
            return k if "(" not in t else tok
    return None


def _stacks(text: str) -> list[list[str]]:
    out = []
    for m in STACK.finditer(text):
        layers = [re.sub(r"\(\s*[\d.]+\s*(?:nm|Å|A)?\s*\)$", "", p.strip()) for p in m.group(1).split("/")]
        layers = [l for l in layers if l]
        if len(layers) >= 3:
            out.append(layers)
    return out


def _near(sentence: str, pattern: str, candidates: list[str], window=60) -> str | None:
    """Candidate closest to a role word in the same sentence."""
    best = None
    for rm in re.finditer(pattern, sentence, re.I):
        for c in candidates:
            for cm in re.finditer(re.escape(c), sentence):
                d = min(abs(cm.start() - rm.end()), abs(rm.start() - cm.end()))
                if d <= window and (best is None or d < best[0]):
                    best = (d, c)
    return best[1] if best else None


def device_type(title: str, text: str) -> str | None:
    t, x = (title or "").lower(), (text or "").lower()
    both = t + " " + x
    if re.search(r"\breview\b|\broadmap\b|\bperspective\b|recent advances|progress in", t) or \
            re.search(r"\bthis review\b|\bwe review\b", x[:3000]):
        return "review"
    mtj = re.search(r"tunnel(?:ing)? junction|\bmtjs?\b|tunnel(?:ing)? magnetoresistance|\btmr\b|\bmram\b", both)
    if mtj:
        # judge "vdW" only from the title + opening text: full papers mention 2D / graphene in passing
        head = t + " " + x[:2500]
        if re.search(r"van der waals|\bvdw\b|\b2d (?:material|magnet|ferromagnet)|two-dimensional|"
                     r"graphene|\bh-?bn\b|fe3gete2|cri3|crbr3|mos2|ws2|wse2|mnbi2te4", head):
            return "vdW MTJ"
        if re.search(r"spin[- –]orbit torque|\bsot\b|spin hall", both):
            return "SOT-MTJ"
        if re.search(r"voltage[- ]control|\bvcma\b|electric[- ]field[- ]control", both):
            return "VCMA-MTJ"
        if re.search(r"spin[- ]transfer torque|\bstt\b|current[- ]induced (?:magnetization )?switching", both):
            return "STT-MTJ"
        return "MTJ (other)"
    if re.search(r"spin[- ]valve|giant magnetoresistance|\bgmr\b", both):
        return "spin valve"
    if re.search(r"spin[- ]fet|spin field[- ]effect transistor|spin transistor", both):
        return "spin-FET"
    if re.search(r"nanoparticle|nanocrystal", both):
        return "nanoparticle"
    first_principles = re.search(r"first[- ]principles|density functional|\bdft\b|ab initio|\bnegf\b", both)
    if re.search(r"thin film|multilayer|heterostructure|bilayer|hall bar|epitaxial|flake|exfoliat|nanolayer|"
                 r"spin[- –]orbit torque|\bsot\b|spin hall|\bdevices?\b", both) and not (
            first_principles and not re.search(r"\bexperiment|\bmeasured|\bfabricat|\bgrown\b", both)):
        return "thin film"
    if re.search(r"single crystal|bulk|polycrystalline|powder", both) and not first_principles:
        return "bulk"
    if first_principles:
        return "simulation/theory"
    return None


def extract(title: str, text: str) -> dict:
    """-> {feature: value or None} for the device & materials features (no Title/Year)."""
    title, text = title or "", text or ""
    ct = _compact(title)
    cx = _compact(text)
    for a, b in COMMON_ALIASES.items():
        cx = re.sub(rf"\b{re.escape(a)}\b", b, cx, flags=re.I)
    sentences = re.split(r"(?<=[.;])\s+", ct + ". " + cx)
    out = {"Device_Type": device_type(title, text), "Main_Material": None, "Free_Layer_Material": None,
           "Tunnel_Barrier_Material": None, "Reference_Layer_Material": None, "Heavy_Metal_SOT_Layer": None,
           "Substrate": None}

    # 1) stacks like Ta/CoFeB/MgO/CoFeB/Ta or Fe/MgO/Fe
    for layers in _stacks(ct + " " + cx):
        idx = next((i for i, l in enumerate(layers) if 0 < i < len(layers) - 1 and _in_list(l, BARRIERS)), None)
        if idx is None:
            continue
        out["Tunnel_Barrier_Material"] = out["Tunnel_Barrier_Material"] or layers[idx]
        left, right = layers[idx - 1], layers[idx + 1]
        if re.sub(r"\(.*?\)", "", left) == re.sub(r"\(.*?\)", "", right):     # symmetric junction
            out["Free_Layer_Material"] = out["Free_Layer_Material"] or left
            out["Reference_Layer_Material"] = out["Reference_Layer_Material"] or right
        break

    # 2) explicit role words in a sentence
    for s in sentences:
        cands = list(dict.fromkeys(formulas(s, allow_single=True)))
        for feat in ("Free_Layer_Material", "Reference_Layer_Material"):
            if not out[feat] and re.search(ROLE_WORDS[feat], s, re.I):
                hit = _near(s, ROLE_WORDS[feat], [c for c in cands if not _in_list(c, BARRIERS)])
                if hit:
                    out[feat] = hit
        if not out["Tunnel_Barrier_Material"] and re.search(ROLE_WORDS["Tunnel_Barrier_Material"], s, re.I):
            known = [b for b in BARRIERS if re.search(rf"(?<![\w-]){re.escape(b)}(?![\w])", s)]
            hit = _near(s, ROLE_WORDS["Tunnel_Barrier_Material"], known)
            if hit:
                out["Tunnel_Barrier_Material"] = hit
        if not out["Substrate"] and re.search(ROLE_WORDS["Substrate"], s, re.I):
            known = [b for b in SUBSTRATES if re.search(rf"(?<![\w-]){re.escape(b)}(?![\w])", s)]
            hit = _near(s, ROLE_WORDS["Substrate"], known, window=40)
            if hit:
                out["Substrate"] = hit

    # 3) heavy metal only for spin-orbit-torque / spin-Hall papers
    if re.search(r"spin[- –]orbit torque|\bsot\b|spin hall", (title + " " + text).lower()):
        for s in sentences:
            if re.search(r"spin hall|heavy[- ]metal|spin[- –]orbit torque|\bsot\b|channel", s, re.I):
                known = [h for h in HEAVY_METALS if re.search(rf"(?<![\w-]){re.escape(h)}(?![\w])", s)]
                hit = _near(s, r"spin hall|heavy[- ]metal|spin[- –]orbit torque|\bSOT\b|channel", known)
                if hit:
                    out["Heavy_Metal_SOT_Layer"] = hit
                    break

    # 4) main material: formulas in the title, else the most frequent in the text
    in_title = list(dict.fromkeys(f for f in formulas(ct) if f.lower() not in NOT_MAIN))
    if in_title:
        out["Main_Material"] = "; ".join(in_title[:3])
    else:
        counts = Counter(f for f in formulas(cx) if f.lower() not in NOT_MAIN and not _in_list(f, SUBSTRATES))
        top = [f for f, n in counts.most_common(2) if n >= 2]
        if top:
            out["Main_Material"] = "; ".join(top)
    if not out["Main_Material"] and out["Free_Layer_Material"]:
        out["Main_Material"] = out["Free_Layer_Material"]
    if out["Device_Type"] is not None:
        out["Device_Type"] = normalize_device_type(out["Device_Type"])
    return out
