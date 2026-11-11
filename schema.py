"""Target feature schema for spintronics paper extraction.

Each paper can describe several devices / material stacks, so the LLM returns a
list of records; every record has all of these fields (null when absent).
"""

FIELDS = {
    # --- bibliographic ---
    "Paper_ID": "Filled by the pipeline (PDF file stem); leave null",
    "Title": "Paper title",
    "Authors": "Authors, semicolon-separated (first 10 max, then 'et al.')",
    "Year": "Publication year (int)",
    "Journal": "Journal name",
    "DOI_Link": "https://doi.org/<doi>",
    "Source_Type": "Experimental | Theoretical/DFT | Simulation | Review | Mixed",
    # --- device / stack ---
    "Device_Type": "EXACTLY one of: STT-MTJ | SOT-MTJ | VCMA-MTJ | vdW MTJ | MTJ (other) | spin valve | "
                   "spin-FET | thin film | bulk | nanoparticle | simulation/theory | review | other. "
                   "Use the device studied even when it is only simulated (a DFT/NEGF transport study of Fe/MgO/Fe "
                   "is 'MTJ (other)'; of a 2D-material junction is 'vdW MTJ'). 'simulation/theory' only when no "
                   "device is modelled (e.g. DFT of a bulk compound). 'review' for review articles",
    "Junction_Geometry": "e.g. circular pillar 50 nm diameter, elliptical 100x200 nm, Hall bar",
    "Junction_Area_nm2": "Junction area in nm^2 (convert from um^2: x1e6)",
    "Anisotropy_Type": "PMA | IMA | Mixed",
    "Free_Layer_Material": "e.g. CoFeB, Co40Fe40B20, Fe3GeTe2",
    "Free_Layer_Thickness_nm": "nm",
    "Reference_Layer_Material": "Pinned/reference layer material",
    "Reference_Layer_Thickness_nm": "nm",
    "Tunnel_Barrier_Material": "e.g. MgO, AlOx, hBN",
    "Tunnel_Barrier_Thickness_nm": "nm",
    "Heavy_Metal_SOT_Layer": "e.g. W, Ta, Pt, Au0.25Pt0.75",
    "Heavy_Metal_Thickness_nm": "nm",
    "Seed_Layer": "Seed/buffer layer stack",
    "Capping_Layer": "Capping layer stack",
    "SAF_Structure": "Synthetic antiferromagnet stack, e.g. [Co/Pt]n/Ru/[Co/Pt]m",
    "Interface_Insertion_Layer": "e.g. 0.4 nm Mg, W dusting, Hf",
    # --- transport ---
    "TMR_percent": "Tunnel magnetoresistance in % (state temperature in Notes if not RT)",
    "RA_product_ohm_um2": "Ohm*um^2",
    "R_parallel_ohm": "Ohm",
    "R_antiparallel_ohm": "Ohm",
    "Conductance_S": "Siemens",
    "Critical_Switching_Voltage_V": "V",
    "Critical_Switching_Current_uA": "uA (convert mA x1000)",
    "Current_Density_A_cm2": "Critical STT switching current density, A/cm^2 (convert A/m^2 x1e-4)",
    "V_half_V": "Bias voltage where TMR drops to half, V",
    "Read_Write_Voltage_Window_V": "V",
    # --- magnetic ---
    "Saturation_Magnetization_Ms_emu_cc": "emu/cm^3 (kA/m == emu/cc; T -> x795.8)",
    "Coercivity_Hc_Oe": "Oe (1 mT = 10 Oe; 1 kA/m = 12.566 Oe)",
    "Anisotropy_Field_Hk_Oe": "Oe",
    "Exchange_Bias_Field_Oe": "Oe",
    "Gilbert_Damping_Alpha": "dimensionless",
    "Spin_Polarization_P": "fraction 0-1 (convert % / 100)",
    "Curie_Temperature_K": "K (convert C + 273.15)",
    "Thermal_Stability_Factor_Delta": "dimensionless",
    # --- SOT ---
    "Spin_Hall_Angle": "dimensionless (convert % / 100)",
    "SOT_Efficiency": "dimensionless damping-like efficiency xi_DL",
    "SOT_Switching_Current_Density_A_cm2": "A/cm^2",
    "Field_Free_Switching_YN": "Y | N",
    "Damping_Field_Like_Torque_Ratio": "dimensionless",
    # --- fabrication / measurement ---
    "Deposition_Method": "e.g. DC/RF magnetron sputtering, MBE, PLD, mechanical exfoliation, CVD",
    "Annealing_Temperature_C": "deg C",
    "Annealing_Duration_min": "minutes",
    "Measurement_Temperature_K": "K",
    "Substrate": "e.g. Si/SiO2, MgO(001), sapphire",
    # --- computational ---
    "Exchange_Correlation_Functional": "e.g. PBE, LDA, HSE06, PBE+U",
    "Hubbard_U_eV": "eV",
    "Spin_Orbit_Coupling_Included_YN": "Y | N",
    "Predicted_Magnetic_Moment_muB": "mu_B (per formula unit or per atom; say which in Notes)",
    "Formation_Energy_eV": "eV (per atom or per f.u.; say which in Notes)",
    "Half_Metallicity_YN": "Y | N",
    "Predicted_Curie_Temp_K": "K",
    "Lattice_Constant_Angstrom": "Angstrom (convert nm x10)",
    # --- provenance / QA ---
    "Value_Source": "Where numbers came from: Text | Table | Figure | Abstract | Mixed",
    "Units_Standardized_YN": "Y if all numeric fields were converted to the column units",
    "Missing_Value_Flag": "Count of null technical fields in this record (int) -- filled by pipeline",
    "Extraction_Confidence": "High | Medium | Low",
    "Notes": "Caveats: conditions (temperature/bias) of values, ranges, which device, conversions done",
}

COLUMNS = list(FIELDS)

# Feature groups (used as presets in the web app); order follows FIELDS
_GROUP_STARTS = [
    ("Bibliographic", "Paper_ID"),
    ("Device / stack", "Device_Type"),
    ("Transport", "TMR_percent"),
    ("Magnetic", "Saturation_Magnetization_Ms_emu_cc"),
    ("SOT", "Spin_Hall_Angle"),
    ("Fabrication / measurement", "Deposition_Method"),
    ("Computational (DFT)", "Exchange_Correlation_Functional"),
    ("Provenance / QA", "Value_Source"),
]
GROUPS: dict[str, list[str]] = {}
for (g, start), nxt in zip(_GROUP_STARTS, _GROUP_STARTS[1:] + [(None, None)]):
    a = COLUMNS.index(start)
    b = COLUMNS.index(nxt[1]) if nxt[1] else len(COLUMNS)
    GROUPS[g] = COLUMNS[a:b]

# Features that are not in the original 64-column sheet (offered in the web app and
# used by the per-category extraction)
ADDITIONAL_FIELDS = {
    "Main_Material": "The main magnetic / spintronic material(s) the paper studies, as chemical formula or name "
                     "(e.g. Fe3GeTe2, Mn3Sn, Co2MnSi, CoFeB, NiO). For a device: its key functional material "
                     "(usually the free layer or the new material). Several: separate with '; '",
}

# "Device type and material" feature set (ma'am's request)
DEVICE_MATERIAL_FEATURES = ["Title", "Year", "Device_Type", "Main_Material", "Free_Layer_Material",
                            "Tunnel_Barrier_Material", "Reference_Layer_Material", "Heavy_Metal_SOT_Layer",
                            "Substrate"]
GROUPS = {"Device & materials": DEVICE_MATERIAL_FEATURES, **GROUPS}

DEVICE_TYPES = ["STT-MTJ", "SOT-MTJ", "VCMA-MTJ", "vdW MTJ", "MTJ (other)", "spin valve", "spin-FET",
                "thin film", "bulk", "nanoparticle", "simulation/theory", "review", "other"]


def normalize_device_type(value) -> str | None:
    """Map whatever the model wrote onto DEVICE_TYPES so categories group cleanly."""
    if value is None or not str(value).strip():
        return None
    v = str(value).strip().lower()
    squash = lambda s: s.replace(" ", "").replace("-", "").replace("_", "")
    for t in DEVICE_TYPES:
        if squash(v) == squash(t.lower()):
            return t
    is_mtj = "mtj" in v or "mram" in v or "tunnel junction" in v or "tunneling junction" in v
    if "review" in v:
        return "review"
    if is_mtj and any(k in v for k in ("vdw", "van der waals", "2d")):
        return "vdW MTJ"
    if is_mtj and ("sot" in v or "spin-orbit torque" in v or "spin orbit torque" in v):
        return "SOT-MTJ"
    if is_mtj and ("vcma" in v or "voltage" in v):
        return "VCMA-MTJ"
    if is_mtj and ("stt" in v or "spin-transfer" in v or "spin transfer" in v):
        return "STT-MTJ"
    if is_mtj:
        return "MTJ (other)"
    if "spin valve" in v or "spin-valve" in v:
        return "spin valve"
    if "spin" in v and ("fet" in v or "transistor" in v):
        return "spin-FET"
    if "nanoparticle" in v or "nanocrystal" in v:
        return "nanoparticle"
    if any(k in v for k in ("film", "multilayer", "heterostructure", "hall bar", "bilayer", "monolayer")):
        return "thin film"
    if any(k in v for k in ("bulk", "single crystal", "crystal", "powder")):
        return "bulk"
    if any(k in v for k in ("dft", "first-principles", "first principles", "simulation", "theory", "theoretical")):
        return "simulation/theory"
    return "other"

# Fields the pipeline fills itself / that don't count as "technical"
META_FIELDS = {
    "Paper_ID", "Title", "Authors", "Year", "Journal", "DOI_Link", "Source_Type",
    "Value_Source", "Units_Standardized_YN", "Missing_Value_Flag",
    "Extraction_Confidence", "Notes",
}

NUMERIC_FIELDS = {
    k for k in COLUMNS
    if any(k.endswith(s) for s in (
        "_nm", "_nm2", "_percent", "_ohm", "_ohm_um2", "_S", "_V", "_uA",
        "_A_cm2", "_emu_cc", "_Oe", "_Alpha", "_P", "_K", "_Delta", "_C",
        "_min", "_eV", "_muB", "_Angstrom", "_Angle", "_Efficiency", "_Ratio",
    ))
} | {"Year"}
