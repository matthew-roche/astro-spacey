import re, math

# aggressive filtering for clustering

ENTITY_RULES = {
    "Mercury": {
        "positives": [r"Mercury", r"Caloris"],
        "aliases":   [r"MESSENGER", r"MDIS", r"MLA", r"XRS", r"GRNS"],
        "negatives": [r"\bcorona\b"]   # dropped: Pluto, New Horizons, Kuiper Belt
    },
    "Venus": {
        "positives": [r"Venus", r"Venera"],
        "aliases":   [r"Akatsuki", r"Magellan"],
        "negatives": []                # dropped: Mars/Earth/Pluto
    },
    "Moon": {
        "positives": [r"\bMoon\b", r"\bLunar\b"],
        "aliases":   [r"LRO", r"LADEE", r"Mini-?RF", r"CRaTER"],
        "negatives": []                # dropped: Mercury/Pluto
    },
    "Mars": {
        "positives": [r"\bMars\b", r"Martian"],
        "aliases":   [r"Curiosity|MSL", r"Opportunity|Spirit", r"MRO|CRISM|MARCI",
                      r"MAVEN", r"Phoenix", r"TGO", r"REMS", r"SAM", r"RAD"],
        "negatives": []
    },
    "Jupiter": {
        "positives": [r"Jupiter", r"GRS|Great Red Spot"],
        "aliases":   [r"Juno\b", r"MWR\b"],
        "negatives": []
    },
    "Saturn": {
        "positives": [r"Saturn"],
        "aliases":   [r"Cassini", r"INMS"],
        "negatives": []
    },
    "Titan": {
        "positives": [r"Titan\b"],
        "aliases":   [r"Cassini", r"Dragonfly"],
        "negatives": []
    },
    "Enceladus": {
        "positives": [r"Enceladus\b"],
        "aliases":   [r"Cassini", r"INMS"],
        "negatives": []
    },
    "Europa": {
        "positives": [r"Europa\b"],
        "aliases":   [r"Clipper", r"Galileo\b"],
        "negatives": []
    },
    "Ganymede": {
        "positives": [r"Ganymede\b"],
        "aliases":   [r"JUICE\b"],
        "negatives": []
    },
    "Io": {
        "positives": [r"\bIo\b"],
        "aliases":   [r"Juno\b", r"Galileo\b"],
        "negatives": []
    },
    "Callisto": {
        "positives": [r"Callisto\b"],
        "aliases":   [r"JUICE\b"],
        "negatives": []
    },
    "Uranus": {
        "positives": [r"Uranus\b"],
        "aliases":   [r"Hubble|JWST|Keck"],
        "negatives": []
    },
    "Neptune": {
        "positives": [r"Neptune\b"],
        "aliases":   [r"Voyager 2", r"Hubble|JWST|Keck|VLT|Gemini\b|MUSE\b"],
        "negatives": []
    },
    "Triton": {
        "positives": [r"Triton\b"],
        "aliases":   [r"Voyager 2"],
        "negatives": []
    },
    "Pluto": {
        "positives": [r"Pluto\b"],
        "aliases":   [r"New Horizons"],
        "negatives": []
    },
    "Ceres": {
        "positives": [r"Ceres\b", r"Occator"],
        "aliases":   [r"Dawn\b"],
        "negatives": []
    }
}

# compile per-planet
COMPILED = {
    k: {
        "pos": re.compile("|".join(v["positives"]), re.I),
        "ali": re.compile("|".join(v["aliases"])) if v["aliases"] else None,
        "neg": re.compile("|".join(v["negatives"]), re.I) if v["negatives"] else None
    }
    for k, v in ENTITY_RULES.items()
}

def passes_entity_filter(planet, title, content, require_alias=False):
    rules = COMPILED.get(planet)
    if rules is None: 
        return True
    txt = f"{title} {content}"
    if rules["neg"] and rules["neg"].search(txt):
        return False
    has_pos = bool(rules["pos"].search(txt))
    if require_alias and rules["ali"]:
        has_alias = bool(rules["ali"].search(txt))
        return has_pos and has_alias
    return has_pos

DETECTION_RX = re.compile(r"\b(found|detect(?:ed|ion)?|evidence|observ(?:e|ed)|confirm(?:ed|ation)?|measured?)\b", re.I)
EXO_RX = re.compile(
    r"\b(exoplanet|super[- ]earth|hot[- ]jupiter|mini[- ]neptune|"
    r"transit|radial velocity|rv signal|toi[- ]\d+|k2[- ]\d+|tess|"
    r"kepler|corot|wasp|hat[- ]p|ogle|moa|tr(es|appist)|gj\s*\d+|hd\s*\d+|lhs\s*\d+)\b",
    re.I
)
EXO_ID_RX = re.compile(
    r"\b("
    r"(?:co)?rot[-\s]?\d+[b-z]|"         # CoRoT-9b, COROT-9b
    r"kepler[-\s]?\d+[b-z]|k2[-\s]?\d+[b-z]|"
    r"toi[-\s]?\d+[b-z]|tess\b|"
    r"wasp[-\s]?\d+[b-z]|hat[-\s]?p[-\s]?\d+[b-z]|"
    r"ogle[-\s]?\d+[b-z]|moa[-\s]?\d+[b-z]|"
    r"trappist[-\s]?-?\d*[b-z]|"
    r"gj\s*\d+\s*[b-z]|hd\s*\d+\s*[b-z]|kic\s*\d+\s*[b-z]|lhs\s*\d+\s*[b-z]|lp\s*\d+\s*[b-z]"
    r")\b",
    re.I
)
EXO_TERMS_RX = re.compile(
    r"\b(exoplanet|super[- ]earth|hot[- ]jupiter|mini[- ]neptune|radial velocity|transit)\b", re.I
)
SIMILE_RX = re.compile(
    r"\b(titan[- ]like|like titan|similar to titan|titan[- ]analog(?:ue)?)\b", re.I
)

PODCAST_RX = re.compile(r'\b(podcast|episode|listen|spotify|apple podcasts|rss|youtube|transcript|interview)\b', re.I)
PLANET_WORDS = ["Mercury","Venus","Moon","Mars","Jupiter","Saturn",
                "Titan","Enceladus","Europa","Ganymede","Io","Callisto",
                "Uranus","Neptune","Triton","Pluto","Ceres","Earth"]

def count_occ(rx, text):  # count occurrences
    return len(list(re.finditer(rx, text)))

def near(rx_a, rx_b, txt, span=140):
    apos = [m.start() for m in re.finditer(rx_a, txt)]
    bpos = [m.start() for m in re.finditer(rx_b, txt)]
    i=j=0
    while i < len(apos) and j < len(bpos):
        if abs(apos[i]-bpos[j]) <= span: return True
        if apos[i] < bpos[j]: i+=1
        else: j+=1
    return False

def window_hit(a_rx, b_rx, text, span=120):
    # any alias/planet near a key term (e.g., 'water ice', 'PH3', 'CRaTER', etc.)
    apos = [m.start() for m in re.finditer(a_rx, text)]
    bpos = [m.start() for m in re.finditer(b_rx, text)]
    if not apos or not bpos: return False
    i, j = 0, 0
    while i < len(apos) and j < len(bpos):
        if abs(apos[i] - bpos[j]) <= span: return True
        if apos[i] < bpos[j]: i += 1
        else: j += 1
    return False

def boost_score(planet, title, content):
    rules = COMPILED.get(planet)
    txt_t = title or ""
    txt_c = content or ""
    txt_all = f"{txt_t} {txt_c}"

    W = {
        "title_planet": 0.25, "alias_any": 0.15,
        "det_title": 0.10, "det_body": 0.05,
        "body_mentions": 0.12, "proximity": 0.10,
        "podcast_pen": 0.20, "exo_pen": 0.18
    }

    b = 0.0
    # title/body cues
    if re.search(rf"\b{re.escape(planet)}\b", txt_t, re.I): b += W["title_planet"]
    if re.search(r"\b(found|detect(?:ed|ion)?|observ(?:e|ed)|confirm(?:ed|ation)?|measured?)\b", txt_t, re.I): b += W["det_title"]
    if re.search(r"\b(found|detect(?:ed|ion)?|observ(?:e|ed)|confirm(?:ed|ation)?|measured?)\b", txt_c, re.I): b += W["det_body"]

    if rules and rules["ali"] and rules["ali"].search(txt_all): b += W["alias_any"]

    # body mentions (strengthen “aboutness” when SimCSE is on content)
    pcount = count_occ(re.compile(rf"\b{re.escape(planet)}\b", re.I), txt_c)
    if pcount >= 1: b += W["body_mentions"] * (1 if pcount == 1 else 1.6)  # 1 mention ok, 2+ stronger

    # proximity of planet/alias to science anchors (customize list if you want)
    ANCHORS = re.compile(r"(water ice|permanently shadowed|phosphine|ph3|ionosphere|magnetosphere|"
                         r"submillimeter|266\.94\s*ghz|ALMA|JCMT|CRaTER|Mini-?RF|SAM|REMS|RAD|CRISM|INMS|MWR)", re.I)
    ali_or_plan = rules["ali"] if (rules and rules["ali"]) else re.compile(rf"\b{re.escape(planet)}\b", re.I)
    if near(ali_or_plan, ANCHORS, txt_c, span=140): b += W["proximity"]

    # soft penalties
    if PODCAST_RX.search(txt_all): b -= W["podcast_pen"]

    # exoplanet talk steals focus? penalize if:
    # - exoplanet terms present
    # - planet NOT in title
    # - planet count in body < 2
    if EXO_RX.search(txt_all) and not re.search(rf"\b{re.escape(planet)}\b", txt_t, re.I) and pcount < 2:
        b -= W["exo_pen"]

    return b

def solar_system_gate(planet, title, content):
    txt_all = f"{title or ''} {content or ''}"
    # allow if target planet appears in title OR ≥2 body mentions OR near its anchors
    rules = COMPILED.get(planet)
    ali_or_plan = rules["ali"] if (rules and rules["ali"]) else re.compile(rf"\b{re.escape(planet)}\b", re.I)
    strong_about = (re.search(rf"\b{re.escape(planet)}\b", title or "", re.I) or
                    count_occ(re.compile(rf"\b{re.escape(planet)}\b", re.I), content or "") >= 2 or
                    near(ali_or_plan, re.compile(r"(Cassini|Dragonfly|MESSENGER|CRaTER|Mini-?RF|JUICE|Juno|INMS|CRISM|LADEE|Akatsuki|Magellan|Huygens|Clipper|MWR)", re.I), content or "", 140))
    if EXO_RX.search(txt_all) and not strong_about:
        return False
    return True