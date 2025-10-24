# --- Categories ---
PLANETS = {
    "mercury", "venus", "earth", "mars",
    "jupiter", "saturn", "uranus", "neptune"
}

DWARF_PLANETS = {
    "pluto", "ceres", "eris", "haumea", "makemake",
    "vesta", "orcus", "quaoar", "sedna", "gonggong"
}

MOONS = {
    # Earth
    "moon",
    # Mars
    "phobos", "deimos",
    # Jupiter (Galilean + important others)
    "io", "europa", "ganymede", "callisto",
    "amalthea", "himalia", "elara",
    # Saturn
    "titan", "enceladus", "rhea", "iapetus",
    "dione", "tethys", "mimas", "phoebe",
    # Uranus
    "miranda", "ariel", "umbriel", "titania", "oberon",
    # Neptune
    "triton", "nereid", "proteus"
}

ASTEROIDS = {
    "bennu", "ryugu", "didymos", "dimorphos",
    "psyche", "eros", "pallas", "hygiea", "lutetia",
    "ceres", "vesta", "mathilde", "gaspra", "ida"
}

COMETS = {
    "comet-1p-halley", "comet-9p-tempel", "comet-67p",  "comet 67p",
    "comet-wild-2", "comet-borrelly", "comet-hyakutake",
    "comet-hale-bopp"
}

# Famous exoplanets & rogue candidates
EXOPLANETS = {
    "kepler-22b", "kepler-62f", "kepler-452b",
    "corot-7b", "corot-9b", "proxima-centauri-b",
    "trappist-1b", "trappist-1c", "trappist-1d",
    "trappist-1e", "trappist-1f", "trappist-1g", "trappist-1h",
    "cfbdsir2149", "gj-1214b", "wasp-12b", "hd-209458b"
}

# --- Unified set ---
PLANETARY_BODIES = PLANETS | DWARF_PLANETS | MOONS | ASTEROIDS | COMETS | EXOPLANETS


# --- Validator ---
def classify_body(name: str) -> str:
    """
    Classify a celestial name into its planetary science category.
    Returns category name (planet, moon, dwarf planet, asteroid, comet, exoplanet)
    or 'unknown' if not in list.
    """
    n = name.strip().lower()
    if n in PLANETS:
        return "planet"
    elif n in MOONS:
        return "moon"
    elif n in DWARF_PLANETS:
        return "dwarf planet"
    elif n in ASTEROIDS:
        return "asteroid"
    elif n in COMETS:
        return "comet"
    elif n in EXOPLANETS:
        return "exoplanet"
    elif n == "sun":
        return "sun"
    else:
        return "unknown"

def classify_sentence(sentence: str = ""):
    """
    Classify a sentence into the celestial body mapping
    """
    words = sentence.lower().split()
    has_planet = False
    if "planet" in words or any(word in PLANETS for word in words) or any(word in DWARF_PLANETS for word in words) or any(word in EXOPLANETS for word in words):
        has_planet = True
    
    matches = set(words) & (PLANETARY_BODIES)

    return has_planet, list(matches)

    
