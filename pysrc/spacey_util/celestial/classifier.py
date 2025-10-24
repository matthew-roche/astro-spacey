import json, re

with open("./data/planetary_vocab.json", "r") as f:
    planetary_vocab = json.load(f)

def celestial_extract(sentence:str):
    words = sentence.lower().split()

    planets_in_text = [p for p in planetary_vocab["planets"] if p.lower() in words]
    moons = [p for p in planetary_vocab["moons"] if p.lower() in words]
    asteroids = [p for p in planetary_vocab["asteroids"] if p.lower() in words]
    dwplanets_in_text = [p for p in planetary_vocab["dwarf_planets"] if p.lower() in words]
    prop_in_text = [p for p in planetary_vocab["planetary_properties"] if p.lower() in words]

    observations = []
    for phrase in planetary_vocab["observations"]:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, sentence.lower()):
            observations.append(phrase)

    body = set(planets_in_text + dwplanets_in_text + moons + asteroids)
    prop = set(prop_in_text + observations)
    
    return list(body), list(prop)