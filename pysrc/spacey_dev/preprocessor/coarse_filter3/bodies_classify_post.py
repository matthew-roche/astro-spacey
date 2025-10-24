# uses bodies classified json

import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from spacey_util.add_path import report_path, data_processed_path, data_raw_path
from spacey_dev.preprocessor.rules.sci import classify_body
from spacey_dev.util.helper import clean_text

REPORT_FILE_PATH = report_path() / "spacenews_bodies_classify.json"

PROCESSED_FILE_SAVE_PATH = data_processed_path() / "spacenews_bodies_classified.json"

df = pd.read_parquet(data_processed_path() / "spacenews_bodies.parquet")

df_org = pd.read_parquet(data_processed_path() / "spacenews.parquet")
df_raw = pd.read_csv(data_raw_path() / "spacenews.csv")

with open(REPORT_FILE_PATH, "r") as f:
    classified_results = json.load(f)

def result2json(text):
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue  # skip malformed lines
    
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in result:
            if "DISCOVERED" in value:
                result[key] = "DISCOVERED"

        if not value:
            # key exists but no value → empty list
            if key == 'STATUS':
                result[key] = 'NONE'
            else:
                result[key] = []
            
        elif ";" in value:
            values = [v.strip().strip('"') for v in value.split(";")]
            result[key] = values
        else:
            # split on semicolons for multiple values
            if value == 'PLANN':
                value = 'PLANNED'
            result[key] = value

    return result

planets_discovery = {'Earth': 0, 'Mars': 0, 'Venus': 0, 'Uranus': 0, 'Mercury': 0, 'Jupiter': 0, 'Saturn': 0}
overall = {
    "DISCOVERED": 0,
    "PLANNED": 0,
    "OTHER": 0
}
bodies_discovery = {'moon': 0, 'planet': 0, 'asteroid': 0, 'dwarf planet': 0, 'comet': 0, 'exoplanet': 0}

consider_titles = [
    'NASA’s blueprint for the Red Planet',
    'Mars rover Opportunity weathering major dust storm'
]

SHOW_SKIPPED = False

processed_results = []
follow = False
filtered_titles = []
for item in classified_results:
    result = result2json(item['result'])
    follow = False

    copy_item = item
    copy_item['result'] = result

    cleaned_item_title = clean_text(item['title'])

    assert result['STATUS'] in ["NONE", "DISCOVERED", "PLANNED", "MIXED"]

    if result['STATUS'] in ['NONE', 'MIXED']:
        overall['OTHER'] += 1

    else:
        target = result['TARGET']
        body_type = classify_body(target)

        if target == 'Exoplanet':
            body_type = 'exoplanet'
        
        # if body_type == "dwarf planet":
        #     print(result)
        #     print(body_type)
        #     print(content)
        
        # todo- refactor: remove
        select = df[df["title"] == cleaned_item_title].get('content')
        if select.empty:
            # try from raw dataset
            raw_content_v = df_raw[df_raw["title"] == item['title']].get('content').values
            if len(raw_content_v) < 1:
                print(item['title'])
                continue

            raw_cleaned = clean_text(raw_content_v[0])
            if len(raw_cleaned) < 1:
                continue

            content = raw_cleaned
        else:
            content = select.values[0]
            
        # if item['title'] == "Curiosity’s Radiation Results":
        #     print(content)

        if cleaned_item_title not in consider_titles:
            if " aim " in content or "ready to launch" in content or "Zhurong" in content or "Ingenuity Mars helicopter" in content or "Perseverance is healthy" in content or "abandoned efforts" in content or "stuck again" in content or "confirmation review" in content:
                if SHOW_SKIPPED:
                    print(f"\nskipping: {cleaned_item_title}")
                    print(content)
                overall["PLANNED"] +=1 
                continue

            if "remaining battery capacity" in content or "spacecraft apparently fell" in content or "lack of sunlight to generate power" in content:
                if SHOW_SKIPPED:
                    print(f"\nskipping: {cleaned_item_title}")
                    print(content)
                overall["OTHER"] +=1
                continue

            if item['title'] in ['Ice giants and icy moons: The planetary science decadal survey looks beyond Mars to the outer solar system']:
                if SHOW_SKIPPED:
                    print(f"\nskipping: {cleaned_item_title}")
                    print(content)
                overall["OTHER"] +=1
                continue

        # not very useful        
        # if target == "Jupiter":
        #     planetary_df = df[df["title"] == item['title']]
        #     print(item['title'])
        #     print(result)
        #     print(planetary_df.get('content').values[0])      
            # print(result)
        
        # osiris rex ignored
        # if body_type == 'sun': # was classified as earth because situation is earth, investigation is sun
        #     planetary_df = df[df["title"] == item['title']]
        #     print(item['title'])
        #     print(result)
        #     print(planetary_df.get('content').values[0])
        if result['STATUS'] == 'DISCOVERED':
            
            increment = True

            if body_type != 'unknown':
                filtered_titles.append(cleaned_item_title)

                processed_results.append({
                    "id": item['id'],
                    "c": item['c'],
                    "title": cleaned_item_title,
                    'body': body_type,
                    'is_planet': "planet" in body_type,
                    'target': target
                })

                if body_type in bodies_discovery:
                    bodies_discovery[body_type] += 1
                else:
                    bodies_discovery[body_type] = 1
                
                overall['DISCOVERED'] +=1
                increment = False
                

            if body_type == "planet":
                if target in planets_discovery:
                    planets_discovery[target] += 1
                else:
                    planets_discovery[target] = 1
                
                if increment:
                    overall['DISCOVERED'] +=1

        else:
            overall[result['STATUS']] += 1

print(overall)
print(bodies_discovery)
print(planets_discovery)
print("check--end")

# visualization
# from coarse filter 2
body_distribution = {'moon': 223, 'planet': 607, 'asteroid': 15, 'dwarf planet': 30, 'comet': 4, 'exoplanet': 3}
planet_distribution = {'Earth': 190, 'Mars': 354, 'Venus': 25, 'Uranus': 2, 'Mercury': 20, 'Jupiter': 12, 'Saturn': 4}

filtered_body_df = df[df["title"].isin(filtered_titles)]

filtered_body_df.to_parquet(data_processed_path() / "spacenews_coarse_filter3.parquet")

with open(data_processed_path() / "spacenews_coarse_filter3.json", "w", encoding="utf-8") as f:
    json.dump(processed_results, f, indent=2, ensure_ascii=False)

print(filtered_body_df)

# helper fn

def plot_extraction():
    body_copy = bodies_discovery.copy()
    body_copy.update(planets_discovery)
    body_copy.pop('planet')

    labels = body_copy.keys()
    values = body_copy.values()

    print(len(labels))
    print(labels)

    fig, ax = plt.subplots()

    colors = [
        "#C0C0C0",  # moon – light gray (lunar surface)
        "#696969",  # asteroid – dim gray (rocky)
        "#708090",  # dwarf planet – slate gray (icy/neutral)
        "#AEEEEE",  # comet – pale cyan (coma glow)
        "#9370DB",  # exoplanet – medium purple (other-worldly)
        "#FFD700",  # sun – goldenrod (solar tone)
        "#4682B4",  # Earth – steel blue (oceanic)
        "#CD5C5C",  # Mars – indian red (red planet)
        "#BDB76B",  # Venus – khaki (sulfuric clouds)
        "#20B2AA",  # Uranus – light sea green (icy/aqua)
        "#B8860B",  # Mercury – dark goldenrod (metallic)
        "#CD853F",  # Jupiter – peru (tan/orange bands)
        "#F4A460"   # Saturn – sandy brown (ring beige)
    ]

    ax.bar(labels, values, color=colors, edgecolor="black")

    # Labels and title
    ax.set_xlabel("Celestial Body")
    ax.set_ylabel("Row Count")
    # ax.set_title("Number of Mentions per Planet")

    # Optional grid
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.rcParams.update({
        'font.size': 12,          # base font size
        'axes.titlesize': 14,     # title
        'axes.labelsize': 12,     # x/y labels
        'xtick.labelsize': 10,    # x ticks
        'ytick.labelsize': 10,    # y ticks
        'legend.fontsize': 10
    })

    plt.tight_layout()
    plt.show()

