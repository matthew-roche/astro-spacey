import pandas as pd, json, numpy as np
from spacey_util.add_path import data_processed_path
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from spacey_dev.preprocessor.rules.sci import classify_body
from spacey_dev.util.helper import clean_text

df = pd.read_parquet(data_processed_path() / "spacenews.parquet")

CLASSIFICATION_TASK_FILE = data_processed_path() / "spacenews_classify.json"

with open(CLASSIFICATION_TASK_FILE, "r") as f:
    classified_results = json.load(f)

bodies = {}
planets = {}
mission_planets = {}
overall = {
    'Spacecraft/Satellite-Ops': 0,
    'Launch/Vehicle': 0,
    'Planetary': 0,
}

planetary_finding_titles = []
planetary_titles = []
bodies_titles = []
def append2overall(kind = "", content = ""):

    if any(x in content.lower() or x in kind.lower() for x in ["budget", "fin", "econ", "inves"]):
        kind = "Budget/Related"
    elif any(x in content.lower() or x in kind.lower() for x in ["business", "agency", "mana", "huma", "hr", "politic", "corp", "com", "gov"]):
        kind = "Management/Related"
    elif "internet" in content.lower() or "internet" in kind.lower():
        kind = "Communications"
    elif "policy" in content.lower() or "policy" in kind.lower():
        kind = "Policy"
    elif "space" in content.lower() or "space" in kind.lower():
        kind = 'Spacecraft/Satellite-Ops'
    elif kind not in overall.keys():
        kind = "other"

    if kind in overall:
        overall[kind] += 1
    else:
        overall[kind] = 1


for item in classified_results:
    result = item['result']

    if not result['IS_PLANETARY']:
        kind = result['NONPLANET'][0]
        if type(kind) == bool:
            if result['EVIDENCE'][0] == False:
                kind = "unknown"
            else:
                content = " ".join(result['EVIDENCE'])
            append2overall(content=content)
        else:
            nonplanet = result['NONPLANET'][0].rstrip().replace(" ", "-").replace(",", "")
            append2overall(kind=nonplanet)


    if result['IS_PLANETARY']:

        primary = result['PRIMARY'].rstrip().replace(" ", "-").replace(",", "")
        primary_type = classify_body(primary)
        
        if primary_type == "unknown":
            append2overall(kind=nonplanet, content=" ".join(result['EVIDENCE']))
            continue

        # planet, Earth but actually sun
        # if item['title'] == "Solar Wind Samples Add Mystery to Earth’s Genesis":
        #     primary_type = 'sun'

        overall['Planetary'] += 1

        # assert result['NONPLANET'][0] == False

        if primary_type == "planet":
            if result['BUCKET'] == "mission" or result['BUCKET'] == "climate":
                planetary_titles.append(item['title'])

                if primary in mission_planets:
                    mission_planets[primary] += 1
                else:
                    mission_planets[primary] = 1


            if primary in planets:
                planets[primary] += 1
            else:
                planets[primary] = 1
        
        if primary_type != "unknown":
            bodies_titles.append(item['title'])
            if primary_type in bodies:
                bodies[primary_type] += 1
            else:
                bodies[primary_type] = 1

# overall['Spacecraft/Satellite-Ops']

print(overall)
print(bodies)
print(planets)
print(mission_planets)


bodies_df = df[df["title"].isin(bodies_titles)]
bodies_df.to_parquet(data_processed_path() / "spacenews_bodies.parquet")


# plt.bar(overall.keys(), overall.values())
# plt.show()

def plot_extraction():

    colors_overall = [
        "#4C72B0",  # muted blue
        "#55A868",  # soft green
        "#C44E52",  # muted red
        "#8172B3",  # lavender purple
        "#CCB974",  # mustard
        "#64B5CD",  # teal
        "#8C8C8C"   # neutral gray
    ]

    colors_types = [
        "#6BA292",  # sage green
        "#A6761D",  # ochre brown
        "#1B7837",  # dark forest green
        "#80B1D3",  # muted sky blue
        "#B2ABD2",  # dusty violet
        "#D6616B"   # rose red
    ]

    colors_planets = [
        "#333333",  # dark gray
        "#666666",  # medium gray
        "#999999",  # light gray
        "#BBBBBB",  # very light gray
        "#4C72B0",  # muted blue highlight
        "#C44E52",  # muted red highlight
        "#55A868"   # muted green highlight
    ]

    def plot(labels, values, colors, title, include_pie = True, include_bar = True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        pie_index = 0
        if include_bar:
            axes[0].bar(labels, values, color=colors)
            axes[0].set_title(f"Bar Chart of {title}")
            axes[0].set_ylabel("Count")
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=45)
            pie_index = 1

        if include_pie:
            wedges, _ = axes[1].pie(
                values,
                labels=None,           # no inner labels
                colors=colors[:len(labels)],
                startangle=90
            )
            axes[pie_index].set_title(f"Pie Chart of {title}", fontsize=12)

            for i, wedge in enumerate(wedges):
                # Angle of the wedge center
                ang = (wedge.theta2 + wedge.theta1) / 2.0
                x = wedge.r * 0.7 * np.cos(np.deg2rad(ang))
                y = wedge.r * 0.7 * np.sin(np.deg2rad(ang))

                # Position outside pie
                x_text = 1.1 * np.cos(np.deg2rad(ang))
                y_text = 1.1 * np.sin(np.deg2rad(ang))

                # Percentage
                pct = values[i] / sum(values) * 100
                axes[pie_index].text(x_text, y_text, f"{pct:.1f}%", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

                # Connector line
                con = ConnectionPatch(
                    xyA=(x, y), coordsA=axes[1].transData,
                    xyB=(x_text, y_text), coordsB=axes[1].transData,
                    arrowstyle="-", lw=0.8, color="black"
                )
                axes[pie_index].add_artist(con)

            # Optionally add legend outside instead of labels inside
            axes[pie_index].legend(
                wedges, labels, title="Labels",
                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=9
            )

    planetary_df = df[df["title"].isin(planetary_titles)]

    # visualize
    plot(list(overall.keys()), list(overall.values()), colors_overall, title="corpus", include_bar=True)
    # plot(list(bodies.keys()), list(bodies.values()), colors_types, title="Bodies")
    # plot(list(planets.keys()), list(planets.values()), colors_planets, title="planets")
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


    # print(planetary_df.get('content').values[0])
    # print(planetary_df.get('content').values[1])