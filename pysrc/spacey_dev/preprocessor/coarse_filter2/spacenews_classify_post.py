# uses classified json

import json
from spacey_util.add_path import report_path, data_processed_path
from spacey_dev.util.helper import clean_text

REPORT_FILE_PATH = report_path() / "spacenews_classify.json"

PROCESSED_FILE_SAVE_PATH = data_processed_path() / "spacenews_classify.json"

with open(REPORT_FILE_PATH, "r") as f:
    classified_results = json.load(f)

def normalize_value(v: str):
    v_upper = v.upper()
    if v_upper in ("NO", "NONE"):
        return False
    elif v_upper == "YES":
        return True
    return v  # keep original string

def result2json(text):
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue  # skip malformed lines
    
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not value:  
            # key exists but no value → empty list
            result[key] = []
        elif key in ["IS_PLANETARY", "IS_EXOPLANET"]:
            result[key] = normalize_value(value)
        elif key in ["PRIMARY", "BUCKET"]:
            result[key] = value
        else:
            # split on semicolons for multiple values
            values = [v.strip().strip('"') for v in value.split(";")]
            result[key] = [normalize_value(v) for v in values]

    return result

classified_parsed = []
for item in classified_results:
    r = item['result']
    try:
        result = result2json(r)

        assert result['IS_PLANETARY'] == True or result['IS_PLANETARY'] == False
        assert result['IS_EXOPLANET'] == True or result['IS_EXOPLANET'] == False

        # only has 1 value
        # assert len(result['BUCKET']) == 1
        assert len(result['NONPLANET']) == 1
        assert len(result['MISSIONS']) <= 1 

        assert len(result.keys()) == 9 # schema key size
    except AssertionError as e:
        print(result)
        raise
        

    classified_parsed.append({
        "id": item['id'],
        "title": clean_text(item['title']),
        'result': result
    })

print("Check--end")


with open(PROCESSED_FILE_SAVE_PATH, "w") as f:
    json.dump(classified_parsed, f, indent=2)