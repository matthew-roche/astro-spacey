from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json, pandas as pd, time
from spacey.add_path import data_processed_path, model_path, report_path
import json, re

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # optional
    bnb_4bit_quant_type="nf4", # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = "cuda" # rtx 4080

model_dir = model_path() / "meta-llama-Meta-Llama-3.1-8B-Instruct"

df = pd.read_parquet(data_processed_path() / "spacenews_bodies.parquet")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config).to(device)

CTX_LEN_DEFAULT = 8192            # fallback if tokenizer.model_max_length is huge
MAX_NEW_TOK     = 64              # you only need ~9 lines
MARGIN_TOK      = 64              # safety room
HARD_BODY_CAP   = 2000             # fast baseline: cap body to 500 tokens

def render_messages(title: str, content: str):
    return [
        {
            "role": "system",
            "content": "You are a precise space-news classifier. Answer ONLY with the required tagged lines. No extra text."
        },
        {
            "role": "user",
            "content": f"""Task: Decide if the article is about (a) an already discovered scientific finding, or (b) a planned/expected future discovery. Also tag the topical focus and target body.

Allowed TARGETS:
Mercury, Venus, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Ceres,
Europa, Ganymede, Callisto, Io, Enceladus, Titan, Triton, Earth, Sun,
Exoplanet, Small-Body (asteroid/comet/meteoroid), Interplanetary, Deep-Space

Allowed TOPICS (pick one that fits best):
water/ice, climate/atmosphere, geology/surface, interior/ocean, radiation/magnetosphere,
biology/biosignatures, habitability, materials/minerals, mission/science-plan, mapping/imaging,
weather, seismology, rings/dust, sample/analysis, mixed/other

OUTPUT FIELDS (exact keys, uppercase, in this order):
STATUS: DISCOVERED | PLANNED | MIXED | NONE
TARGET: <one from list, or NONE>
TOPIC: <one from list>
EVIDENCE: "short quote #1"; "short quote #2"

Rules:
- DISCOVERED: reports results/measurements already obtained (e.g., “detected…”, “found…”, “measurements show…”).
- PLANNED: proposals, upcoming missions/instruments/observations aiming to discover/confirm something (“will”, “aims to”, “plan to measure…”).
- MIXED: clearly contains both prior discovery results and substantial new/planned discovery aims.
- If launch/ops only and no science aim/result → STATUS: NONE; TOPIC: mixed/other; TARGET: NONE.
- Evidence quotes must be verbatim snippets <=100 chars, from the article body/title.
- If target not explicit, infer the closest valid option; otherwise NONE.

Example Output (DISCOVERED):
STATUS: DISCOVERED
TARGET: Mars
TOPIC: water/ice
EVIDENCE: "hydrated salts detected"; "recurring slope lineae"

Example Output (PLANNED):
STATUS: PLANNED
TARGET: Europa
TOPIC: mission/science-plan
EVIDENCE: "will probe subsurface ocean"; "ice-penetrating radar"

Example Output (MIXED):
STATUS: MIXED
TARGET: Mercury
TOPIC: climate/atmosphere
EVIDENCE: "exospheric sodium observed"; "mission will map dawn/dusk variations"

Now classify this article. Answer ONLY with tagged lines in the same format.

Title: {title}
Body: {content}"""
        }
    ]

def compute_prompt_tokens_without_body(title: str) -> int:
    """Measure the token cost of the chat template + title (empty body)."""
    msgs = render_messages(title, content=="")
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
    return ids.shape[1]

def truncate_body_to_budget(body: str, budget: int) -> str:
    """Truncate body to `budget` tokens and decode back to text."""
    enc = tokenizer(body, truncation=True, max_length=budget, return_tensors="pt")
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

def make_inputs(title: str, body: str):
    # 1) compute available context
    ctx_len = getattr(tokenizer, "model_max_length", CTX_LEN_DEFAULT)
    if ctx_len > 100_000:  # some tokenizers use a giant sentinel; clamp
        ctx_len = CTX_LEN_DEFAULT

    # 2) cost of prompt (no body)
    P = compute_prompt_tokens_without_body(title)

    # 3) body budget = min(HARD_CAP, ctx_len - P - MAX_NEW - MARGIN)
    body_budget = max(64, min(HARD_BODY_CAP, ctx_len - P - MAX_NEW_TOK - MARGIN_TOK))

    # 4) truncate body to budget
    body_trunc = truncate_body_to_budget(body, body_budget)

    # 5) render final prompt with truncated body
    msgs = render_messages(title, content=body_trunc)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # 6) tokenize final prompt; keep truncation=True as a last-resort safety
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    return inputs

REPORT_FILE_PATH = report_path() / "spacenews_bodies_classify.json"
classifications = []
with open(REPORT_FILE_PATH, "w") as f:
    json.dump(classifications, f)

def append_to_report(new_data):
    try:
        with open(REPORT_FILE_PATH, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        existing_data = []
    except json.JSONDecodeError:
        # Handle cases where the file might be empty or malformed
        existing_data = []

    existing_data.extend(new_data)
    with open(REPORT_FILE_PATH, 'w') as file:
        json.dump(existing_data, file, indent=4)
    

start_time = time.time()
counter = 0
for i, row in df.iterrows():
    title = row['title']
    content = row['content']

    inputs = make_inputs(title, body=content)

    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen = model.generate(**inputs,
                         do_sample=True, 
                         max_new_tokens=128,
                         temperature=0.7,
                         top_p=0.9)
    out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    classifications.append({"id": i, "c":counter, "title": title, "result": out})
    print(f"Processed {counter+1} of {len(df)}")
    print(out)

    if counter%50 == 0 or (counter == len(df)-1):
        append_to_report(classifications)
        classifications = [] # reset
    
    counter+=1

    # exit(0)
end_time = time.time()
print(f"Classification time: {end_time - start_time:.4f} seconds")

# with open(report_path() / "spacenews_classify.json", "w") as f:
#     json.dump(classifications, f)

# Classification time: 5523.3165 seconds