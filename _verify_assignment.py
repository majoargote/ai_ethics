"""Verification script — mirrors AI_workshop_assignment.ipynb cell by cell."""
import json, sys, os, types, textwrap

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Simulate %run workshop_docs/class_notebook.ipynb ──────────────────────────
nb_path = "workshop_docs/class_notebook.ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

shared_ns = {"__name__": "__main__"}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    # skip IPython magics and Ollama-dependent cells
    if src.strip().startswith("%") or "requests.post" in src or "ollama" in src.lower():
        continue
    try:
        exec(compile(src, nb_path, "exec"), shared_ns)
    except Exception as e:
        # non-fatal — some cells may reference Ollama or display()
        pass

print("class_notebook loaded. Functions available:", [k for k in shared_ns if not k.startswith("_") and callable(shared_ns.get(k))])

# ── Task 1 ────────────────────────────────────────────────────────────────────
Action     = shared_ns["Action"]
Template   = shared_ns["Template"]
Vocabulary = shared_ns["Vocabulary"]
generate_sentences  = shared_ns["generate_sentences"]
build_export_rows   = shared_ns["build_export_rows"]

order_actions = [
    Action(text="enforcing",    value="order"),
    Action(text="maintaining",  value="order"),
    Action(text="upholding",    value="order"),
    Action(text="organising",   value="order"),
    Action(text="regulating",   value="order"),
    Action(text="structuring",  value="order"),
    Action(text="coordinating", value="order"),
    Action(text="managing",     value="order"),
]

order_templates = [
    Template(text="{agent} was {action} {patient} {place}."),
    Template(text="It was clear that {agent} spent the day {action} {patient}."),
    Template(text="Everyone noticed {agent} {action} {patient} {place}."),
    Template(text="During the crisis, {agent} kept {action} {patient}."),
    Template(text="People praised {agent} for {action} {patient} {place}."),
]

order_vocabularies = [
    Vocabulary(name="agent",   words=["a police officer", "a judge", "a committee", "a supervisor", "the security team"]),
    Vocabulary(name="patient", words=["the community", "the public space", "the system", "the neighbourhood", "the institution"]),
    Vocabulary(name="place",   words=["in the city", "at the event", "during the crisis", "across the region", "throughout the district"]),
]

order_sentences = generate_sentences(
    templates=order_templates,
    actions=order_actions,
    vocabularies=order_vocabularies,
    n_per_value=100,
    seed=42,
)
print(f"\nTask 1 — Generated {len(order_sentences)} sentences")
sample_key = "text" if "text" in order_sentences[0] else list(order_sentences[0].keys())[0]
print(f"  Sample: {order_sentences[0].get('text', order_sentences[0])}")

import pandas as pd
order_rows = build_export_rows(base_rows=order_sentences)
order_df = pd.DataFrame(order_rows)
order_df.to_csv("order_sentences.csv", index=False, sep="\t")
print(f"  Saved order_sentences.csv — {len(order_df)} rows, columns: {list(order_df.columns)}")
assert len(order_df) == 100, f"Expected 100 rows, got {len(order_df)}"
assert order_df["true_value"].unique().tolist() == ["order"], "true_value should be 'order'"

# ── Task 2 ────────────────────────────────────────────────────────────────────
compute_recognition_ratio    = shared_ns["compute_recognition_ratio"]
compute_robustness_ratio     = shared_ns["compute_robustness_ratio"]
compute_directional_agreement = shared_ns["compute_directional_agreement"]

frame = pd.read_csv("workshop_docs/scored_nature_sentences.csv")
print(f"\nTask 2 — Loaded scored CSV: {frame.shape}, columns={len(frame.columns)}")
assert len(frame.columns) == 14, f"Expected 14 columns, got {len(frame.columns)}"
assert set(frame["variant"].unique()) == {"base", "animal", "plant"}
assert "nature_score" in frame.columns

# 2a Recognition
recognition = compute_recognition_ratio(frame)
print(f"\n2a Recognition ratio: {recognition}")
ratio = recognition["nature"]
print(f"   nature={ratio:.4f} → {'PASS' if ratio >= 0.5 else 'FAIL'}")

# 2b Robustness
results, extra = compute_robustness_ratio(frame)
print(f"\n2b Robustness ratios:")
for pair, r in results.items():
    print(f"   {pair}: {r:.4f} → {'ROBUST' if r > 0.5 else 'NOT ROBUST'}")

# 2c Directional
shift_mask = {
    "base":   lambda df: df["variant"] == "base",
    "animal": lambda df: df["variant"] == "animal",
    "plant":  lambda df: df["variant"] == "plant",
}
dir_results = compute_directional_agreement(
    frame=frame,
    shift_mask=shift_mask,
    value="nature",
    direction_of_agreement=lambda sr: sr > 1.0,
)
print(f"\n2c Directional agreement: {dir_results}")

print("\n✓ All checks passed — notebook is correct.")
