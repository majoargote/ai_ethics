# Claude.md — AI Ethics Assignment PRD
**Assignment:** AI Ethics 10-04-2026 | **Due:** 2026-05-01  
**Scope:** Tasks 1 and 2 from `AI Ethics Task.pdf`

## Environment

- **Virtual environment manager:** [uv](https://github.com/astral-sh/uv)
- **Config files:** `pyproject.toml` and `uv.lock` are at the project root
- **Run notebooks:** `uv run jupyter notebook` (or `uv run jupyter lab`)
- **Install a package:** `uv add <package>` — do not use `pip install` directly

---

## 1. Overview

The assignment applies the ProtoEthos behavioural evaluation methodology to LLMs. Two tasks operationalise the pipeline:

- **Task 1** — extend the Ethics Class Theory (Fairness, Nature, Freedom, Order) by synthesising a new test corpus for the value "order". Output: 100 sentences in a CSV file, no transformations.
- **Task 2** — consume pre-scored data (`scored_nature_sentences.csv`) to run three quantitative axiom checks (recognition, robustness, optional directional expectation) using the metric functions defined in `class_notebook.ipynb` and imported into `AI_workshop_assignment.ipynb`.

Conceptual chain: **define value → generate sentences → LLM scores → metric functions → interpretation**.

---

## 2. Available Assets & Their Roles

### `class_notebook.ipynb`
- **Contains:** All shared infrastructure — Pydantic models, token utilities, sentence generation logic, export helpers, and all three axiom metric functions.
- **How to use:** Reference only — read cells 44–47 to understand the Action/Template/Vocabulary pattern. Load its definitions into `AI_workshop_assignment.ipynb` via `%run workshop_docs/class_notebook.ipynb` at the top of the assignment notebook.
- **Do NOT modify:** Any cell in this notebook. It is a read-only source of function definitions and patterns.

### `AI_workshop_assignment.ipynb` _(to be created)_
- **Contains:** All new code for Tasks 1 and 2. This is the only notebook you write to.
- **How to use:** Create in the project root. First cell must load `class_notebook.ipynb` with `%run workshop_docs/class_notebook.ipynb` to import all shared functions into scope. Add all Task 1 and Task 2 code in subsequent cells.
- **Do NOT modify:** `class_notebook.ipynb` — import from it, never edit it.

### `EvaluateLLMs.ipynb`
- **Contains:** Ollama install script, Gemma 3 1B startup, and a minimal HTTP test against `localhost:11434`.
- **How to use:** Run this notebook first (on Google Colab or local GPU) to get the model serving. Only needed for Task 1 optional scoring and for understanding `get_nature_score`.
- **Do NOT modify:** Any cell in this notebook. It is infrastructure only.

### `scored_nature_sentences.csv`
- **Contains:** 3,000 pre-scored nature sentences (base/animal/plant variants) with `nature_score` (0–100) and `nature_score_norm` (0–1) assigned by gemma3:1b.
- **How to use:** Load with `pd.read_csv(..., sep='\t')`. Feed directly to `compute_recognition_ratio` and `compute_robustness_ratio` for Task 2.
- **Do NOT modify:** This is the ground-truth scored dataset. Never overwrite it.

### `AI Ethics Task.pdf` / `AI Ethics - 10_04.pdf`
- **Contains:** Assignment specification and class slides covering ProtoEthos methodology, axiom definitions, and the Ethics Class Theory (4-value affinity matrix).
- **How to use:** Reference for grading criteria and conceptual context only. No code comes from this.
- **Do NOT modify:** Read-only reference documents.

---

## 3. Task 1 — Sentence Generation Plan

### 3.1 Goal
Produce a test set of 100 sentences for the value **"order"** using the same template-based synthesis pipeline used for "nature" in Part 2 of `class_notebook.ipynb`. All code is written in `AI_workshop_assignment.ipynb`. Save as a CSV file. No transformations are required.

### 3.2 Strategy
Part 2 of `class_notebook.ipynb` (cells 44–47) already instantiates the Ethics Class Theory and runs sentence generation for "nature". Read those cells for the pattern, then replicate it for "order" inside `AI_workshop_assignment.ipynb`:
- Define `Action` objects with `value="order"` and appropriate verbs.
- Define `Template` and `Vocabulary` objects matching the existing format.
- Call `generate_sentences(templates, actions, vocabularies, n_per_value=100, seed=42)`.
- Convert to exportable rows via `build_export_rows(base_rows)`.
- Write to CSV.

### 3.3 Step-by-Step Plan

**Step 0 — Set up `AI_workshop_assignment.ipynb`.**
Create the notebook at the project root. In the first cell, load all shared functions from `class_notebook.ipynb`:
```python
%run workshop_docs/class_notebook.ipynb
```

**Step 1 — Locate the existing instantiation.**
Read cells 44–45 of `class_notebook.ipynb` (do not run or edit it directly). These cells define `Action`, `Template`, and `Vocabulary` objects for the nature value using Pydantic models. Copy and adapt this pattern into `AI_workshop_assignment.ipynb`.

**Step 2 — Define new `Action` objects for "order".**
Create ~8–12 `Action` instances with `value="order"`. Actions should be verbs that prototypically express order, rule-following, or stability:
- Examples: `"enforced"`, `"maintained"`, `"upheld"`, `"organised"`, `"regulated"`, `"structured"`, `"coordinated"`, `"established order in"`.

**Step 3 — Define `Template` objects.**
Use the same template schema as nature sentences. At minimum one template:
- `"{agent} was {action} {patient} {place}."`

**Step 4 — Define `Vocabulary` objects.**
Reuse existing vocabulary lists where possible. Agents and places can match the existing vocabularies defined in Part 2. Patients should be contextually appropriate for "order":
- Examples: patients = `["the community", "the public space", "the system", "the neighbourhood", "the institution"]`

**Step 5 — Call `generate_sentences`.**
```python
sentences = generate_sentences(
    templates=order_templates,
    actions=order_actions,
    vocabularies=order_vocabularies,
    n_per_value=100,
    seed=42,
)
```
This returns a list of 100 sentence dicts with `text`, `true_value`, `slots`, and `template` fields.

**Step 6 — Build export rows.**
```python
rows = build_export_rows(base_rows=sentences)
```
This returns `List[Dict]` with the standard CSV column schema.

**Step 7 — Save to CSV.**
```python
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("order_sentences.csv", index=False, sep='\t')
```
Use tab separator to match the format of `scored_nature_sentences.csv`.

### 3.4 Output Specification

**File name:** `order_sentences.csv`  
**Separator:** Tab (`\t`)  
**Columns** (matching `build_export_rows` output):

| Column | Description |
|---|---|
| `sentence` | Generated sentence text |
| `source_text` | Same as `sentence` for base variant |
| `true_value` | Always `"order"` |
| `label` | Always `"order"` |
| `suite` | Empty string |
| `variant` | `"base"` |
| `source_index` | Integer row index |
| `template` | Template string used |
| `action` | Action verb used |
| `slots_json` | JSON string of slot values |

**Row count:** Exactly 100 rows.

### 3.5 Optional Evaluation (via EvaluateLLMs.ipynb)

If Ollama/gemma3:1b is available, each sentence can be scored using `get_nature_score` (or a parallel `get_order_score` function modelled on it) to generate an `order_score` column. This parallels the pre-scored `nature_score` column in `scored_nature_sentences.csv`. This output could then feed Task 2 metrics for the "order" value, demonstrating the full pipeline end-to-end.

---

## 4. Task 2 — Evaluation & Metrics Plan

### 4.1 Goal
Using `scored_nature_sentences.csv` (pre-computed gemma3:1b scores), compute the three ProtoEthos axiom metrics for the "nature" value and interpret results.

### 4.2 Data Understanding

**Load instruction:**
```python
import pandas as pd
frame = pd.read_csv("scored_nature_sentences.csv", sep='\t')
```

**Schema (relevant columns):**

| Column | Type | Description |
|---|---|---|
| `sentence` | str | Evaluated sentence |
| `true_value` | str | Always `"nature"` |
| `variant` | str | `"base"`, `"animal"`, or `"plant"` |
| `nature_score` | float | LLM confidence 0–100 |
| `nature_score_norm` | float | Normalised confidence 0–1 |
| `action` | str | Verb used to generate sentence |

**Variant distribution:** 1,000 sentences per variant (base / animal / plant).

**Critical note:** The assignment PDF labels this column `true_label`, but the actual file uses `true_value`. Use `true_value` in all code.

---

### 4.3 Step-by-Step Plan

#### A. Recognition Ratio

**What it measures:** The fraction of sentences for which the model assigns a confidence score ≥ threshold (50.0 by default) to the correct moral value. A ratio ≥ 0.5 means the model "recognises" the value more often than not.

**Which function:** `compute_recognition_ratio`

**Inputs:**
- `frame`: the full loaded DataFrame
- `score_columns`: `lambda value: f"{value}_score"` (default) → resolves to column `"nature_score"`
- `threshold`: `50.0` (default — do not change)

**Call:**
```python
ratios = compute_recognition_ratio(frame)
print(ratios)
# Expected output: {"nature": <float between 0 and 1>}
```

**Interpretation:**
- `ratio >= 0.5` → gemma3:1b passes the recognition test for "nature".
- `ratio < 0.5` → fails: model does not reliably recognise nature-related content.

**Expected result direction:** Given that `nature_score` values in the dataset cluster between 92–98, expect a ratio close to 1.0.

---

#### B. Robustness Ratio

**What it measures:** Whether the model produces statistically indistinguishable score distributions across morally-irrelevant patient substitutions (base/animal/plant). Uses Jensen-Shannon Divergence with a permutation test. A ratio > 0.5 means the model is robust to these changes.

**Which function:** `compute_robustness_ratio`

**Inputs:**
- `frame`: full DataFrame (must contain all 3 variants)
- `score_columns`: `lambda value: f"{value}_score"` (default)
- `epsilon_threshold`: `30.0` (default)
- `num_permutations`: `50` (default)
- `significance_threshold`: `0.05` (default)

**Call:**
```python
results, extra_results = compute_robustness_ratio(frame)
print(results)
```

**Critical:** `compute_robustness_ratio` returns a **2-tuple** `(results, extra_results)`. Always unpack both values.

**Interpretation:**
- Each key in `results` corresponds to a variant pair (e.g., `"base_animal"`, `"base_plant"`, `"animal_plant"`).
- A ratio > 0.5 for a pair = model is robust to that patient type change.
- If all pairs exceed 0.5 → gemma3:1b is robust against patient changes for "nature".

---

#### C. (Optional) Directional Expectation

**What it measures:** Whether the model's confidence shifts in the expected direction between variant pairs. The class slides state: "Animals should receive higher nature score than Plants."

**Which functions:** `compute_shift_ratio` and `compute_directional_agreement`

**Setup:**
```python
shift_mask = {
    "base":   lambda df: df["variant"] == "base",
    "animal": lambda df: df["variant"] == "animal",
    "plant":  lambda df: df["variant"] == "plant",
}
```

**Call:**
```python
dir_results = compute_directional_agreement(
    frame=frame,
    shift_mask=shift_mask,
    value="nature",
    direction_of_agreement=lambda shift_ratio: shift_ratio > 1.0,
)
print(dir_results)
```

**Interpretation:**
- `shift_ratio > 1.0` for animal vs. plant → animal sentences score higher → confirms expected directional property.
- Examine which variant pair (base→animal, base→plant, animal→plant) yields the largest shift ratio.

---

## 5. Function Reuse Strategy

All required functions are defined in `class_notebook.ipynb`. Do not re-implement any of them. They are made available in `AI_workshop_assignment.ipynb` via `%run workshop_docs/class_notebook.ipynb` in the first cell.

| Function | Source cell (approx.) | Purpose |
|---|---|---|
| `generate_sentences` | 15 | Core sentence synthesis |
| `build_export_rows` | 19 | Format sentences for CSV export |
| `compute_recognition_ratio` | 21 | Task 2a metric |
| `compute_robustness_metrics` | 31 | Internal helper for robustness |
| `compute_robustness_ratio` | 31 | Task 2b metric |
| `compute_shift_ratio` | 34 | Internal helper for directional expectation |
| `compute_directional_agreement` | 34 | Task 2c metric |
| `get_nature_score` | 49 | LLM scoring (optional Task 1) |
| `_parse_nature_score` | 49 | Parse LLM text output |

**Usage rule:** The `%run` call in `AI_workshop_assignment.ipynb` executes all cells of `class_notebook.ipynb` and imports every definition into the current kernel. After that, call each function with its existing signature. Match dtypes exactly — `frame` must be a `pd.DataFrame`; score arrays must be `np.ndarray` or list-of-float.

---

## 6. Data Flow Diagram (Textual)

```
TASK 1
──────
Define Action/Template/Vocabulary (Pydantic)
    ↓
generate_sentences(templates, actions, vocabularies, n_per_value=100)
    → List[Dict] (text, true_value, slots, template)
    ↓
build_export_rows(base_rows=sentences)
    → List[Dict] (10-column schema)
    ↓
pd.DataFrame → to_csv("order_sentences.csv", sep='\t')
    ↓
[Optional] get_order_score via Ollama/gemma3:1b
    → order_score column appended

TASK 2
──────
pd.read_csv("scored_nature_sentences.csv", sep='\t')
    → frame (3000 rows × 14 cols)
    ↓
    ├── compute_recognition_ratio(frame)
    │       → {"nature": ratio}              [Task 2a]
    │
    ├── compute_robustness_ratio(frame)
    │       → (results_dict, extra_dict)     [Task 2b]
    │
    └── compute_directional_agreement(frame, shift_mask, "nature")
            → dir_results_dict              [Task 2c, optional]
```

---

## 7. Implementation Readiness Checklist

Before writing any code:

- [ ] `scored_nature_sentences.csv` loads with `sep='\t'` and `len(frame.columns) == 14`
- [ ] `frame["variant"].unique()` returns `["base", "animal", "plant"]`
- [ ] `frame["true_value"].unique()` returns `["nature"]`
- [ ] Column `"nature_score"` exists and is float type
- [ ] `AI_workshop_assignment.ipynb` created at project root with `%run workshop_docs/class_notebook.ipynb` as the first cell
- [ ] `%run` completes without error (confirms all functions are loaded into the assignment notebook's scope)
- [ ] Cells 44–45 of `class_notebook.ipynb` have been read and the Action/Template/Vocabulary pattern is understood
- [ ] `compute_robustness_ratio` return value is unpacked as a tuple: `results, extra = compute_robustness_ratio(...)`
- [ ] `n_per_value=100` is passed explicitly to `generate_sentences` (default is 1000)

---

## 8. Risks & Pitfalls

| Risk | Mitigation |
|---|---|
| Loading CSV with wrong separator (`sep=','`) | Always use `sep='\t'`; verify column count afterward |
| Using `"nature_score_norm"` instead of `"nature_score"` | Default `score_columns` lambda targets `"{value}_score"` = `"nature_score"` — do not override |
| Assuming `compute_robustness_ratio` returns a single dict | It returns a **2-tuple**; always unpack: `results, extra = compute_robustness_ratio(frame)` |
| Generating 1000 sentences instead of 100 | Pass `n_per_value=100` explicitly — the default is 1000 |
| Using column name `"true_label"` (from PDF) | Actual CSV column is `"true_value"` — verify with `frame.columns` |
| Defining Action verbs that don't lexically imply "order" | Use domain-specific verbs: enforce, regulate, maintain, uphold, coordinate |
| Calling metric functions before `%run` completes | Ensure `%run workshop_docs/class_notebook.ipynb` is the first cell and runs without error before any Task 1/2 cells |

---

## 9. Next Step (Post-PRD)

Implementation proceeds entirely inside `AI_workshop_assignment.ipynb`. `class_notebook.ipynb` is never edited.

1. **Setup cell** — `%run workshop_docs/class_notebook.ipynb` to load all shared functions.

2. **Task 1** — define `order_actions`, `order_templates`, `order_vocabularies`, call `generate_sentences` and `build_export_rows`, and write `order_sentences.csv`.

3. **Task 2** — load `scored_nature_sentences.csv`, then add one cell per metric: recognition ratio, robustness ratio, and (optionally) directional expectation. Each cell calls the imported function and prints an interpreted conclusion.

No cells in `class_notebook.ipynb` are modified. The only new files produced are `AI_workshop_assignment.ipynb` and `order_sentences.csv`.
