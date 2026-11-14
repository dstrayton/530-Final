# Filament Finder

A small class project: three main code files, local CSVs, and no separate backend service.

## Files to understand, in order

1. `01_prepare_data.ipynb`: imports three sources with pandas; shows verification tables;
   removes open printers; cleans names; fits and evaluates random forest regression;
   compares a material-mean baseline; exports prepared datasets and evaluation evidence.
2. `app.py`: Streamlit interface and smolagents `ToolCallingAgent` in one file.
   Like the class examples, creates a model, defines tools, creates an agent and runs it.
   Tools compare ML predictions and retrieve descriptions/review excerpts. The output
   validator checks structure, supported materials and source IDs. The app allows users
   to edit their request and retry. No live scraping or separate database is needed.
3. `benchmark.py`: 24 documented cases; the same model with evidence upfront versus the
   tool-using agent. Saves raw answers, evidence, tool calls, cases, prompt configuration,
   versions, data hashes, automatic metrics, latency and a comparison chart.

`requirements.txt` is installation support, not another application component.
`data/` contains both original and generated tables. The notebook has already been run;
rerun it to reproduce the included outputs.

## Install and run (Mac terminal)

Unzip the project, then open a terminal inside the `filament_finder` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook 01_prepare_data.ipynb
```

Choose **Kernel > Restart Kernel and Run All Cells**, then save the notebook. In Colab,
upload `01_prepare_data.ipynb` and `performance_raw.csv`, `descriptions_raw.csv`, and
`reviews_raw.csv` from `data/` to the same working directory. If needed, install pandas,
scikit-learn and matplotlib in Colab first. Download the prepared outputs into local
`data/` afterward; the actual website runs locally with Streamlit, not inside Colab.

In the terminal, set your OWN new key (do not paste it into the Python files):

```bash
export OPENAI_API_KEY="YOUR_NEW_API_KEY"
export OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

The browser opens the local app. Keep the terminal running. To stop, press Ctrl+C.
Optional: set `OPENAI_BASE_URL` if your approved OpenAI-compatible class endpoint
supports chat completions and tool calling. Model access depends on your account.
No `.env` file or model_setup.py is required. Do not reuse the exposed class-example key.

## What the ML actually does

- Target: specimen peak stress, converted from kPa to MPa.
- Inputs: material, infill pattern, layer thickness, infill density, and speed.
- Closed-only filter retains 453 of 500 original experiments (47 open rows removed).
- Text inputs are one-hot encoded using training rows only.
- 80/20 held-out split, stratified by material, random seed 42.
- Fixed forest: 200 trees, minimum leaf size 3, all features considered per split.
- Compare against each material's training mean; choose lower held-out MAE and refit.
- `predicted_peak_stress_MPa` in `performance_prepared.csv` is a fitted estimate at an
  observed setting, NOT a new observation or an independent test prediction.
- Only `ml_held_out_predictions.csv` supports the reported test error.
- The app uses the nearest observed configuration for each material and displays actual
  settings and whether it was an exact match. It does not predict arbitrary new combinations.
- Strength-oriented recommendations must compare these predictions. Descriptions and
  reviews inform other priorities. The comparison table also exposes the ML values.

This is a deliberately conservative first version. Better unseen-configuration modeling
can come later after checking experimental design and coverage.

## Data documentation

Original user-provided filenames are mapped to portable names:

| Original | Included copy | Generated main output |
|---|---|---|
| 3D Printing Data(2).csv | data/performance_raw.csv | data/performance_prepared.csv |
| Material Data(1).csv | data/descriptions_raw.csv | data/descriptions_clean.csv |
| filament_reviews_combined_707(1).csv | data/reviews_raw.csv | data/reviews_clean.csv |

The three original files are preserved byte-for-byte. Two unnamed columns contain
isolated stray numbers; all unnamed columns are dropped from the working data.
Normalize whitespace/case and infill aliases. Keep missing strain-at-break values because
that variable is not used. Preserve distinct polymers and brands. Remove unknown review
materials and duplicate material/product/review records. Preserve other review-only
materials in the cleaned file; the app does not treat them as experimental candidates.

The `PLA phosphor` description explicitly identifies glow-in-the-dark filament. Mapping
it and `PLA phosphorus` to `PLA-GLOW` is a working assumption requiring source confirmation.
Ten materials remain in experimental data; eight also have descriptions and are eligible
for recommendations. rPET and transparent PLA remain in data but are outside that app catalog.
Review excerpt retrieval shows one lower- and one higher-rated example when available;
this is not an exhaustive theme analysis or an unbiased estimate of customer sentiment.
IDs EXP/ DESC/ REV identify rows in supplied CSVs, not external website citations.

Before final submission, add original experimental provenance, test protocol/standards,
brand information, Xometry page URLs, Amazon listing URLs, collection dates, and permitted
redistribution details. These were not provided in the CSVs and have not been invented.
Descriptions are secondary supplied text, not verified specifications. Reviews may be
product-specific, imbalanced or otherwise unrepresentative. No claims of causal material
superiority follow from the experiment table: settings and material coverage are uneven.

## Benchmark instructions

Offline checks (no API calls):

```bash
python benchmark.py --check
```

Live evaluation (24 cases x 2 approaches; makes paid API calls):

```bash
python benchmark.py --run
```

Results go to a new timestamped folder under `benchmark_results/`. The baseline uses the
same model and full available evidence upfront, with no retrieval tools. smolagents still
uses its final-answer mechanism; this is a no-retrieval-tool baseline, not a claim of one
network request. Each case is independent. One run is a snapshot; repeat full runs if you
need stability estimates. Temperature zero does not guarantee identical API outputs.

Cases include 8 ordinary, 6 tradeoff, 4 ambiguous and 6 failure/edge requests. Each has a
predefined expected status. Recommendation cases permit multiple defensible materials.
Explicit safety certification, open printers and unavailable products test scope handling.
These scenarios were authored from the product's intended use and limits, not selected
after seeing model responses. Review expected labels before evaluation; they are an
authored rubric, not independently established ground truth.

Automatic scores: valid output/source IDs, expected-status match, and mean/median latency.
Errors stay in the denominator. A valid source ID does NOT establish factual grounding.
Overall task quality requires human review of each saved response against its evidence:

| Human field | 0 | 1 | 2 |
|---|---|---|---|
| material_fit | Wrong material/action or violates requirement | Plausible but misses a priority | Appropriate choice or appropriate clarification/refusal |
| grounding | Invented/misrepresented claims | Mostly supported, some overstatement | Material factual claims supported and evidence types distinguished |
| tradeoffs | Missing/misleading limits | Generic limitations | Relevant tradeoffs, actual settings mismatch, and evidence gaps explained |

Fill these columns in `results.csv`; leave unreviewed fields blank, not zero. Add notes
and references to source IDs. Have another person independently score a subset if possible,
record disagreements, and reconcile the rubric. There is no LLM judge. Recompute:

```bash
python benchmark.py --summarize benchmark_results/YOUR_RUN_FOLDER
```

The summary shows counts of human-scored cases and means; the chart covers automated
checks only. Inspect at least two successes and two failures and document two concrete
improvements based on actual results. Potential changes to investigate include fuller
review retrieval and stronger claim-level validation, but do not call them findings until
the evaluation runs. The benchmark is not yet live-run in this delivered prototype.

## Limits and next course deliverables

The app checks source membership and structure, not semantic truth of every sentence.
The tool-check confirms both selected materials were retrieved. It cannot guarantee the
LLM weighed evidence correctly. It refuses to display an unchecked recommendation when
validation fails. No medical, safety, food-contact or load certification is provided.
No heat-resistance or impact model is trained. Closed-printer data do not establish that
all closed printers or filament brands behave alike. The small per-material test samples
and missing batch IDs limit generalization evidence.

This package is the requested simple implementation, not the entire final submission:
you still need live API testing, human benchmark scoring and analysis, usability testing
with two documented UX revisions, a portfolio page, a walkthrough video and final source
provenance. See VERIFICATION.md for what was actually run in the creation environment.

Official API reference used for the class-style tool agent:
https://huggingface.co/docs/smolagents/en/reference/agents
