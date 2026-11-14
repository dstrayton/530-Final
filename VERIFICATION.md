# Verification completed

- Executed all eight code cells, in order, from a fresh shared namespace. Saved stdout,
  dataframe displays and figures into the notebook using a local Python execution harness.
  The environment did not have a Jupyter kernel execution package available, so this was
  not a Jupyter UI test. Notebook cells contain ordinary executable Python.
- Confirmed 500 raw experiments -> 453 closed-printer rows; 47 open rows removed.
- The notebook exported all three prepared datasets. Cleaned reviews contain 704 rows:
  unknown labels and duplicate material/product/review records were removed.
- Held-out material-mean baseline MAE: 4.366117 MPa.
- Held-out random forest MAE: 2.857697 MPa. The forest was selected and refit.
- These scores use the same stratified 80/20 split (362 train / 91 test), seed 42.
  They are not LLM benchmark scores or confidence intervals.
- All Python application/benchmark files parsed successfully.
- Ran `python benchmark.py --check` successfully: closed-only data, nonmissing
  predictions, eight-material catalog, observed-row lookup, fabricated source rejection,
  unsupported material rejection, distinct alternatives, schema behavior, and 24 cases.
- Live Streamlit rendering, smolagents imports/tool calls and API recommendations were
  NOT run. These packages were unavailable here and installation returned no available
  distributions from this environment. `smolagents[openai]==1.26.0` matches the supplied
  class requirements; the documented agent/model API signatures were checked against
  the official reference. This is not a substitute for local integration testing.
- No private key was included or used. No live benchmark results were fabricated.

Execution environment: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5,
scikit-learn 1.8.0, Matplotlib 3.10.8. The notebook fixes its random seed; other
library versions may produce slightly different results.

## First checks on your machine

1. Install requirements in a fresh virtual environment as described in README.md.
2. Open the notebook; restart the kernel, run all cells and save.
3. Run `python benchmark.py --check`.
4. Set a new personal API key and start `streamlit run app.py`.
5. Try the enclosure example, edit inputs and resubmit, try an empty request, and select
   an open printer. Confirm loading feedback, evidence IDs and actual settings display.
6. Run `python benchmark.py --run`; inspect outputs and complete human scoring.
