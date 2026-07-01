# RAG_scorers — DeepEval-style eval scorers for Braintrust

Ready-to-push Braintrust **scorers** that implement DeepEval's RAG & LLM metrics.
Push them once and your whole team can pick them in the **Playground**, attach
them to **experiments**, or run them on **production logs** — no eval code needed.

## Pick a flavor

| Folder | Slugs | What runs | Choose it when |
|---|---|---|---|
| [`native/`](native/) | `bt-native-*` | DeepEval's algorithm reimplemented in plain Python (lean: `openai` + `pydantic`) | You want fast, low‑maintenance, Braintrust‑owned scoring — **recommended for most teams** |
| [`deepeval/`](deepeval/) | `bt-deepeval-*` | the real `deepeval` library, bundled into the function | You need **exact parity** with your existing DeepEval runs |

They behave identically in the UI once pushed and can coexist in the same project.

## Get started (~5 minutes)

```bash
# 0. Prereqs: a Braintrust account, the `bt` CLI (https://www.braintrust.dev/docs),
#    and Python >= 3.10.  Then log in:
bt auth login

# 1. Tell the scorers which project to live in.  IMPORTANT: a pushed scorer lands
#    in the project named here, so always set this (not just the -p flag).
export BRAINTRUST_DEFAULT_PROJECT="Your Project"
# export BRAINTRUST_ORG_NAME="Your Org"     # only if it's not your default org

# 2. Set up ONE folder (native shown; same idea for deepeval/).
cd native
# Use Python >= 3.10 (a plain `python -m venv` may pick an older default and the
# push runner will crash with "unsupported operand type(s) for |"). uv fetches it:
uv venv --python 3.12 .venv        # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install braintrust -r requirements.txt

# 3. Push the SCORER files only — do NOT include run_eval.py / rag_dataset.py
#    (they import the scorers, which would register each slug twice → "duplicate slug").
#    Copy the exact file list from the folder's README; for native/ it's:
bt functions push \
  answer_relevancy.py faithfulness.py contextual_precision.py contextual_recall.py \
  contextual_relevancy.py hallucination.py exact_match.py summarization.py \
  -p "$BRAINTRUST_DEFAULT_PROJECT" --language python --runner .venv/bin/python \
  --requirements requirements.txt --if-exists replace -y
```

**Now use them — two ways:**
- **In the UI (no code):** open a Playground or experiment, add a `bt-native-*` /
  `bt-deepeval-*` scorer from the project's scorer library, and run.
- **From code:** `python run_eval.py` scores a built-in sample dataset and creates
  an experiment you can open in Braintrust.
- **Playground dataset:** each folder ships a standalone `rag_dataset.jsonl`
  (JSON‑lines: `input` / `expected` / `metadata` / `tags`). Upload it independently
  of the scorers — `bt datasets create "<Name>" --file <folder>/rag_dataset.jsonl -p "<Project>"`
  (or `python <folder>/rag_dataset.py`) — then select it in the Playground.

## Metrics at a glance

- **`native/`** (8): AnswerRelevancy, Faithfulness, ContextualPrecision,
  ContextualRecall, ContextualRelevancy, Hallucination, ExactMatch, Summarization.
- **`deepeval/`** (5): ExactMatch, AnswerRelevancy, Faithfulness,
  ContextualRelevancy, and a G‑Eval custom‑criteria judge (Correctness).

## Dataset fields the scorers read

Scorers read these fields off each row; RAG context lives in `metadata`:

```python
{ "input": "...", "output": "...", "expected": "...",
  "metadata": { "retrieval_context": ["chunk", ...],   # RAG metrics
                "context":           ["ground-truth", ...] } }   # Hallucination
```

## Good to know

- **Push goes to the env‑var project.** The scorer file does
  `projects.create(name=os.environ["BRAINTRUST_DEFAULT_PROJECT"] or "RAG-Scorers-Demo")`.
  If you forget to set it, scorers land in a `RAG-Scorers-Demo` project instead.
- **Runner needs Python ≥ 3.10.** Use the folder's `.venv`.
- **Different org than your default?** Add `-o "Your Org"` and set
  `BRAINTRUST_API_KEY=<your key>` on the push.

Full per‑folder details (metric tables, the exact push command, running an eval,
and — for `deepeval/` — how to wrap your *own* DeepEval metric) are in
[`native/README.md`](native/README.md) and [`deepeval/README.md`](deepeval/README.md).
