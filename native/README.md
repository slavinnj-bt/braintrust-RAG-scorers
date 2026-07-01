# native — DeepEval metrics as native Braintrust scorers (`bt-native-*`)

DeepEval's RAG/LLM metrics reimplemented in plain Python — no `deepeval`
dependency, just `openai` + `pydantic`, so they're lightweight and fast. Push
once and use them in the Playground, experiments, or online scoring. (Want the
real DeepEval library instead? See [`../deepeval/`](../deepeval/).)

## Quick start

```bash
cd native
# Use Python >= 3.10 (a plain `python -m venv` may pick an older default and the
# push runner will crash with "unsupported operand type(s) for |"). uv fetches it:
uv venv --python 3.12 .venv        # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install braintrust -r requirements.txt
export BRAINTRUST_DEFAULT_PROJECT="Your Project"           # pushes land here

# Push all 8 scorers
bt functions push \
  answer_relevancy.py faithfulness.py contextual_precision.py contextual_recall.py \
  contextual_relevancy.py hallucination.py exact_match.py summarization.py \
  -p "$BRAINTRUST_DEFAULT_PROJECT" --language python --runner .venv/bin/python \
  --requirements requirements.txt --if-exists replace -y

# Try them on a sample dataset (creates an experiment)
python run_eval.py
```

Then open a Playground or experiment in Braintrust and add any `bt-native-*`
scorer — no code needed. (Different org than your default? add `-o "Your Org"`
and set `BRAINTRUST_API_KEY` on the push.)

## Metrics

| File | Slug | Type | Reads |
|---|---|---|---|
| `answer_relevancy.py` | `bt-native-answer-relevancy` | LLM | input, output |
| `faithfulness.py` | `bt-native-faithfulness` | LLM | output, `metadata.retrieval_context` |
| `contextual_precision.py` | `bt-native-contextual-precision` | LLM | input, expected, `metadata.retrieval_context` |
| `contextual_recall.py` | `bt-native-contextual-recall` | LLM | expected, `metadata.retrieval_context` |
| `contextual_relevancy.py` | `bt-native-contextual-relevancy` | LLM | input, `metadata.retrieval_context` |
| `hallucination.py` | `bt-native-hallucination` | LLM | output, `metadata.context` |
| `exact_match.py` | `bt-native-exact-match` | code | output, expected |
| `summarization.py` | `bt-native-summarization` | LLM | input (source), output (summary) |

All return 0–1 where **higher is better**. (Hallucination returns `1 − rate`, so
1.0 = no hallucination; the raw DeepEval value is kept in the result metadata.)

## Dataset fields

```python
{ "input": "the question", "output": "the system's answer", "expected": "ideal answer",
  "metadata": { "retrieval_context": ["chunk", ...],   # RAG metrics
                "context":           ["ground-truth doc", ...],   # Hallucination
                "document":          "longer source text" } }     # Summarization
```

## Run locally / configure

- `python answer_relevancy.py` — run a single scorer's built‑in demo.
- `python run_eval.py` — score all 8 over the sample dataset (`bt eval run_eval.py` also works).
- LLM judges call the Braintrust AI proxy. Override with `DEEPEVAL_SCORER_MODEL`
  (default `claude-sonnet-4-6`); auth uses `BRAINTRUST_API_KEY` (or `OPENAI_API_KEY`).

## Notes

- Each metric mirrors DeepEval's decomposition (extract → per‑item verdict →
  aggregate), so scores track DeepEval closely. Each file is self‑contained — copy
  one out if you only need a single metric.
- `ExactMatch` is strict; on full‑sentence answers it usually reads 0 — it shines
  on short‑answer datasets.
- Each scorer file registers itself via `projects.create(name=...)` + a handler;
  push only the scorer files (not `run_eval.py` / `rag_dataset.py`).
