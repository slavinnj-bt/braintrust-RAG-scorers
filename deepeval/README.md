# deepeval — run the real DeepEval library in Braintrust (`bt-deepeval-*`)

These scorers `import deepeval` and run the actual metrics. DeepEval is bundled
into the Braintrust function at push time, so it executes server‑side — same
Playground / experiment / online‑scoring behavior as any scorer, with **exact
DeepEval parity**. Heavier than the native versions in [`../native/`](../native/)
(bigger bundle, slower first run); pick whichever fits.

## Quick start

```bash
cd deepeval
# Use Python >= 3.10 (a plain `python -m venv` may pick an older default and the
# push runner will crash with "unsupported operand type(s) for |"). uv fetches it:
uv venv --python 3.12 .venv        # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install braintrust -r requirements.txt                 # installs deepeval too
export BRAINTRUST_DEFAULT_PROJECT="Your Project"           # pushes land here

# Push all 5 scorers (bundles deepeval into each function)
bt functions push \
  exact_match.py answer_relevancy.py faithfulness.py \
  contextual_relevancy.py g_eval_correctness.py \
  -p "$BRAINTRUST_DEFAULT_PROJECT" --language python --runner .venv/bin/python \
  --requirements requirements.txt --if-exists replace -y

# Try them on a sample dataset (creates an experiment)
python run_eval.py
```

Then add any `bt-deepeval-*` scorer in a Playground or experiment — no code
needed. The first run is slower while DeepEval cold‑starts. (Different org than
your default? add `-o "Your Org"` and set `BRAINTRUST_API_KEY` on the push.)

## Metrics

| File | Slug | DeepEval metric | Reads |
|---|---|---|---|
| `exact_match.py` | `bt-deepeval-exact-match` | `ExactMatchMetric` (deterministic) | output, expected |
| `answer_relevancy.py` | `bt-deepeval-answer-relevancy` | `AnswerRelevancyMetric` | input, output |
| `faithfulness.py` | `bt-deepeval-faithfulness` | `FaithfulnessMetric` | input, output, `metadata.retrieval_context` |
| `contextual_relevancy.py` | `bt-deepeval-contextual-relevancy` | `ContextualRelevancyMetric` (RAG) | input, output, `metadata.retrieval_context` |
| `g_eval_correctness.py` | `bt-deepeval-geval-correctness` | `GEval` (custom criteria) | input, output, expected |

The LLM judges route through Braintrust's auto‑injected proxy via a custom
`DeepEvalBaseLLM` (model `claude-sonnet-4-6` by default), so no model key is
needed server‑side.

## native vs. deepeval

| | [`../native/`](../native/) `bt-native-*` | this folder `bt-deepeval-*` |
|---|---|---|
| What runs | DeepEval's algorithm, hand‑written | the `deepeval` library |
| Deps bundled | `openai` + `pydantic` (light) | full `deepeval` tree (heavy) |
| Cold start | fast | slower |
| Fidelity | close reimplementation | exact |

---

## Advanced

### How the push works
`bt functions push <file.py> --requirements requirements.txt` (1) imports the
file with `--runner` Python and registers every `project.scorers.create(...)`
against the project from `projects.create(name=...)`; (2) bundles the handler +
`requirements.txt` deps (here, the whole `deepeval` tree) into the function; (3)
runs it server‑side, where Braintrust injects `BRAINTRUST_API_KEY` + proxy access
so LLM judges work with no key in the bundle. Push only scorer files (not
`run_eval.py`). The push runner needs `braintrust` + `pydantic`; `deepeval` is
bundled, not needed to push.

### Wrap your OWN DeepEval metric as a Braintrust scorer
Already have a DeepEval metric? Wrap it once — the scoring stays 100% DeepEval:

1. **Map fields:** `actual_output`→`output`, `expected_output`→`expected`,
   `input`→`input`, `retrieval_context`/`context`→ from `metadata`.
2. **Write a handler** + declare its inputs with a pydantic model:

```python
import os
from pydantic import BaseModel
from braintrust import projects

project = projects.create(name=os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo"))

class Inp(BaseModel):
    input: str
    output: str

def answer_relevancy(input, output, **_):
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    m = AnswerRelevancyMetric(model=ProxyJudge(), async_mode=False)   # ProxyJudge: copy from any file here
    m.measure(LLMTestCase(input=input, actual_output=output))
    return {"name": "AnswerRelevancy (DeepEval)", "score": float(m.score), "metadata": {"reason": m.reason}}

project.scorers.create(name="AnswerRelevancy (DeepEval)", slug="bt-deepeval-answer-relevancy",
                       handler=answer_relevancy, parameters=Inp, if_exists="replace")
```

3. **Route the judge through Braintrust:** pass `model=ProxyJudge(), async_mode=False`
   (copy the `ProxyJudge` `DeepEvalBaseLLM` from any file in this folder).
4. **Add `deepeval` to `requirements.txt`** and push (command above).

**Gotchas:** scores must be floats in `[0,1]`; for Hallucination return
`1 - m.score` (DeepEval's hallucination score is higher‑is‑worse); set
`os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")` in the handler; keep
all `import`s inside the handler so it bundles.
