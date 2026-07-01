#!/usr/bin/env python3
"""ExactMatch scorer that runs the REAL deepeval library inside Braintrust.

Unlike the `rag-*` scorers (which reimplement DeepEval's algorithm as native BT
code), this imports `deepeval` and runs `deepeval.metrics.ExactMatchMetric`.
Deterministic, no LLM. Pushed with deepeval bundled via --requirements.

Push:
    bt functions push deepeval/exact_match.py \
      -o "My Org" -p RAG-Scorers-Demo \
      --language python --runner .venv-deepeval/bin/python \
      --requirements deepeval/requirements.txt --if-exists replace -y
"""

import os

from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ExactMatchInput(BaseModel):
    output: str
    expected: str


def deepeval_exact_match(output, expected, **kwargs):
    import os

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

    from deepeval.metrics import ExactMatchMetric
    from deepeval.test_case import LLMTestCase

    metric = ExactMatchMetric()
    tc = LLMTestCase(input="n/a", actual_output=str(output), expected_output=str(expected))
    metric.measure(tc)
    return {
        "name": "ExactMatch (DeepEval)",
        "score": float(metric.score),
        "metadata": {"reason": getattr(metric, "reason", None), "ran": "deepeval-lib"},
    }


project.scorers.create(
    name="ExactMatch (DeepEval)",
    slug="bt-deepeval-exact-match",
    description="Runs the real deepeval ExactMatchMetric (strict string equality) inside Braintrust. Pick it in the Playground over any dataset with output + expected.",
    handler=deepeval_exact_match,
    parameters=ExactMatchInput,
    if_exists="replace",
    metadata={"framework": "deepeval", "runs": "deepeval-library", "scorer_type": "code"},
)


if __name__ == "__main__":
    print(deepeval_exact_match("Paris", "Paris"))
    print(deepeval_exact_match("paris", "Paris"))
