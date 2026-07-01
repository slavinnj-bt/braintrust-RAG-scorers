#!/usr/bin/env python3
"""ExactMatch scorer (DeepEval `ExactMatchMetric`) for the RAG-Scorers-Demo project.

DeepEval reference: https://deepeval.com/docs/metrics-introduction
ExactMatch is deterministic: it scores 1.0 when ``actual_output`` equals
``expected_output`` and 0.0 otherwise. This implementation applies light
normalization (case, surrounding whitespace, and trailing punctuation) so that
cosmetically-identical answers still match; set ``DEEPEVAL_EXACT_STRICT=1`` to
compare raw strings instead.

Code scorer — no LLM call.

Push to Braintrust (uses your bt OAuth profile; needs a Python >=3.10 runner):
    bt functions push exact_match.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ExactMatchInput(BaseModel):
    output: str
    expected: str


def exact_match(output, expected) -> dict:
    """1.0 when output == expected (normalized), else 0.0. Self-contained."""
    import json
    import os
    import re

    strict = os.environ.get("DEEPEVAL_EXACT_STRICT", "") not in ("", "0", "false", "False")

    def norm(value):
        if value is None:
            return ""
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        if strict:
            return text
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.?!,;:\s]+$", "", text)
        text = text.strip("\"'")
        return text

    o, e = norm(output), norm(expected)
    score = 1.0 if (e != "" and o == e) else 0.0
    return {
        "name": "ExactMatch",
        "score": score,
        "metadata": {
            "strict": strict,
            "normalized_output": o,
            "normalized_expected": e,
            "reason": "Exact match" if score == 1.0 else "Output does not exactly match the expected answer",
        },
    }


project.scorers.create(
    name="ExactMatch",
    slug="bt-native-exact-match",
    description=(
        "DeepEval ExactMatchMetric: deterministic 1.0 when the output equals the expected "
        "answer (case / whitespace / trailing punctuation normalized; set DEEPEVAL_EXACT_STRICT=1 "
        "for raw comparison), else 0.0. Best for short-answer LLM evals."
    ),
    handler=exact_match,
    parameters=ExactMatchInput,
    if_exists="replace",
    metadata={"category": "accuracy", "scorer_type": "code", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(exact_match("Paris", "Paris"))
    print(exact_match("paris.", "Paris"))
    print(exact_match("the city of Paris", "Paris"))
