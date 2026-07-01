#!/usr/bin/env python3
"""ContextualRecall scorer (DeepEval `ContextualRecallMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-contextual-recall
Measures whether the ``retrieval_context`` contains enough information to support
the ideal answer. Uses ``expected_output`` and ``retrieval_context``.

Algorithm (reimplemented, LLM-as-judge):
  1. Break the expected_output into atomic statements.
  2. For each statement, judge whether it can be attributed to (is supported by)
     any node in the retrieval_context (verdict yes/no).
  3. Score = attributable statements / total statements.

Reads nodes from ``metadata.retrieval_context`` (list of strings). RAG metric.

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push contextual_recall.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ContextualRecallInput(BaseModel):
    expected: str
    metadata: dict = {}


def contextual_recall(expected, metadata=None) -> dict:
    import json
    import os
    import re

    from openai import OpenAI

    def judge(prompt, max_tokens=1500):
        api_key = os.environ.get("BRAINTRUST_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        base_url = os.environ.get("BRAINTRUST_PROXY_URL", "https://api.braintrust.dev/v1/proxy")
        model = os.environ.get("DEEPEVAL_SCORER_MODEL", "claude-sonnet-4-6")
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are a meticulous evaluation engine. Respond with ONLY valid JSON. No prose, no markdown fences."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text)

    md = metadata or {}
    rc = md.get("retrieval_context") or []
    if isinstance(rc, str):
        rc = [rc]
    ideal = expected if isinstance(expected, str) else json.dumps(expected)

    if not rc:
        return {"name": "ContextualRecall", "score": 0.0, "metadata": {"reason": "No retrieval_context provided in metadata."}}
    if not ideal.strip():
        return {"name": "ContextualRecall", "score": 0.0, "metadata": {"reason": "No expected output provided."}}

    statements = judge(
        "Break the following expected answer into a list of atomic statements. "
        "Return JSON: {\"statements\": [\"...\", ...]}.\n\nExpected answer:\n" + ideal
    ).get("statements", [])
    if not statements:
        statements = [ideal]

    nodes_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(rc))
    verdicts = judge(
        "Given retrieved context and a list of statements from the ideal answer, decide for each "
        "statement whether it can be attributed to (is supported by) the retrieved context. "
        "Return verdict 'yes' if supported, 'no' otherwise, with a short reason. Return JSON: "
        "{\"verdicts\": [{\"verdict\": \"yes|no\", \"reason\": \"...\"}, ...]} in statement order.\n\n"
        "Retrieved context:\n" + nodes_text + "\n\nStatements:\n" + json.dumps(statements)
    ).get("verdicts", [])

    total = len(verdicts) or len(statements)
    attributable = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() == "yes")
    score = attributable / total if total else 0.0
    missing = [v.get("reason", "") for v in verdicts if str(v.get("verdict", "")).strip().lower() != "yes"]
    return {
        "name": "ContextualRecall",
        "score": score,
        "metadata": {
            "statement_count": total,
            "attributable_count": attributable,
            "unsupported_reasons": missing,
            "reason": "All parts of the ideal answer are supported by the retrieved context." if not missing else f"{len(missing)} statement(s) of the ideal answer are not found in the retrieved context.",
        },
    }


project.scorers.create(
    name="ContextualRecall",
    slug="bt-native-contextual-recall",
    description=(
        "DeepEval ContextualRecallMetric: splits the expected answer into statements and scores the "
        "fraction supported by the retrieved context (metadata.retrieval_context). 1.0 = retriever "
        "surfaced everything the ideal answer needs."
    ),
    handler=contextual_recall,
    parameters=ContextualRecallInput,
    if_exists="replace",
    metadata={"category": "retrieval", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(contextual_recall(
        "Paris is the capital of France and is home to the Eiffel Tower.",
        {"retrieval_context": [
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower is a landmark located in Paris.",
        ]},
    ))
