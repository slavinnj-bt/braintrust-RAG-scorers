#!/usr/bin/env python3
"""ContextualPrecision scorer (DeepEval `ContextualPrecisionMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-contextual-precision
Measures whether RELEVANT nodes in ``retrieval_context`` are ranked higher than
irrelevant ones (a re-ranker quality signal). Uses ``input``, ``expected_output``,
and ``retrieval_context``.

Algorithm (reimplemented, LLM-as-judge):
  1. For each retrieved node, judge relevance (1/0) to the input using the ideal
     expected_output as the objective reference.
  2. Weighted cumulative precision:
       score = (1 / #relevant) * Σ_k [ (#relevant in first k / k) * r_k ]
     which rewards relevant nodes appearing earlier in the ranking.

Reads nodes from ``metadata.retrieval_context`` (list of strings). RAG metric.

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push contextual_precision.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ContextualPrecisionInput(BaseModel):
    input: str
    expected: str
    metadata: dict = {}


def contextual_precision(input, expected, metadata=None) -> dict:
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
    question = input if isinstance(input, str) else json.dumps(input)
    ideal = expected if isinstance(expected, str) else json.dumps(expected)

    if not rc:
        return {"name": "ContextualPrecision", "score": 0.0, "metadata": {"reason": "No retrieval_context provided in metadata."}}

    nodes_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(rc))
    verdicts = judge(
        "Given a question and the ideal (expected) answer, judge each retrieved context node for "
        "whether it is relevant to answering the question. Use the expected answer to stay objective. "
        "For each node return verdict 'yes' (relevant) or 'no' (not relevant) with a short reason. "
        "Return JSON: {\"verdicts\": [{\"verdict\": \"yes|no\", \"reason\": \"...\"}, ...]} in node order.\n\n"
        "Question:\n" + question + "\n\nExpected answer:\n" + ideal + "\n\nRetrieved nodes:\n" + nodes_text
    ).get("verdicts", [])

    rel = [1 if str(v.get("verdict", "")).strip().lower() == "yes" else 0 for v in verdicts]
    rel = rel[: len(rc)] + [0] * (len(rc) - len(rel))
    total_relevant = sum(rel)
    if total_relevant == 0:
        return {
            "name": "ContextualPrecision",
            "score": 0.0,
            "metadata": {"node_count": len(rc), "relevant_count": 0,
                         "reason": "No retrieved node was judged relevant to the question."},
        }

    cumulative = 0.0
    running_relevant = 0
    for k, r in enumerate(rel, start=1):
        if r:
            running_relevant += 1
            cumulative += (running_relevant / k)
    score = cumulative / total_relevant
    return {
        "name": "ContextualPrecision",
        "score": score,
        "metadata": {
            "node_count": len(rc),
            "relevant_count": total_relevant,
            "relevance_mask": rel,
            "reason": f"{total_relevant}/{len(rc)} nodes relevant; weighted precision rewards relevant nodes ranked earlier.",
        },
    }


project.scorers.create(
    name="ContextualPrecision",
    slug="bt-native-contextual-precision",
    description=(
        "DeepEval ContextualPrecisionMetric: judges each retrieved node's relevance (using input + "
        "expected) and computes weighted cumulative precision that rewards relevant nodes ranked "
        "higher. Reads metadata.retrieval_context. A re-ranker quality signal."
    ),
    handler=contextual_precision,
    parameters=ContextualPrecisionInput,
    if_exists="replace",
    metadata={"category": "retrieval", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(contextual_precision(
        "What is the capital of France?",
        "Paris",
        {"retrieval_context": [
            "Paris is the capital and largest city of France.",
            "France is a country in Western Europe.",
            "Bananas are a popular fruit.",
        ]},
    ))
