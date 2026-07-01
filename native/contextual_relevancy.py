#!/usr/bin/env python3
"""ContextualRelevancy scorer (DeepEval `ContextualRelevancyMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-contextual-relevancy
Measures how much of the ``retrieval_context`` is actually relevant to the
``input`` (signal vs. noise in what the retriever returned).

Algorithm (reimplemented, LLM-as-judge):
  1. Split each retrieved node into atomic statements.
  2. For each statement, judge whether it is relevant to the input question.
  3. Score = relevant statements / total statements.

Reads nodes from ``metadata.retrieval_context`` (list of strings). RAG metric.

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push contextual_relevancy.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ContextualRelevancyInput(BaseModel):
    input: str
    metadata: dict = {}


def contextual_relevancy(input, metadata=None) -> dict:
    import json
    import os
    import re

    from openai import OpenAI

    def judge(prompt, max_tokens=1800):
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

    if not rc:
        return {"name": "ContextualRelevancy", "score": 0.0, "metadata": {"reason": "No retrieval_context provided in metadata."}}

    nodes_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(rc))
    result = judge(
        "Given a question and retrieved context nodes, split the context into atomic statements "
        "and judge each statement's relevance to the question. Return verdict 'yes' (relevant) or "
        "'no' (irrelevant) with a short reason. Return JSON: "
        "{\"verdicts\": [{\"statement\": \"...\", \"verdict\": \"yes|no\", \"reason\": \"...\"}, ...]}.\n\n"
        "Question:\n" + question + "\n\nRetrieved context:\n" + nodes_text
    )
    verdicts = result.get("verdicts", [])
    total = len(verdicts)
    if total == 0:
        return {"name": "ContextualRelevancy", "score": 0.0, "metadata": {"reason": "No statements extracted from context."}}
    relevant = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() == "yes")
    score = relevant / total
    irrelevant = [v.get("statement", v.get("reason", "")) for v in verdicts if str(v.get("verdict", "")).strip().lower() != "yes"]
    return {
        "name": "ContextualRelevancy",
        "score": score,
        "metadata": {
            "statement_count": total,
            "relevant_count": relevant,
            "irrelevant_statements": irrelevant,
            "reason": "All retrieved statements are relevant to the question." if not irrelevant else f"{len(irrelevant)}/{total} retrieved statements are irrelevant (retrieval noise).",
        },
    }


project.scorers.create(
    name="ContextualRelevancy",
    slug="bt-native-contextual-relevancy",
    description=(
        "DeepEval ContextualRelevancyMetric: splits the retrieved context into statements and scores "
        "the fraction relevant to the input question (metadata.retrieval_context). 1.0 = no retrieval noise."
    ),
    handler=contextual_relevancy,
    parameters=ContextualRelevancyInput,
    if_exists="replace",
    metadata={"category": "retrieval", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(contextual_relevancy(
        "What is the capital of France?",
        {"retrieval_context": [
            "Paris is the capital and largest city of France.",
            "Bananas are a popular fruit grown in tropical climates.",
        ]},
    ))
