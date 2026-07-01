#!/usr/bin/env python3
"""Faithfulness scorer (DeepEval `FaithfulnessMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-faithfulness
Algorithm (reimplemented, LLM-as-judge):
  1. Extract the claims made in ``actual_output``.
  2. For each claim, judge whether it contradicts the ``retrieval_context``.
     A claim is truthful if it does NOT contradict the retrieved facts.
  3. Score = truthful claims / total claims.

Reads the retrieved chunks from ``metadata.retrieval_context`` (a list of
strings, or a single string). RAG metric.

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push faithfulness.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class FaithfulnessInput(BaseModel):
    output: str
    metadata: dict = {}


def faithfulness(output, metadata=None) -> dict:
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
    rc = md.get("retrieval_context") or md.get("context") or []
    if isinstance(rc, str):
        rc = [rc]
    context_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(rc))
    answer = output if isinstance(output, str) else json.dumps(output)

    if not rc:
        return {"name": "Faithfulness", "score": 0.0, "metadata": {"reason": "No retrieval_context provided in metadata."}}
    if not answer.strip():
        return {"name": "Faithfulness", "score": 1.0, "metadata": {"reason": "Empty output makes no claims."}}

    claims = judge(
        "Extract the list of factual claims asserted by the following text. "
        "Return JSON: {\"claims\": [\"...\", ...]}.\n\nText:\n" + answer
    ).get("claims", [])
    if not claims:
        return {"name": "Faithfulness", "score": 1.0, "metadata": {"reason": "No factual claims found in output."}}

    verdicts = judge(
        "You are given retrieved context and a list of claims from an answer. For each claim, "
        "decide whether it CONTRADICTS the context. Return 'no' if the claim contradicts or is "
        "unsupported by the context, 'yes' if the context supports it, 'idk' if the context is "
        "silent but the claim does not contradict it. Treat 'yes' and 'idk' as truthful (a claim "
        "is unfaithful only if it contradicts the context). Return JSON: "
        "{\"verdicts\": [{\"verdict\": \"yes|idk|no\", \"reason\": \"...\"}, ...]} in claim order.\n\n"
        "Context:\n" + context_text + "\n\nClaims:\n" + json.dumps(claims)
    ).get("verdicts", [])

    total = len(verdicts) or len(claims)
    truthful = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() != "no")
    score = truthful / total if total else 1.0
    contradictions = [v.get("reason", "") for v in verdicts if str(v.get("verdict", "")).strip().lower() == "no"]
    return {
        "name": "Faithfulness",
        "score": score,
        "metadata": {
            "claim_count": total,
            "truthful_count": truthful,
            "contradiction_reasons": contradictions,
            "reason": "All claims grounded in the retrieved context." if not contradictions else f"{len(contradictions)} claim(s) contradict or are unsupported by the context.",
        },
    }


project.scorers.create(
    name="Faithfulness",
    slug="bt-native-faithfulness",
    description=(
        "DeepEval FaithfulnessMetric: extracts claims from the output and scores the fraction "
        "that do not contradict the retrieved context (metadata.retrieval_context). 1.0 = fully grounded."
    ),
    handler=faithfulness,
    parameters=FaithfulnessInput,
    if_exists="replace",
    metadata={"category": "grounding", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(faithfulness(
        "Paris is the capital of France and has a population of 50 million.",
        {"retrieval_context": ["Paris is the capital and largest city of France.",
                               "The population of Paris is about 2.1 million."]},
    ))
