#!/usr/bin/env python3
"""Hallucination scorer (DeepEval `HallucinationMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-hallucination
Compares ``actual_output`` against the supplied ground-truth ``context`` (NOT
the retrieved chunks). Uses ``context`` and ``actual_output``.

Algorithm (reimplemented, LLM-as-judge):
  1. For each context document, judge whether the output contradicts it
     (verdict 'no' = output agrees with the context, 'yes' = output contradicts
     / introduces facts that conflict with it).
  2. DeepEval hallucination_score = contradicted contexts / total contexts
     (HIGHER = MORE hallucination = worse).

Braintrust convention is "higher is better", so the returned ``score`` is the
faithful-alignment rate ``1 - hallucination_rate`` (1.0 = no hallucination). The
raw DeepEval ``hallucination_score`` is included in metadata.

Reads ground-truth docs from ``metadata.context`` (list of strings). LLM metric.

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push hallucination.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class HallucinationInput(BaseModel):
    output: str
    metadata: dict = {}


def hallucination(output, metadata=None) -> dict:
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
    ctx = md.get("context") or md.get("retrieval_context") or []
    if isinstance(ctx, str):
        ctx = [ctx]
    answer = output if isinstance(output, str) else json.dumps(output)

    if not ctx:
        return {"name": "Hallucination", "score": 1.0, "metadata": {"reason": "No ground-truth context provided in metadata; nothing to contradict.", "hallucination_score": 0.0}}

    ctx_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctx))
    verdicts = judge(
        "You are given ground-truth context documents and an output. For EACH context document, "
        "decide whether the output AGREES with it or CONTRADICTS it (introduces information that "
        "conflicts with that document). Return verdict 'no' if the output agrees / does not "
        "contradict the document, 'yes' if the output contradicts it. Include a short reason. "
        "Return JSON: {\"verdicts\": [{\"verdict\": \"yes|no\", \"reason\": \"...\"}, ...]} in document order.\n\n"
        "Output:\n" + answer + "\n\nContext documents:\n" + ctx_text
    ).get("verdicts", [])

    total = len(verdicts) or len(ctx)
    contradicted = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() == "yes")
    hallucination_score = contradicted / total if total else 0.0
    score = 1.0 - hallucination_score
    reasons = [v.get("reason", "") for v in verdicts if str(v.get("verdict", "")).strip().lower() == "yes"]
    return {
        "name": "Hallucination",
        "score": score,
        "metadata": {
            "hallucination_score": hallucination_score,
            "context_count": total,
            "contradicted_count": contradicted,
            "contradiction_reasons": reasons,
            "reason": "Output agrees with all ground-truth context (no hallucination)." if contradicted == 0 else f"Output contradicts {contradicted}/{total} context document(s).",
        },
    }


project.scorers.create(
    name="Hallucination",
    slug="bt-native-hallucination",
    description=(
        "DeepEval HallucinationMetric: checks whether the output contradicts the ground-truth "
        "context (metadata.context). Returns faithful-alignment rate (1.0 = no hallucination); "
        "raw DeepEval hallucination_score (higher = worse) is in metadata."
    ),
    handler=hallucination,
    parameters=HallucinationInput,
    if_exists="replace",
    metadata={"category": "safety", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(hallucination(
        "Paris is the capital of France and has a population of 50 million.",
        {"context": ["Paris is the capital of France.",
                     "The population of Paris is about 2.1 million."]},
    ))
