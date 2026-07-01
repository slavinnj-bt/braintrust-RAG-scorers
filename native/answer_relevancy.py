#!/usr/bin/env python3
"""AnswerRelevancy scorer (DeepEval `AnswerRelevancyMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-answer-relevancy
Algorithm (reimplemented, LLM-as-judge):
  1. Extract the atomic statements made in ``actual_output`` (the model answer).
  2. For each statement, judge whether it is relevant to the ``input`` question
     (verdict yes / idk / no, with a reason).
  3. Score = (statements that are NOT irrelevant) / (total statements).
Referenceless RAG/LLM metric: uses only ``input`` and ``output``.

LLM calls go through the Braintrust AI proxy (OpenAI-compatible). Set the model
with DEEPEVAL_SCORER_MODEL (default claude-sonnet-4-6); auth comes from
BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push answer_relevancy.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class AnswerRelevancyInput(BaseModel):
    input: str
    output: str


def answer_relevancy(input, output) -> dict:
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
            model=model,
            temperature=0,
            max_tokens=max_tokens,
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

    answer = output if isinstance(output, str) else json.dumps(output)
    question = input if isinstance(input, str) else json.dumps(input)

    if not answer.strip():
        return {"name": "AnswerRelevancy", "score": 0.0, "metadata": {"reason": "Empty output."}}

    statements = judge(
        "Break the following text into a list of atomic statements (claims/assertions it makes). "
        "Ignore pure filler. Return JSON: {\"statements\": [\"...\", ...]}.\n\nText:\n" + answer
    ).get("statements", [])
    if not statements:
        statements = [answer]

    verdicts = judge(
        "For the question below, decide whether each statement from an answer is relevant to "
        "addressing the question. For each statement return a verdict: 'yes' (relevant), "
        "'idk' (cannot tell / borderline), or 'no' (irrelevant), with a short reason. "
        "Return JSON: {\"verdicts\": [{\"verdict\": \"yes|idk|no\", \"reason\": \"...\"}, ...]} "
        "in the same order as the statements.\n\n"
        "Question:\n" + question + "\n\nStatements:\n" + json.dumps(statements)
    ).get("verdicts", [])

    total = len(verdicts) or len(statements)
    if total == 0:
        return {"name": "AnswerRelevancy", "score": 0.0, "metadata": {"reason": "No statements to score."}}
    relevant = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() != "no")
    score = relevant / total
    irrelevant = [v.get("reason", "") for v in verdicts if str(v.get("verdict", "")).strip().lower() == "no"]
    return {
        "name": "AnswerRelevancy",
        "score": score,
        "metadata": {
            "statement_count": total,
            "relevant_count": relevant,
            "irrelevant_reasons": irrelevant,
            "reason": "All statements relevant." if not irrelevant else f"{len(irrelevant)} statement(s) judged irrelevant to the question.",
        },
    }


project.scorers.create(
    name="AnswerRelevancy",
    slug="bt-native-answer-relevancy",
    description=(
        "DeepEval AnswerRelevancyMetric: extracts atomic statements from the output and scores "
        "the fraction that are relevant to the input question. Referenceless. 1.0 = fully on-topic."
    ),
    handler=answer_relevancy,
    parameters=AnswerRelevancyInput,
    if_exists="replace",
    metadata={"category": "relevancy", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(answer_relevancy(
        "What is the capital of France and what is it known for?",
        "The capital of France is Paris. Paris is known for the Eiffel Tower. Bananas are yellow.",
    ))
