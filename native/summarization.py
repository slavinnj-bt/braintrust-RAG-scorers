#!/usr/bin/env python3
"""Summarization scorer (DeepEval `SummarizationMetric`) for RAG-Scorers-Demo.

DeepEval reference: https://deepeval.com/docs/metrics-summarization
Evaluates a summary (``actual_output``) of an original text (``input``) on two
axes and takes the minimum:

  - Alignment: does the summary contain hallucinated/contradictory info relative
    to the original? (claims in summary supported by original)
  - Coverage: generate N closed-ended yes/no questions answerable 'yes' from the
    original, then check the fraction the summary also answers 'yes'.

  Summarization = min(alignment_score, coverage_score)

Uses ``input`` (original text) and ``output`` (summary). LLM metric. The number
of coverage questions is DEEPEVAL_SUMMARIZATION_QUESTIONS (default 5).

LLM via the Braintrust AI proxy; model = DEEPEVAL_SCORER_MODEL (default
claude-sonnet-4-6); auth = BRAINTRUST_API_KEY (or OPENAI_API_KEY).

Push:
    bt functions push summarization.py \
        -o "My Org" -p RAG-Scorers-Demo \
        --language python --runner .venv-push/bin/python \
        --requirements requirements.txt --if-exists replace -y
"""

import os
from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class SummarizationInput(BaseModel):
    input: str
    output: str


def summarization(input, output) -> dict:
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

    original = input if isinstance(input, str) else json.dumps(input)
    summary = output if isinstance(output, str) else json.dumps(output)
    n_questions = int(os.environ.get("DEEPEVAL_SUMMARIZATION_QUESTIONS", "5"))

    if not summary.strip():
        return {"name": "Summarization", "score": 0.0, "metadata": {"reason": "Empty summary."}}
    if not original.strip():
        return {"name": "Summarization", "score": 0.0, "metadata": {"reason": "Empty original text."}}

    # --- Alignment: summary claims must be supported by the original text ---
    claims = judge(
        "Extract the list of factual claims asserted by this summary. "
        "Return JSON: {\"claims\": [\"...\", ...]}.\n\nSummary:\n" + summary
    ).get("claims", [])
    if claims:
        align_verdicts = judge(
            "Given the ORIGINAL text and a list of claims taken from a SUMMARY of it, decide for each "
            "claim whether the original text SUPPORTS it. Return verdict 'yes' if supported, 'no' if it "
            "contradicts or is not supported by the original. Include a short reason. Return JSON: "
            "{\"verdicts\": [{\"verdict\": \"yes|no\", \"reason\": \"...\"}, ...]} in claim order.\n\n"
            "Original:\n" + original + "\n\nClaims:\n" + json.dumps(claims)
        ).get("verdicts", [])
        a_total = len(align_verdicts) or len(claims)
        a_supported = sum(1 for v in align_verdicts if str(v.get("verdict", "")).strip().lower() == "yes")
        alignment = a_supported / a_total if a_total else 1.0
    else:
        alignment, a_total, a_supported = 1.0, 0, 0

    # --- Coverage: questions answerable 'yes' from the original ---
    questions = judge(
        f"Generate {n_questions} closed-ended yes/no questions that capture the key information in "
        "the ORIGINAL text and whose answer is 'yes' according to the original. "
        "Return JSON: {\"questions\": [\"...\", ...]}.\n\nOriginal:\n" + original
    ).get("questions", [])
    if questions:
        ans = judge(
            "Answer each yes/no question using ONLY the SUMMARY below. If the summary does not "
            "contain the information, answer 'no'. Return JSON: {\"answers\": [\"yes|no\", ...]} in "
            "question order.\n\nSummary:\n" + summary + "\n\nQuestions:\n" + json.dumps(questions)
        ).get("answers", [])
        c_total = len(questions)
        c_yes = sum(1 for a in ans if str(a).strip().lower() == "yes")
        coverage = c_yes / c_total if c_total else 0.0
    else:
        coverage, c_total, c_yes = 0.0, 0, 0

    score = min(alignment, coverage)
    return {
        "name": "Summarization",
        "score": score,
        "metadata": {
            "alignment_score": alignment,
            "coverage_score": coverage,
            "claims_supported": f"{a_supported}/{a_total}",
            "questions_covered": f"{c_yes}/{c_total}",
            "reason": f"min(alignment={alignment:.2f}, coverage={coverage:.2f}). "
                      "Alignment = summary claims supported by the original; coverage = key questions the summary still answers.",
        },
    }


project.scorers.create(
    name="Summarization",
    slug="bt-native-summarization",
    description=(
        "DeepEval SummarizationMetric: score = min(alignment, coverage). Alignment = fraction of "
        "summary claims supported by the original (input); coverage = fraction of key yes/no "
        "questions about the original that the summary (output) still answers 'yes'."
    ),
    handler=summarization,
    parameters=SummarizationInput,
    if_exists="replace",
    metadata={"category": "summarization", "scorer_type": "llm", "framework": "deepeval", "owner": "ai-team"},
)


if __name__ == "__main__":
    print(summarization(
        "The Apollo program was a series of human spaceflight missions run by NASA between 1961 "
        "and 1972. It achieved the first crewed Moon landing in 1969 with Apollo 11, and is "
        "remembered as a landmark of 20th-century engineering.",
        "NASA's Apollo program landed the first humans on the Moon in 1969 during Apollo 11.",
    ))
