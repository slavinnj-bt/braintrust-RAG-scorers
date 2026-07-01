#!/usr/bin/env python3
"""G-Eval (Correctness) scorer that runs the REAL deepeval library in Braintrust.

Imports `deepeval` and runs `deepeval.metrics.GEval` — the flexible "define your
own criteria" LLM judge (chain-of-thought + form-filling). This instance scores
factual correctness of the output vs the expected answer for the given input.
Judge routed through Braintrust's injected proxy via a custom DeepEvalBaseLLM.

Edit `CRITERIA` / `EVAL_PARAMS` below to make any custom G-Eval scorer.
"""

import os

from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)

CRITERIA = (
    "Determine whether the actual output is factually correct and fully addresses the "
    "input question, using the expected output as the reference answer. Penalize "
    "missing key facts, contradictions, and unsupported additions."
)


class GEvalInput(BaseModel):
    input: str
    output: str
    expected: str


def deepeval_geval_correctness(input, output, expected, **kwargs):
    import json
    import os
    import re

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

    from openai import OpenAI

    from deepeval.metrics import GEval
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    class ProxyJudge(DeepEvalBaseLLM):
        def __init__(self, name="claude-sonnet-4-6"):
            self._name = name
            super().__init__(name)

        def load_model(self):
            return OpenAI(
                api_key=os.environ.get("BRAINTRUST_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("BRAINTRUST_PROXY_URL", "https://api.braintrust.dev/v1/proxy"),
            )

        def get_model_name(self):
            return self._name

        def supports_log_probs(self):
            return False

        def _c(self, prompt, schema=None):
            system = "You are a precise evaluation engine."
            if schema is not None:
                system += " Respond ONLY with a single JSON object matching the requested fields. No markdown."
            r = self.model.chat.completions.create(
                model=self._name, temperature=0, max_tokens=2000,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            t = (r.choices[0].message.content or "").strip()
            if schema is None:
                return t
            m = re.search(r"(\{.*\}|\[.*\])", t, re.DOTALL)
            return schema.model_validate(json.loads(m.group(1) if m else t))

        def generate(self, prompt, schema=None):
            return self._c(prompt, schema)

        async def a_generate(self, prompt, schema=None):
            return self._c(prompt, schema)

    metric = GEval(
        name="Correctness",
        criteria=CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=ProxyJudge(),
        async_mode=False,
    )
    metric.measure(LLMTestCase(input=str(input), actual_output=str(output), expected_output=str(expected)))
    return {
        "name": "G-Eval Correctness (DeepEval)",
        "score": float(metric.score),
        "metadata": {"reason": metric.reason, "criteria": CRITERIA, "ran": "deepeval-lib"},
    }


project.scorers.create(
    name="G-Eval Correctness (DeepEval)",
    slug="bt-deepeval-geval-correctness",
    description="Runs deepeval's GEval (custom-criteria LLM judge) for factual correctness vs the expected answer. Edit the criteria to make any bespoke G-Eval scorer. Pick it in the Playground over a dataset with input + output + expected.",
    handler=deepeval_geval_correctness,
    parameters=GEvalInput,
    if_exists="replace",
    metadata={"framework": "deepeval", "runs": "deepeval-library", "scorer_type": "code", "category": "g-eval"},
)


if __name__ == "__main__":
    print(deepeval_geval_correctness(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Paris",
    ))
    print(deepeval_geval_correctness(
        "How tall is Mount Everest?",
        "Mount Everest is about 9,500 meters tall.",
        "About 8,849 meters (29,032 ft).",
    ))
