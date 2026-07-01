#!/usr/bin/env python3
"""AnswerRelevancy scorer that runs the REAL deepeval library inside Braintrust.

Imports `deepeval` and runs `deepeval.metrics.AnswerRelevancyMetric`. The judge
is routed through Braintrust's auto-injected proxy (Claude) via a custom
DeepEvalBaseLLM, so no separate model key is needed server-side. Pushed with
deepeval bundled via --requirements.
"""

import os

from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class AnswerRelevancyInput(BaseModel):
    input: str
    output: str


def deepeval_answer_relevancy(input, output, **kwargs):
    import json
    import os
    import re

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

    from openai import OpenAI

    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

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

    metric = AnswerRelevancyMetric(model=ProxyJudge(), async_mode=False)
    metric.measure(LLMTestCase(input=str(input), actual_output=str(output)))
    return {
        "name": "AnswerRelevancy (DeepEval)",
        "score": float(metric.score),
        "metadata": {"reason": metric.reason, "ran": "deepeval-lib"},
    }


project.scorers.create(
    name="AnswerRelevancy (DeepEval)",
    slug="bt-deepeval-answer-relevancy",
    description="Runs the real deepeval AnswerRelevancyMetric inside Braintrust (judge via injected proxy). Pick it in the Playground over any dataset with input + output.",
    handler=deepeval_answer_relevancy,
    parameters=AnswerRelevancyInput,
    if_exists="replace",
    metadata={"framework": "deepeval", "runs": "deepeval-library", "scorer_type": "code"},
)


if __name__ == "__main__":
    print(deepeval_answer_relevancy(
        "What is the capital of France?",
        "The capital of France is Paris. Also, bananas are yellow.",
    ))
