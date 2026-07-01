#!/usr/bin/env python3
"""Faithfulness scorer that runs the REAL deepeval library inside Braintrust.

Imports `deepeval` and runs `deepeval.metrics.FaithfulnessMetric`. Reads the
retrieved chunks from metadata.retrieval_context. Judge routed through
Braintrust's injected proxy (Claude) via a custom DeepEvalBaseLLM. Pushed with
deepeval bundled via --requirements.
"""

import os

from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class FaithfulnessInput(BaseModel):
    input: str
    output: str
    metadata: dict = {}


def deepeval_faithfulness(input, output, metadata=None, **kwargs):
    import json
    import os
    import re

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

    from openai import OpenAI

    from deepeval.metrics import FaithfulnessMetric
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

    md = metadata or {}
    rc = md.get("retrieval_context") or md.get("context") or []
    if isinstance(rc, str):
        rc = [rc]
    if not rc:
        return {"name": "Faithfulness (DeepEval)", "score": 1.0,
                "metadata": {"reason": "No retrieval_context in metadata; nothing to contradict.", "ran": "deepeval-lib"}}

    metric = FaithfulnessMetric(model=ProxyJudge(), async_mode=False)
    metric.measure(LLMTestCase(input=str(input), actual_output=str(output), retrieval_context=[str(c) for c in rc]))
    return {
        "name": "Faithfulness (DeepEval)",
        "score": float(metric.score),
        "metadata": {"reason": metric.reason, "ran": "deepeval-lib"},
    }


project.scorers.create(
    name="Faithfulness (DeepEval)",
    slug="bt-deepeval-faithfulness",
    description="Runs the real deepeval FaithfulnessMetric inside Braintrust (judge via injected proxy). Reads metadata.retrieval_context. Pick it in the Playground over a RAG dataset.",
    handler=deepeval_faithfulness,
    parameters=FaithfulnessInput,
    if_exists="replace",
    metadata={"framework": "deepeval", "runs": "deepeval-library", "scorer_type": "code"},
)


if __name__ == "__main__":
    print(deepeval_faithfulness(
        "How tall is Mount Everest?",
        "Mount Everest is about 9,500 meters tall.",
        {"retrieval_context": ["Mount Everest's peak is 8,849 metres (29,032 ft) above sea level."]},
    ))
