#!/usr/bin/env python3
"""ContextualRelevancy (RAG) scorer that runs the REAL deepeval library in Braintrust.

Imports `deepeval` and runs `deepeval.metrics.ContextualRelevancyMetric` — a RAG
retrieval-quality metric: of the statements in the retrieved context, what
fraction are relevant to the question. Reads metadata.retrieval_context. Judge
routed through Braintrust's injected proxy via a custom DeepEvalBaseLLM.
"""

import os

from pydantic import BaseModel

from braintrust import projects

PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
project = projects.create(name=PROJECT_NAME)


class ContextualRelevancyInput(BaseModel):
    input: str
    output: str
    metadata: dict = {}


def deepeval_contextual_relevancy(input, output, metadata=None, **kwargs):
    import json
    import os
    import re

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

    from openai import OpenAI

    from deepeval.metrics import ContextualRelevancyMetric
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

    md = metadata or {}
    rc = md.get("retrieval_context") or md.get("context") or []
    if isinstance(rc, str):
        rc = [rc]
    if not rc:
        return {"name": "ContextualRelevancy (DeepEval)", "score": 0.0,
                "metadata": {"reason": "No retrieval_context in metadata.", "ran": "deepeval-lib"}}

    metric = ContextualRelevancyMetric(model=ProxyJudge(), async_mode=False)
    metric.measure(LLMTestCase(input=str(input), actual_output=str(output), retrieval_context=[str(c) for c in rc]))
    return {
        "name": "ContextualRelevancy (DeepEval)",
        "score": float(metric.score),
        "metadata": {"reason": metric.reason, "ran": "deepeval-lib"},
    }


project.scorers.create(
    name="ContextualRelevancy (DeepEval)",
    slug="bt-deepeval-contextual-relevancy",
    description="Runs the real deepeval ContextualRelevancyMetric (RAG retrieval quality) inside Braintrust. Reads metadata.retrieval_context. Pick it in the Playground over a RAG dataset.",
    handler=deepeval_contextual_relevancy,
    parameters=ContextualRelevancyInput,
    if_exists="replace",
    metadata={"framework": "deepeval", "runs": "deepeval-library", "scorer_type": "code", "category": "rag"},
)


if __name__ == "__main__":
    print(deepeval_contextual_relevancy(
        "What gas do plants absorb during photosynthesis?",
        "Plants absorb carbon dioxide.",
        {"retrieval_context": [
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            "The stock market closed higher on Tuesday.",
        ]},
    ))
