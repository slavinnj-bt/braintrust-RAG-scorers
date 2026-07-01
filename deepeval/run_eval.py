#!/usr/bin/env python3
"""Run a basic RAG dataset against ALL five DeepEval-library scorers via Eval().

Companion to the standalone scorer files in this folder. Those files implement +
push the scorers (for the Playground / online scoring); THIS script runs them
locally over a small dataset so you get an experiment with every metric scored,
without clicking through the UI.

Each scorer imports the real `deepeval` library, so run this with a Python env
that has deepeval installed (e.g. the .venv-deepeval used to push):

    BRAINTRUST_API_KEY=<org key or bt OAuth token> \
    BRAINTRUST_DEFAULT_PROJECT=RAG-Scorers-Demo \
    BRAINTRUST_ORG_NAME="My Org" \
    /path/to/.venv-deepeval/bin/python deepeval/run_eval.py

The LLM judges call the Braintrust AI proxy (model DEEPEVAL_SCORER_MODEL,
default claude-sonnet-4-6); ExactMatch is deterministic (no LLM).
"""

import os
import sys

import braintrust

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_relevancy import deepeval_answer_relevancy
from contextual_relevancy import deepeval_contextual_relevancy
from exact_match import deepeval_exact_match
from faithfulness import deepeval_faithfulness
from g_eval_correctness import deepeval_geval_correctness
from rag_dataset import RAG_ROWS

ORG_NAME = os.environ.get("BRAINTRUST_ORG_NAME", "My Org")
PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
EXPERIMENT_NAME = "DeepEval lib scorers demo"

# The dataset lives in rag_dataset.py (RAG_ROWS) so it can be uploaded to Braintrust
# independently for the Playground:  python rag_dataset.py
# Here the task just echoes each row's pre-baked `output` so every metric scores.
_OUTPUT_BY_INPUT = {r["input"]: r["output"] for r in RAG_ROWS}


def task(input):
    return _OUTPUT_BY_INPUT.get(input, "")


# Adapters: Braintrust passes (input, output, expected, metadata); forward what
# each DeepEval scorer needs (RAG context lives in metadata.retrieval_context).
def ExactMatch(output, expected, **_):
    return deepeval_exact_match(output, expected)


def AnswerRelevancy(input, output, **_):
    return deepeval_answer_relevancy(input, output)


def Faithfulness(input, output, metadata, **_):
    return deepeval_faithfulness(input, output, metadata)


def ContextualRelevancy(input, output, metadata, **_):
    return deepeval_contextual_relevancy(input, output, metadata)


def GEvalCorrectness(input, output, expected, **_):
    return deepeval_geval_correctness(input, output, expected)


def data():
    return [
        {"input": r["input"], "expected": r["expected"], "metadata": r["metadata"]}
        for r in RAG_ROWS
    ]


if __name__ == "__main__":
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("ERROR: set BRAINTRUST_API_KEY (org key or bt OAuth token).")
        sys.exit(1)
    braintrust.login(org_name=ORG_NAME)

braintrust.Eval(
    PROJECT_NAME,
    data=data,
    task=task,
    scores=[ExactMatch, AnswerRelevancy, Faithfulness, ContextualRelevancy, GEvalCorrectness],
    experiment_name=EXPERIMENT_NAME,
    metadata={"framework": "deepeval", "scorers": "deepeval-lib (real library)"},
)
