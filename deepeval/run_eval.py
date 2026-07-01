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

ORG_NAME = os.environ.get("BRAINTRUST_ORG_NAME", "My Org")
PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
EXPERIMENT_NAME = "DeepEval lib scorers demo"

# A small RAG dataset: faithful, hallucinated, and noisy-retrieval cases so every
# metric shows a real spread. `output` is a pre-baked answer (the task just echoes
# it) — swap in your own generator/prompt as needed.
ROWS = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "expected": "Paris",
        "retrieval_context": ["Paris is the capital and most populous city of France."],
    },
    {
        "input": "How tall is Mount Everest?",
        "output": "Mount Everest is about 9,500 meters tall.",
        "expected": "About 8,849 meters (29,032 ft).",
        "retrieval_context": ["Mount Everest's peak is 8,849 metres (29,032 ft) above sea level."],
    },
    {
        "input": "What gas do plants absorb during photosynthesis?",
        "output": "Plants absorb carbon dioxide.",
        "expected": "Carbon dioxide",
        "retrieval_context": [
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            "The stock market closed higher on Tuesday.",
        ],
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "output": "Romeo and Juliet was written by William Shakespeare.",
        "expected": "William Shakespeare",
        "retrieval_context": ["Romeo and Juliet is a tragedy written by William Shakespeare."],
    },
]

_OUTPUT_BY_INPUT = {r["input"]: r["output"] for r in ROWS}


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
        {
            "input": r["input"],
            "expected": r["expected"],
            "metadata": {"retrieval_context": r["retrieval_context"]},
        }
        for r in ROWS
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
