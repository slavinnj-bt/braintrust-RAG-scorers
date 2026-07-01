#!/usr/bin/env python3
"""Sample Braintrust Eval() that runs ALL DeepEval-style scorers over a RAG dataset.

Demonstrates the 8 scorers (6 RAG-oriented + ExactMatch + Summarization) against
the small mixed RAG dataset in ``rag_dataset.py``. Each row already carries a
simulated RAG answer (``output``) plus the retrieved chunks, ground-truth
context, and a source document, so every metric produces a real, non-trivial
score. Running this creates an experiment in the RAG-Scorers-Demo project.

Setup:
    pip install braintrust -r requirements.txt
    export BRAINTRUST_API_KEY=<org key or bt OAuth token>

Run:
    python run_eval.py
    # or with the Braintrust runner:
    bt eval run_eval.py

The scorers call the Braintrust AI proxy for their LLM steps; set
DEEPEVAL_SCORER_MODEL to change the judge model (default claude-sonnet-4-6).
"""

import os
import sys

import braintrust

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_relevancy import answer_relevancy
from contextual_precision import contextual_precision
from contextual_recall import contextual_recall
from contextual_relevancy import contextual_relevancy
from exact_match import exact_match
from faithfulness import faithfulness
from hallucination import hallucination
from rag_dataset import RAG_ROWS
from summarization import summarization

ORG_NAME = os.environ.get("BRAINTRUST_ORG_NAME", "My Org")
PROJECT_NAME = os.environ.get("BRAINTRUST_DEFAULT_PROJECT", "RAG-Scorers-Demo")
EXPERIMENT_NAME = "DeepEval scorers demo"

# Map each input to its simulated RAG answer so the task is deterministic.
_OUTPUT_BY_INPUT = {row["input"]: row["output"] for row in RAG_ROWS}


def task(input):
    return _OUTPUT_BY_INPUT.get(input, "")


# --- Scorer adapters: map Braintrust's (input, output, expected, metadata) -----
# to each DeepEval scorer's signature. RAG metrics read context from metadata.
def AnswerRelevancy(input, output, **_):
    return answer_relevancy(input, output)


def Faithfulness(output, metadata, **_):
    return faithfulness(output, metadata)


def ContextualPrecision(input, expected, metadata, **_):
    return contextual_precision(input, expected, metadata)


def ContextualRecall(expected, metadata, **_):
    return contextual_recall(expected, metadata)


def ContextualRelevancy(input, metadata, **_):
    return contextual_relevancy(input, metadata)


def Hallucination(output, metadata, **_):
    return hallucination(output, metadata)


def ExactMatch(output, expected, **_):
    return exact_match(output, expected)


def Summarization(output, metadata, **_):
    # Treat the row's source document as the original text being summarized.
    md = metadata or {}
    original = md.get("document") or "\n".join(md.get("retrieval_context", []))
    return summarization(original, output)


def data():
    return [
        {"input": row["input"], "expected": row["expected"], "metadata": row["metadata"]}
        for row in RAG_ROWS
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
    scores=[
        AnswerRelevancy,
        Faithfulness,
        ContextualPrecision,
        ContextualRecall,
        ContextualRelevancy,
        Hallucination,
        ExactMatch,
        Summarization,
    ],
    experiment_name=EXPERIMENT_NAME,
    metadata={"framework": "deepeval", "scorers": "rag+llm suite"},
)
