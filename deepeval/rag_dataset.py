#!/usr/bin/env python3
"""A small RAG demo dataset for the DeepEval-style scorers.

Each row carries every field the scorers need:
  - input    : the user question
  - expected : the ideal (ground-truth) answer
  - output   : a simulated RAG system answer (so scores show a real spread)
  - metadata.retrieval_context : the chunks the retriever returned  (RAG metrics)
  - metadata.context           : the ground-truth context           (Hallucination)
  - metadata.document          : a longer source text               (Summarization)

The rows are intentionally mixed: some answers are faithful and on-topic, some
hallucinate, and some retrievals contain noise or miss information — so each
metric produces a meaningful, non-trivial score.

`upload(project)` pushes the rows into a Braintrust dataset; `RAG_ROWS` is also
importable directly for a local Eval() run.
"""

RAG_ROWS = [
    {
        "input": "What is the capital of France and what famous tower is located there?",
        "expected": "Paris is the capital of France, and the Eiffel Tower is located there.",
        "output": "The capital of France is Paris, home to the Eiffel Tower.",
        "metadata": {
            "retrieval_context": [
                "Paris is the capital and most populous city of France.",
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            ],
            "context": [
                "Paris is the capital of France.",
                "The Eiffel Tower is located in Paris.",
            ],
            "document": (
                "Paris is the capital and most populous city of France. Situated on the Seine "
                "River, it is a major center for art, fashion, and culture. Among its landmarks "
                "is the Eiffel Tower, a wrought-iron lattice tower built in 1889 on the Champ de Mars."
            ),
            "case": "faithful_on_topic",
        },
    },
    {
        "input": "Who wrote the play Romeo and Juliet?",
        "expected": "William Shakespeare wrote Romeo and Juliet.",
        "output": "Romeo and Juliet was written by William Shakespeare in the late 16th century.",
        "metadata": {
            "retrieval_context": [
                "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                "The play was likely written between 1591 and 1595.",
            ],
            "context": [
                "Romeo and Juliet was written by William Shakespeare.",
            ],
            "document": (
                "Romeo and Juliet is a tragedy written by the English playwright William "
                "Shakespeare early in his career, likely between 1591 and 1595. It is among his "
                "most frequently performed plays and tells the story of two young star-crossed lovers."
            ),
            "case": "faithful_on_topic",
        },
    },
    {
        "input": "How tall is Mount Everest?",
        "expected": "Mount Everest is about 8,849 meters (29,032 feet) tall.",
        "output": "Mount Everest is approximately 9,500 meters tall, making it the tallest mountain in the solar system.",
        "metadata": {
            "retrieval_context": [
                "Mount Everest's peak is 8,849 metres (29,032 ft) above sea level.",
                "It is Earth's highest mountain above sea level, located in the Himalayas.",
            ],
            "context": [
                "Mount Everest is 8,849 meters (29,032 feet) tall.",
                "Mount Everest is the highest mountain on Earth.",
            ],
            "document": (
                "Mount Everest, located in the Mahalangur Himal sub-range of the Himalayas, is "
                "Earth's highest mountain above sea level. Its peak rises to 8,849 metres "
                "(29,032 ft). It straddles the border between Nepal and the Tibet Autonomous Region of China."
            ),
            "case": "hallucinated_answer",
        },
    },
    {
        "input": "What gas do plants absorb during photosynthesis?",
        "expected": "Plants absorb carbon dioxide during photosynthesis.",
        "output": "During photosynthesis, plants absorb carbon dioxide from the air.",
        "metadata": {
            "retrieval_context": [
                "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
                "The stock market closed higher on Tuesday amid strong earnings.",
            ],
            "context": [
                "Plants absorb carbon dioxide during photosynthesis.",
            ],
            "document": (
                "Photosynthesis is the process by which green plants and some other organisms use "
                "sunlight to synthesize foods from carbon dioxide and water. It generally involves "
                "the green pigment chlorophyll and generates oxygen as a byproduct."
            ),
            "case": "noisy_retrieval",
        },
    },
    {
        "input": "What is the boiling point of water at sea level?",
        "expected": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "output": "At sea level, water boils at 100 degrees Celsius.",
        "metadata": {
            "retrieval_context": [
                "Atmospheric pressure at sea level is about 101.3 kilopascals.",
                "Water is composed of two hydrogen atoms and one oxygen atom.",
            ],
            "context": [
                "Water boils at 100 degrees Celsius at sea level.",
            ],
            "document": (
                "At standard atmospheric pressure (sea level), pure water boils at 100 degrees "
                "Celsius, equivalent to 212 degrees Fahrenheit. The boiling point decreases at "
                "higher altitudes where atmospheric pressure is lower."
            ),
            "case": "missing_retrieval",
        },
    },
]


def upload(project_name="RAG-Scorers-Demo", dataset_name="DeepEval RAG demo dataset"):
    """Create/replace the dataset in Braintrust and return its record ids."""
    import os

    import braintrust

    org = os.environ.get("BRAINTRUST_ORG_NAME", "My Org")
    braintrust.login(org_name=org)
    ds = braintrust.init_dataset(project=project_name, name=dataset_name)
    existing = list(ds.fetch())
    for rec in existing:
        ds.delete(rec["id"])
    if existing:
        ds.flush()
    ids = []
    for row in RAG_ROWS:
        ids.append(ds.insert(
            input=row["input"],
            expected=row["expected"],
            metadata={**row["metadata"], "reference_output": row["output"]},
            tags=[row["metadata"].get("case", "demo")],
        ))
    ds.flush()
    print(f"Uploaded {len(RAG_ROWS)} rows to '{dataset_name}' (cleared {len(existing)} old rows).")
    return ids


if __name__ == "__main__":
    upload()
