# from art.utils import iterate_dataset
from email_deep_research.types_enron import SyntheticQuery
from typing import List, Optional
from datasets import load_dataset, Dataset
import random

# Define the Hugging Face repository ID
HF_REPO_ID = "corbt/enron_emails_sample_questions"

# A few spot-checked synthetic queries that we found to be ambiguous or
# contradicted by other source emails. We'll exclude them for a more accurate
# model.
bad_queries = [49, 101, 129, 171, 208, 327]


def load_synthetic_queries(
    split: str = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
) -> List[SyntheticQuery]:
    dataset: Dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    dataset = dataset.filter(lambda x: x["id"] not in bad_queries)

    if shuffle:
        dataset = dataset.shuffle()

    # Convert each row (dict) in the dataset to a SyntheticQuery object
    # Apply the limit *after* conversion if specified
    queries = [SyntheticQuery(**row) for row in dataset]  # type: ignore

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if shuffle:
        random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    else:
        return queries
