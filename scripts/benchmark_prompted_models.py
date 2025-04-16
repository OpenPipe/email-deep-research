# To run:
# uv run scripts/benchmark_prompted_models.py

import asyncio
import polars as pl
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from email_deep_research.query_iterators import load_synthetic_queries
from email_deep_research.rollout import rollout
from email_deep_research.local_email_db import generate_database

load_dotenv()
generate_database()

MODELS_TO_BENCHMARK = [
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/o3-mini",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.5-pro-preview-03-25",
]

TEST_SET_ENTRIES = 100


async def benchmark_model(model: str, limit: int = 5) -> pl.DataFrame:
    """Benchmark a model on the test dataset"""
    scenarios = load_synthetic_queries(split="test", limit=limit)
    trajectories = await tqdm.gather(
        *[
            rollout(model, scenario, trainable=False, log_to_openpipe=False)
            for scenario in scenarios
        ],
        desc=f"Benchmarking {model}",
    )

    metrics = pl.DataFrame([{**t.metrics, "reward": t.reward} for t in trajectories])

    avg_metrics = metrics.select([pl.mean(c).alias(c) for c in metrics.columns])

    return avg_metrics


async def main():
    results = await asyncio.gather(
        *[benchmark_model(model, TEST_SET_ENTRIES) for model in MODELS_TO_BENCHMARK]
    )

    df: pl.DataFrame = pl.concat(results)
    df = df.transpose(include_header=True)

    col_names = {"column": "metric"}
    for i, model in enumerate(MODELS_TO_BENCHMARK):
        col_names[f"column_{i}"] = model

    df = df.rename(col_names)
    with open("data/benchmark_prompted_models.html", "w") as f:
        f.write(df.to_pandas().to_html())

    print(df.to_pandas().to_markdown())


asyncio.run(main())
