# To run:
# uv run scripts/benchmark_prompted_models.py

import art
import asyncio
import polars as pl
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from email_deep_research.query_iterators import load_synthetic_queries
from email_deep_research.rollout import rollout
from email_deep_research.local_email_db import generate_database
from email_deep_research.project_types import ProjectPolicyConfig
from email_deep_research.benchmark import benchmark_model
import os

load_dotenv()
generate_database()

MODELS_TO_BENCHMARK = [
    # "openai/gpt-4o",
    "openai/gpt-4.1",
    # "openai/o4-mini",
    # "openai/o3",
    # "gemini/gemini-2.0-flash",
    # "gemini/gemini-2.5-pro-preview-03-25",
]

TEST_SET_ENTRIES = 100


async def main():
    api = art.LocalAPI()
    models = [
        art.Model(
            name=model_name,
            project="email-deep-research",
            config=ProjectPolicyConfig(litellm_model_name=model_name, use_tools=True),
        )
        for model_name in MODELS_TO_BENCHMARK
    ]
    for model in models:
        await model.register(api)
    results = await asyncio.gather(
        *[benchmark_model(model, TEST_SET_ENTRIES) for model in models]
    )
    for model in models:
        await model.push_to_s3(
            os.environ["BACKUP_BUCKET"],
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
