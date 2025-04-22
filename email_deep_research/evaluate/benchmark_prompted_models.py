# To run:
# uv run scripts/benchmark_prompted_models.py

import art
import asyncio
import polars as pl
from dotenv import load_dotenv
from email_deep_research.data.local_email_db import generate_database
from email_deep_research.project_types import ProjectPolicyConfig
from email_deep_research.evaluate.benchmark import benchmark_model
import os

load_dotenv()
generate_database()

MODELS_TO_BENCHMARK = [
    ("gpt-4o", "openai/gpt-4o", False),
    ("gpt-4.1", "openai/gpt-4.1", False),
    ("o4-mini", "openai/o4-mini", True),
    ("o3", "openai/o3", True),
    ("gemini-2.0-flash", "gemini/gemini-2.0-flash", False),
    ("gemini-2.5-pro", "gemini/gemini-2.5-pro-preview-03-25", False),
]

TEST_SET_ENTRIES = 100


async def main():
    api = art.LocalAPI()
    models = []
    for model_name, model_id, use_tools in MODELS_TO_BENCHMARK:
        model = art.Model(
            name=model_name.split("/")[-1],
            project="email_agent",
            config=ProjectPolicyConfig(
                litellm_model_name=model_id, use_tools=use_tools
            ),
        )
        await model.register(api)
        models.append(model)

    results = await asyncio.gather(
        *[
            benchmark_model(model, TEST_SET_ENTRIES, swallow_exceptions=False)
            for model in models
        ]
    )
    for model in models:
        await api._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )

    df: pl.DataFrame = pl.concat(results)
    df = df.transpose(include_header=True)

    col_names = {"column": "metric"}
    for i, model in enumerate(MODELS_TO_BENCHMARK):
        col_names[f"column_{i}"] = model[0]

    df = df.rename(col_names)
    with open("data/benchmark_prompted_models.html", "w") as f:
        f.write(df.to_pandas().to_html())

    print(df.to_pandas().to_markdown())


asyncio.run(main())
