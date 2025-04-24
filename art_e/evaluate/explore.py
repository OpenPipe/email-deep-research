# %%
import polars as pl

from art_e.evaluate.load_trajectories import load_trajectories
from art_e.data.local_email_db import generate_database


df = await load_trajectories(".art/email_agent")  # type: ignore


# %%
from art_e.data.query_iterators import load_synthetic_queries

scenarios = load_synthetic_queries(
    split="test", limit=100, exclude_known_bad_queries=False
)
# %% Let's look at the ones that 008 is still getting wrong
from art_e.email_search_tools import read_email
from art_e.data.local_email_db import generate_database

generate_database()

failures = (
    df.filter(pl.col("model") == "email-agent-008")
    .filter(pl.col("metric_answer_correct") == 0)
    .filter(pl.col("split") == "val")
    .filter(pl.col("step") == 594)
)
print(len(failures))

for i, row in enumerate(failures.rows(named=True)):
    scenario_id = int(row["metadata_scenario_id"])

    scenario = next((s for s in scenarios if s.id == scenario_id), None)

    if scenario is None:
        print(f"Scenario {i} not found")
        break

    message = read_email(scenario.message_ids[0])
    assert message is not None

    print(f"=== Failure {i} ===")
    print(f"**Scenario ID**: {scenario.id}")
    print(f"**Question**: {scenario.question}")
    print(f"**Expected Answer**: {scenario.answer}")
    print(f"**Source email**: {message.subject} ({message.message_id})\n{message.body}")
    print()
    print("**Model's Response**")
    print(row)
    for message in row["messages"]:
        # print(message)
        print(f" --- {message['role']} --- ")
        print(message["content"] or message.get("tool_calls", {}))
    print()

# %%

max_scores = (
    df.filter(pl.col("model") == "email-agent-008")
    .filter(pl.col("split") == "val")
    .group_by("step")
    .agg(pl.col("metric_answer_correct").mean().alias("avg_score"))
    .sort("step")
)
print("Scores by step for agent 8:")
print(max_scores)


# %%

df.filter(pl.col("model") == "email-agent-008").filter(pl.col("split") == "val").filter(
    pl.col("step") > 300
).filter(pl.col("metric_answer_correct") == 0).group_by(
    "metadata_scenario_id"
).count().sort("count", descending=True)

# %%

df.filter(pl.col("step") == 0).filter(pl.col("split") == "val").filter(
    pl.col("metric_answer_correct") == 0
).group_by("metadata_scenario_id").count().sort("count", descending=True)

# %%

df.filter(pl.col("step") == 0).filter(pl.col("split") == "val").filter(
    pl.col("metric_answer_correct") == 0
).group_by("metadata_scenario_id").count().sort("count", descending=True)
