import art
import asyncio
from dotenv import load_dotenv
from typing import List
from rollout import rollout
from query_iterators import load_synthetic_queries
from types_enron import SyntheticQuery
from local_email_db import generate_database
from art.utils import iterate_dataset
from email_deep_research.project_types import ProjectPolicyConfig, TrainingConfig
from email_deep_research.benchmark import benchmark_model

load_dotenv()

agent_002 = art.TrainableModel(
    name="email-agent-002",
    project="email_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    config=ProjectPolicyConfig(
        max_turns=10,
        log_to_openpipe=True,
        training_config=TrainingConfig(
            trajectories_per_group=6,
            groups_per_step=8,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=4000,
            num_epochs=1,
        ),
    ),
)

agent_004 = agent_002.model_copy(deep=True)
assert isinstance(agent_004.config, ProjectPolicyConfig)
agent_004.name = "email-agent-004"
agent_004.config.max_turns = 30

agent_005 = agent_002.model_copy(deep=True)
assert isinstance(agent_005.config, ProjectPolicyConfig)
agent_005.name = "email-agent-005"

agent_006 = agent_005.model_copy(deep=True)
agent_006.name = "email-agent-006"

agent_007 = agent_005.model_copy(deep=True)
agent_007.name = "email-agent-007"
assert isinstance(agent_007.config, ProjectPolicyConfig)
agent_007.config.use_tools = True

agent_008 = agent_005.model_copy(deep=True)
agent_008.name = "email-agent-008"
assert isinstance(agent_008.config, ProjectPolicyConfig)
assert agent_008.config.training_config is not None
agent_008.config.use_tools = True
agent_008.config.training_config.trajectories_per_group = 4
agent_008.config.training_config.groups_per_step = 12
agent_008.config.training_config.num_epochs = 2

agent_009 = agent_008.model_copy(deep=True)
agent_009.name = "email-agent-009"
assert isinstance(agent_009.config, ProjectPolicyConfig)
agent_009.base_model = "Qwen/Qwen2.5-32B-Instruct"


async def run_training(model: art.TrainableModel):
    generate_database()

    assert isinstance(model.config, ProjectPolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")
    api = art.LocalAPI()
    await model.register(api)
    print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await api._experimental_pull_from_s3(
        model,
        s3_bucket=os.environ["BACKUP_BUCKET"],
        verbose=True,
    )

    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=model.config.training_config.training_dataset_size
    )
    print("Loading validation data...")
    val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="test", limit=model.config.training_config.val_set_size
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=model.config.training_config.groups_per_step,
        num_epochs=model.config.training_config.num_epochs,
        initial_step=await model.get_step(),
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        if global_step % model.config.training_config.eval_steps == 0:
            print(f"\n--- Evaluating at Iteration {global_step} ---")
            await benchmark_model(model)
            await model.delete_checkpoints()
            await api._experimental_push_to_s3(
                model,
                s3_bucket=os.environ["BACKUP_BUCKET"],
            )

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(
                            model.config.training_config.trajectories_per_group
                        )
                    )
                )
                for scenario in batch
            )
        )

        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate
            ),
        )

    await benchmark_model(model)
    await api._experimental_push_to_s3(
        model,
        s3_bucket=os.environ["BACKUP_BUCKET"],
    )
    print("Training finished.")


if __name__ == "__main__":
    import os

    config = None
    training_config = os.environ.get("RUN_ID")
    if training_config == "002":
        config = agent_002
    elif training_config == "004":
        config = agent_004
    elif training_config == "005":
        config = agent_005
    elif training_config == "006":
        config = agent_006
    elif training_config == "007":
        config = agent_007
    elif training_config == "008":
        config = agent_008
    elif training_config == "009":
        config = agent_009
    else:
        raise ValueError(f"Invalid RUN_ID: {training_config}")

    asyncio.run(run_training(config))
