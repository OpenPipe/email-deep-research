import art
import asyncio
from dotenv import load_dotenv
from typing import List
from rollout import rollout
from query_iterators import load_synthetic_queries
from types_enron import SyntheticQuery
from local_email_db import generate_database
from art.utils import iterate_dataset

load_dotenv()

MODEL_NAME = "email-agent-002"
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LEARNING_RATE = 1.2e-5
EVAL_STEPS = 30
VAL_SET_SIZE = 100
TRAINING_DATASET_SIZE = 4000
BATCH_SIZE = 8
TRAJECTORIES_PER_GROUP = 6
PROJECT = "email_agent"
NUM_EPOCHS = 4


async def main():
    generate_database()

    api = art.LocalAPI()
    model = await api.get_or_create_model(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
    )

    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=TRAINING_DATASET_SIZE
    )
    print("Loading validation data...")
    val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="test", limit=VAL_SET_SIZE
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    train_iterator = iterate_dataset(
        train_scenarios, batch_size=4, initial_step=await model.get_step()
    )

    openai_client = await model.openai_client()
    api_key = openai_client.api_key
    base_url = str(openai_client.base_url)
    model_name = f"hosted_vllm/{MODEL_NAME}"
    print(f"API Key: {api_key}")
    print(f"Base URL: {base_url}")
    print(f"Model Name: {model_name}")

    for batch, epoch, global_step, epoch_step in train_iterator:
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(
                            model=model_name,
                            scenario=scenario,
                            base_url=base_url,
                            api_key=api_key,
                            trainable=True,
                        )
                        for _ in range(TRAJECTORIES_PER_GROUP)
                    )
                )
                for scenario in batch
            )
        )

        if val_scenarios and global_step > 0 and global_step % EVAL_STEPS == 0:
            print(f"\n--- Evaluating at Iteration {global_step} ---")
            print(f"Running validation rollouts on {len(val_scenarios)} samples...")

            val_trajectories = await art.gather_trajectories(
                (
                    rollout(
                        model=model_name,
                        scenario=item,
                        base_url=base_url,
                        api_key=api_key,
                        trainable=True,
                    )
                    for item in val_scenarios
                ),
                pbar_desc="Validation Rollout",
                max_exceptions=100,
            )

            valid_trajectories = [
                t for t in val_trajectories if isinstance(t, art.Trajectory)
            ]

            await model.log(valid_trajectories)
            await model.delete_checkpoints()

        await model.train(
            groups,
            config=art.TrainConfig(learning_rate=LEARNING_RATE),
        )

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(main())
