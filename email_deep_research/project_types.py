from pydantic import BaseModel


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    batch_size: int = 8
    num_epochs: int = 4


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    reward_extra_turns: bool = True
    log_to_openpipe: bool = False
    litellm_model_name: str | None = None
    use_tools: bool = False

    training_config: TrainingConfig | None = None
