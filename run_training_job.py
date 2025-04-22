import argparse
import sys
from pathlib import Path
from typing import Dict

import sky

# Usage:
# uv run scripts/run_training_job.py 002 --fast


def load_env_file(env_path: str) -> Dict[str, str]:
    """Load a simple dotenv style file (KEY=VALUE per line)."""
    envs: Dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        print(f"Warning: env file {env_path} does not exist – continuing without it.")
        return envs

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue  # skip malformed lines
            key, val = line.split("=", 1)
            envs[key.strip()] = val.strip()
    return envs


def main():
    parser = argparse.ArgumentParser(
        description="Launch a SkyPilot training job for email agent using the Python SDK.)",
    )
    parser.add_argument(
        "run_id",
        help="The identifier for this training run (e.g., '002', '3').",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the --fast flag for sky.launch (skip provisioning if cluster is already up).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file (default: .env).",
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=60,
        help="Idle minutes before autostop (default: 60).",
    )
    parser.add_argument(
        "--accelerator",
        default="H100-SXM:1",
        help=(
            "Accelerator specification in the format '<TYPE>:<COUNT>' to pass to SkyPilot. "
            "Examples: 'A100-80GB:2', 'T4:1'. Default: 'H100-SXM:1'. If the count is omitted, "
            "it defaults to 1 (e.g., 'A100' == 'A100:1')."
        ),
    )

    args = parser.parse_args()

    # --- Construct Cluster Name and Format RUN_ID ---
    try:
        run_id_int = int(args.run_id)
        formatted_run_id = f"{run_id_int:03d}"
    except ValueError:
        print(
            f"Error: run_id '{args.run_id}' must be an integer or convertible to one.",
            file=sys.stderr,
        )
        sys.exit(1)

    cluster_name = f"kyle-email-agent-{args.run_id}"

    # --- Define Task Inline ---
    setup_script = """
    curl -LsSf https://astral.sh/uv/install.sh | sh

    source $HOME/.local/bin/env

    uv remove openpipe-art
    uv add --editable ~/ART
    uv add awscli
    uv sync
    """

    run_script = (
        'echo "Running training script..."\nuv run python email_deep_research/train.py'
    )

    # Base env skeleton matching the original YAML (values will be filled from env file)
    base_envs = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "WANDB_API_KEY": "",
        "OPENPIPE_API_KEY": "",
        "RUN_ID": "",
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_REGION": "",
        "BACKUP_BUCKET": "",
    }

    task = sky.Task(
        workdir=".",
        setup=setup_script,
        run=run_script,
        envs=base_envs,
    )

    # --- Configure Resources ---

    def _parse_accelerator(spec: str) -> Dict[str, int]:
        """Parse an accelerator spec of the form 'TYPE:COUNT' (COUNT optional)."""
        if ":" in spec:
            name, count_str = spec.split(":", 1)
            try:
                count = int(count_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid accelerator count in spec '{spec}'. Must be an int."
                ) from exc
        else:
            name, count = spec, 1
        return {name: count}

    acc_dict = _parse_accelerator(args.accelerator)

    task.set_resources(sky.Resources(accelerators=acc_dict))

    # File mounts
    task.set_file_mounts({"~/ART": "../ART"})

    # --- Prepare environment variables ---
    envs: Dict[str, str] = load_env_file(args.env_file)
    envs["RUN_ID"] = formatted_run_id
    task.update_envs(envs)

    # --- Attempt to cancel existing jobs on cluster (if any) ---
    try:
        pending_jobs = sky.queue(cluster_name)  # type: ignore[attr-defined]
        if pending_jobs:
            print(f"Found existing jobs on {cluster_name}, cancelling...")
            # Cancel all jobs for current user on this cluster
            sky.cancel(cluster_name, all=True)  # type: ignore[attr-defined]
    except Exception:
        # Either the cluster does not exist yet or queue/cancel failed; ignore.
        pass

    # --- Launch the task ---
    print(
        f"Launching cluster '{cluster_name}' with RUN_ID={formatted_run_id} (fast={args.fast})"
    )

    try:
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=args.idle_minutes,
            down=True,
            fast=args.fast,
        )

        # Stream logs until completion and get result (blocking).
        sky.stream_and_get(request_id)  # type: ignore[attr-defined]
        print("\nJob launched successfully.")
    except Exception as e:
        print(f"\nJob launch failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
