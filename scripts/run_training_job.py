import argparse
import subprocess
import sys

# Usage:
# uv run scripts/run_training_job.py 002 --fast


def run_command(command, shell=False):
    """Helper function to run a shell command and print output."""
    print(
        f"Running command: {' '.join(command) if isinstance(command, list) else command}"
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=shell,  # Use shell=True if command is a string
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
        # Stream output
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(
                f"Command failed with exit code {process.returncode}", file=sys.stderr
            )
            # Optionally exit here if needed: sys.exit(process.returncode)
        return process.returncode
    except FileNotFoundError:
        print(
            f"Error: Command not found: {command[0] if isinstance(command, list) else command.split()[0]}",
            file=sys.stderr,
        )
        print(
            "Please ensure 'sky' and 'uv' are installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch a SkyPilot training job for email agent."
    )
    parser.add_argument(
        "run_id", help="The identifier for this training run (e.g., '002', '3')."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Include the --fast flag for sky launch."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file (default: .env).",
    )
    parser.add_argument(
        "--skypilot-yaml",
        default="skypilot.yaml",
        help="Path to the SkyPilot YAML file (default: skypilot.yaml).",
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=60,
        help="Idle minutes before autostop (default: 60).",
    )

    args = parser.parse_args()

    # --- Construct Cluster Name and Format Run ID ---
    try:
        # Ensure run_id can be treated as an integer for formatting, but keep original for cluster name if needed
        run_id_int = int(args.run_id)
        formatted_run_id = f"{run_id_int:03d}"  # Pad with leading zeros to 3 digits
    except ValueError:
        print(
            f"Error: run_id '{args.run_id}' must be an integer or convertible to one.",
            file=sys.stderr,
        )
        sys.exit(1)

    cluster_name = f"kyle-email-agent-{args.run_id}"  # Use original run_id for name consistency if desired

    # --- Attempt to Cancel Existing Cluster ---
    print(f"Attempting to cancel existing cluster: {cluster_name}")
    cancel_command = ["sky", "cancel", cluster_name, "--yes"]
    # We don't exit if cancel fails, as the cluster might not exist
    run_command(cancel_command)

    # --- Construct Launch Command ---
    print(
        f"Constructing launch command for cluster: {cluster_name} with RUN_ID={formatted_run_id}"
    )
    launch_command_list = [
        "uv",
        "run",
        "sky",
        "launch",
        args.skypilot_yaml,
        "--env-file",
        args.env_file,
        "--yes",
        "--retry-until-up",
        "--down",
        f"--idle-minutes-to-autostop={args.idle_minutes}",
        f"--cluster={cluster_name}",
        f"--env",
        f"RUN_ID={formatted_run_id}",
    ]

    if args.fast:
        launch_command_list.append("--fast")

    # --- Execute Launch Command ---
    print(f"\nLaunching new cluster: {cluster_name}")
    launch_return_code = run_command(launch_command_list)

    if launch_return_code == 0:
        print("\nJob launched successfully.")
    else:
        print("\nJob launch failed.")
        sys.exit(launch_return_code)


if __name__ == "__main__":
    main()
