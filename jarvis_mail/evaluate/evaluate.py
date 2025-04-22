# %%

import polars as pl
import yaml
from pathlib import Path


def load_trajectories(project_path: str, debug: bool = False) -> pl.DataFrame:
    """
    Load and flatten trajectory YAML files into a Polars DataFrame.

    The expected on‑disk layout is::

        {project_path}/models/{model_name}/trajectories/{split}/{step_number}.yaml

    Each YAML file contains a list of *TrajectoryGroups* (see `art`), and each
    group in turn contains a list of *Trajectories*.  This helper walks the
    directory tree, reads every YAML file, and converts every single trajectory
    into one row of a tabular dataset.

    For every trajectory we record a handful of fixed columns plus two dynamic
    families of columns:

    Fixed columns
    -------------
    model : str
        Name of the model that produced the trajectory (taken from the folder
        name under ``models/``).
    step : int
        Training / evaluation step extracted from the YAML filename.
    reward : float | None
        Reward associated with the trajectory.
    group_number : int
        Running counter that uniquely identifies the surrounding trajectory
        group within the current parsing session (useful for debugging / joins).
    messages : list[str] | None
        Raw list of messages & choices for the dialogue.
    logs : list[str] | None
        Internal log lines captured during rollout.

    Dynamic columns
    ---------------
    metric_* : float
        One column for every distinct metric key found in the dataset.  Missing
        values are filled with nulls.
    metadata_* : str
        One column for every distinct metadata key (after merging group‑ and
        trajectory‑level metadata).  Values are coerced to strings so that the
        resulting table is rectangular.

    Parameters
    ----------
    project_path : str
        Path to the ART project on disk. Typically found in `.art/{project_name}`.
    debug : bool, optional
        If *True*, the function prints progress information while parsing.  The
        default is *False*.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing one row per trajectory with the schema
        described above.
    """
    rows: list[dict] = []
    metric_cols: set[str] = set()
    metadata_cols: set[str] = set()

    root = Path(project_path) / "models"
    group_number = 0

    # Walk through all models and their trajectory files
    for model_dir in root.iterdir():
        if debug:
            print(f"Processing {model_dir}")
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        traj_root = model_dir / "trajectories"
        if not traj_root.exists():
            continue

        # Iterate over splits (e.g. train/val)
        for split_dir in traj_root.iterdir():
            if not split_dir.is_dir():
                continue

            for yaml_path in split_dir.glob("*.yaml"):
                step = int(yaml_path.stem)
                if debug:
                    print(f"Processing {yaml_path}")

                # Each YAML file is a list of trajectory groups
                trajectory_groups = yaml.safe_load(yaml_path.read_text())

                for group in trajectory_groups:
                    group_number += 1
                    group_meta = group.get("metadata", {})
                    for traj in group.get("trajectories", []):
                        # Merge metadata downward (group metadata < traj metadata wins)
                        merged_meta = {**group_meta, **traj.get("metadata", {})}

                        metrics = traj.get("metrics", {})

                        prepped_metrics = {f"metric_{k}": v for k, v in metrics.items()}
                        prepped_metadata = {
                            f"metadata_{k}": str(v) for k, v in merged_meta.items()
                        }
                        metric_cols.update(prepped_metrics.keys())
                        metadata_cols.update(prepped_metadata.keys())
                        messages = []
                        for message in traj.get("messages_and_choices", []):
                            if "message" in message:
                                messages.append(
                                    {**message["message"], "trainable": True}
                                )
                            else:
                                messages.append({**message, "trainable": False})

                        row: dict[str, object] = {
                            "model": model_name,
                            "split": split_dir.name,
                            "step": step,
                            "reward": traj.get("reward"),
                            "group_number": group_number,
                            "messages": messages,
                            "logs": traj.get("logs"),
                            **prepped_metrics,
                            **prepped_metadata,
                        }

                        rows.append(row)

    schema = (
        {
            "model": pl.Utf8,
            "split": pl.Utf8,
            "step": pl.Int64,
            "reward": pl.Float64,
            "group_number": pl.Int64,
            "messages": pl.Object,
            "logs": pl.List(pl.Utf8),
        }
        | {k: pl.Float64 for k in metric_cols}
        | {k: pl.Utf8 for k in metadata_cols}
    )

    return pl.DataFrame(rows, schema=schema)


df = load_trajectories("../.art/email_agent", debug=False)
df.head()
# %%


def graph_metric(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    save_to: str,
    models: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
):
    """Plot and save a line chart of *metric_name* over training *step* for every model.

    Parameters
    ----------
    df : pl.DataFrame
        Table produced by :func:`load_trajectories`.
    split : str
        Which evaluation split to visualise (e.g. ``"train"`` / ``"val"``).
    save_to : str
        Path where the generated plot should be written to (usually a ``.png``
        or ``.pdf`` file).
    models : list[str] | None
        If provided, restrict the chart to the given subset of model names; the
        legend order will follow the order in this list.
    title : str | None, optional
        Custom figure title.  If *None* (default) a sensible title will be
        generated automatically.
    x_label : str | None, optional
        Custom label for the *x*-axis.  If *None* (default) the label will be
        set to ``"Step"`` (the previous hard‑coded behaviour).  Pass an empty
        string (``""``) to remove the label entirely.
    y_label : str | None, optional
        Custom label for the *y*-axis.  If *None* (default) the *y*-axis will
        remain unlabeled (replicating the previous behaviour).  Pass an empty
        string (``""``) to explicitly clear the label.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*‑value and
        annotates it with the text "Perfect score" on the left‑hand side of
        the plot.  The reference line is excluded from the legend.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure so that it can be displayed inline in IPython /
        Jupyter notebooks.
    """
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.dpi"] = 300

    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )

    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    df = df.filter(pl.col("split") == split)

    # ------------------------------------------------------------------
    # Determine plotting / legend order
    # ------------------------------------------------------------------
    if models is not None:
        # Keep the explicit order provided by the caller; also acts as a filter
        plot_models = [m for m in models if m in df["model"].unique().to_list()]
        df = df.filter(pl.col("model").is_in(plot_models))
    else:
        # Separate trained models (multiple distinct steps) from comparison
        # models (exactly one step) and list trained models first.
        step_counts = (
            df.group_by("model")
            .agg(pl.col("step").n_unique().alias("n_steps"))
            .to_dict(as_series=False)
        )

        trained_models: list[str] = []
        comparison_models: list[str] = []
        for model_name, n_steps in zip(step_counts["model"], step_counts["n_steps"]):
            if n_steps > 1:
                trained_models.append(model_name)
            else:
                comparison_models.append(model_name)

        # Sort alphabetically within each group to keep things predictable
        plot_models = sorted(trained_models) + sorted(comparison_models)

    # Guard against empty selection
    if df.is_empty():
        raise ValueError(
            "No rows left after applying the split filter. Check your 'split' argument or the DataFrame contents."
        )

    agg_df = (
        df.group_by(["model", "step"])
        .agg(pl.col(full_metric_col).mean().alias(full_metric_col))
        .sort("step")
    )

    # Use a clean theme without background grid lines
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Explicitly remove grid lines that might be enabled by default
    ax.grid(False)

    # Determine global x‑range so that we can draw horizontal reference lines
    all_steps = agg_df["step"].unique().to_list()
    if not all_steps:
        raise ValueError(
            "Unable to determine the range of training steps – no data available."
        )
    min_step, max_step = min(all_steps), max(all_steps)

    # ------------------------------------------------------------------
    # Optional: draw a reference line indicating a perfect score
    # ------------------------------------------------------------------
    if perfect_score is not None:
        # Draw the line *before* plotting the model curves so that it sits
        # behind them; exclude it from the legend by using a special label.
        ax.plot(
            [min_step, max_step],
            [perfect_score, perfect_score],
            linestyle="-",
            linewidth=1,
            color="dimgray",
            label="_nolegend_",
        )

        # Place the annotation slightly below the line to avoid overlap.
        # Calculate a small offset based on the y-axis limits
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.1  # Adjust multiplier as needed
        ax.text(
            min_step,
            perfect_score - offset,  # Adjusted y-position
            "Perfect score",
            verticalalignment="top",
            horizontalalignment="left",
            # fontsize="small", # Removed to use default size
            color="dimgray",
        )

    # ------------------------------------------------------------------
    # Draw trained models first so that prompted comparison models (usually
    # baselines) are rendered *on top* and therefore remain visible even if
    # lines overlap.
    # ------------------------------------------------------------------

    # Identify models with a single evaluation step (= prompted / comparison)
    step_counts = (
        df.group_by("model")
        .agg(pl.col("step").n_unique().alias("n_steps"))
        .to_dict(as_series=False)
    )
    step_count_map = dict(zip(step_counts["model"], step_counts["n_steps"]))

    trained_first = [m for m in plot_models if step_count_map.get(m, 2) > 1]
    comparison_last = [m for m in plot_models if step_count_map.get(m, 1) == 1]

    # ------------------------------------------------------------------
    # Colour assignment
    # ------------------------------------------------------------------
    # We want the most distinctive colours to belong to the trained models even
    # though they are drawn *after* the comparison models.  Therefore we
    # construct a palette and assign colours explicitly: first to the trained
    # models, then to the comparison / baseline models.

    ordered_for_palette = trained_first + comparison_last
    palette = sns.color_palette(n_colors=len(ordered_for_palette))
    model_colors = {m: c for m, c in zip(ordered_for_palette, palette)}  # type: ignore

    for model in comparison_last + trained_first:
        model_df = agg_df.filter(pl.col("model") == model).sort("step")
        x = model_df["step"].to_list()
        y = model_df[full_metric_col].to_list()

        if len(x) == 1:
            # Prompted comparison model – draw a dashed horizontal line across the full range
            ax.plot(
                [min_step, max_step],
                [y[0], y[0]],
                linestyle="--",
                linewidth=2,
                color=model_colors[model],
                label=model,
            )
        else:
            # Trained model
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=model_colors[model],
                label=model,
            )

    if title is None:
        title = f"{split}/{metric_name}"
    ax.set_title(title)

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------

    # X‑axis label
    if x_label is None:
        ax.set_xlabel("Step")
    else:
        ax.set_xlabel(x_label)

    # Y‑axis label
    if y_label is not None:
        ax.set_ylabel(y_label)

    # ------------------------------------------------------------------
    # Assemble the legend: trained models first, comparison models afterwards
    # so that the legend mirrors the visual layering (trained on top).
    # ------------------------------------------------------------------

    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lbl: h for lbl, h in zip(labels, handles)}
    ordered_labels = trained_first + comparison_last
    ordered_handles = [handle_map[lbl] for lbl in ordered_labels]

    ax.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        frameon=False,
        loc="lower right",
        borderaxespad=0.0,
    )

    # Remove the top and right spines for a cleaner presentation
    sns.despine()

    # ------------------------------------------------------------------
    # Output handling (save + return)
    # ------------------------------------------------------------------
    save_path = Path(save_to)
    if save_path.suffix == "":
        # Default to PNG if no extension given
        save_path = save_path.with_suffix(".png")
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Set higher resolution for matplotlib display in notebook

fig = graph_metric(
    df.filter(pl.col("step").ne(592)).with_columns(
        pl.col("model").str.replace("email-agent-008", "qwen-2.5-14b (trained)")
    ),
    "val",
    "answer_correct",
    "plots/accuracy_train.png",
    models=[
        "qwen-2.5-14b (trained)",
        "o4-mini",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gpt-4o",
    ],
    title="Email Agent Success Rate",
    perfect_score=1.0,
)

# %%
from jarvis_mail.data.query_iterators import load_synthetic_queries

scenarios = load_synthetic_queries(
    split="test", limit=100, exclude_known_bad_queries=False
)
# %% Let's look at the ones that 008 is still getting wrong
import importlib
import jarvis_mail.email_search_tools

jarvis_mail.email_search_tools = importlib.reload(jarvis_mail.email_search_tools)

from jarvis_mail.email_search_tools import read_email

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

df.filter(pl.col("model") == "email-agent-008").filter(
    pl.col("split") == "val"
).group_by("step").count().sort("step")


def graph_metric_bar(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    save_to: str,
    models: list[str] | None = None,
    title: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
):
    """Create a bar chart visualising *metric_name* for all *models* at the
    **first** and **last** available training step.

    For *trained* models (those that have been evaluated across multiple training
    steps) the bar is *stacked*: the base segment represents the score at the
    **first** step while the hatched segment on top encodes the improvement (or
    decline) by the **last** step.  Prompted comparison models that were only
    evaluated once are rendered as a single, solid bar.

    Parameters
    ----------
    df : pl.DataFrame
        Table produced by :func:`load_trajectories`.
    split : str
        Which evaluation split to visualise (e.g. ``"train"`` / ``"val"``).
    metric_name : str
        Name of the metric (without the ``"metric_"`` prefix) or the full
        column name as it appears in *df*.
    save_to : str
        Output path for the figure.  The file type is inferred from the suffix
        and defaults to PNG if none is given.
    models : list[str] | None, optional
        Optional explicit ordering / subset of model names.  Works the same way
        as in :func:`graph_metric`.
    title : str | None, optional
        Custom figure title.  If *None*, a sensible default is generated.
    y_label : str | None, optional
        Label for the *y*‑axis.  If *None* (default) the axis remains
        unlabeled.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*‑value and
        annotates it with the text "Perfect score".

    Returns
    -------
    matplotlib.figure.Figure
        The created figure so that it can be displayed inline in IPython /
        Jupyter notebooks.
    """
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.rcParams["figure.dpi"] = 300

    # Resolve column name (support both raw + prefixed)
    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )
    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Filter + determine model ordering (same logic as *graph_metric*)
    # ------------------------------------------------------------------
    df = df.filter(pl.col("split") == split)

    if models is not None:
        plot_models = [m for m in models if m in df["model"].unique().to_list()]
        df = df.filter(pl.col("model").is_in(plot_models))
    else:
        step_counts = (
            df.group_by("model")
            .agg(pl.col("step").n_unique().alias("n_steps"))
            .to_dict(as_series=False)
        )
        trained_models, comparison_models = [], []
        for mdl, n in zip(step_counts["model"], step_counts["n_steps"]):
            (trained_models if n > 1 else comparison_models).append(mdl)
        plot_models = sorted(trained_models) + sorted(comparison_models)

    if df.is_empty():
        raise ValueError("No rows left after filtering – check 'split' or model list.")

    # ------------------------------------------------------------------
    # Aggregate first / last metrics per model
    # ------------------------------------------------------------------
    agg = (df.group_by(["model", "step"]).agg(pl.col(full_metric_col).mean())).sort(
        ["model", "step"]
    )

    model_stats: list[tuple[str, float, float]] = []  # (model, base, improvement)
    for model in plot_models:
        sub = agg.filter(pl.col("model") == model)
        metric_vals = sub[full_metric_col].to_list()
        if len(metric_vals) == 0:
            continue
        base_score = float(metric_vals[0])
        final_score = float(metric_vals[-1])
        improvement = final_score - base_score if len(metric_vals) > 1 else 0.0
        model_stats.append((str(model), base_score, improvement))

    if not model_stats:
        raise ValueError("No data available for the requested models.")

    # Convert to percentages if the data appears to be in [0, 1] range
    max_val = max(base + imp for _, base, imp in model_stats)
    scale = 100.0 if max_val <= 1.01 else 1.0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)

    trained_set = {m for m, b, imp in model_stats if imp != 0}

    # ------------------------------------------------------------------
    # Colour scheme – match the example shared by the caller
    # ------------------------------------------------------------------
    ORANGE = "#e67a30"
    GREY = "#e0dcd5"

    # Colour assignment: use same approach (trained first) so the most
    # distinctive colours are given to the trained models.
    ordered_for_palette = [m for m in plot_models if m in trained_set] + [
        m for m in plot_models if m not in trained_set
    ]
    palette = sns.color_palette(n_colors=len(ordered_for_palette))
    model_colors = {m: c for m, c in zip(ordered_for_palette, palette)}  # type: ignore

    bar_width = 0.7
    x_positions = np.arange(len(model_stats))

    for idx, (model, base, imp) in enumerate(model_stats):
        base_scaled = base * scale
        imp_scaled = imp * scale
        color = ORANGE if imp_scaled else GREY

        # Base segment (always)
        ax.bar(
            x_positions[idx],
            base_scaled,
            width=bar_width,
            color=color if imp_scaled else "#e0dcd5",
            edgecolor="white",
        )

        # Improvement segment (only for trained models)
        if imp_scaled != 0:
            ax.bar(
                x_positions[idx],
                imp_scaled,
                width=bar_width,
                bottom=base_scaled,
                color=color,
                hatch="///",
                edgecolor="white",
                alpha=0.95,
            )
            # Annotate improvement in the middle of the improvement bar
            ax.text(
                x_positions[idx],
                base_scaled + imp_scaled / 2,
                f"RL +{imp_scaled:.0f}{'%' if scale == 100 else ''}",
                ha="center",
                va="center",
                color="#c05a20",  # slightly darker orange for contrast
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    boxstyle="round,pad=0.3",
                    alpha=0.95,
                    edgecolor="#e67a30",
                    linewidth=1,
                ),
            )

        # Total label on top
        total = base_scaled + imp_scaled
        ax.text(
            x_positions[idx],
            total + (max_val * scale) * 0.02,
            f"{total:.0f}{'%' if scale == 100 else ''}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # ------------------------------------------------------------------
    # Optional perfect score reference line
    # ------------------------------------------------------------------
    if perfect_score is not None:
        score_scaled = perfect_score * scale
        ax.plot(
            [-0.5, len(model_stats) - 0.5],
            [score_scaled, score_scaled],
            linestyle="-",
            linewidth=1,
            color="dimgray",
            label="_nolegend_",
        )
        ax.text(
            -0.4,
            score_scaled - (max_val * scale) * 0.05,
            "Perfect score",
            va="top",
            ha="left",
            color="dimgray",
        )

    # ------------------------------------------------------------------
    # Axes & legend
    # ------------------------------------------------------------------
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m for m, _, _ in model_stats], rotation=15, ha="right")
    # Remove y‑axis label, ticks, and spine as requested
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    if title is None:
        title = f"{split}/{metric_name}"
    ax.set_title(title)

    # Clean up remaining spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save + return
    # ------------------------------------------------------------------
    save_path = Path(save_to)
    if save_path.suffix == "":
        save_path = save_path.with_suffix(".png")
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


graph_metric_bar(
    df.with_columns(pl.col("model").str.replace("email-agent-008", "qwen-2.5-14b")),
    "val",
    "answer_correct",
    "plots/accuracy_train.png",
    models=[
        "gpt-4o",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "o4-mini",
        "qwen-2.5-14b",
    ],
    title="Email Agent Success Rate",
    y_label="Val set success rate",
)

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
