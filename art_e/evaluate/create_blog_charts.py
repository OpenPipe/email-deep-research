# %%

import importlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import art_e.evaluate.charts

importlib.reload(art_e.evaluate.charts)

import polars as pl
from art_e.evaluate.load_trajectories import load_trajectories
from art_e.evaluate.charts import comparison_models_bar_chart, training_progress_chart
from art_e.evaluate.save_figure_for_blog import save_figure_for_blog

df = await load_trajectories(".art/email_agent")  # type: ignore

# Create the first chart (bar chart)
fig1 = comparison_models_bar_chart(
    df,
    split="val",
    metric_name="answer_correct",
    models=[
        ("gpt-4.1", "GPT-4.1"),
        ("gemini-2.5-pro", "Gemini\n2.5 Pro"),
        # ("gemini-2.0-flash", "Gemini\n2.0 Flash"),
        ("o4-mini", "o4-mini"),
        "o3",
        ("email-agent-008", "ART·E\n(Qwen 2.5 14B)"),
    ],
    title="Percentage of Questions Answered Correctly",
)

fig1.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-accuracy-comparison.svg"
)

# Create the second chart (line chart)
fig2 = training_progress_chart(
    df.filter(pl.col("step").ne(592)),
    "val",
    "answer_correct",
    models=[
        "o3",
        ("o4-mini", "o4-mini"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gpt-4.1", "GPT-4.1"),
        ("email-agent-008", "ART·E (Qwen 2.5 14B)"),
    ],
    title="Fraction of Questions Answered Correctly",
    y_label="Val set success rate",
)
# Save the second chart using the helper
fig2.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-accuracy-training-progress.svg"
)

# %%
# --- Create Latency and Cost Charts ---
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="ticks")

# models_cost_latency = ["o4-mini", "o3", "ART·E"] # Old order
models_cost_latency = ["o3", "o4-mini", "ART·E"]  # New order
# latency_values = [3.4, 5.6, 1.1] # Old order
latency_values = [5.6, 3.4, 1.1]  # New order
# cost_values = [7.88, 55.19, 0.85] # Old order
cost_values = [55.19, 7.88, 0.85]  # New order

# Define colors - using the specific hex codes from comparison_models_bar_chart
ORANGE = "#e67a30"
GREY = "#e0dcd5"
model_colors = {"o4-mini": GREY, "o3": GREY, "ART·E": ORANGE}
# colors = sns.color_palette("muted", n_colors=len(models_cost_latency)) # Old way
# model_colors = {model: color for model, color in zip(models_cost_latency, colors)} # Old way
bar_colors = [model_colors[model] for model in models_cost_latency]


fig3, axes = plt.subplots(1, 2, figsize=(8, 3.5))  # Adjusted figsize slightly

# Latency Chart
ax_latency = axes[0]
x_pos = np.arange(len(models_cost_latency))
bar_width = 0.6

bars_latency = ax_latency.bar(x_pos, latency_values, width=bar_width, color=bar_colors)
ax_latency.set_title("Full-Run Latency (Seconds)", pad=15)
ax_latency.set_xticks(x_pos)
ax_latency.set_xticklabels(models_cost_latency, rotation=0, ha="center")
ax_latency.spines["left"].set_visible(False)
ax_latency.spines["top"].set_visible(False)
ax_latency.spines["right"].set_visible(False)
ax_latency.set_yticks([])
ax_latency.grid(False)

# Add latency value labels
max_latency = max(latency_values)
for bar in bars_latency:
    yval = bar.get_height()
    ax_latency.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + max_latency * 0.02,  # Position label slightly above bar
        f"{yval:.1f}s",
        ha="center",
        va="bottom",
        fontweight="bold",
    )


# Cost Chart
ax_cost = axes[1]
bars_cost = ax_cost.bar(x_pos, cost_values, width=bar_width, color=bar_colors)
ax_cost.set_title("Cost per 1K Runs", pad=15)
ax_cost.set_xticks(x_pos)
ax_cost.set_xticklabels(models_cost_latency, rotation=0, ha="center")
ax_cost.spines["left"].set_visible(False)
ax_cost.spines["top"].set_visible(False)
ax_cost.spines["right"].set_visible(False)
ax_cost.set_yticks([])
ax_cost.grid(False)

# Add cost value labels
max_cost = max(cost_values)
for bar in bars_cost:
    yval = bar.get_height()
    ax_cost.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + max_cost * 0.02,  # Position label slightly above bar
        f"${yval:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Bold the 'ART·E' label on both axes
for ax in axes:
    labels = ax.get_xticklabels()
    for label in labels:
        if label.get_text() == "ART·E":
            label.set_fontweight("bold")
    ax.set_xticklabels(labels)

plt.tight_layout(pad=2.0)  # Add padding between plots and title
# save_figure_for_blog(fig3, "art-e-cost-latency-comparison")

fig3.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-cost-latency-comparison.svg"
)

print("Chart generation and conversion finished.")

# %%
