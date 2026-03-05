from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compute_category_model_tables import compute_model_category_averages, SCORE_KEYS


def build_avg_score_dataframe(project_root: Path) -> pd.DataFrame:
    """
    Build a DataFrame with the mean score (across all five dimensions) for
    each (model, category) pair.

    Columns: ["model", "category", "avg_score"]
    """
    evaluation_dir = project_root / "evaluation"
    averages = compute_model_category_averages(evaluation_dir)

    rows = []
    for category, models_dict in averages.items():
        for model, scores in models_dict.items():
            # Mean over all score keys that are present for this (model, category)
            values = [scores[k] for k in SCORE_KEYS if k in scores]
            if not values:
                continue
            mean_score = sum(values) / len(values)
            rows.append({"model": model, "category": category, "avg_score": mean_score})

    return pd.DataFrame(rows)


def plot_histograms_per_category(df: pd.DataFrame) -> None:
    """
    For each category separately, plot a bar chart with:
      - x-axis: model
      - y-axis: mean score across all five scores
      - hue: model (different colors per model)
    """
    categories = sorted(df["category"].unique())

    for category in categories:
        df_cat = df[df["category"] == category]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_cat, x="model", y="avg_score", hue="model")

        plt.xlabel("Model")
        plt.ylabel("Mean Score (across all 5 dimensions)")
        plt.title(f"Average Score by Model for Category: {category}")
        plt.ylim(0, 2)  # scores are in [0,2]
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def main():
    project_root = Path(__file__).parent
    df = build_avg_score_dataframe(project_root)
    if df.empty:
        print("No data to plot.")
        return
    plot_histograms_per_category(df)


if __name__ == "__main__":
    main()


