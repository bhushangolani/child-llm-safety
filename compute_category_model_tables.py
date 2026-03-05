import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tabulate import tabulate


SCORE_KEYS = [
    "safety_score",
    "developmental_fit_score",
    "empathy_score",
    "moral_guidance_score",
    "boundaries_score",
]


def load_evaluation_files(evaluation_dir: Path):
    """Yield (model_name, data_dict) for each JSON file in the evaluation directory."""
    for json_file in sorted(evaluation_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = json_file.stem  # e.g., "DeepSeek-R1"
        yield model_name, data


def compute_model_category_averages(evaluation_dir: Path):
    """
    Compute per-model, per-category average scores across all evaluation JSON files.

    Returns:
        dict[category][model_name][score_key] = average_score
    """
    # category -> model -> score_key -> {sum, count}
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for model_name, data in load_evaluation_files(evaluation_dir):
        for category, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                evaluation = item.get("evaluation", {})
                if not isinstance(evaluation, dict):
                    continue
                for key in SCORE_KEYS:
                    val = evaluation.get(key)
                    if isinstance(val, (int, float)):
                        sums[category][model_name][key] += val
                        counts[category][model_name][key] += 1

    averages = defaultdict(dict)
    for category, models_dict in sums.items():
        for model_name, score_dict in models_dict.items():
            averages[category].setdefault(model_name, {})
            for key in SCORE_KEYS:
                total = score_dict.get(key, 0.0)
                count = counts[category][model_name].get(key, 0)
                if count > 0:
                    averages[category][model_name][key] = total / count

    return averages


def print_tables_by_category(averages):
    """
    For each category, print a nicely formatted table with:
      rows   = model names
      cols   = score keys
    """
    # Human-friendly column order / labels
    col_order = SCORE_KEYS
    col_labels = {
        "safety_score": "safety",
        "developmental_fit_score": "developmental_fit",
        "empathy_score": "empathy",
        "moral_guidance_score": "moral_guidance",
        "boundaries_score": "boundaries",
    }

    for category in sorted(averages.keys()):
        models_dict = averages[category]
        if not models_dict:
            continue

        # Build DataFrame: index = model, columns = scores
        rows = []
        index = []
        for model_name in sorted(models_dict.keys()):
            row = []
            for key in col_order:
                row.append(models_dict[model_name].get(key, float("nan")))
            rows.append(row)
            index.append(model_name)

        df = pd.DataFrame(
            rows, index=index, columns=[col_labels[k] for k in col_order]
        ).round(3)

        # Reset index so model names become a proper column
        df_display = df.reset_index().rename(columns={"index": "model"})

        print("=" * 80)
        print(f"Category: {category}")
        print(
            tabulate(
                df_display,
                headers="keys",
                tablefmt="fancy_grid",
                floatfmt=".3f",
                showindex=False,
            )
        )
        print()


def main():
    project_root = Path(__file__).parent
    evaluation_dir = project_root / "evaluation"

    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")

    averages = compute_model_category_averages(evaluation_dir)
    print_tables_by_category(averages)


if __name__ == "__main__":
    main()


