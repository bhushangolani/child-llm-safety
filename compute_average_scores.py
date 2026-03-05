import json
from collections import defaultdict
from pathlib import Path


def compute_averages(json_path: str) -> None:
    """
    Compute and print the average scores per category from a DeepSeek evaluation JSON.

    Expected JSON structure (simplified):
    {
        "category_name": [
            {
                "prompt": "...",
                "response": "...",
                "evaluation": {
                    "safety_score": 2,
                    "developmental_fit_score": 1,
                    "empathy_score": 2,
                    "moral_guidance_score": 1,
                    "boundaries_score": 2
                }
            },
            ...
        ],
        ...
    }
    """
    json_file = Path(json_path)
    if not json_file.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    score_keys = [
        "safety_score",
        "developmental_fit_score",
        "empathy_score",
        "moral_guidance_score",
        "boundaries_score",
    ]

    # category -> score_key -> {sum, count}
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    # Iterate over top-level categories (e.g., "sexual", "harm", etc.)
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            evaluation = item.get("evaluation", {})
            if not isinstance(evaluation, dict):
                continue
            for key in score_keys:
                if key in evaluation and isinstance(evaluation[key], (int, float)):
                    sums[category][key] += evaluation[key]
                    counts[category][key] += 1

    # Print results
    header = [
        "category",
        "safety",
        "developmental_fit",
        "empathy",
        "moral_guidance",
        "boundaries",
    ]
    print("\t".join(header))

    for category in sorted(sums.keys()):
        row = [category]
        for key in score_keys:
            total = sums[category].get(key, 0.0)
            count = counts[category].get(key, 0)
            avg = total / count if count > 0 else float("nan")
            row.append(f"{avg:.3f}" if count > 0 else "nan")
        print("\t".join(row))


if __name__ == "__main__":
    # Default to DeepSeek-R1.json in the evaluation directory
    default_path = Path(__file__).parent / "evaluation" / "DeepSeek-R1.json"
    compute_averages(str(default_path))


