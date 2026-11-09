"""Evaluate frozen HW3 checkpoints without fitting parameters or thresholds."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.data import concatenate_cases, load_mat_case, load_web_case
from training.hw3.models import make_model
from training.train import average_loss, evaluate, make_loader, save_confusion_plot


def evaluate_checkpoint(path, x, y):
    # Files are checkpoints produced by this project's own training pipeline.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config, name = saved["config"], saved["model"]
    if config["dataSeed"] != 7400 or config["testCase"] != "case_0004":
        raise ValueError("Unexpected test split or sampling seed")
    settings = config.get("modelSettings", {}).get(name, {})
    model = make_model(name, settings)
    model.load_state_dict(saved["modelState"], strict=True)
    normalized = (x - saved["featureMean"]) / saved["featureStd"]
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite standardized test features")
    loader = make_loader(normalized.astype(np.float32), y, config["batchSize"], False)
    metrics = evaluate(model, loader)  # Sets eval mode: context dropout is disabled.
    metrics["loss"] = average_loss(model, loader, nn.CrossEntropyLoss())
    return {"model": name, "seed": saved["seed"], "checkpoint": str(path),
            "checkpointSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": config, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate one own checkpoint for a local smoke check")
    parser.add_argument("--prototype-web-case", type=Path)
    args = parser.parse_args()
    if args.prototype_web_case and not args.checkpoint:
        parser.error("Prototype mode requires --checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    case = load_web_case(args.prototype_web_case) if args.prototype_web_case else load_mat_case(
        "/project/jxie_group/Canyu/B2H/data/pointcloud_mat_9.16/case_0004.mat")
    # Exactly the same window sampling seed as the original held-out evaluation.
    x, y, data_summary = concatenate_cases([case], 7400 + 200000)
    paths = []
    if args.checkpoint:
        paths = [args.checkpoint]
    else:
        for index, seed in enumerate([7400, 7401, 7402]):
            paths.extend([
                Path(f"artifacts_hw3/context-22681331-{index}/gated_context50-seed{seed}/best_model.pt"),
                Path(f"artifacts_hw3/context-22681331-{index}/gated_dual_branch-seed{seed}/best_model.pt"),
                Path(f"artifacts_hw3/followup-22679009-0/baseline-seed{seed}/best_model.pt"),
            ])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    (args.output_dir / "evaluation_plan.json").write_text(json.dumps({
        "prototype": bool(args.prototype_web_case), "case": case.case_id,
        "samplingSeed": 207400, "threshold": 0.5, "dataSummary": data_summary,
        "selection": "Context dropout 0.5 selected by mean validation cross-entropy across three seeds; frozen before test scoring.",
        "references": "60-epoch baseline and original gated model, each using its minimum-validation-loss checkpoint.",
        "limitation": "Case_0004 was examined in the initial suite; this is an exploratory follow-up test evaluation, not a fresh untouched test set.",
        "checkpoints": [str(path) for path in paths]}, indent=2))
    rows = []
    for path in paths:
        row = evaluate_checkpoint(path, x, y)
        rows.append(row)
        output = args.output_dir / f"{row['model']}-seed{row['seed']}"
        output.mkdir()
        (output / "metrics.json").write_text(json.dumps(row, indent=2))
        save_confusion_plot(row["metrics"]["confusionMatrix"], output / "test_confusion_matrix.png")
        (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({"model": row["model"], "seed": row["seed"], **row["metrics"]}), flush=True)
    fields = ["accuracy", "errorRate", "f1", "recall", "precision", "auroc", "loss"]
    with (args.output_dir / "test_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", *fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": row["model"], "seed": row["seed"], **{field: row["metrics"][field] for field in fields}})
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        records = [row for row in rows if row["model"] == model]
        summary[model] = {field: {
            "mean": statistics.mean([r["metrics"][field] for r in records]),
            "sampleStd": statistics.stdev([r["metrics"][field] for r in records]) if len(records) > 1 else None,
        } for field in fields}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Completed frozen test evaluation: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
