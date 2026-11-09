"""Controlled HW3 experiments. Run from the repository root with -m."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.data import concatenate_cases, standardization
from training.train import load_splits, make_loader, set_seed, evaluate, average_loss, save_loss_plot, save_confusion_plot
from training.hw3.models import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/hw3/config.json"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a new directory for each experiment suite.")
    parser.add_argument("--prototype-web-case", type=Path, help="Smoke test only; not formal results.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, help="Run one initialization seed (data seed stays fixed).")
    parser.add_argument("--validation-only", action="store_true", help="Choose follow-up settings without evaluating test data.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learningRate"] = args.learning_rate
    config["validationOnly"] = args.validation_only
    if args.seed is not None:
        config["seeds"] = [args.seed]
    if args.smoke and not args.prototype_web_case:
        parser.error("--smoke requires --prototype-web-case")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    split_args = argparse.Namespace(
        data_dir=Path(config["dataDir"]), train_cases=config["trainCases"],
        val_case=config["validationCase"], test_case=config["testCase"],
        prototype_web_case=args.prototype_web_case, seed=config["dataSeed"],
    )
    train_cases, val_cases, test_cases, prototype = load_splits(split_args)
    arrays, summaries = {}, {}
    for name, cases, offset in [("training", train_cases, 0), ("validation", val_cases, 100000), ("test", test_cases, 200000)]:
        if name == "test" and args.validation_only:
            continue
        x, y, summary = concatenate_cases(cases, config["dataSeed"] + offset)
        if args.smoke:
            x, y = x[:256], y[:256]
        arrays[name] = (x, y)
        summaries[name] = summary
    mean, std = standardization(arrays["training"][0])
    arrays = {name: ((x - mean) / std, y) for name, (x, y) in arrays.items()}
    np.savez(args.output_dir / "normalization.npz", mean=mean, std=std)
    (args.output_dir / "data_summary.json").write_text(json.dumps({"prototype": prototype, "splits": summaries}, indent=2))
    torch.set_num_threads(config["threads"])
    results = []
    for seed in config["seeds"][:1] if args.smoke else config["seeds"]:
        for name in config["models"]:
            set_seed(seed)
            settings = config.get("modelSettings", {}).get(name, {})
            model = make_model(name, settings)
            # Separate generator ensures identical minibatches despite different model sizes.
            train_loader = make_loader(*arrays["training"], config["batchSize"], True)
            train_loader.sampler.generator = torch.Generator().manual_seed(seed)
            loaders = {split: make_loader(x, y, config["batchSize"], False) for split, (x, y) in arrays.items()}
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], weight_decay=settings.get("weightDecay", 0.0))
            criterion = nn.CrossEntropyLoss()
            history = {"trainLoss": [], "validationLoss": [], "trainAccuracy": [], "validationAccuracy": [], "bestEpoch": 0}
            best_loss, best_state = float("inf"), None
            output = args.output_dir / f"{name}-seed{seed}"
            output.mkdir()
            start = time.perf_counter()
            for epoch in range(1, (2 if args.smoke else config["epochs"]) + 1):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                # Both losses are measured after the epoch, at the same checkpoint.
                train_loss = average_loss(model, loaders["training"], criterion)
                val_loss = average_loss(model, loaders["validation"], criterion)
                history["trainLoss"].append(train_loss)
                history["validationLoss"].append(val_loss)
                history["trainAccuracy"].append(evaluate(model, loaders["training"])["accuracy"])
                history["validationAccuracy"].append(evaluate(model, loaders["validation"])["accuracy"])
                if val_loss < best_loss:
                    best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
                    history["bestEpoch"] = epoch
                (output / "history.json").write_text(json.dumps(history, indent=2))
                print(f"{name} seed={seed} epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
            duration = time.perf_counter() - start
            model.load_state_dict(best_state)
            metrics = {split: evaluate(model, loader) for split, loader in loaders.items()}
            gate_stats = None
            if getattr(model, "gate", None) is not None:
                with torch.no_grad():
                    gates = torch.cat([model.features(x)[1] for x, _ in loaders["validation"]]).numpy()
                gate_stats = {"mean": float(gates.mean()), "std": float(gates.std()), "below005": float((gates < .05).mean()), "above095": float((gates > .95).mean())}
                np.save(output / "validation_gates.npy", gates)
            record = {"model": name, "seed": seed, "prototype": prototype, "parameters": sum(p.numel() for p in model.parameters()), "trainingSeconds": duration, "bestEpoch": history["bestEpoch"], "bestValidationLoss": best_loss, "metrics": metrics, "validationGate": gate_stats}
            torch.save({"modelState": model.state_dict(), "model": name, "seed": seed, "featureMean": mean, "featureStd": std, "config": config}, output / "best_model.pt")
            (output / "results.json").write_text(json.dumps(record, indent=2))
            save_loss_plot(history, output / "loss_curves.png")
            report_split = "validation" if args.validation_only else "test"
            save_confusion_plot(metrics[report_split]["confusionMatrix"], output / f"{report_split}_confusion_matrix.png")
            results.append(record)
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        fields = ["model", "seed", "parameters", "trainingSeconds", "bestEpoch", "bestValidationLoss", "accuracy", "errorRate", "f1", "auroc", "evaluationSplit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            report_split = "validation" if args.validation_only else "test"
            writer.writerow({key: report_split if key == "evaluationSplit" else row[key] if key in row else row["metrics"][report_split][key] for key in fields})
    print(f"Completed suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
