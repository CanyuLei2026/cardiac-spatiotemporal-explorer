from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.data import (
    HISTORY_FRAMES,
    HORIZON_FRAMES,
    concatenate_cases,
    load_mat_case,
    load_web_case,
    standardization,
    subset_case,
)
from training.model import ActivationMLP, LinearBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the next-20-frame cardiac activation classifier.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing full case_*.mat files.")
    parser.add_argument("--train-cases", nargs="+", help="Three training case IDs, without .mat.")
    parser.add_argument("--val-case", help="Validation case ID, without .mat.")
    parser.add_argument("--test-case", help="Held-out test case ID, without .mat.")
    parser.add_argument(
        "--prototype-web-case",
        type=Path,
        help="Development-only compact web case. Splits points instead of cases.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--browser-model", type=Path, default=Path("public/model/activation_model.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7400)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(args: argparse.Namespace):
    if args.prototype_web_case:
        full = load_web_case(args.prototype_web_case)
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(full.points.shape[0])
        first = int(order.size * 0.6)
        second = int(order.size * 0.8)
        return (
            [subset_case(full, order[:first], "train-prototype")],
            [subset_case(full, order[first:second], "val-prototype")],
            [subset_case(full, order[second:], "test-prototype")],
            True,
        )

    required = [args.data_dir, args.train_cases, args.val_case, args.test_case]
    if not all(required):
        raise SystemExit(
            "Provide --data-dir, --train-cases, --val-case, and --test-case, "
            "or use --prototype-web-case for local development."
        )
    if len(args.train_cases) != 3:
        raise SystemExit("This assignment plan expects exactly three training cases.")
    case_ids = [*args.train_cases, args.val_case, args.test_case]
    if len(set(case_ids)) != 5:
        raise SystemExit("Training, validation, and test case IDs must be five distinct cases.")

    def load(case_id: str):
        path = args.data_dir / f"{case_id}.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        return load_mat_case(path)

    return (
        [load(case_id) for case_id in args.train_cases],
        [load(args.val_case)],
        [load(args.test_case)],
        False,
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def average_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total = 0.0
    count = 0
    for inputs, labels in loader:
        loss = criterion(model(inputs), labels)
        total += loss.item() * labels.size(0)
        count += labels.size(0)
    return total / max(count, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> dict[str, list[float] | int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    remaining_patience = patience
    history = {"trainLoss": [], "validationLoss": [], "bestEpoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            count += labels.size(0)

        train_loss = running_loss / max(count, 1)
        val_loss = average_loss(model, val_loader, criterion)
        history["trainLoss"].append(train_loss)
        history["validationLoss"].append(val_loss)
        print(f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["bestEpoch"] = epoch
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(best_state)
    return history


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.size + 1, dtype=np.float64)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float | list[list[int]]]:
    model.eval()
    labels_all = []
    probabilities_all = []
    for inputs, labels in loader:
        probabilities = torch.softmax(model(inputs), dim=1)[:, 1]
        labels_all.append(labels.numpy())
        probabilities_all.append(probabilities.numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    accuracy = (tp + tn) / labels.size
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "errorRate": 1 - accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": binary_auc(labels, probabilities),
        "confusionMatrix": [[tn, fp], [fn, tp]],
        "sampleCount": int(labels.size),
    }


def save_loss_plot(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["trainLoss"]) + 1)
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.plot(epochs, history["trainLoss"], label="Training loss", linewidth=2)
    axis.plot(epochs, history["validationLoss"], label="Validation loss", linewidth=2)
    axis.axvline(history["bestEpoch"], color="#d9534f", linestyle="--", label="Best epoch")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="MLP training history")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_plot(matrix: list[list[int]], path: Path) -> None:
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center", fontsize=12)
    axis.set_xticks([0, 1], ["No", "Yes"])
    axis.set_yticks([0, 1], ["No", "Yes"])
    axis.set(xlabel="Predicted", ylabel="Actual", title="Held-out test confusion matrix")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_browser_model(
    model: ActivationMLP,
    mean: np.ndarray,
    std: np.ndarray,
    metrics: dict,
    split: dict,
    history: dict,
    prototype: bool,
    path: Path,
) -> None:
    state = model.state_dict()
    layers = []
    for index, activation in [(0, "relu"), (2, "relu"), (4, "linear")]:
        layers.append(
            {
                "weight": state[f"network.{index}.weight"].cpu().numpy().tolist(),
                "bias": state[f"network.{index}.bias"].cpu().numpy().tolist(),
                "activation": activation,
            }
        )
    payload = {
        "schemaVersion": 1,
        "task": "Activation within the next 20 frames",
        "prototype": prototype,
        "input": {
            "historyFrames": HISTORY_FRAMES,
            "horizonFrames": HORIZON_FRAMES,
            "featureMean": mean.tolist(),
            "featureStd": std.tolist(),
        },
        "network": {"architecture": [44, 64, 32, 2], "layers": layers},
        "caseSplit": split,
        "metrics": metrics,
        "training": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_cases, val_cases, test_cases, prototype = load_splits(args)
    split = {
        "training": [case.case_id for case in train_cases],
        "validation": [case.case_id for case in val_cases],
        "test": [case.case_id for case in test_cases],
    }
    print(json.dumps({"caseSplit": split, "prototype": prototype}, indent=2))

    train_x, train_y, train_summary = concatenate_cases(train_cases, args.seed)
    val_x, val_y, val_summary = concatenate_cases(val_cases, args.seed + 100_000)
    test_x, test_y, test_summary = concatenate_cases(test_cases, args.seed + 200_000)
    mean, std = standardization(train_x)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    print(json.dumps({"training": train_summary, "validation": val_summary, "test": test_summary}, indent=2))

    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, args.batch_size, False)
    test_loader = make_loader(test_x, test_y, args.batch_size, False)

    mlp = ActivationMLP(train_x.shape[1])
    print("\nTraining MLP")
    history = fit_model(mlp, train_loader, val_loader, args.epochs, args.learning_rate, args.patience)
    mlp_metrics = evaluate(mlp, test_loader)

    baseline = LinearBaseline(train_x.shape[1])
    print("\nTraining logistic-regression baseline")
    baseline_history = fit_model(
        baseline,
        train_loader,
        val_loader,
        args.epochs,
        args.learning_rate,
        args.patience,
    )
    baseline_metrics = evaluate(baseline, test_loader)
    metrics = {"mlp": mlp_metrics, "logisticRegression": baseline_metrics}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelState": mlp.state_dict(),
            "featureMean": mean,
            "featureStd": std,
            "caseSplit": split,
            "prototype": prototype,
        },
        args.output_dir / "activation_mlp.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "case_split.json").write_text(json.dumps(split, indent=2))
    save_loss_plot(history, args.output_dir / "loss_curves.png")
    save_confusion_plot(mlp_metrics["confusionMatrix"], args.output_dir / "confusion_matrix.png")
    export_browser_model(mlp, mean, std, metrics, split, history, prototype, args.browser_model)
    print("\nFinal held-out test metrics")
    print(json.dumps(metrics, indent=2))
    print(f"Saved artifacts to {args.output_dir}")
    print(f"Saved browser model to {args.browser_model}")


if __name__ == "__main__":
    main()
