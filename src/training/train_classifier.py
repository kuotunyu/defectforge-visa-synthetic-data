"""Train leakage-safe per-object ConvNeXt classifiers for M16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.training.classifier_data import (
    ClassificationGroup,
    ClassificationSample,
    build_classification_group,
    group_payload_sha256,
    resolve_sample_path,
    summarize_samples,
)

RESULT_COLUMNS = (
    "run_name",
    "run_signature",
    "requested_group",
    "canonical_group",
    "object",
    "seed",
    "model",
    "model_revision",
    "input_size",
    "total_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "train_total",
    "train_real",
    "train_synthetic",
    "sampled_real_good",
    "sampled_real_bad",
    "sampled_synthetic_bad",
    "test_total",
    "test_good",
    "test_bad",
    "macro_f1",
    "anomaly_f1",
    "auroc",
    "normal_false_positive_rate",
    "peak_vram_gib",
    "training_seconds",
    "data_manifest_sha256",
    "split_manifest_sha256",
    "selection_sha256",
)
SAMPLER_STRATEGIES = ("class_balanced", "domain_balanced")


class ClassifierTrainingError(RuntimeError):
    """Raised when M16 training violates the frozen experiment contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassifierTrainingError(message)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid classifier config: {path}")
    require(value.get("schema_version") == 1, "Unsupported classifier config schema")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_transforms(
    config: Mapping[str, Any],
    *,
    standard_augmentation: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    model_config = config["model"]
    size = int(model_config["input_size"])
    mean = tuple(float(value) for value in model_config["mean"])
    std = tuple(float(value) for value in model_config["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    evaluation = transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if not standard_augmentation:
        return evaluation, evaluation

    augmentation = config["standard_augmentation"]
    jitter = [float(value) for value in augmentation["color_jitter"]]
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=tuple(float(value) for value in augmentation["random_resized_crop_scale"]),
                ratio=tuple(float(value) for value in augmentation["random_resized_crop_ratio"]),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                translate=(
                    float(augmentation["translation_fraction"]),
                    float(augmentation["translation_fraction"]),
                ),
                scale=tuple(float(value) for value in augmentation["scale"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=jitter[0],
                contrast=jitter[1],
                saturation=jitter[2],
                hue=jitter[3],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return training, evaluation


class ClassificationDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Resolve portable sample records only when an item is loaded."""

    def __init__(
        self,
        paths: Paths,
        samples: Sequence[ClassificationSample],
        transform: transforms.Compose,
    ) -> None:
        self.paths = paths
        self.samples = tuple(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        with Image.open(resolve_sample_path(self.paths, sample)) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label, index


def balanced_sampler(
    samples: Sequence[ClassificationSample],
    *,
    num_samples: int,
    seed: int,
) -> WeightedRandomSampler:
    counts = Counter(sample.label for sample in samples)
    require(set(counts) == {0, 1}, "Training data must contain both classes")
    weights = torch.tensor(
        [1.0 / counts[sample.label] for sample in samples],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def _exposure_bucket(sample: ClassificationSample) -> str:
    if sample.kind == "synthetic":
        require(sample.label == 1, "Synthetic classifier samples must be anomalies")
        return "synthetic_bad"
    return "real_bad" if sample.label == 1 else "real_good"


def domain_balanced_sampler(
    samples: Sequence[ClassificationSample],
    *,
    num_samples: int,
    seed: int,
    real_bad_share: float,
) -> WeightedRandomSampler:
    """Balance good/bad classes, then balance real/synthetic within bad."""
    require(0.0 < real_bad_share < 1.0, "real_bad_share must be between zero and one")
    buckets = Counter(_exposure_bucket(sample) for sample in samples)
    required = {"real_good", "real_bad", "synthetic_bad"}
    require(set(buckets) == required, "Domain-balanced sampling requires all three domains")
    target_probability = {
        "real_good": 0.5,
        "real_bad": 0.5 * real_bad_share,
        "synthetic_bad": 0.5 * (1.0 - real_bad_share),
    }
    weights = torch.tensor(
        [
            target_probability[_exposure_bucket(sample)]
            / buckets[_exposure_bucket(sample)]
            for sample in samples
        ],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def build_sampler(
    samples: Sequence[ClassificationSample],
    *,
    strategy: str,
    num_samples: int,
    seed: int,
    real_bad_share: float,
) -> WeightedRandomSampler:
    require(strategy in SAMPLER_STRATEGIES, f"Unknown sampler strategy: {strategy}")
    if strategy == "class_balanced":
        require(
            real_bad_share == 0.5,
            "real_bad_share only applies to domain_balanced sampling",
        )
        return balanced_sampler(samples, num_samples=num_samples, seed=seed)
    return domain_balanced_sampler(
        samples,
        num_samples=num_samples,
        seed=seed,
        real_bad_share=real_bad_share,
    )


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    require(len(labels) == len(scores) and bool(labels), "Metric inputs are empty or mismatched")
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    require(set(labels_array.tolist()) == {0, 1}, "Metrics require both classes")
    predictions = (scores_array >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(labels_array.size),
        "good": int((labels_array == 0).sum()),
        "bad": int((labels_array == 1).sum()),
        "macro_f1": float(f1_score(labels_array, predictions, average="macro")),
        "anomaly_f1": float(f1_score(labels_array, predictions, pos_label=1)),
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "normal_false_positive_rate": float(fp / (fp + tn)),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    for images, batch_labels, _ in loader:
        images = images.to(device, non_blocking=False)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        labels.extend(int(value) for value in batch_labels.tolist())
        scores.extend(float(value) for value in probabilities.cpu().tolist())
    return classification_metrics(labels, scores)


def download_locked_weights(paths: Paths, model_config: Mapping[str, Any]) -> Path:
    cache_dir = paths.cache / "huggingface"
    weight_path = Path(
        hf_hub_download(
            repo_id=str(model_config["repository"]),
            filename=str(model_config["filename"]),
            revision=str(model_config["revision"]),
            cache_dir=str(cache_dir),
        )
    )
    observed = sha256_file(weight_path)
    require(
        observed == model_config["sha256"],
        f"Classifier base weight hash mismatch: {observed}",
    )
    return weight_path


def load_model(
    paths: Paths,
    model_config: Mapping[str, Any],
) -> tuple[nn.Module, Path]:
    weight_path = download_locked_weights(paths, model_config)
    model = timm.create_model(str(model_config["name"]), pretrained=False)
    state = load_file(str(weight_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.reset_classifier(int(model_config["num_classes"]))
    return model, weight_path


def cosine_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: WeightedRandomSampler | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _save_model(model: nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(path))
    return sha256_file(path)


def _exposure_counts(
    samples: Sequence[ClassificationSample],
    sampled_indices: Sequence[int],
) -> dict[str, int]:
    counts = {
        "real_good": 0,
        "real_bad": 0,
        "synthetic_bad": 0,
    }
    for index in sampled_indices:
        sample = samples[index]
        if sample.kind == "synthetic":
            counts["synthetic_bad"] += 1
        elif sample.label == 1:
            counts["real_bad"] += 1
        else:
            counts["real_good"] += 1
    return counts


def train(
    *,
    paths: Paths,
    config: Mapping[str, Any],
    group: ClassificationGroup,
    output_dir: Path,
    seed: int,
    total_steps: int,
    learning_rate: float,
    weight_decay: float,
    sampler_strategy: str,
    real_bad_share: float,
    experimental_synthetic_development: bool,
    smoke: bool,
) -> dict[str, Any]:
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    require(num_workers == 0, "Windows-native M16 requires num_workers=0")
    amp = bool(training_config["amp"])
    set_determinism(seed)
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "M16 requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    training_transform, evaluation_transform = build_transforms(
        config,
        standard_augmentation=group.standard_augmentation,
    )
    train_dataset = ClassificationDataset(paths, group.train, training_transform)
    sampler = build_sampler(
        group.train,
        strategy=sampler_strategy,
        num_samples=total_steps * batch_size,
        seed=seed,
        real_bad_share=real_bad_share,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
    evaluation_samples = group.validation if group.mode == "development" else group.test
    evaluation_dataset = ClassificationDataset(paths, evaluation_samples, evaluation_transform)
    evaluation_loader = _loader(
        evaluation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    started_wall = time.perf_counter()
    model, base_weight_path = load_model(paths, config["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = min(int(training_config["warmup_steps"]), max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    sampled_indices: list[int] = []
    loss_values: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_validation: dict[str, Any] | None = None
    best_step = 0
    stale_evaluations = 0
    executed_steps = 0
    model_path = output_dir / "model.safetensors"
    training_started = time.perf_counter()
    eval_every = max(1, int(training_config["eval_every_steps"]))
    patience = int(training_config["early_stopping_patience"])

    for step, (images, labels, indices) in enumerate(train_loader, start=1):
        if step > total_steps:
            break
        model.train()
        images = images.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training_config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        sampled_indices.extend(int(index) for index in indices.tolist())
        loss_values.append(float(loss.detach().cpu()))
        executed_steps = step

        should_evaluate = group.mode == "development" and (
            step % eval_every == 0 or step == total_steps
        )
        if should_evaluate:
            metrics = evaluate(model, evaluation_loader, device=device, amp=amp)
            record = {"step": step, **metrics}
            validation_history.append(record)
            score = (metrics["macro_f1"], metrics["auroc"])
            best_score = (
                (-1.0, -1.0)
                if best_validation is None
                else (best_validation["macro_f1"], best_validation["auroc"])
            )
            if score > best_score:
                best_validation = record
                best_step = step
                _save_model(model, model_path)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if not smoke and stale_evaluations >= patience:
                break

    training_seconds = time.perf_counter() - training_started
    if group.mode == "development":
        require(best_validation is not None and model_path.is_file(), "No best validation model")
        model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
        model.to(device)
        final_metrics = best_validation
    else:
        model_sha256 = _save_model(model, model_path)
        final_metrics = evaluate(model, evaluation_loader, device=device, amp=amp)

    if group.mode == "development":
        model_sha256 = sha256_file(model_path)
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    exposure = _exposure_counts(group.train, sampled_indices)
    report = {
        "status": "passed",
        "pipeline_version": config["pipeline_version"],
        "mode": group.mode,
        "smoke": smoke,
        "object": group.object_name,
        "requested_group": group.requested_group,
        "canonical_group": group.canonical_group,
        "standard_augmentation": group.standard_augmentation,
        "seed": seed,
        "model": config["model"]["name"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "base_weight_sha256": config["model"]["sha256"],
        "base_weight_bytes": base_weight_path.stat().st_size,
        "input_size": config["model"]["input_size"],
        "batch_size": batch_size,
        "requested_total_steps": total_steps,
        "executed_steps": executed_steps,
        "best_step": best_step if group.mode == "development" else executed_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sampler_strategy": sampler_strategy,
        "real_bad_share": real_bad_share,
        "experimental_synthetic_development": experimental_synthetic_development,
        "warmup_steps": warmup_steps,
        "last_loss": loss_values[-1],
        "mean_loss": float(np.mean(loss_values)),
        "training_seconds": training_seconds,
        "wall_clock_seconds": time.perf_counter() - started_wall,
        "peak_vram_gib": peak_vram_gib,
        "sample_exposure": exposure,
        "train_counts": summarize_samples(group.train),
        "evaluation_counts": summarize_samples(evaluation_samples),
        "validation_history": validation_history,
        "metrics": final_metrics,
        "model_sha256": model_sha256,
    }
    return report


def append_result(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(tuple(reader.fieldnames or ()) == RESULT_COLUMNS, "Classification CSV schema changed")
            existing = list(reader)
    duplicate = [item for item in existing if item["run_name"] == row["run_name"]]
    require(not duplicate, f"Classification result already exists: {row['run_name']}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({name: row[name] for name in RESULT_COLUMNS})
    os.replace(temporary, path)


def result_row(
    *,
    run_name: str,
    run_signature: str,
    group: ClassificationGroup,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    data_manifest_sha256: str,
) -> dict[str, Any]:
    metrics = report["metrics"]
    train_counts = report["train_counts"]
    evaluation_counts = report["evaluation_counts"]
    exposure = report["sample_exposure"]
    return {
        "run_name": run_name,
        "run_signature": run_signature,
        "requested_group": group.requested_group,
        "canonical_group": group.canonical_group,
        "object": group.object_name,
        "seed": report["seed"],
        "model": config["model"]["name"],
        "model_revision": config["model"]["revision"],
        "input_size": config["model"]["input_size"],
        "total_steps": report["executed_steps"],
        "batch_size": report["batch_size"],
        "learning_rate": report["learning_rate"],
        "weight_decay": report["weight_decay"],
        "train_total": train_counts["total"],
        "train_real": train_counts["kinds"].get("real", 0),
        "train_synthetic": train_counts["kinds"].get("synthetic", 0),
        "sampled_real_good": exposure["real_good"],
        "sampled_real_bad": exposure["real_bad"],
        "sampled_synthetic_bad": exposure["synthetic_bad"],
        "test_total": evaluation_counts["total"],
        "test_good": evaluation_counts["labels"].get("good", 0),
        "test_bad": evaluation_counts["labels"].get("bad", 0),
        "macro_f1": metrics["macro_f1"],
        "anomaly_f1": metrics["anomaly_f1"],
        "auroc": metrics["auroc"],
        "normal_false_positive_rate": metrics["normal_false_positive_rate"],
        "peak_vram_gib": report["peak_vram_gib"],
        "training_seconds": report["training_seconds"],
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": group.manifest_sha256,
        "selection_sha256": group.selection_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--object", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--mode", choices=("development", "final"), default="final")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--sampler-strategy",
        choices=SAMPLER_STRATEGIES,
        default="class_balanced",
    )
    parser.add_argument("--real-bad-share", type=float, default=0.5)
    parser.add_argument("--experimental-synthetic-development", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths)
    config_path = args.config.resolve(strict=True)
    config = load_config(config_path)
    if args.mode == "final" and not args.smoke and not args.dry_run:
        require(
            bool(config.get("hyperparameters_frozen")),
            "Formal runs require frozen Real-only hyperparameters",
        )
    training_config = config["training"]
    total_steps = 1 if args.smoke else int(args.total_steps or training_config["total_steps"])
    learning_rate = float(args.learning_rate or training_config["learning_rate"])
    weight_decay = float(
        training_config["weight_decay"] if args.weight_decay is None else args.weight_decay
    )
    require(total_steps > 0, "total_steps must be positive")
    require(learning_rate > 0, "learning_rate must be positive")
    require(weight_decay >= 0, "weight_decay must be non-negative")
    require(0.0 < args.real_bad_share < 1.0, "real_bad_share must be between zero and one")
    if args.sampler_strategy == "class_balanced":
        require(
            args.real_bad_share == 0.5,
            "real_bad_share only applies to domain_balanced sampling",
        )
    if args.mode == "final":
        require(
            args.sampler_strategy == "class_balanced",
            "v2 sampler is development-only until the confirmatory contract is frozen",
        )
        require(
            not args.experimental_synthetic_development,
            "Experimental synthetic development flag is invalid for final mode",
        )

    group = build_classification_group(
        paths,
        config,
        group_name=args.group,
        object_name=args.object,
        seed=args.seed,
        mode=args.mode,
        allow_synthetic_development=args.experimental_synthetic_development,
    )
    run_name = args.run_name or (
        f"{group.requested_group}__{group.object_name}__seed_{args.seed}__{args.mode}"
    )
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve(strict=False)
    else:
        suffix = "cls_smoke" if args.smoke else str(config["output"]["run_subdirectory"])
        output_dir = paths.runs / suffix / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    preexisting = [path for path in output_dir.iterdir()]
    require(not preexisting, f"Output directory is not empty: {output_dir}")

    data_payload = group.payload()
    data_manifest_sha256 = group_payload_sha256(group)
    data_payload["sha256"] = data_manifest_sha256
    atomic_write_json(output_dir / "data_manifest.json", data_payload)
    config_sha256 = sha256_file(config_path)
    signature_payload = {
        "pipeline_version": config["pipeline_version"],
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "model_sha256": config["model"]["sha256"],
        "seed": args.seed,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sampler_strategy": args.sampler_strategy,
        "real_bad_share": args.real_bad_share,
        "experimental_synthetic_development": args.experimental_synthetic_development,
        "mode": args.mode,
        "smoke": args.smoke,
    }
    run_signature = canonical_json_sha256(signature_payload)
    atomic_write_json(
        output_dir / "run_config.json",
        {
            **signature_payload,
            "run_name": run_name,
            "run_signature": run_signature,
            "requested_group": group.requested_group,
            "canonical_group": group.canonical_group,
            "object": group.object_name,
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "dry_run": True,
                    "run_name": run_name,
                    "run_signature": run_signature,
                    "output_dir": str(output_dir),
                    "counts": data_payload["counts"],
                    "sampler_strategy": args.sampler_strategy,
                    "real_bad_share": args.real_bad_share,
                    "experimental_synthetic_development": (
                        args.experimental_synthetic_development
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    report = train(
        paths=paths,
        config=config,
        group=group,
        output_dir=output_dir,
        seed=args.seed,
        total_steps=total_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        sampler_strategy=args.sampler_strategy,
        real_bad_share=args.real_bad_share,
        experimental_synthetic_development=args.experimental_synthetic_development,
        smoke=args.smoke,
    )
    report = {
        **report,
        "run_name": run_name,
        "run_signature": run_signature,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": group.manifest_sha256,
        "selection_sha256": group.selection_sha256,
    }
    atomic_write_json(output_dir / "training_report.json", report)
    if args.mode == "final" and not args.smoke:
        results_path = paths.project_root / str(config["output"]["results_csv"])
        append_result(
            results_path,
            result_row(
                run_name=run_name,
                run_signature=run_signature,
                group=group,
                config=config,
                report=report,
                data_manifest_sha256=data_manifest_sha256,
            ),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
