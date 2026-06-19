r"""Local TrOCR fine-tuning on the bundled handwritten medicine datasets.

This script exists because the repository's original OCR fine-tuning path was a
Colab notebook. For paper work, we also need a reproducible local training
entrypoint that:

1. uses the real bundled handwriting datasets under ``<project>/dataset``
2. does not depend on notebook-only tooling
3. writes checkpoints + metrics that the evaluation harness can compare

Example quick sanity run:

    cd backend
    .venv\Scripts\python training\train_trocr_local.py ^
        --epochs 1 --train-limit 64 --val-limit 32 --test-limit 32

Example full run (recommended to leave running for hours):

    cd backend
    .venv\Scripts\python training\train_trocr_local.py ^
        --dataset both --epochs 6 --batch-size 1 --grad-accum 16 ^
        --output-dir results\trocr-rx-finetuned
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.eval.metrics import corpus_wer_cer

PROJECT_ROOT = BACKEND_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "results" / "trocr-rx-finetuned"


@dataclass
class Sample:
    image_path: Path
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATASET_ROOT,
        help="Project dataset root containing 'Handwritten Rx' and the BD dataset",
    )
    parser.add_argument(
        "--dataset",
        choices=("rx", "bd", "both"),
        default="both",
        help="Which dataset(s) to train on",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/trocr-base-handwritten",
        help="Base TrOCR checkpoint or local directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where checkpoints and metrics are written",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of Handwritten Rx train split used for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Preferred device",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to lower VRAM usage",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model only from the local Hugging Face cache",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CSVs with validation/test predictions",
    )
    parser.add_argument(
        "--eval-only",
        type=Path,
        help="Skip training and evaluate an existing checkpoint directory",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_rows(rows: list[Sample], limit: int | None, seed: int) -> list[Sample]:
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def _rx_split(root: Path, split: str) -> list[Sample]:
    base = root / "Handwritten Rx"
    label_file = "Train_Label.csv" if split == "train" else "Test_Label.csv"
    image_dir = "Train_Set" if split == "train" else "Test_Set"
    path = base / label_file
    if not path.exists():
        return []

    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Images") or "").strip()
            text = (row.get("Text") or "").strip()
            image_path = base / image_dir / name
            if name and text and image_path.exists():
                rows.append(Sample(image_path, text))
    return rows


def _bd_split(root: Path, split: str) -> list[Sample]:
    folder = {"train": "Training", "val": "Validation", "test": "Testing"}[split]
    words_dir = {
        "train": "training_words",
        "val": "validation_words",
        "test": "testing_words",
    }[split]
    label_file = {
        "train": "training_labels.csv",
        "val": "validation_labels.csv",
        "test": "testing_labels.csv",
    }[split]
    matches = list(root.glob("*Handwritten Prescription BD*"))
    if not matches:
        return []

    base = matches[0] / folder
    path = base / label_file
    if not path.exists():
        return []

    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("IMAGE") or "").strip()
            text = (row.get("MEDICINE_NAME") or "").strip()
            image_path = base / words_dir / name
            if name and text and image_path.exists():
                rows.append(Sample(image_path, text))
    return rows


def build_splits(args: argparse.Namespace) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(args.seed)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    if args.dataset in ("rx", "both"):
        rx_train = _rx_split(args.data_root, "train")
        rng.shuffle(rx_train)
        split = int(len(rx_train) * args.val_fraction)
        val.extend(rx_train[:split])
        train.extend(rx_train[split:])
        test.extend(_rx_split(args.data_root, "test"))

    if args.dataset in ("bd", "both"):
        train.extend(_bd_split(args.data_root, "train"))
        val.extend(_bd_split(args.data_root, "val"))
        test.extend(_bd_split(args.data_root, "test"))

    train = _sample_rows(train, args.train_limit, args.seed)
    val = _sample_rows(val, args.val_limit, args.seed + 1)
    test = _sample_rows(test, args.test_limit, args.seed + 2)
    return train, val, test


class RxWordDataset(Dataset):
    def __init__(self, rows: list[Sample], processor: TrOCRProcessor, max_len: int):
        self.rows = rows
        self.processor = processor
        self.max_len = max_len
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        sample = self.rows[index]
        image = Image.open(sample.image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        tokenized = self.processor.tokenizer(
            sample.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)
        labels = labels.masked_fill(labels == self.pad_token_id, -100)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": sample.text,
            "path": str(sample.image_path),
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    base_model: str,
    processor: TrOCRProcessor,
    device: torch.device,
    max_len: int,
    num_beams: int,
    local_files_only: bool,
    gradient_checkpointing: bool,
) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_len
    model.generation_config.num_beams = num_beams
    model.generation_config.early_stopping = True
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    return model


def _word_accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    hits = sum(
        1
        for ref, hyp in pairs
        if " ".join(ref.lower().split()) == " ".join(hyp.lower().split())
    )
    return hits / len(pairs)


def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    loader: DataLoader,
    device: torch.device,
    max_len: int,
    num_beams: int,
    save_predictions_to: Path | None = None,
) -> dict:
    if len(loader.dataset) == 0:
        return {
            "loss": None,
            "word_acc": 0.0,
            "wer": 0.0,
            "cer": 0.0,
            "n_samples": 0,
        }

    model.eval()
    total_loss = 0.0
    total_items = 0
    pairs: list[tuple[str, str]] = []
    prediction_rows: list[dict] = []
    pad_token_id = processor.tokenizer.pad_token_id

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            batch_size = pixel_values.size(0)
            total_loss += float(outputs.loss.item()) * batch_size
            total_items += batch_size

            generated = model.generate(
                pixel_values,
                max_new_tokens=max_len,
                num_beams=num_beams,
            )
            predictions = [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]

            label_ids = labels.detach().cpu().clone()
            label_ids[label_ids == -100] = pad_token_id
            references = [text.strip() for text in processor.batch_decode(label_ids.tolist(), skip_special_tokens=True)]

            pairs.extend(zip(references, predictions))
            prediction_rows.extend(
                {
                    "image_path": path,
                    "reference": reference,
                    "prediction": prediction,
                }
                for path, reference, prediction in zip(batch["paths"], references, predictions)
            )

    wer, cer = corpus_wer_cer(pairs)
    metrics = {
        "loss": round(total_loss / total_items, 4) if total_items else None,
        "word_acc": round(_word_accuracy(pairs), 4),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "n_samples": len(pairs),
    }
    if save_predictions_to is not None:
        write_predictions_csv(save_predictions_to, prediction_rows)
    return metrics


def save_checkpoint(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    path: Path,
    metrics: dict,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def print_metrics(label: str, metrics: dict) -> None:
    loss = metrics["loss"]
    loss_str = f"{loss:.4f}" if isinstance(loss, float) else "n/a"
    print(
        f"{label}: n={metrics['n_samples']} "
        f"loss={loss_str} "
        f"word_acc={metrics['word_acc'] * 100:.2f}% "
        f"wer={metrics['wer'] * 100:.2f}% "
        f"cer={metrics['cer'] * 100:.2f}%"
    )


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor_source = str(args.eval_only) if args.eval_only else args.base_model
    processor = TrOCRProcessor.from_pretrained(
        processor_source,
        local_files_only=args.local_files_only,
    )

    train_rows, val_rows, test_rows = build_splits(args)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"Split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"(dataset={args.dataset})"
    )
    if not train_rows and not args.eval_only:
        raise RuntimeError("No training samples found. Check --data-root and dataset layout.")

    train_loader = DataLoader(
        RxWordDataset(train_rows, processor, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        RxWordDataset(val_rows, processor, args.max_len),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        RxWordDataset(test_rows, processor, args.max_len),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    base_model = str(args.eval_only) if args.eval_only else args.base_model
    model = build_model(
        base_model=base_model,
        processor=processor,
        device=device,
        max_len=args.max_len,
        num_beams=args.num_beams,
        local_files_only=args.local_files_only,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    run_summary = {
        "config": {
            **vars(args),
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "eval_only": str(args.eval_only) if args.eval_only else None,
            "device_resolved": str(device),
        },
        "dataset_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "epochs": [],
    }

    if args.eval_only:
        print(f"Evaluating checkpoint: {args.eval_only}")
        val_metrics = evaluate_model(
            model,
            processor,
            val_loader,
            device,
            args.max_len,
            args.num_beams,
            save_predictions_to=(args.output_dir / "val_predictions.csv") if args.save_predictions else None,
        )
        test_metrics = evaluate_model(
            model,
            processor,
            test_loader,
            device,
            args.max_len,
            args.num_beams,
            save_predictions_to=(args.output_dir / "test_predictions.csv") if args.save_predictions else None,
        )
        run_summary["best_val"] = val_metrics
        run_summary["test"] = test_metrics
        print_metrics("Validation", val_metrics)
        print_metrics("Test", test_metrics)
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )
        return run_summary

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_cer = float("inf")
    best_dir = args.output_dir / "best"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * args.grad_accum
            should_step = step % args.grad_accum == 0 or step == len(train_loader)
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_step = scaler.get_scale() >= prev_scale
                else:
                    optimizer.step()
                    did_step = True
                if did_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 25 == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"avg_loss={avg_loss:.4f}"
                )

        val_predictions = args.output_dir / f"val_predictions_epoch{epoch:02d}.csv" if args.save_predictions else None
        val_metrics = evaluate_model(
            model,
            processor,
            val_loader,
            device,
            args.max_len,
            args.num_beams,
            save_predictions_to=val_predictions,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, len(train_loader)), 4),
            "val": val_metrics,
            "elapsed_sec": round(time.perf_counter() - epoch_start, 1),
        }
        run_summary["epochs"].append(epoch_metrics)
        print_metrics(f"Epoch {epoch} validation", val_metrics)

        epoch_dir = args.output_dir / f"epoch_{epoch:02d}"
        save_checkpoint(model, processor, epoch_dir, epoch_metrics)

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            save_checkpoint(model, processor, best_dir, epoch_metrics)
            print(f"Saved new best checkpoint to {best_dir}")

        (args.output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )

    print("Reloading best checkpoint for final evaluation.")
    best_model = build_model(
        base_model=str(best_dir),
        processor=processor,
        device=device,
        max_len=args.max_len,
        num_beams=args.num_beams,
        local_files_only=True,
        gradient_checkpointing=False,
    )
    test_metrics = evaluate_model(
        best_model,
        processor,
        test_loader,
        device,
        args.max_len,
        args.num_beams,
        save_predictions_to=(args.output_dir / "test_predictions.csv") if args.save_predictions else None,
    )
    run_summary["best_val"] = json.loads((best_dir / "metrics.json").read_text(encoding="utf-8"))
    run_summary["test"] = test_metrics
    print_metrics("Held-out test", test_metrics)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    return run_summary


def main() -> None:
    args = parse_args()
    summary = train(args)
    print(f"Summary written to {args.output_dir / 'run_summary.json'}")
    if summary.get("best_val"):
        print_metrics("Best validation", summary["best_val"]["val"] if "val" in summary["best_val"] else summary["best_val"])
    if summary.get("test"):
        print_metrics("Test", summary["test"])


if __name__ == "__main__":
    main()
