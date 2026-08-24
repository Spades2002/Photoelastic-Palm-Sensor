"""
Training script for the optical tactile sensor force-regression pipeline.

Run with:
    python train.py

All checkpoints, logs, plots and prediction CSVs are written under
config.OUTPUT_DIR; nothing is written back into the original CSV or image
folders.
"""
from __future__ import annotations

import csv
import datetime
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import TactileSensorDataset
from splits import get_group_split_indices, make_group_ids
from model import TactileForceNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_dataloaders():
    full_dataset = TactileSensorDataset()
    train_idx, val_idx, test_idx = get_group_split_indices(full_dataset.frame)
    logger.info(
        "Group split by %s: train=%d val=%d test=%d rows (%d/%d/%d groups)",
        config.GROUP_SPLIT_COLS, len(train_idx), len(val_idx), len(test_idx),
        len(set(make_group_ids(full_dataset.frame, config.GROUP_SPLIT_COLS)[train_idx])),
        len(set(make_group_ids(full_dataset.frame, config.GROUP_SPLIT_COLS)[val_idx])),
        len(set(make_group_ids(full_dataset.frame, config.GROUP_SPLIT_COLS)[test_idx])),
    )

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def compute_batch_loss(model, batch, device, depth_loss_fn, location_loss_fn, force_loss_fn):
    fringe_image = batch["fringe_image"].to(device)
    indent_image = batch["indent_image"].to(device)
    depth_target = batch["depth_mm"].to(device)
    xy_target = batch["xy_mm"].to(device)
    force_target = batch["force"].to(device)

    outputs = model(fringe_image, indent_image)
    depth_loss = depth_loss_fn(outputs["depth_pred"], depth_target)
    location_loss = location_loss_fn(outputs["xy_pred"], xy_target)
    force_loss = force_loss_fn(outputs["force_pred"], force_target)
    total_loss = (
        config.DEPTH_LOSS_WEIGHT * depth_loss
        + config.LOCATION_LOSS_WEIGHT * location_loss
        + config.FORCE_LOSS_WEIGHT * force_loss
    )
    return total_loss, depth_loss, location_loss, force_loss, outputs


def run_epoch(model, loader, device, depth_loss_fn, location_loss_fn, force_loss_fn,
              optimizer=None, epoch=None, phase_label="train"):
    is_train = optimizer is not None
    model.train(is_train)

    total, total_depth, total_location, total_force = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    total_batches = len(loader)
    log_every = max(1, total_batches // 20)  # roughly 20 progress updates per epoch
    start_time = time.time()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            loss, depth_loss, location_loss, force_loss, _ = compute_batch_loss(
                model, batch, device, depth_loss_fn, location_loss_fn, force_loss_fn
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()
            total_depth += depth_loss.item()
            total_location += location_loss.item()
            total_force += force_loss.item()
            n_batches += 1

            if n_batches % log_every == 0 or n_batches == total_batches:
                elapsed = time.time() - start_time
                rate = n_batches / elapsed if elapsed > 0 else 0
                remaining = (total_batches - n_batches) / rate if rate > 0 else float("nan")
                logger.info(
                    "Epoch %s [%s] batch %d/%d (%.1f%%) | %.2f batches/sec | elapsed %.1f min | ETA %.1f min",
                    epoch, phase_label, n_batches, total_batches, 100 * n_batches / total_batches,
                    rate, elapsed / 60, remaining / 60,
                )

    return (
        total / n_batches, total_depth / n_batches, total_location / n_batches, total_force / n_batches
    )


def evaluate_test_set(model, loader, device) -> list:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            fringe_image = batch["fringe_image"].to(device)
            indent_image = batch["indent_image"].to(device)
            outputs = model(fringe_image, indent_image)

            depth_pred = outputs["depth_pred"].cpu().numpy()
            xy_pred = outputs["xy_pred"].cpu().numpy()
            force_pred = outputs["force_pred"].cpu().numpy()
            depth_true = batch["depth_mm"].numpy()
            xy_true = batch["xy_mm"].numpy()
            force_true = batch["force"].numpy()

            meta = batch["meta"]
            rows = meta["row"].tolist() if torch.is_tensor(meta["row"]) else list(meta["row"])
            cols = meta["col"].tolist() if torch.is_tensor(meta["col"]) else list(meta["col"])
            cycles = meta["cycle"].tolist() if torch.is_tensor(meta["cycle"]) else list(meta["cycle"])
            phases = meta["phase"].tolist() if torch.is_tensor(meta["phase"]) else list(meta["phase"])

            for i in range(len(depth_pred)):
                depth_err = float(depth_pred[i]) - float(depth_true[i])
                x_err = float(xy_pred[i][0]) - float(xy_true[i][0])
                y_err = float(xy_pred[i][1]) - float(xy_true[i][1])
                fx_err = float(force_pred[i][0]) - float(force_true[i][0])
                fy_err = float(force_pred[i][1]) - float(force_true[i][1])
                fz_err = float(force_pred[i][2]) - float(force_true[i][2])
                mag_err = float(force_pred[i][3]) - float(force_true[i][3])

                records.append({
                    "row": rows[i], "col": cols[i], "cycle": cycles[i], "phase": phases[i],
                    "depth_mm_true": depth_true[i], "depth_mm_pred": depth_pred[i], "depth_mm_error": depth_err,
                    "x_mm_true": xy_true[i][0], "x_mm_pred": xy_pred[i][0], "x_mm_error": x_err,
                    "y_mm_true": xy_true[i][1], "y_mm_pred": xy_pred[i][1], "y_mm_error": y_err,
                    "Fx_true": force_true[i][0], "Fx_pred": force_pred[i][0], "Fx_error": fx_err,
                    "Fy_true": force_true[i][1], "Fy_pred": force_pred[i][1], "Fy_error": fy_err,
                    "Fz_true": force_true[i][2], "Fz_pred": force_pred[i][2], "Fz_error": fz_err,
                    "magnitude_true": force_true[i][3], "magnitude_pred": force_pred[i][3], "magnitude_error": mag_err,
                })
    return records


def save_predictions_csv(records: list, path: Path) -> None:
    if not records:
        logger.warning("No test records to save.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved test predictions to %s", path)


def plot_loss_curves(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss, combined")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved loss curve to %s", path)


def main():
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = build_dataloaders()
    logger.info(
        "Train/val/test sizes: %d / %d / %d",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )

    model = TactileForceNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    depth_loss_fn = torch.nn.SmoothL1Loss(beta=config.HUBER_DELTA)
    location_loss_fn = torch.nn.SmoothL1Loss(beta=config.HUBER_DELTA)
    force_loss_fn = torch.nn.SmoothL1Loss(beta=config.HUBER_DELTA)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch_durations = []

    log_path = config.LOG_DIR / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "epoch", "train_loss", "train_depth_loss", "train_location_loss", "train_force_loss",
            "val_loss", "val_depth_loss", "val_location_loss", "val_force_loss",
        ])

        for epoch in range(1, config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_depth_loss, train_location_loss, train_force_loss = run_epoch(
                model, train_loader, device, depth_loss_fn, location_loss_fn, force_loss_fn, optimizer,
                epoch=epoch, phase_label="train",
            )
            val_loss, val_depth_loss, val_location_loss, val_force_loss = run_epoch(
                model, val_loader, device, depth_loss_fn, location_loss_fn, force_loss_fn, optimizer=None,
                epoch=epoch, phase_label="val",
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            log_writer.writerow([
                epoch, train_loss, train_depth_loss, train_location_loss, train_force_loss,
                val_loss, val_depth_loss, val_location_loss, val_force_loss,
            ])
            log_file.flush()

            logger.info(
                "Epoch %03d | train %.4f (depth %.4f, loc %.4f, force %.4f) | "
                "val %.4f (depth %.4f, loc %.4f, force %.4f)",
                epoch, train_loss, train_depth_loss, train_location_loss, train_force_loss,
                val_loss, val_depth_loss, val_location_loss, val_force_loss,
            )

            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
            remaining_epochs = config.NUM_EPOCHS - epoch
            eta_seconds = avg_epoch_duration * remaining_epochs
            eta_clock = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            logger.info(
                "Epoch %03d took %.1f min (avg %.1f min/epoch so far) | if it runs to "
                "NUM_EPOCHS=%d without early stopping: ETA in %.1f min, around %s "
                "(early stopping may finish sooner)",
                epoch, epoch_duration / 60, avg_epoch_duration / 60,
                config.NUM_EPOCHS, eta_seconds / 60, eta_clock.strftime("%Y-%m-%d %H:%M"),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, checkpoint_path)
                logger.info("New best model saved to %s", checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_without_improvement,
                    )
                    break

    plot_loss_curves(history, config.PLOT_DIR / "loss_curve.png")

    best_checkpoint = torch.load(config.CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logger.info("Loaded best model from epoch %d for test evaluation.", best_checkpoint["epoch"])

    test_records = evaluate_test_set(model, test_loader, device)
    save_predictions_csv(test_records, config.PREDICTIONS_DIR / "test_predictions.csv")


if __name__ == "__main__":
    main()
