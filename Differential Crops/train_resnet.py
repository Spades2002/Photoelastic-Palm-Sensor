"""
Deep model: a two-branch network built on pretrained ResNet-18 backbones,
trained on the GPU if one is available.

    fringe crop -> ResNet18 -> \\
                                  concat -> force_head -> Fz (default: Fz only)
    indent crop -> ResNet18 -> /       \\
                                         -> [+ force_pred] -> location_head -> x_mm, y_mm

By default the force prediction feeds into the location head too (detached,
so a bad localisation gradient can't corrupt force accuracy), since force
correlates with indentation depth, and depth affects how much of the dimple
is visible. Disable with --no-force-conditions-location for the original
two-independent-heads architecture.

Also by default, trains on delta crops (current frame minus that point's
own 0mm baseline, see build_dataset.py) rather than raw crops, since the
raw fringe crop barely changes visibly across the whole force range, delta
crops isolate what actually changed. Use --crop-source raw for the original
behaviour. And by default predicts Fz only, the one force axis that's been
reliable throughout every model tried so far; use --force-targets Fx,Fy,Fz
for the original 3-axis behaviour.

GPU setup (do this once, before running this script):
    1. Run `nvidia-smi` in a terminal and check the CUDA version your driver supports.
    2. Go to https://pytorch.org/get-started/locally/, pick your OS/pip/CUDA version,
       and run the exact command it gives you, e.g. something like:
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
       (the cuXXX tag changes as new CUDA/PyTorch versions are released, so use
       whatever the site currently recommends rather than copying an old command)
    3. Verify it worked: python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
       If this prints False, this script will still run, just on the CPU, much
       slower, and it will tell you so at startup.

Prints one line per epoch (train/val loss, time taken, and an ETA in
minutes for the next epoch, not a stream of per-batch progress lines), and
stops early if validation loss hasn't improved for --patience epochs
(default 10), so you're not left running epochs that only get thrown away.

Run build_dataset.py with --save-crops first (needed for both raw and delta
crop paths to exist in dataset.csv).

Usage:
    python train_resnet.py --epochs 30 --batch-size 32
    python train_resnet.py --epochs 30 --batch-size 32 --patience 15
    python train_resnet.py --epochs 30 --batch-size 32 --crop-source raw --force-targets Fx,Fy,Fz --no-force-conditions-location  # original behaviour, for comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

import config

logger = logging.getLogger(__name__)

FORCE_TARGETS = ["Fx", "Fy", "Fz"]
LOCATION_TARGETS = ["x_mm", "y_mm"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TactileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, force_mean, force_std, loc_mean, loc_std, train: bool,
                 force_targets: list[str] | None = None, crop_source: str = "raw"):
        self.df = df.reset_index(drop=True)
        self.force_mean, self.force_std = force_mean, force_std
        self.loc_mean, self.loc_std = loc_mean, loc_std
        self.force_targets = force_targets or FORCE_TARGETS
        self.crop_source = crop_source
        fringe_col, indent_col = ("fringe_delta_path", "indent_delta_path") if crop_source == "delta" else ("fringe_crop_path", "indent_crop_path")
        self.fringe_col, self.indent_col = fringe_col, indent_col
        if crop_source == "delta" and fringe_col not in self.df.columns:
            raise ValueError(f"{fringe_col} not in dataset.csv, re-run build_dataset.py --save-crops (it now saves delta crops too)")

        aug = [transforms.RandomHorizontalFlip()] if train else []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fringe = self._load(row[self.fringe_col])
        indent = self._load(row[self.indent_col])

        force = (row[self.force_targets].values.astype(np.float32) - self.force_mean) / self.force_std
        loc = (row[LOCATION_TARGETS].values.astype(np.float32) - self.loc_mean) / self.loc_std

        return fringe, indent, torch.tensor(force, dtype=torch.float32), torch.tensor(loc, dtype=torch.float32)


class TactileNet(nn.Module):
    def __init__(self, n_force_outputs=3, n_location_outputs=2, pretrained=True, force_conditions_location=False):
        super().__init__()
        self.force_conditions_location = force_conditions_location
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.fringe_backbone = models.resnet18(weights=weights)
        self.indent_backbone = models.resnet18(weights=weights)
        embed_dim = self.fringe_backbone.fc.in_features  # 512
        self.fringe_backbone.fc = nn.Identity()
        self.indent_backbone.fc = nn.Identity()

        self.force_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_force_outputs),
        )
        location_in_dim = embed_dim * 2 + (n_force_outputs if force_conditions_location else 0)
        self.location_head = nn.Sequential(
            nn.Linear(location_in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_location_outputs),
        )

    def forward(self, fringe, indent):
        f_embed = self.fringe_backbone(fringe)
        i_embed = self.indent_backbone(indent)
        combined = torch.cat([f_embed, i_embed], dim=1)
        force_pred = self.force_head(combined)
        if self.force_conditions_location:
            # detached: a bad localisation gradient shouldn't be able to
            # degrade force accuracy, force is trained purely to be correct,
            # location is trained to make good use of it.
            loc_input = torch.cat([combined, force_pred.detach()], dim=1)
        else:
            loc_input = combined
        loc_pred = self.location_head(loc_input)
        return force_pred, loc_pred


def load_tactile_checkpoint(path, device):
    """
    Shared loader used by predict.py, predict_batch.py, and evaluate_resnet.py,
    so a checkpoint trained with a non-default architecture (Fz-only, delta
    crops, the force-conditions-location cascade) always gets reconstructed
    correctly instead of each caller guessing from hardcoded defaults.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    force_targets = ckpt.get("force_targets", FORCE_TARGETS)  # older checkpoints predate this key
    model = TactileNet(
        n_force_outputs=len(force_targets),
        n_location_outputs=len(LOCATION_TARGETS),
        pretrained=False,
        force_conditions_location=ckpt.get("force_conditions_location", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _group_split(df, group_col, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA available, training on GPU: %s", name)
        return torch.device("cuda")
    logger.warning(
        "CUDA is NOT available, falling back to CPU. Training will be much slower. "
        "See the GPU setup instructions in this file's docstring if you have an NVIDIA GPU."
    )
    return torch.device("cpu")


def train(
    epochs=30, batch_size=32, lr=1e-4, test_size=0.25, seed=0,
    force_weight=1.0, location_weight=1.0, num_workers=4, amp=True,
    device_arg="auto", patience=10, pretrained=True,
    crop_source="delta", force_targets=None, force_conditions_location=True,
):
    force_targets = force_targets or ["Fz"]
    if not config.DATASET_CSV_PATH.exists():
        raise SystemExit(f"{config.DATASET_CSV_PATH} not found, run build_dataset.py first")

    df = pd.read_csv(config.DATASET_CSV_PATH)
    crop_cols = ["fringe_delta_path", "indent_delta_path"] if crop_source == "delta" else ["fringe_crop_path", "indent_crop_path"]
    required = set(crop_cols)
    if not required.issubset(df.columns):
        hint = "re-run build_dataset.py --save-crops (delta crops are the default now)" if crop_source == "delta" else "re-run build_dataset.py --save-crops --crop-types raw,delta (raw crops are opt-in now, to save disk space)"
        raise SystemExit(f"dataset.csv has no {crop_cols}, {hint}")
    df = df.dropna(subset=crop_cols + force_targets + LOCATION_TARGETS)

    train_df, val_df = _group_split(df, "point_id", test_size, seed)
    logger.info("Train points (%d): %s", train_df["point_id"].nunique(), sorted(train_df["point_id"].unique()))
    logger.info("Val points   (%d): %s", val_df["point_id"].nunique(), sorted(val_df["point_id"].unique()))
    logger.info("Train rows: %d   Val rows: %d", len(train_df), len(val_df))
    logger.info("crop_source=%s  force_targets=%s  force_conditions_location=%s", crop_source, force_targets, force_conditions_location)

    force_mean = train_df[force_targets].values.astype(np.float32).mean(axis=0)
    force_std = train_df[force_targets].values.astype(np.float32).std(axis=0) + 1e-8
    loc_mean = train_df[LOCATION_TARGETS].values.astype(np.float32).mean(axis=0)
    loc_std = train_df[LOCATION_TARGETS].values.astype(np.float32).std(axis=0) + 1e-8

    train_ds = TactileDataset(train_df, force_mean, force_std, loc_mean, loc_std, train=True, force_targets=force_targets, crop_source=crop_source)
    val_ds = TactileDataset(val_df, force_mean, force_std, loc_mean, loc_std, train=False, force_targets=force_targets, crop_source=crop_source)

    device = _resolve_device(device_arg) if device_arg == "auto" else torch.device(device_arg)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # crop size is fixed at 224x224, so this is a safe speedup

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda, persistent_workers=num_workers > 0,
    )

    model = TactileNet(n_force_outputs=len(force_targets), pretrained=pretrained, force_conditions_location=force_conditions_location).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.1fM", n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    use_amp = amp and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info("Mixed precision: %s", "on" if use_amp else "off")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    config.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    all_force_pred = all_force_true = all_loc_pred = all_loc_true = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for fringe, indent, force, loc in train_loader:
            fringe = fringe.to(device, non_blocking=use_cuda)
            indent = indent.to(device, non_blocking=use_cuda)
            force = force.to(device, non_blocking=use_cuda)
            loc = loc.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                force_pred, loc_pred = model(fringe, indent)
                loss = force_weight * loss_fn(force_pred, force) + location_weight * loss_fn(loc_pred, loc)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * fringe.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_force_pred, all_force_true, all_loc_pred, all_loc_true = [], [], [], []
        with torch.no_grad():
            for fringe, indent, force, loc in val_loader:
                fringe = fringe.to(device, non_blocking=use_cuda)
                indent = indent.to(device, non_blocking=use_cuda)
                force = force.to(device, non_blocking=use_cuda)
                loc = loc.to(device, non_blocking=use_cuda)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    force_pred, loc_pred = model(fringe, indent)
                    loss = force_weight * loss_fn(force_pred, force) + location_weight * loss_fn(loc_pred, loc)

                val_loss += loss.item() * fringe.size(0)
                all_force_pred.append(force_pred.float().cpu().numpy())
                all_force_true.append(force.float().cpu().numpy())
                all_loc_pred.append(loc_pred.float().cpu().numpy())
                all_loc_true.append(loc.float().cpu().numpy())
        val_loss /= max(len(val_ds), 1)

        epoch_time_min = (time.time() - epoch_start) / 60
        logger.info(
            "epoch %d/%d done in %.1fmin  train_loss=%.4f  val_loss=%.4f  next epoch ETA ~%.1fmin",
            epoch + 1, epochs, epoch_time_min, train_loss, val_loss, epoch_time_min,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "force_mean": force_mean, "force_std": force_std,
                "loc_mean": loc_mean, "loc_std": loc_std,
                "force_targets": force_targets,
                "crop_source": crop_source,
                "force_conditions_location": force_conditions_location,
                "epoch": epoch + 1,
            }, config.MODELS_ROOT / "resnet_tactile_best.pt")
            logger.info("  -> new best, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Stopping early: val_loss hasn't improved for %d epochs (best was epoch %d).",
                    patience, epoch + 1 - epochs_without_improvement,
                )
                break

    if all_force_pred and len(val_ds):
        force_pred = np.concatenate(all_force_pred) * force_std + force_mean
        force_true = np.concatenate(all_force_true) * force_std + force_mean
        loc_pred = np.concatenate(all_loc_pred) * loc_std + loc_mean
        loc_true = np.concatenate(all_loc_true) * loc_std + loc_mean

        metrics = {
            "force_r2": {t: float(r2_score(force_true[:, i], force_pred[:, i])) for i, t in enumerate(force_targets)},
            "location_r2": {t: float(r2_score(loc_true[:, i], loc_pred[:, i])) for i, t in enumerate(LOCATION_TARGETS)},
        }
        logger.info("Final validation metrics (last epoch, not necessarily the best checkpoint):\n%s", json.dumps(metrics, indent=2))
        with open(config.MODELS_ROOT / "resnet_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info("Best model saved to %s", config.MODELS_ROOT / "resnet_tactile_best.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4, help="Set to 0 if you hit DataLoader/multiprocessing errors on Windows")
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision on GPU (default on, ignored on CPU)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=10,
                         help="Stop early if val_loss hasn't improved for this many epochs (default 10; your last real run's best epoch was 6/30, so this saves real time without much risk)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    parser.add_argument("--crop-source", choices=["raw", "delta"], default="delta",
                         help="delta = baseline-subtracted crops (default, needs build_dataset.py --save-crops re-run); raw = original behaviour")
    parser.add_argument("--force-targets", default="Fz",
                         help="Comma-separated force axes to predict, e.g. 'Fz' (default) or 'Fx,Fy,Fz' for the original 3-axis behaviour")
    parser.add_argument("--force-conditions-location", dest="force_conditions_location", action="store_true", default=True,
                         help="Feed the force prediction into the location head (default on)")
    parser.add_argument("--no-force-conditions-location", dest="force_conditions_location", action="store_false")
    args = parser.parse_args()
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, test_size=args.test_size,
        seed=args.seed, num_workers=args.num_workers, amp=args.amp, device_arg=args.device,
        patience=args.patience, pretrained=args.pretrained,
        crop_source=args.crop_source,
        force_targets=[t.strip() for t in args.force_targets.split(",")],
        force_conditions_location=args.force_conditions_location,
    )
