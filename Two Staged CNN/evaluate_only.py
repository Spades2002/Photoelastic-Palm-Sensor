r"""
Regenerates predictions/test_predictions.csv from an existing checkpoint,
with no training involved. Use this whenever you want the full test-set
predictions CSV in the current column format without waiting through a
full train.py run.

Same held-out test split as train.py (same seed, same fractions), so this
never evaluates on data the model was trained or tuned on.

Run with:
    python evaluate_only.py
    python evaluate_only.py --checkpoint path\to\some_other_model.pt
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader, random_split

import config
from dataset import TactileSensorDataset
from model import TactileForceNet
from train import evaluate_test_set, save_predictions_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rebuild_test_loader(full_dataset: TactileSensorDataset) -> DataLoader:
    n = len(full_dataset)
    n_val = int(n * config.VAL_SPLIT)
    n_test = int(n * config.TEST_SPLIT)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, _, test_set = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    return DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--out", default=str(config.PREDICTIONS_DIR / "test_predictions.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    full_dataset = TactileSensorDataset()
    test_loader = rebuild_test_loader(full_dataset)
    logger.info("Test split size: %d", len(test_loader.dataset))

    model = TactileForceNet(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d (val_loss %.4f)", checkpoint["epoch"], checkpoint["val_loss"])

    records = evaluate_test_set(model, test_loader, device)
    save_predictions_csv(records, args.out)


if __name__ == "__main__":
    main()
