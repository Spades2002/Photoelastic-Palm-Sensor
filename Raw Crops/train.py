"""
Raw-crop method: two ResNet18 branches (fringe region, indent region),
TWO INDEPENDENT heads, no conditioning between them.

    fringe crop -> ResNet18 -> \\
                                  concat -> force_head    -> Fx, Fy, Fz
    indent crop -> ResNet18 -> /        -> location_head  -> x_mm, y_mm

Both heads read the same concatenated embedding, but neither head's output
feeds into the other. Trained end to end with equal-weighted summed MSE:

    L = MSE(force_pred, force_true) + MSE(loc_pred, loc_true)

Input crops are the raw camera frames (fringe region and indent/mirror
region), not background-subtracted. This is the FIRST of the two methods
in this repo; contrast with the differential_method/ folder, which
additionally subtracts a per-point baseline frame from each crop and
conditions the location head on the force prediction.

This script is a thin, locked entry point around this folder's own
train_resnet.py (the training engine; an identical copy also lives in
../differential_method/, each folder is fully self-contained and doesn't
depend on the other). It exists so the method this repo describes as "raw
crop, two independent heads" can never accidentally be run with the
wrong settings.

Requires build_dataset.py --save-crops --crop-types raw,delta (raw crops
are opt-in there, since the differential method's delta crops are the
default, this method specifically needs the raw ones).

Usage:
    python train.py --epochs 30 --batch-size 32
"""
from __future__ import annotations

import argparse
import logging

from train_resnet import train

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    # Locked, not exposed as flags: this is what makes it "the raw-crop method"
    # rather than something a mistyped flag could silently turn into the
    # differential method.
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, test_size=args.test_size,
        seed=args.seed, num_workers=args.num_workers, patience=args.patience, device_arg=args.device,
        crop_source="raw",
        force_targets=["Fx", "Fy", "Fz"],
        force_conditions_location=False,
    )
