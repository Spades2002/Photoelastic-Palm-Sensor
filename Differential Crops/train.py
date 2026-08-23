"""
Differential-crop method: two ResNet18 branches (fringe region, indent
region), with the location head CONDITIONED on the force prediction.

    fringe crop -> ResNet18 -> \\
                                  concat -> force_head -> Fz
    indent crop -> ResNet18 -> /       \\
                                         -> [+ force_pred] -> location_head -> x_mm, y_mm

The force prediction is concatenated (detached, so its gradient can't flow
back and corrupt force accuracy) into the location head's input, on the
theory that force correlates with indentation depth, and depth affects how
much of the dimple is visible. Trained end to end with equal-weighted
summed MSE, same loss form as the raw-crop method:

    L = MSE(force_pred, force_true) + MSE(loc_pred, loc_true)

Only Fz is predicted (not Fx/Fy), the weakest, noisiest targets in every
method tried. Input crops are DIFFERENTIAL: each crop is the current frame
minus that point's own unloaded (0mm) baseline frame, shifted to mid-grey
so the sign of the change is preserved. This is the SECOND of the two
methods in this repo; contrast with the raw_crop_method/ folder, which uses
unmodified raw crops and two fully independent heads.

This script is a thin, locked entry point around this folder's own
train_resnet.py (the training engine; an identical copy also lives in
../raw_crop_method/, each folder is fully self-contained and doesn't
depend on the other).

Requires build_dataset.py --save-crops (delta crops are saved by default).

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

    # Locked, not exposed as flags: this is what makes it "the differential
    # method" rather than something a mistyped flag could silently turn
    # into the raw-crop method.
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, test_size=args.test_size,
        seed=args.seed, num_workers=args.num_workers, patience=args.patience, device_arg=args.device,
        crop_source="delta",
        force_targets=["Fz"],
        force_conditions_location=True,
    )
