"""
Export a small set of MVTec demo images for the f-AnoGAN notebook (run on server).

The load-only notebook ``C3-3-AnoGAN.ipynb`` needs a handful of real test images
to demonstrate anomaly scoring without shipping the full ~5 GB MVTec dataset.
This script copies a few **normal** (test/good) and a few **defective**
(test/<defect>) images from one MVTec category, resized to the model resolution,
into a single tensor file.

Expected input layout (standard MVTec-AD), e.g. bottle:

    <data-dir>/test/good/*.png
    <data-dir>/test/<defect>/*.png

Output (written next to this script):
    fanogan_demo.pt — dict with
        "images" : (N, 3, H, W) float tensor in [-1, 1]
        "labels" : list[int]  (0 = normal, 1 = anomalous)
        "names"  : list[str]  (e.g. "good/000", "broken_large/003")

Usage:
    python export_fanogan_demo.py --data-dir bottle --n-good 5 --n-bad 5
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
import torchvision.transforms as T
from PIL import Image

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def list_images(folder: str) -> list[str]:
    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def main(args):
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"No test/ folder under {args.data_dir}.")

    imgs, labels, names = [], [], []

    # Normal images: test/good
    good = list_images(os.path.join(test_dir, "good"))[: args.n_good]
    for p in good:
        imgs.append(transform(Image.open(p).convert("RGB")))
        labels.append(0)
        names.append("good/" + os.path.splitext(os.path.basename(p))[0])

    # Defective images: spread across the defect subfolders.
    defect_dirs = sorted(
        d for d in glob.glob(os.path.join(test_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "good"
    )
    per_defect = max(1, args.n_bad // max(1, len(defect_dirs)))
    taken = 0
    for d in defect_dirs:
        for p in list_images(d)[:per_defect]:
            if taken >= args.n_bad:
                break
            imgs.append(transform(Image.open(p).convert("RGB")))
            labels.append(1)
            names.append(os.path.basename(d) + "/" + os.path.splitext(os.path.basename(p))[0])
            taken += 1

    batch = torch.stack(imgs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"images": batch, "labels": labels, "names": names}, args.out)
    print(f"Saved {len(imgs)} images "
          f"({labels.count(0)} good, {labels.count(1)} defective) to {args.out}.")
    print("names:", names)


def parse_args():
    p = argparse.ArgumentParser(description="Export MVTec demo images for the f-AnoGAN notebook.")
    p.add_argument("--data-dir", required=True, help="MVTec category folder (e.g. 'bottle').")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-good", type=int, default=5, help="Number of normal images (default: 5).")
    p.add_argument("--n-bad", type=int, default=5, help="Number of defective images (default: 5).")
    p.add_argument("--out", default="fanogan_demo_assets/fanogan_demo.pt",
                   help="Output tensor file (default: fanogan_demo_assets/fanogan_demo.pt).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
