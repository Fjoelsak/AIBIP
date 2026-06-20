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


def _score_paths(paths, transform, G, D, E, device):
    """Return the f-AnoGAN anomaly score for each image path."""
    from fanogan import anomaly_score
    scores = []
    for p in paths:
        x = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        s, _ = anomaly_score(x, G, D, E)
        scores.append(s.item())
    return scores


def main(args):
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"No test/ folder under {args.data_dir}.")

    defect_dirs = sorted(
        d for d in glob.glob(os.path.join(test_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "good"
    )
    good_paths = list_images(os.path.join(test_dir, "good"))

    if args.curate:
        # Score every candidate with the trained model and pick the images that
        # separate most cleanly: good images with the LOWEST scores, defective
        # images with the HIGHEST scores. This avoids hard/ambiguous samples.
        import torch as _torch
        from fanogan import Generator, Discriminator, Encoder
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        G = Generator(args.latent_dim, args.channels).to(device)
        D = Discriminator(args.channels).to(device)
        E = Encoder(args.latent_dim, args.channels).to(device)
        G.load_state_dict(_torch.load(os.path.join(args.model_dir, "g.pth"), map_location=device))
        D.load_state_dict(_torch.load(os.path.join(args.model_dir, "d.pth"), map_location=device))
        E.load_state_dict(_torch.load(os.path.join(args.model_dir, "e.pth"), map_location=device))
        G.eval(); D.eval(); E.eval()

        good_scored = sorted(zip(good_paths, _score_paths(good_paths, transform, G, D, E, device)),
                             key=lambda t: t[1])
        chosen_good = [p for p, _ in good_scored[: args.n_good]]

        defect_paths = [p for d in defect_dirs for p in list_images(d)]
        defect_scored = sorted(zip(defect_paths, _score_paths(defect_paths, transform, G, D, E, device)),
                               key=lambda t: t[1], reverse=True)
        chosen_defect = [p for p, _ in defect_scored[: args.n_bad]]
    else:
        chosen_good = good_paths[: args.n_good]
        per_defect = max(1, args.n_bad // max(1, len(defect_dirs)))
        chosen_defect = []
        for d in defect_dirs:
            chosen_defect.extend(list_images(d)[:per_defect])
        chosen_defect = chosen_defect[: args.n_bad]

    imgs, labels, names = [], [], []
    for p in chosen_good:
        imgs.append(transform(Image.open(p).convert("RGB")))
        labels.append(0)
        names.append("good/" + os.path.splitext(os.path.basename(p))[0])
    for p in chosen_defect:
        imgs.append(transform(Image.open(p).convert("RGB")))
        labels.append(1)
        names.append(os.path.basename(os.path.dirname(p)) + "/"
                     + os.path.splitext(os.path.basename(p))[0])

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
    p.add_argument("--curate", action="store_true",
                   help="Score all candidates with the trained model and pick the "
                        "cleanest-separating images (lowest-score good, highest-score "
                        "defective). Requires --model-dir.")
    p.add_argument("--model-dir", default="models/fanogan",
                   help="Folder with g.pth/d.pth/e.pth, used by --curate "
                        "(default: models/fanogan).")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
