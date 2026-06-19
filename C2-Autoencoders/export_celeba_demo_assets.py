"""
Export small demo assets for the CelebA VAE notebook (run on the server).

The load-only notebook ``C2-4-VAE_CelebA.ipynb`` needs two things to run on
Colab without downloading the full CelebA dataset:

1. a handful of real example faces (for reconstruction / interpolation), and
2. pre-computed latent attribute directions (for attribute arithmetic).

Both are tiny and are committed to the repository. This script produces them
from a local copy of CelebA (images + the ``list_attr_celeba.txt`` label file),
so the heavy data never has to be checked in.

Outputs (written next to this script):
    sample_faces.pt        — dict with "images" (N, 3, 64, 64) float tensor in
                             [0, 1] and "filenames" (list[str]).
    attribute_vectors.pt   — dict mapping attribute name -> latent direction
                             (latent_dim,) float tensor.

Usage:
    python export_celeba_demo_assets.py \
        --images-dir <folder-with-jpgs> \
        --attr-file  list_attr_celeba.txt \
        --weights    models/vae_celeba.pth
"""

from __future__ import annotations

import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image

from VAE_CelebA import VAE

# Attributes whose latent directions are exported. Kept small and visually
# obvious so the demo reads clearly.
DEFAULT_ATTRIBUTES = ["Smiling", "Eyeglasses", "Male", "Blond_Hair"]


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist yet."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_transform() -> T.Compose:
    """Same preprocessing as training: center-crop 178 then resize to 64x64."""
    return T.Compose([
        T.CenterCrop(178),
        T.Resize(64),
        T.ToTensor(),
    ])


def read_attr_file(attr_file: str) -> tuple[list[str], dict[str, int], dict[str, list[int]]]:
    """
    Parse a CelebA attribute file in either of the two common formats.

    Original ``list_attr_celeba.txt`` (space-separated):
        line 1: number of images
        line 2: attribute names
        line 3+: <filename> <+1/-1 per attribute>

    Kaggle ``list_attr_celeba.csv`` (comma-separated):
        line 1: image_id,<attr1>,<attr2>,...
        line 2+: <filename>,<-1/1 per attribute>

    The format is detected from the first line: a CSV header starts with the
    ``image_id`` column, whereas the original file starts with the image count.

    Returns:
        filenames    — ordered list of image filenames
        attr_index   — attribute name -> column index
        labels       — filename -> list of {0, 1} flags (1 == attribute present)
    """
    with open(attr_file, "r") as f:
        first = f.readline()
        is_csv = first.lower().lstrip().startswith("image_id")
        sep = "," if is_csv else None          # None => split on any whitespace

        if is_csv:
            attr_names = [c.strip() for c in first.split(sep)][1:]   # drop "image_id"
        else:
            attr_names = f.readline().split()                       # 2nd line holds names

        attr_index = {name: i for i, name in enumerate(attr_names)}
        filenames: list[str] = []
        labels: dict[str, list[int]] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            fname, values = parts[0], parts[1:]
            filenames.append(fname)
            labels[fname] = [1 if v.strip() == "1" else 0 for v in values]
    return filenames, attr_index, labels


def export_sample_faces(
    images_dir: str,
    filenames: list[str],
    transform: T.Compose,
    n: int,
    out_path: str,
) -> None:
    """Preprocess the first ``n`` available images and save them as a tensor."""
    imgs, used = [], []
    for fname in filenames:
        path = os.path.join(images_dir, fname)
        if not os.path.exists(path):
            continue
        imgs.append(transform(Image.open(path).convert("RGB")))
        used.append(fname)
        if len(imgs) >= n:
            break
    batch = torch.stack(imgs)
    _ensure_parent_dir(out_path)
    torch.save({"images": batch, "filenames": used}, out_path)
    print(f"Saved {len(used)} sample faces to {out_path} ({tuple(batch.shape)}).")


def export_attribute_vectors(
    images_dir: str,
    filenames: list[str],
    attr_index: dict[str, int],
    labels: dict[str, list[int]],
    transform: T.Compose,
    model: VAE,
    device: str,
    attributes: list[str],
    n_images: int,
    out_path: str,
) -> None:
    """
    Estimate and save the latent direction of each requested attribute.

    For each attribute the direction is the difference between the mean latent
    code of images that have it and those that lack it, estimated over up to
    ``n_images`` encoded images.
    """
    # Encode a pool of images once, tracking their attribute labels.
    z_all, label_rows, seen = [], [], 0
    model.eval()
    with torch.no_grad():
        for fname in filenames:
            path = os.path.join(images_dir, fname)
            if not os.path.exists(path):
                continue
            x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            z_all.append(model.encode(x).cpu())
            label_rows.append(labels[fname])
            seen += 1
            if seen >= n_images:
                break
    z = torch.cat(z_all)                       # (seen, latent_dim)
    lab = torch.tensor(label_rows)             # (seen, 40)

    vectors = {}
    for name in attributes:
        col = attr_index[name]
        has = lab[:, col] == 1
        if has.sum() == 0 or (~has).sum() == 0:
            print(f"  skipping {name}: not enough positive/negative samples.")
            continue
        vectors[name] = z[has].mean(0) - z[~has].mean(0)
        print(f"  {name}: norm {vectors[name].norm().item():.3f} "
              f"({has.sum().item()} pos / {(~has).sum().item()} neg)")

    _ensure_parent_dir(out_path)
    torch.save(vectors, out_path)
    print(f"Saved {len(vectors)} attribute vectors to {out_path}.")


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = build_transform()
    filenames, attr_index, labels = read_attr_file(args.attr_file)

    model = VAE(latent_dim=args.latent_dim).to(device)
    model.load_model(path=args.weights, device=device)

    export_sample_faces(
        args.images_dir, filenames, transform, args.n_samples, args.out_samples
    )
    export_attribute_vectors(
        args.images_dir, filenames, attr_index, labels, transform, model, device,
        args.attributes, args.n_attr_images, args.out_vectors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo assets for the CelebA VAE notebook.")
    parser.add_argument("--images-dir", required=True,
                        help="Folder containing the aligned CelebA .jpg images.")
    parser.add_argument("--attr-file", required=True,
                        help="Path to list_attr_celeba.txt.")
    parser.add_argument("--weights", default="models/vae_celeba.pth",
                        help="Path to the trained checkpoint (default: models/vae_celeba.pth).")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="Latent dimensionality (must match the checkpoint, default: 256).")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Number of example faces to export (default: 16).")
    parser.add_argument("--n-attr-images", type=int, default=2000,
                        help="Images to encode when estimating attribute directions (default: 2000).")
    parser.add_argument("--attributes", nargs="+", default=DEFAULT_ATTRIBUTES,
                        help=f"Attributes to export (default: {DEFAULT_ATTRIBUTES}).")
    parser.add_argument("--out-samples", default="celeba_demo_assets/sample_faces.pt",
                        help="Output path for the example faces "
                             "(default: celeba_demo_assets/sample_faces.pt).")
    parser.add_argument("--out-vectors", default="celeba_demo_assets/attribute_vectors.pt",
                        help="Output path for the attribute vectors "
                             "(default: celeba_demo_assets/attribute_vectors.pt).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
