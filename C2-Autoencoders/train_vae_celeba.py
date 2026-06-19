"""
Training script for the CelebA face VAE (RGB 64x64, latent_dim=256).

This is a standalone script meant to be run on a GPU server (or a Colab GPU
runtime) — training on CPU is impractical. It downloads CelebA, trains the VAE
defined in ``VAE_CelebA.py``, and writes the checkpoint to
``models/vae_celeba.pth``.

After training, upload that checkpoint wherever the demo notebook's
``WEIGHTS_URL`` points (e.g. a GitHub release) so the load-only notebook
``C2-4-VAE_CelebA.ipynb`` works for everyone.

Usage:
    python train_vae_celeba.py --epochs 25 --batch-size 128 --lr 1e-3

Note on the CelebA download:
    ``torchvision.datasets.CelebA`` pulls the data from a Google Drive mirror
    that frequently hits a quota limit. If the download fails, retry later or
    place the aligned images manually under ``<data-root>/celeba/``.
"""

from __future__ import annotations

import argparse

import glob

import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA

from VAE_CelebA import VAE, vae_loss

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class FlatImageFolder(Dataset):
    """
    Dataset over a flat folder of images (no class subdirectories required).

    Unlike ``torchvision.datasets.ImageFolder``, this loads images that sit
    directly inside ``root``. If ``root`` itself contains no images but has a
    single subdirectory that does, it transparently descends into it — this
    handles the common ``celeba/img_align_celeba/img_align_celeba/`` nesting.

    Args:
        root      (str):        Folder containing the image files.
        transform (T.Compose):  Preprocessing applied to each PIL image.
    """

    def __init__(self, root: str, transform: T.Compose):
        self.transform = transform
        self.paths = self._collect(root)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root} (or its single subfolder).")
        print(f"Found {len(self.paths)} images under {root}.")

    @staticmethod
    def _collect(root: str) -> list[str]:
        paths: list[str] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(f"{root}/{ext}"))
        if paths:
            return sorted(paths)
        # Descend one level if images are nested in a single subfolder.
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(f"{root}/*/{ext}"))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0   # dummy label (unused during training)


def build_transform() -> T.Compose:
    """Center-crop the aligned 178x218 CelebA images and resize to 64x64."""
    return T.Compose([
        T.CenterCrop(178),
        T.Resize(64),
        T.ToTensor(),          # -> [0, 1], shape (3, 64, 64)
    ])


def build_dataset(args: argparse.Namespace, transform: T.Compose):
    """
    Build the training dataset, preferring a local image folder over download.

    The ``torchvision`` CelebA wrapper downloads from a Google Drive mirror that
    is frequently rate-limited ("quota exceeded"). To avoid that, set
    ``--data-dir`` to a folder of pre-downloaded face images (e.g. the aligned
    images from the Kaggle CelebA mirror). Training only needs the pixels, not
    the attribute labels, so a flat folder of images is sufficient.

    Args:
        args      (argparse.Namespace): Parsed command-line arguments.
        transform (T.Compose):          Image preprocessing pipeline.

    Returns:
        torch.utils.data.Dataset: A dataset yielding (image, label) pairs.
            The label is unused during training.
    """
    if args.data_dir is not None:
        # FlatImageFolder handles both layouts: images directly inside
        # --data-dir, or nested one level deeper in a single subfolder.
        print(f"Loading images from folder: {args.data_dir}")
        return FlatImageFolder(root=args.data_dir, transform=transform)

    print("No --data-dir given; falling back to torchvision CelebA download.")
    return CelebA(
        root=args.data_root,
        split="train",
        target_type="attr",
        download=True,
        transform=transform,
    )


def train(args: argparse.Namespace) -> None:
    """Run the full training loop and save the resulting checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cpu":
        print("WARNING: no GPU detected — training CelebA on CPU is impractical.")

    transform = build_transform()
    train_set = build_dataset(args, transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = running_recon = running_kl = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(imgs)
            loss, recon, kl = vae_loss(imgs, x_hat, mu, log_var, beta=args.beta)
            loss.backward()
            optimizer.step()
            running       += loss.item()
            running_recon += recon.item()
            running_kl    += kl.item()
        n = len(train_loader)
        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"loss {running / n:8.2f} | "
            f"recon {running_recon / n:8.2f} | "
            f"kl {running_kl / n:6.2f}"
        )

    model.save_model(args.out)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the CelebA face VAE.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3).")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="Latent space dimensionality (default: 256).")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight on the KL term (beta-VAE, default: 1.0).")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root for the torchvision CelebA download (default: ./data). "
                             "Only used when --data-dir is not given.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to a local folder of pre-downloaded face images "
                             "(loaded via ImageFolder, no download, no attribute labels). "
                             "Use this to avoid the rate-limited Google Drive mirror.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2).")
    parser.add_argument("--out", type=str, default="models/vae_celeba.pth",
                        help="Output path for the checkpoint (default: models/vae_celeba.pth).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
