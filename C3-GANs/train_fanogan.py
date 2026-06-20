"""
Two-phase f-AnoGAN training on a single MVTec-AD category (run on a GPU server).

f-AnoGAN is trained on **defect-free images only**, in two phases:

  Phase 1 — WGAN-GP: train Generator G and Discriminator D with the Wasserstein
            loss and a gradient penalty, so G learns the manifold of normal
            images and D becomes a good feature extractor / critic.
  Phase 2 — Encoder (izi_f): freeze G and D, train the encoder E so that
            G(E(x)) reconstructs x in both image space and discriminator-feature
            space. This gives a single forward-pass mapping x -> z at test time.

Expected data layout (standard MVTec-AD), e.g. for the bottle category:

    <data-dir>/
        train/good/*.png        <- used for training (defect-free)
        test/good/*.png         <- normal test images
        test/<defect>/*.png     <- anomalous test images

Only train/good is used here. Outputs three checkpoints under --out-dir:
    g.pth, d.pth, e.pth

Usage:
    python train_fanogan.py --data-dir bottle --epochs-gan 300 --epochs-enc 150
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fanogan import Generator, Discriminator, Encoder, _save

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


class GoodImages(Dataset):
    """Flat dataset over the defect-free training images of one MVTec category."""

    def __init__(self, data_dir: str, image_size: int = 128):
        good_dir = os.path.join(data_dir, "train", "good")
        root = good_dir if os.path.isdir(good_dir) else data_dir
        self.paths = []
        for ext in IMAGE_EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}.")
        print(f"Found {len(self.paths)} training images under {root}.")
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def gradient_penalty(D: Discriminator, real: torch.Tensor, fake: torch.Tensor,
                     device: str) -> torch.Tensor:
    """WGAN-GP gradient penalty on interpolations between real and fake."""
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(
        outputs=d_inter, inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(args, loader, device):
    """Phase 1: WGAN-GP training of G and D."""
    G = Generator(args.latent_dim, args.channels).to(device)
    D = Discriminator(args.channels).to(device)
    opt_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    for epoch in range(args.epochs_gan):
        for real in loader:
            real = real.to(device)
            b = real.size(0)
            # --- train D (n_critic steps) ---
            for _ in range(args.n_critic):
                z = torch.randn(b, args.latent_dim, device=device)
                fake = G(z).detach()
                gp = gradient_penalty(D, real, fake, device)
                loss_d = D(fake).mean() - D(real).mean() + args.gp_weight * gp
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            # --- train G ---
            z = torch.randn(b, args.latent_dim, device=device)
            loss_g = -D(G(z)).mean()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        print(f"[GAN] epoch {epoch + 1:3d}/{args.epochs_gan} | "
              f"loss_D {loss_d.item():8.3f} | loss_G {loss_g.item():8.3f}")

    _save(G, os.path.join(args.out_dir, "g.pth"))
    _save(D, os.path.join(args.out_dir, "d.pth"))
    return G, D


def train_encoder(args, loader, G, D, device):
    """Phase 2: train E so that G(E(x)) matches x in image and feature space."""
    for p in G.parameters(): p.requires_grad_(False)
    for p in D.parameters(): p.requires_grad_(False)
    G.eval(); D.eval()

    E = Encoder(args.latent_dim, args.channels).to(device)
    opt_e = optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.999))
    mse = nn.MSELoss()

    for epoch in range(args.epochs_enc):
        for real in loader:
            real = real.to(device)
            z = E(real)
            rec = G(z)
            # image-space + discriminator-feature-space reconstruction (izi_f)
            loss_img = mse(rec, real)
            loss_feat = mse(D.features(rec), D.features(real))
            loss_e = loss_img + args.kappa * loss_feat
            opt_e.zero_grad(); loss_e.backward(); opt_e.step()
        print(f"[ENC] epoch {epoch + 1:3d}/{args.epochs_enc} | "
              f"loss_E {loss_e.item():.5f} (img {loss_img.item():.5f}, "
              f"feat {loss_feat.item():.5f})")

    _save(E, os.path.join(args.out_dir, "e.pth"))
    return E


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cpu":
        print("WARNING: no GPU detected — f-AnoGAN training on CPU is impractical.")

    dataset = GoodImages(args.data_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"),
                        drop_last=True)

    G, D = train_gan(args, loader, device)
    train_encoder(args, loader, G, D, device)
    print("Done. Checkpoints written to", args.out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Train f-AnoGAN on one MVTec category.")
    p.add_argument("--data-dir", required=True,
                   help="Category folder (expects train/good/ inside, e.g. 'bottle').")
    p.add_argument("--out-dir", default="models/fanogan",
                   help="Where to write g.pth, d.pth, e.pth (default: models/fanogan).")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-gan", type=int, default=300,
                   help="WGAN-GP epochs (default: 300).")
    p.add_argument("--epochs-enc", type=int, default=150,
                   help="Encoder epochs (default: 150).")
    p.add_argument("--n-critic", type=int, default=5,
                   help="Discriminator updates per generator update (default: 5).")
    p.add_argument("--gp-weight", type=float, default=10.0)
    p.add_argument("--kappa", type=float, default=1.0,
                   help="Weight of the feature-residual term (default: 1.0).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
