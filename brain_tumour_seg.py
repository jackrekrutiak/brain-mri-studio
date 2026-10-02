#!/usr/bin/env python3
"""
brain_tumour_seg.py - 3D brain-tumour segmentation (BraTS / Medical Segmentation Decathlon)
=========================================================================================

Trains a 3D U-Net that outlines a brain tumour in a multi-sequence MRI and splits it into
the three regions radiologists use:

    label 1  edema               swelling around the tumour          (FLAIR-bright)
    label 2  non-enhancing core  tumour tissue that does not take up contrast
    label 3  enhancing tumour    active rim that lights up after contrast dye

Standard BraTS evaluation regions, reported as Dice scores (1.0 = perfect overlap with the
expert outline):
    WT  whole tumour     = labels 1+2+3
    TC  tumour core      = labels 2+3
    ET  enhancing tumour = label 3

Data: Medical Segmentation Decathlon Task01_BrainTumour - 484 expert-labelled BraTS cases,
four MRI sequences per case (FLAIR, T1, T1 with contrast, T2) stored as one 4D NIfTI.
License CC-BY-SA 4.0 (Simpson et al. 2019, arXiv:1902.09063; Menze et al. 2015; Bakas et al. 2017).

Commands (run in order):
    python brain_tumour_seg.py selftest
    python brain_tumour_seg.py download --out data
    python brain_tumour_seg.py prepare  --data data/Task01_BrainTumour --out data/prep
    python brain_tumour_seg.py train    --prep data/prep --out runs/seg
    python brain_tumour_seg.py evaluate --prep data/prep --ckpt runs/seg/best_seg.pt
    python brain_tumour_seg.py predict  --ckpt runs/seg/best_seg.pt --image case.nii.gz --out case_mask.nii.gz

Research / education code - not a medical device.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # run any op the Mac GPU lacks on the CPU

import argparse
import hashlib
import json
import logging
import random
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

LOG = logging.getLogger("tumour_seg")

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_MD5 = "240a19d752f0d9e9101544901065d872"
MODALITIES = ["FLAIR", "T1", "T1gd", "T2"]            # channel order in the MSD 4D files
LABELS = {0: "background", 1: "edema", 2: "non-enhancing core", 3: "enhancing tumour"}
REGIONS = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    forced = os.environ.get("SEG_DEVICE")                      # e.g. SEG_DEVICE=cpu
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# =============================================================================
# 1. DOWNLOAD
# =============================================================================
def download(out_dir: Path) -> Path:
    """Download (resumable) and extract Task01_BrainTumour (~7 GB tar)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "Task01_BrainTumour.tar"
    target = out_dir / "Task01_BrainTumour"
    if target.exists() and any((target / "imagesTr").glob("*.nii.gz")):
        LOG.info("Dataset already extracted at %s", target)
        return target

    have = tar_path.stat().st_size if tar_path.exists() else 0
    req = urllib.request.Request(MSD_URL, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + have
            if have and resp.status != 206:          # server ignored Range: restart
                have, mode = 0, "wb"
            else:
                mode = "ab" if have else "wb"
            LOG.info("Downloading %.2f GB to %s (resuming at %.2f GB)", total / 1e9, tar_path, have / 1e9)
            t0, done = time.time(), have
            with open(tar_path, mode) as f:
                while True:
                    buf = resp.read(1 << 22)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total:
                        rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"\r  {done / 1e9:5.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)", end="", flush=True)
            print()
    except urllib.error.HTTPError as e:
        if e.code != 416:                           # 416 = already complete
            raise
    LOG.info("Verifying checksum ...")
    if md5sum(tar_path) != MSD_MD5:
        raise SystemExit("Checksum mismatch - delete the .tar and run download again.")
    LOG.info("Extracting ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(out_dir)
    LOG.info("Done. You can delete %s to free 7 GB.", tar_path)
    return target


# =============================================================================
# 2. PREPROCESSING (shared by training and inference)
# =============================================================================
def load_case(image_path: str, extra: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """Return (C,X,Y,Z) float32 image, affine and spatial shape, reoriented to RAS.

    `image_path` is either one 4D NIfTI in MSD channel order (FLAIR, T1, T1gd, T2), or - with
    `extra` - the FLAIR file plus [T1, T1gd, T2] as three extra 3D files (BraTS naming).
    """
    if nib is None:
        raise ImportError("pip install nibabel")
    paths = [image_path] + list(extra or [])
    vols, affine = [], None
    for p in paths:
        img = nib.as_closest_canonical(nib.load(p))
        affine = img.affine if affine is None else affine
        v = np.asarray(img.dataobj, dtype=np.float32)
        vols.extend([v[..., c] for c in range(v.shape[-1])] if v.ndim == 4 else [v])
    if len(vols) != 4:
        raise ValueError(f"Expected 4 MRI sequences ({', '.join(MODALITIES)}), got {len(vols)}.")
    arr = np.nan_to_num(np.stack(vols))
    return arr, affine, arr.shape[1:]


def brain_bbox(arr: np.ndarray, margin: int = 4) -> Tuple[slice, slice, slice]:
    mask = (arr != 0).any(0)
    if not mask.any():
        return tuple(slice(0, s) for s in arr.shape[1:])  # type: ignore[return-value]
    idx = np.nonzero(mask)
    return tuple(slice(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, s))
                 for i, s in zip(idx, arr.shape[1:]))  # type: ignore[return-value]


def zscore(arr: np.ndarray) -> np.ndarray:
    """Per-channel z-score over brain voxels (MRI intensities have no absolute scale)."""
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        m = arr[c] != 0
        if m.sum() > 100:
            mu, sd = arr[c][m].mean(), arr[c][m].std() + 1e-6
            out[c][m] = (arr[c][m] - mu) / sd
    return out


def resize_img(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(x[None], size=tuple(size), mode="trilinear", align_corners=False)[0]


def resize_lbl(y: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(y[None, None].float(), size=tuple(size), mode="nearest")[0, 0].to(torch.uint8)


def preprocess(arr: np.ndarray, size: Sequence[int]):
    bb = brain_bbox(arr)
    crop = zscore(arr[(slice(None),) + bb])
    x = resize_img(torch.from_numpy(crop), size)
    return x, bb, crop.shape[1:]


def prepare(data_dir: Path, out_dir: Path, size: Sequence[int]) -> None:
    """Crop to the brain, normalise and resample every case once -> small .npz files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted((data_dir / "imagesTr").glob("*.nii.gz"))
    images = [p for p in images if not p.name.startswith("._")]
    if not images:
        raise SystemExit(f"No images in {data_dir / 'imagesTr'}")
    for i, p in enumerate(images, 1):
        dst = out_dir / (p.name.replace(".nii.gz", ".npz"))
        if dst.exists():
            continue
        arr, _, _ = load_case(str(p))
        lbl = np.asarray(nib.as_closest_canonical(nib.load(data_dir / "labelsTr" / p.name)).dataobj).astype(np.uint8)
        x, bb, _ = preprocess(arr, size)
        y = resize_lbl(torch.from_numpy(lbl[bb]), size)
        np.savez_compressed(dst, image=x.numpy().astype(np.float16), label=y.numpy())
        if i % 20 == 0 or i == len(images):
            LOG.info("Prepared %d / %d", i, len(images))
    (out_dir / "prep.json").write_text(json.dumps({"size": list(size), "modalities": MODALITIES,
                                                   "source": str(data_dir)}, indent=2))


# =============================================================================
# 3. DATASET + AUGMENTATION
# =============================================================================
class SegDataset(Dataset):
    def __init__(self, files: List[Path], train: bool):
        self.files, self.train = files, train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        d = np.load(self.files[i])
        x = torch.from_numpy(d["image"].astype(np.float32))
        y = torch.from_numpy(d["label"].astype(np.int64))
        if self.train:
            for ax in (1, 2, 3):                          # random flips on each axis
                if random.random() < 0.5:
                    x, y = x.flip(ax), y.flip(ax - 1)
            scale = torch.empty(x.shape[0], 1, 1, 1).uniform_(0.9, 1.1)
            shift = torch.empty(x.shape[0], 1, 1, 1).uniform_(-0.1, 0.1)
            brain = (x != 0).any(0, keepdim=True)
            x = torch.where(brain, x * scale + shift, x)   # per-sequence intensity jitter
        return x, y


# =============================================================================
# 4. MODEL: 3D U-Net
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Classic encoder-decoder with skip connections (Cicek et al. 2016), InstanceNorm as in nnU-Net."""

    def __init__(self, in_ch: int = 4, n_classes: int = 4, base: int = 16, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(in_ch, chs[0])] + [ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv3d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


# =============================================================================
# 5. LOSS + METRICS
# =============================================================================
def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + soft Dice over the tumour classes (standard for BraTS)."""
    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * oh).sum(dims)
    denom = probs.sum(dims) + oh.sum(dims)
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return ce + (1 - dice[1:].mean())


def region_dice(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    out = {}
    for name, labs in REGIONS.items():
        p, g = np.isin(pred, labs), np.isin(gt, labs)
        s = p.sum() + g.sum()
        out[name] = 1.0 if s == 0 else float(2 * (p & g).sum() / s)
    return out


# =============================================================================
# 6. TRAINING
# =============================================================================
def split_cases(files: List[Path], seed: int, val: float = 0.1, test: float = 0.1):
    files = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    nt, nv = max(1, int(n * test)), max(1, int(n * val))
    return files[nt + nv:], files[nt:nt + nv], files[:nt]


def run_epoch(model, loader, device, opt=None) -> Tuple[float, Dict[str, float]]:
    train = opt is not None
    model.train(train)
    losses, dices = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = dice_ce_loss(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 12)
                opt.step()
            losses.append(loss.item())
            if not train:
                pred = logits.argmax(1).cpu().numpy()
                for p, g in zip(pred, y.cpu().numpy()):
                    dices.append(region_dice(p, g))
    mean = {k: float(np.mean([d[k] for d in dices])) for k in REGIONS} if dices else {}
    return float(np.mean(losses)), mean


def train(args) -> Path:
    set_seed(args.seed)
    prep = Path(args.prep)
    meta = json.loads((prep / "prep.json").read_text())
    files = sorted(prep.glob("*.npz"))
    tr, va, te = split_cases(files, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"train": [f.name for f in tr], "val": [f.name for f in va],
                                                "test": [f.name for f in te]}, indent=2))
    LOG.info("Cases: %d train / %d val / %d test | input size %s", len(tr), len(va), len(te), meta["size"])

    device = get_device()
    model = UNet3D(base=args.base).to(device)
    LOG.info("Device %s | 3D U-Net with %.1fM parameters", device, sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(tr) // args.batch),
                                                pct_start=0.1)
    tl = DataLoader(SegDataset(tr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(SegDataset(va, False), batch_size=1, num_workers=args.workers)

    best, stale, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = dice_ce_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
            sched.step()
            losses.append(loss.item())
        vloss, vd = run_epoch(model, vl, device)
        mean_d = float(np.mean(list(vd.values())))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vloss, **{f"val_{k}": v for k, v in vd.items()}})
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        LOG.info("Epoch %03d/%03d  train loss %.3f | val loss %.3f  Dice WT %.3f  TC %.3f  ET %.3f | %.0fs",
                 ep, args.epochs, np.mean(losses), vloss, vd["WT"], vd["TC"], vd["ET"], time.time() - t0)
        if mean_d > best:
            best, stale = mean_d, 0
            torch.save({"state_dict": model.state_dict(), "base": args.base, "size": meta["size"],
                        "modalities": MODALITIES, "epoch": ep, "val_dice": vd}, out / "best_seg.pt")
            LOG.info("  -> saved best model (mean val Dice %.3f)", best)
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                LOG.info("Early stopping.")
                break
    return out / "best_seg.pt"


# =============================================================================
# 7. INFERENCE
# =============================================================================
def load_seg_model(ckpt: str, device: Optional[torch.device] = None):
    device = device or get_device()
    c = torch.load(ckpt, map_location=device, weights_only=True)
    model = UNet3D(base=c["base"]).to(device).eval()
    model.load_state_dict(c["state_dict"])
    return model, c, device


@torch.no_grad()
def segment(arr: np.ndarray, model, meta: Dict, device, tta: bool = True) -> np.ndarray:
    """(4,X,Y,Z) raw MRI -> (X,Y,Z) uint8 label map in the original grid."""
    x, bb, crop_shape = preprocess(arr, meta["size"])
    xb = x[None].to(device)
    prob = model(xb).softmax(1)
    if tta:                                            # average over left-right flip
        prob = prob + model(xb.flip(2)).softmax(1).flip(2)
    prob = F.interpolate(prob, size=tuple(crop_shape), mode="trilinear", align_corners=False)
    lab = prob.argmax(1)[0].to(torch.uint8).cpu().numpy()
    full = np.zeros(arr.shape[1:], dtype=np.uint8)
    full[bb] = lab
    full[(arr == 0).all(0)] = 0                       # nothing outside the head
    return full


def predict(args) -> None:
    model, meta, device = load_seg_model(args.ckpt)
    arr, affine, _ = load_case(args.image, args.extra)
    mask = segment(arr, model, meta, device)
    nib.save(nib.Nifti1Image(mask, affine), args.out)
    vox = abs(np.linalg.det(affine[:3, :3])) / 1000.0      # cm^3 per voxel
    vols = {LABELS[k]: round(float((mask == k).sum() * vox), 2) for k in (1, 2, 3)}
    print(json.dumps({"mask": args.out, "volumes_cm3": vols}, indent=2))


def evaluate(args) -> None:
    """Dice on the held-out test cases (downsampled grid) + example overlay PNGs."""
    model, meta, device = load_seg_model(args.ckpt)
    split = json.loads((Path(args.ckpt).parent / "split.json").read_text())
    files = [Path(args.prep) / f for f in split["test"]]
    scores = []
    out_dir = Path(args.ckpt).parent
    for i, f in enumerate(files):
        d = np.load(f)
        x = torch.from_numpy(d["image"].astype(np.float32))[None].to(device)
        with torch.no_grad():
            pred = (model(x).softmax(1) + model(x.flip(2)).softmax(1).flip(2)).argmax(1)[0].cpu().numpy()
        s = region_dice(pred, d["label"])
        scores.append(s)
        if i < 4:
            save_overlay_png(d["image"][2].astype(np.float32), d["label"], pred, out_dir / f"example_{i}.png")
    res = {k: {"mean": round(float(np.mean([s[k] for s in scores])), 4),
               "median": round(float(np.median([s[k] for s in scores])), 4)} for k in REGIONS}
    res["n_test_cases"] = len(scores)
    (out_dir / "test_dice.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def save_overlay_png(t1gd: np.ndarray, gt: np.ndarray, pred: np.ndarray, path: Path) -> None:
    from PIL import Image
    z = int(np.argmax((gt > 0).sum((0, 1)))) if gt.any() else gt.shape[2] // 2
    base = t1gd[:, :, z]
    nz = base[base != 0]
    lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if nz.size > 50 else (base.min(), base.max())
    base = np.where(base != 0, np.clip((base - lo) / (hi - lo + 1e-6), 0, 1), 0)
    colors = {1: (255, 214, 10), 2: (48, 120, 255), 3: (255, 59, 48)}
    panels = []
    for lab in (None, gt[:, :, z], pred[:, :, z]):
        rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, 2)
        if lab is not None:
            for k, c in colors.items():
                m = lab == k
                rgb[m] = (0.45 * rgb[m] + 0.55 * np.array(c)).astype(np.uint8)
        panels.append(np.rot90(rgb))
    img = Image.fromarray(np.concatenate(panels, 1))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(path)


def selftest(size=(64, 64, 48), base: int = 16) -> None:
    """Check the model trains on this machine's device and time one step."""
    device = get_device()
    model = UNet3D(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x = torch.randn(2, 4, *size, device=device)
    y = torch.randint(0, 4, (2, *size), device=device)
    for i in range(3):
        t0 = time.time()
        loss = dice_ce_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
    full = dt * (128 * 128 * 96) / (size[0] * size[1] * size[2])
    print(f"OK on {device}: one training step at {size} took {dt:.2f}s "
          f"(roughly {full:.1f}s per step at the default 128x128x96 grid; ~194 steps per epoch).")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="3D brain tumour segmentation (BraTS / MSD Task01).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="check the model runs on this machine and estimate speed")
    d = sub.add_parser("download")
    d.add_argument("--out", default="data")

    p = sub.add_parser("prepare")
    p.add_argument("--data", default="data/Task01_BrainTumour")
    p.add_argument("--out", default="data/prep")
    p.add_argument("--size", type=int, nargs=3, default=[128, 128, 96], help="resampled grid (X Y Z); multiples of 16")

    t = sub.add_parser("train")
    t.add_argument("--prep", default="data/prep")
    t.add_argument("--out", default="runs/seg")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--base", type=int, default=16, help="channels in the first U-Net level")
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("evaluate")
    e.add_argument("--prep", default="data/prep")
    e.add_argument("--ckpt", default="runs/seg/best_seg.pt")

    r = sub.add_parser("predict")
    r.add_argument("--ckpt", default="runs/seg/best_seg.pt")
    r.add_argument("--image", required=True, help="4D NIfTI (FLAIR,T1,T1gd,T2) or the FLAIR file with --extra")
    r.add_argument("--extra", nargs=3, metavar=("T1", "T1GD", "T2"), help="BraTS-style separate files")
    r.add_argument("--out", default="tumour_mask.nii.gz")

    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "download":
        download(Path(a.out))
    elif a.cmd == "prepare":
        if any(s % 16 for s in a.size):
            raise SystemExit("--size values must be multiples of 16")
        prepare(Path(a.data), Path(a.out), a.size)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "evaluate":
        evaluate(a)
    else:
        predict(a)


if __name__ == "__main__":
    main()
