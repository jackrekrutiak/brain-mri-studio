#!/usr/bin/env python3
"""
brain_mri_classifier.py
=======================

Transfer-learning pipeline for classifying brain MRI scans with PyTorch.

Supports
--------
* 3D volumes in NIfTI format (.nii / .nii.gz) via nibabel. Each volume is
  reoriented to RAS, intensity-normalised at the *volume* level, and sampled
  into several 2.5D axial "slabs" (slice i-1, i, i+1 stacked as the 3 input
  channels a 2D ImageNet CNN expects).
* 2D slices exported as PNG / JPEG / BMP / TIFF (8- or 16-bit, grey or RGB).

Expected directory layout (one sub-folder per class; folder name = label):

    data_dir/
        healthy/
            sub-001_T1w.nii.gz
            sub-002_T1w.nii.gz
        tumor/
            sub-101_T1w.nii.gz
            slice_0042.png
            ...

Usage
-----
Train:
    python brain_mri_classifier.py train --data-dir ./mri_data --output-dir ./runs/exp1 \
        --arch resnet50 --epochs 30 --positive-class tumor \
        --subject-regex "(sub-[A-Za-z0-9]+)"

Predict one scan:
    python brain_mri_classifier.py predict --checkpoint ./runs/exp1/best_model.pt \
        --image ./new_scan.nii.gz

IMPORTANT: this is a research tool. Its outputs are not a medical diagnosis and
it has not been validated for clinical use.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

try:
    import nibabel as nib
except ImportError:  # nibabel is only required if NIfTI files are present
    nib = None

LOG = logging.getLogger("brain_mri")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    data_dir: str = "./mri_data"
    output_dir: str = "./runs/exp1"
    arch: str = "resnet50"                 # resnet50 | efficientnet_b0 ... efficientnet_b7
    pretrained: bool = True
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 30                       # total epochs (head-only + fine-tuning)
    head_epochs: int = 5                   # phase 1: frozen backbone, train new head only
    unfreeze: str = "last"                 # phase 2: none | last (last block) | all
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4              # backbone LR in phase 2 (head gets 10x)
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    patience: int = 8                      # early stopping on val loss (0 = off)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4
    amp: bool = True                       # mixed precision on CUDA
    slices_per_volume: int = 5             # 2.5D slabs sampled per NIfTI volume
    slice_range: Tuple[float, float] = (0.35, 0.65)  # fraction of axial extent to sample from
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    positive_class: Optional[str] = None   # class treated as "positive" for sensitivity/specificity
    subject_regex: Optional[str] = None    # regex whose 1st group is the patient ID (prevents leakage)
    hflip: bool = True                     # disable if laterality (left vs right) matters for your label


# =============================================================================
# 2. FILE DISCOVERY & PATIENT-LEVEL STRATIFIED SPLITTING
# =============================================================================
@dataclass
class Record:
    path: str
    label: int
    group: str = ""   # patient / subject ID; all files of a group stay in the same split


def is_nifti(path: Path | str) -> bool:
    name = str(path).lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_supported(path: Path) -> bool:
    return is_nifti(path) or path.suffix.lower() in IMAGE_EXTS


def discover_dataset(data_dir: Path) -> Tuple[List[Record], List[str]]:
    """Scan data_dir/<class_name>/**/<file> and return records + sorted class names."""
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"Need >= 2 class sub-folders in {data_dir}, found {len(class_dirs)}.")
    class_names = [d.name for d in class_dirs]
    records: List[Record] = []
    for label, d in enumerate(class_dirs):
        files = sorted(p for p in d.rglob("*") if p.is_file() and is_supported(p))
        records += [Record(path=str(p), label=label) for p in files]
        LOG.info("Class %-20s -> %d files", f"'{d.name}'", len(files))
    if any(is_nifti(r.path) for r in records) and nib is None:
        raise ImportError("NIfTI files found but nibabel is not installed: pip install nibabel")
    return records, class_names


def assign_groups(records: List[Record], data_dir: Path, subject_regex: Optional[str]) -> None:
    """Attach a patient/subject ID to every record.

    Without a regex every file is its own group. That is fine for one volume per
    patient, but if you exported many 2D slices per patient you MUST supply a
    regex, otherwise near-identical slices of one patient end up in both train
    and test and your metrics will be badly inflated (data leakage).
    """
    if subject_regex:
        pattern = re.compile(subject_regex)
        unmatched = 0
        for r in records:
            rel = Path(r.path).relative_to(data_dir).as_posix()
            m = pattern.search(rel)
            if m:
                r.group = m.group(1) if m.groups() else m.group(0)
            else:
                r.group, unmatched = r.path, unmatched + 1
        if unmatched:
            LOG.warning("%d files did not match --subject-regex; each is treated as its own subject.", unmatched)
    else:
        for r in records:
            r.group = r.path
        if any(not is_nifti(r.path) for r in records):
            LOG.warning("2D images found but no --subject-regex given. If several slices come from the "
                        "same patient, set --subject-regex to avoid train/test leakage.")
    LOG.info("Found %d unique subjects/groups.", len({r.group for r in records}))


def stratified_group_split(records: List[Record], val_frac: float, test_frac: float, seed: int
                           ) -> Tuple[List[Record], List[Record], List[Record]]:
    """Split at the group (patient) level, stratified by each group's majority label."""
    by_group: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_group[r.group][r.label] += 1
    groups = sorted(by_group)
    g_labels = []
    for g in groups:
        (lab, _), *rest = by_group[g].most_common()
        if rest:
            LOG.warning("Subject %s has files in several classes; stratifying on majority label.", g)
        g_labels.append(lab)

    def _split(items, labels, size, rs):
        try:
            return train_test_split(items, labels, test_size=size, stratify=labels, random_state=rs)
        except ValueError:
            LOG.warning("Too few subjects for a stratified split; falling back to a random split.")
            return train_test_split(items, labels, test_size=size, random_state=rs)

    g_trval, g_test, l_trval, _ = _split(groups, g_labels, test_frac, seed)
    g_train, g_val, _, _ = _split(g_trval, l_trval, val_frac / (1.0 - test_frac), seed)

    sets = [set(g_train), set(g_val), set(g_test)]
    return tuple([r for r in records if r.group in s] for s in sets)  # type: ignore[return-value]


def expand_to_items(records: Sequence[Record], slices_per_volume: int, slice_range: Tuple[float, float]
                    ) -> List[Tuple[str, int, Optional[float]]]:
    """Turn records into training items. A NIfTI volume becomes `slices_per_volume` items,
    each pointing at a relative axial position in [slice_range]. 2D images stay 1 item."""
    items = []
    for r in records:
        if is_nifti(r.path):
            for frac in slice_positions(slices_per_volume, slice_range):
                items.append((r.path, r.label, frac))
        else:
            items.append((r.path, r.label, None))
    return items


def slice_positions(k: int, slice_range: Tuple[float, float]) -> List[float]:
    return [0.5] if k <= 1 else np.linspace(slice_range[0], slice_range[1], k).tolist()


# =============================================================================
# 3. LOADING & DETERMINISTIC PREPROCESSING (shared by training and inference)
# =============================================================================
def normalize_intensity(arr: np.ndarray, clip: Tuple[float, float]) -> np.ndarray:
    """Robust percentile clipping + min-max scaling to [0, 1].

    MRI intensities are arbitrary (scanner/sequence dependent), so percentiles are
    computed on foreground voxels (> background minimum) to ignore empty air.
    """
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fg = arr[arr > arr.min()]
    if fg.size < 100:
        fg = arr.ravel()
    lo, hi = np.percentile(fg, clip)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=4)  # small per-worker cache: consecutive slabs of one volume reuse the load
def load_nifti_volume(path: str, clip: Tuple[float, float]) -> np.ndarray:
    """Load a NIfTI volume, reorient to RAS (axis 2 = axial), normalise the whole volume."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4:          # e.g. multi-echo / time series: use first volume
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape} for {path}")
    return normalize_intensity(vol, clip)


def extract_25d_slab(vol: np.ndarray, frac: float) -> np.ndarray:
    """Return a (3, H, W) array of axial slices z-1, z, z+1 at relative depth `frac`."""
    depth = vol.shape[2]
    z = int(round(frac * (depth - 1)))
    idx = np.clip([z - 1, z, z + 1], 0, depth - 1)
    slab = vol[:, :, idx]                      # (X, Y, 3)
    slab = np.rot90(slab, k=1, axes=(0, 1))    # radiological-style display orientation
    return np.ascontiguousarray(slab.transpose(2, 0, 1))


def load_2d_image(path: str, clip: Tuple[float, float]) -> np.ndarray:
    """Load a 2D image (8/16-bit, grey or RGB) as a normalised (3, H, W) array."""
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:                          # RGB(A) export of a greyscale scan -> grey
        arr = arr[..., :3].mean(axis=-1)
    arr = normalize_intensity(arr, clip)
    return np.repeat(arr[None], 3, axis=0)


def pad_and_resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad to square (preserves anatomy aspect ratio) then resize to size x size."""
    _, h, w = x.shape
    s = max(h, w)
    pad = ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2)
    x = F.pad(x, pad, value=0.0)
    return F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[0]


def load_and_preprocess(path: str, slice_frac: Optional[float], img_size: int,
                        clip: Tuple[float, float]) -> torch.Tensor:
    """Single source of truth for preprocessing -> float tensor (3, S, S) in [0, 1]."""
    if is_nifti(path):
        if nib is None:
            raise ImportError("pip install nibabel to read NIfTI files")
        arr = extract_25d_slab(load_nifti_volume(path, tuple(clip)), 0.5 if slice_frac is None else slice_frac)
    else:
        arr = load_2d_image(path, tuple(clip))
    return pad_and_resize(torch.from_numpy(arr), img_size)


# =============================================================================
# 4. MEDICAL-SPECIFIC AUGMENTATION
# =============================================================================
class RandomIntensity(nn.Module):
    """Simulate scanner/protocol variability: gamma, contrast scaling, brightness shift."""

    def __init__(self, gamma=(0.8, 1.25), scale=(0.9, 1.1), shift=(-0.05, 0.05), p=0.8):
        super().__init__()
        self.gamma, self.scale, self.shift, self.p = gamma, scale, shift, p

    @staticmethod
    def _u(lo_hi):
        return float(torch.empty(1).uniform_(*lo_hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x = x.clamp(0, 1) ** self._u(self.gamma)
        return (x * self._u(self.scale) + self._u(self.shift)).clamp(0, 1)


class RandomGaussianNoise(nn.Module):
    """Mild additive noise (MRI thermal noise proxy)."""

    def __init__(self, std=(0.0, 0.03), p=0.3):
        super().__init__()
        self.std, self.p = std, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        sigma = float(torch.empty(1).uniform_(*self.std))
        return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def build_transforms(train: bool, hflip: bool = True) -> T.Compose:
    """Augmentations are deliberately subtle: large rotations/elastic warps produce
    anatomically implausible brains and hurt more than they help."""
    ops: List[nn.Module] = []
    if train:
        ops += [
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))], p=0.8),
            *( [T.RandomHorizontalFlip(p=0.5)] if hflip else [] ),
            RandomIntensity(),
            RandomGaussianNoise(),
        ]
    ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))  # match pretraining statistics
    return T.Compose(ops)


# =============================================================================
# 5. PYTORCH DATASET
# =============================================================================
class BrainMRIDataset(Dataset):
    """Loads NIfTI volumes (as 2.5D slabs) or 2D slices and returns (tensor, label)."""

    def __init__(self, items: List[Tuple[str, int, Optional[float]]], img_size: int,
                 clip: Tuple[float, float], transform: Optional[nn.Module] = None):
        self.items, self.img_size, self.clip, self.transform = items, img_size, tuple(clip), transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, frac = self.items[idx]
        x = load_and_preprocess(path, frac, self.img_size, self.clip)
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(label, dtype=torch.long)


# =============================================================================
# 6. MODEL: PRETRAINED BACKBONE + NEW HEAD (TRANSFER LEARNING)
# =============================================================================
def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Load an ImageNet-pretrained CNN and replace its classifier with a new head.

    ResNet50:        model.fc (2048 -> 1000)          becomes Dropout + Linear(2048 -> num_classes)
    EfficientNet-Bx: model.classifier[1] (C -> 1000)  becomes Dropout + Linear(C -> num_classes)
    The new Linear layer is randomly initialised; everything before it keeps the
    pretrained weights and acts as a generic feature extractor.
    """
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch.startswith("efficientnet_b"):
        weights = models.get_model_weights(arch).DEFAULT if pretrained else None
        model = getattr(models, arch)(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Use resnet50 or efficientnet_b0..b7.")
    return model


def get_head(model: nn.Module, arch: str) -> nn.Module:
    return model.fc if arch == "resnet50" else model.classifier


def get_last_block(model: nn.Module, arch: str) -> nn.Module:
    return model.layer4 if arch == "resnet50" else model.features[-3:]


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Phase 1: disable gradients everywhere except the new head.

    With few scans, training all ~25M ResNet50 weights from the start lets the
    randomly-initialised head send large, noisy gradients into good pretrained
    filters. Training the head alone first is faster and far more stable.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head(model, arch).parameters():
        p.requires_grad = True


def unfreeze_backbone(model: nn.Module, arch: str, mode: str) -> None:
    """Phase 2: let the deepest (most task-specific) block adapt to MRI texture/contrast."""
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "last":
        for p in get_last_block(model, arch).parameters():
            p.requires_grad = True


def keep_frozen_bn_in_eval(model: nn.Module) -> None:
    """requires_grad=False does NOT stop BatchNorm running-stat updates in train mode.
    Put frozen BN layers in eval mode so tiny MRI batches don't corrupt ImageNet statistics."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 7. METRICS
# =============================================================================
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, pos_idx: int) -> Dict[str, float]:
    """Accuracy, sensitivity (recall), specificity, balanced accuracy, ROC-AUC.

    Binary: sensitivity/specificity w.r.t. `pos_idx` (the pathology class).
    Multi-class: macro-averaged one-vs-rest sensitivity and specificity.
    """
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    acc = _safe_div(np.trace(cm), total)

    def ovr(k):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        return _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)

    if num_classes == 2:
        sens, spec = ovr(pos_idx)
        try:
            auc = roc_auc_score((y_true == pos_idx).astype(int), y_prob[:, pos_idx])
        except ValueError:
            auc = float("nan")
    else:
        pairs = [ovr(k) for k in range(num_classes)]
        sens, spec = float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(num_classes)))
        except ValueError:
            auc = float("nan")
    return {"acc": acc, "sens": sens, "spec": spec, "bal_acc": (sens + spec) / 2, "auc": float(auc)}


# =============================================================================
# 8. TRAINING / VALIDATION LOOP
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False
              ) -> Tuple[float, np.ndarray, np.ndarray]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_bn_in_eval(model)
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_probs.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return total_loss / max(n, 1), np.concatenate(all_probs), np.concatenate(all_labels)


def make_optimizer(model, arch, lr_backbone, lr_head, wd):
    """AdamW with a separate (higher) LR for the new head."""
    head_ids = {id(p) for p in get_head(model, arch).parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": lr_head}]
    if body:
        groups.append({"params": body, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=wd)


def save_splits(path: Path, splits: Dict[str, List[Record]], class_names: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label", "class", "subject"])
        for name, recs in splits.items():
            for r in recs:
                w.writerow([name, r.path, r.label, class_names[r.label], r.group])


def train(cfg: Config) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()
    use_amp = cfg.amp and device.type == "cuda"
    LOG.info("Device: %s | AMP: %s", device, use_amp)

    # ---- data --------------------------------------------------------------
    data_dir = Path(cfg.data_dir).resolve()
    records, class_names = discover_dataset(data_dir)
    num_classes = len(class_names)
    pos_idx = class_names.index(cfg.positive_class) if cfg.positive_class else num_classes - 1
    LOG.info("Classes: %s | positive class for sens/spec: '%s'", class_names, class_names[pos_idx])

    assign_groups(records, data_dir, cfg.subject_regex)
    train_r, val_r, test_r = stratified_group_split(records, cfg.val_frac, cfg.test_frac, cfg.seed)
    save_splits(out / "splits.csv", {"train": train_r, "val": val_r, "test": test_r}, class_names)
    for name, rs in (("train", train_r), ("val", val_r), ("test", test_r)):
        dist = Counter(class_names[r.label] for r in rs)
        LOG.info("%-5s: %4d files | %s", name, len(rs), dict(dist))

    items = {k: expand_to_items(v, cfg.slices_per_volume, cfg.slice_range)
             for k, v in (("train", train_r), ("val", val_r), ("test", test_r))}
    datasets = {
        "train": BrainMRIDataset(items["train"], cfg.img_size, cfg.clip_percentiles, build_transforms(True, cfg.hflip)),
        "val": BrainMRIDataset(items["val"], cfg.img_size, cfg.clip_percentiles, build_transforms(False)),
        "test": BrainMRIDataset(items["test"], cfg.img_size, cfg.clip_percentiles, build_transforms(False)),
    }
    pin = device.type == "cuda"
    loaders = {k: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(k == "train"), num_workers=cfg.num_workers,
                             pin_memory=pin, drop_last=(k == "train" and len(ds) > cfg.batch_size),
                             persistent_workers=cfg.num_workers > 0)
               for k, ds in datasets.items()}

    # ---- class-weighted loss (handles class imbalance) ----------------------
    counts = np.bincount([it[1] for it in items["train"]], minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device),
                                    label_smoothing=cfg.label_smoothing)
    LOG.info("Class weights: %s", np.round(weights, 3).tolist())

    # ---- model: phase 1 = frozen backbone ----------------------------------
    model = build_model(cfg.arch, num_classes, cfg.pretrained).to(device)
    freeze_backbone(model, cfg.arch)
    head_epochs = cfg.epochs if cfg.unfreeze == "none" else min(cfg.head_epochs, cfg.epochs)
    optimizer = make_optimizer(model, cfg.arch, cfg.lr_finetune, cfg.lr_head, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(head_epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    LOG.info("Phase 1 (head only): %s trainable params", f"{count_trainable(model):,}")

    meta = {"arch": cfg.arch, "class_names": class_names, "positive_class_idx": pos_idx,
            "img_size": cfg.img_size, "clip_percentiles": list(cfg.clip_percentiles),
            "slices_per_volume": cfg.slices_per_volume, "slice_range": list(cfg.slice_range)}
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    history, best_loss, stale, phase = [], float("inf"), 0, "head"
    hist_path = out / "history.csv"

    for epoch in range(1, cfg.epochs + 1):
        # ---- switch to phase 2: unfreeze and rebuild optimizer/scheduler ----
        if epoch == head_epochs + 1 and cfg.unfreeze != "none":
            unfreeze_backbone(model, cfg.arch, cfg.unfreeze)
            optimizer = make_optimizer(model, cfg.arch, cfg.lr_finetune, cfg.lr_finetune * 10, cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(cfg.epochs - head_epochs, 1), eta_min=1e-7)
            phase, stale = "finetune", 0
            LOG.info("Phase 2 (fine-tune, unfreeze=%s): %s trainable params",
                     cfg.unfreeze, f"{count_trainable(model):,}")

        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, use_amp)
        va_loss, va_p, va_y = run_epoch(model, loaders["val"], criterion, device, use_amp=use_amp)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        tr_m = compute_metrics(tr_y, tr_p, num_classes, pos_idx)
        va_m = compute_metrics(va_y, va_p, num_classes, pos_idx)
        LOG.info("Epoch %03d/%03d [%-8s] train loss %.4f acc %.3f sens %.3f spec %.3f | "
                 "val loss %.4f acc %.3f sens %.3f spec %.3f auc %.3f | lr %.2e | %.0fs",
                 epoch, cfg.epochs, phase, tr_loss, tr_m["acc"], tr_m["sens"], tr_m["spec"],
                 va_loss, va_m["acc"], va_m["sens"], va_m["spec"], va_m["auc"], lr, time.time() - t0)

        row = {"epoch": epoch, "phase": phase, "lr": lr, "train_loss": tr_loss, "val_loss": va_loss,
               **{f"train_{k}": v for k, v in tr_m.items()}, **{f"val_{k}": v for k, v in va_m.items()}}
        history.append(row)
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history)

        # ---- checkpoint on best validation loss + early stopping ------------
        if va_loss < best_loss:
            best_loss, stale = va_loss, 0
            torch.save({"state_dict": model.state_dict(), "meta": meta, "epoch": epoch,
                        "val_metrics": va_m, "val_loss": va_loss}, out / "best_model.pt")
            LOG.info("  -> new best model saved (val loss %.4f)", va_loss)
        else:
            stale += 1
            if cfg.patience and stale >= cfg.patience and phase == ("finetune" if cfg.unfreeze != "none" else "head"):
                LOG.info("Early stopping: no val-loss improvement for %d epochs.", cfg.patience)
                break

    # ---- final, untouched test-set evaluation --------------------------------
    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_p, te_y = run_epoch(model, loaders["test"], criterion, device, use_amp=use_amp)
    te_m = compute_metrics(te_y, te_p, num_classes, pos_idx)
    te_m["loss"] = te_loss
    te_m["best_epoch"] = ckpt["epoch"]
    (out / "test_metrics.json").write_text(json.dumps(te_m, indent=2))
    LOG.info("TEST (slice level, best epoch %d): loss %.4f acc %.3f sens %.3f spec %.3f bal_acc %.3f auc %.3f",
             ckpt["epoch"], te_loss, te_m["acc"], te_m["sens"], te_m["spec"], te_m["bal_acc"], te_m["auc"])
    return out / "best_model.pt"


# =============================================================================
# 9. INFERENCE
# =============================================================================
def load_model_for_inference(checkpoint_path: str | Path, device: Optional[torch.device] = None):
    """Rebuild the architecture from checkpoint metadata and load trained weights."""
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], len(meta["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta, device


@torch.no_grad()
def predict_mri(image_path: str | Path, checkpoint_path: str | Path | None = None,
                bundle: Optional[tuple] = None) -> Dict:
    """Predict class probabilities for ONE new scan.

    Preprocessing is identical to training (same loader, normalisation, resize and
    ImageNet stats, no augmentation). For a NIfTI volume the same set of axial slabs
    used in training is scored and the softmax probabilities are averaged, giving a
    single volume-level prediction.

    Pass `bundle=load_model_for_inference(...)` to reuse a loaded model across calls.
    """
    if bundle is None:
        if checkpoint_path is None:
            raise ValueError("Provide checkpoint_path or a preloaded bundle.")
        bundle = load_model_for_inference(checkpoint_path)
    model, meta, device = bundle
    path = str(image_path)
    tfm = build_transforms(train=False)
    clip = tuple(meta["clip_percentiles"])

    fracs = slice_positions(meta["slices_per_volume"], tuple(meta["slice_range"])) if is_nifti(path) else [None]
    batch = torch.stack([tfm(load_and_preprocess(path, f, meta["img_size"], clip)) for f in fracs]).to(device)
    probs = torch.softmax(model(batch).float(), dim=1).mean(dim=0).cpu().numpy()

    names = meta["class_names"]
    pos = meta["positive_class_idx"]
    result = {
        "image": path,
        "predicted_class": names[int(probs.argmax())],
        "probabilities": {n: round(float(p), 4) for n, p in zip(names, probs)},
        "n_slices_scored": len(fracs),
    }
    if len(names) == 2:
        result["positive_class"] = names[pos]
        result["positive_class_probability"] = round(float(probs[pos]), 4)
    return result


# =============================================================================
# 10. CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brain MRI classification with transfer learning.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a model.")
    d = Config()
    t.add_argument("--data-dir", required=True)
    t.add_argument("--output-dir", default=d.output_dir)
    t.add_argument("--arch", default=d.arch)
    t.add_argument("--no-pretrained", action="store_true", help="Random init (debugging only).")
    t.add_argument("--img-size", type=int, default=d.img_size)
    t.add_argument("--batch-size", type=int, default=d.batch_size)
    t.add_argument("--epochs", type=int, default=d.epochs)
    t.add_argument("--head-epochs", type=int, default=d.head_epochs)
    t.add_argument("--unfreeze", choices=["none", "last", "all"], default=d.unfreeze)
    t.add_argument("--lr-head", type=float, default=d.lr_head)
    t.add_argument("--lr-finetune", type=float, default=d.lr_finetune)
    t.add_argument("--weight-decay", type=float, default=d.weight_decay)
    t.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    t.add_argument("--patience", type=int, default=d.patience)
    t.add_argument("--val-frac", type=float, default=d.val_frac)
    t.add_argument("--test-frac", type=float, default=d.test_frac)
    t.add_argument("--seed", type=int, default=d.seed)
    t.add_argument("--num-workers", type=int, default=d.num_workers)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--slices-per-volume", type=int, default=d.slices_per_volume)
    t.add_argument("--slice-range", type=float, nargs=2, default=list(d.slice_range))
    t.add_argument("--positive-class", default=None)
    t.add_argument("--subject-regex", default=None)
    t.add_argument("--no-hflip", action="store_true")

    pr = sub.add_parser("predict", help="Predict a single scan.")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--image", required=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    if args.cmd == "train":
        cfg = Config(
            data_dir=args.data_dir, output_dir=args.output_dir, arch=args.arch, pretrained=not args.no_pretrained,
            img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs, head_epochs=args.head_epochs,
            unfreeze=args.unfreeze, lr_head=args.lr_head, lr_finetune=args.lr_finetune,
            weight_decay=args.weight_decay, label_smoothing=args.label_smoothing, patience=args.patience,
            val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed, num_workers=args.num_workers,
            amp=not args.no_amp, slices_per_volume=args.slices_per_volume, slice_range=tuple(args.slice_range),
            positive_class=args.positive_class, subject_regex=args.subject_regex, hflip=not args.no_hflip)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        logging.getLogger().addHandler(logging.FileHandler(Path(cfg.output_dir) / "train.log"))
        train(cfg)
    else:
        print(json.dumps(predict_mri(args.image, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
