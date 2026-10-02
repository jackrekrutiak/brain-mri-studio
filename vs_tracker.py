#!/usr/bin/env python3
"""
vs_tracker.py - longitudinal tumour volume tracker (vestibular schwannoma, NF2-related schwannomatosis)
=====================================================================================================

What it does
------------
1. Segments vestibular schwannomas (VS) on contrast-enhanced T1 MRI with a patch-based 3D U-Net.
2. Measures tumour volume per side (left / right - bilateral VS is the hallmark of NF2-related
   schwannomatosis) at every timepoint and tracks change over time.
3. Measures how reliable those volumes are: on held-out patients it compares automated volumes
   with the expert outlines and derives a size-dependent *minimum detectable change* (MDC95).
   Growth is flagged only when a change exceeds BOTH the clinical 20 % threshold AND the MDC for
   that tumour size - the open problem noted in the dataset paper ("for very small tumours, higher
   limits should be employed ... as inter-observer variability can exceed the standard criteria").

Data
----
Vestibular-Schwannoma-MC-RC2 (TCIA, CC BY 4.0, DOI 10.7937/BQ0Z-XA62): 184 patients, 533 annotated
contrast T1 scans over ~5 years. Files: VS_MC_RC2_{id}_{YYYY-MM-DD}_{T1C|T1|T2}[_seg].nii.gz

Commands
--------
    python vs_tracker.py index    --data ~/vs_data                     # find scans + outlines
    python vs_tracker.py prepare  --out data/vs_prep                    # resample + normalise once
    python vs_tracker.py train    --prep data/vs_prep --out runs/vs
    python vs_tracker.py evaluate --prep data/vs_prep --ckpt runs/vs/best_vs.pt   # Dice + MDC analysis
    python vs_tracker.py track    --ckpt runs/vs/best_vs.pt --patient VS_MC_RC2_001 --out tracker_001.html

Research / education code - not a medical device.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from brain_tumour_seg import UNet3D, dice_ce_loss, get_device, set_seed

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

LOG = logging.getLogger("vs_tracker")

SPACING = (0.8, 0.8, 1.0)          # mm, target voxel size after resampling
PATCH = (96, 96, 64)               # training / inference patch (multiples of 16)
GROWTH_THRESHOLD = 0.20            # clinical convention for VS growth (20 % volume increase)
MIN_COMPONENT_MM3 = 20.0           # ignore specks smaller than this
NAME_RE = re.compile(r"(?P<pid>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<seq>T1C|T1CE|T1|T2)(?P<seg>_seg)?\.nii(\.gz)?$", re.I)
SIZE_BINS = [(0.0, 0.5), (0.5, 2.0), (2.0, 1e9)]   # cm^3


# =============================================================================
# 1. INDEX: find scans + outlines and group them by patient and date
# =============================================================================
def parse_name(path: Path) -> Optional[Dict[str, str]]:
    m = NAME_RE.search(path.name)
    if not m:
        return None
    seq = m["seq"].upper().replace("T1CE", "T1C")
    return {"pid": m["pid"], "date": m["date"], "seq": seq, "seg": bool(m["seg"])}


def index(data_dir: Path, out_csv: Path, seq: str = "T1C") -> List[Dict[str, str]]:
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    for p in sorted(data_dir.rglob("*.nii*")):
        if p.name.startswith("._"):
            continue
        info = parse_name(p)
        if not info or info["seq"] != seq:
            continue
        key = (info["pid"], info["date"])
        r = rows.setdefault(key, {"patient": info["pid"], "date": info["date"], "image": "", "label": ""})
        r["label" if info["seg"] else "image"] = str(p.resolve())
    out = [r for r in rows.values() if r["image"]]
    out.sort(key=lambda r: (r["patient"], r["date"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "date", "image", "label"])
        w.writeheader()
        w.writerows(out)
    n_lab = sum(bool(r["label"]) for r in out)
    LOG.info("Indexed %d %s scans from %d patients (%d with expert outlines) -> %s",
             len(out), seq, len({r['patient'] for r in out}), n_lab, out_csv)
    return out


def read_index(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


# =============================================================================
# 2. PREPROCESSING (shared by training and inference)
# =============================================================================
def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    return np.nan_to_num(data), img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def resample(arr: np.ndarray, zooms: Sequence[float], target: Sequence[float] = SPACING, order: int = 1) -> np.ndarray:
    factors = [z / t for z, t in zip(zooms, target)]
    return ndimage.zoom(arr, factors, order=order, mode="nearest", grid_mode=False)


def foreground(arr: np.ndarray) -> np.ndarray:
    thr = 0.1 * np.percentile(arr, 99)
    return arr > thr


def normalise_u8(arr: np.ndarray) -> np.ndarray:
    """z-score over the head, clipped to [-3, 5], stored as uint8 (compact, fast to load)."""
    fg = foreground(arr)
    vals = arr[fg] if fg.sum() > 100 else arr.ravel()
    z = (arr - vals.mean()) / (vals.std() + 1e-6)
    return np.clip((z + 3) / 8 * 255, 0, 255).round().astype(np.uint8)


def u8_to_float(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255 * 8 - 3


def prepare(index_csv: Path, out_dir: Path) -> None:
    rows = [r for r in read_index(index_csv) if r["label"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for i, r in enumerate(rows, 1):
        dst = out_dir / f"{r['patient']}_{r['date']}.npz"
        if not dst.exists():
            img, _, zooms = load_nifti(r["image"])
            lab, _, lzooms = load_nifti(r["label"])
            if lab.shape != img.shape:
                LOG.warning("Shape mismatch image/label for %s %s - skipped", r["patient"], r["date"])
                continue
            x = normalise_u8(resample(img, zooms, order=1))
            y = (resample((lab > 0).astype(np.float32), zooms, order=0) > 0.5).astype(np.uint8)
            np.savez_compressed(dst, image=x, label=y, voxel_mm3=float(np.prod(SPACING)),
                                gt_cm3=float((lab > 0).sum() * np.prod(zooms) / 1000))
        kept.append({"file": dst.name, "patient": r["patient"], "date": r["date"]})
        if i % 25 == 0 or i == len(rows):
            LOG.info("Prepared %d / %d", i, len(rows))
    (out_dir / "prep.json").write_text(json.dumps({"spacing": SPACING, "patch": PATCH, "cases": kept}, indent=2))


# =============================================================================
# 3. PATIENT-LEVEL SPLIT + PATCH DATASET
# =============================================================================
def split_patients(cases: List[Dict], seed: int, val: float = 0.15, test: float = 0.15):
    """All timepoints of one patient stay in the same split (otherwise scores are inflated)."""
    pids = sorted({c["patient"] for c in cases})
    random.Random(seed).shuffle(pids)
    n = len(pids)
    nt, nv = max(1, round(n * test)), max(1, round(n * val))
    te, va = set(pids[:nt]), set(pids[nt:nt + nv])
    pick = lambda s: [c for c in cases if c["patient"] in s]
    return pick(set(pids) - te - va), pick(va), pick(te)


def pad_to(arr: np.ndarray, shape: Sequence[int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    pads = [(max(0, (s - a) // 2), max(0, s - a - max(0, (s - a) // 2))) for a, s in zip(arr.shape, shape)]
    return (np.pad(arr, pads) if any(p != (0, 0) for p in pads) else arr), pads


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, prep: Path, cases: List[Dict], iters: int, pos_frac: float = 0.67):
        self.prep, self.cases, self.iters, self.pos_frac = prep, cases, iters, pos_frac

    def __len__(self) -> int:
        return self.iters

    def __getitem__(self, _):
        c = random.choice(self.cases)
        d = np.load(self.prep / c["file"])
        img, lab = pad_to(d["image"], PATCH)[0], pad_to(d["label"], PATCH)[0]
        if random.random() < self.pos_frac and lab.any():
            pts = np.argwhere(lab)
        else:
            pts = np.argwhere(foreground(img.astype(np.float32)))
            if len(pts) == 0:
                pts = np.argwhere(np.ones_like(lab))
        ctr = pts[random.randrange(len(pts))]
        lo = [int(np.clip(c_ - p // 2 + random.randint(-p // 4, p // 4), 0, s - p))
              for c_, p, s in zip(ctr, PATCH, img.shape)]
        sl = tuple(slice(l, l + p) for l, p in zip(lo, PATCH))
        x = torch.from_numpy(u8_to_float(img[sl]))[None]
        y = torch.from_numpy(lab[sl].astype(np.int64))
        if random.random() < 0.5:                       # left-right flip (VS occurs on either side)
            x, y = x.flip(1), y.flip(0)
        x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return x, y


# =============================================================================
# 4. SLIDING-WINDOW INFERENCE + POST-PROCESSING
# =============================================================================
@torch.no_grad()
def sliding_window(model, vol: np.ndarray, device, batch: int = 4) -> np.ndarray:
    """Tumour probability for a whole volume from overlapping patches (50 % overlap)."""
    padded, pads = pad_to(vol, PATCH)
    shape = padded.shape
    steps = [sorted(set(list(range(0, s - p + 1, p // 2)) + [s - p])) for s, p in zip(shape, PATCH)]
    starts = [(a, b, c) for a in steps[0] for b in steps[1] for c in steps[2]]
    prob = np.zeros(shape, np.float32)
    count = np.zeros(shape, np.float32)
    fg = foreground(padded)
    for i in range(0, len(starts), batch):
        chunk = [s for s in starts[i:i + batch]
                 if fg[s[0]:s[0] + PATCH[0], s[1]:s[1] + PATCH[1], s[2]:s[2] + PATCH[2]].any()]
        if not chunk:
            continue
        x = torch.stack([torch.from_numpy(padded[a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]].copy())
                         for a, b, c in chunk])[:, None].to(device)
        p = model(x).softmax(1)[:, 1]
        p = (p + model(x.flip(2)).softmax(1)[:, 1].flip(1)) / 2          # flip test-time augmentation
        p = p.cpu().numpy()
        for (a, b, c), pp in zip(chunk, p):
            prob[a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]] += pp
            count[a:a + PATCH[0], b:b + PATCH[1], c:c + PATCH[2]] += 1
    prob /= np.maximum(count, 1)
    unpad = tuple(slice(p0, s - p1) for (p0, p1), s in zip(pads, shape))
    return prob[unpad]


def split_sides(mask: np.ndarray, head: np.ndarray, voxel_mm3: float) -> np.ndarray:
    """Keep the (up to) two largest tumours and label them 1 = left, 2 = right (RAS: +x is right)."""
    comp, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, np.uint8)
    sizes = ndimage.sum(mask, comp, range(1, n + 1))
    order = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] * voxel_mm3 >= MIN_COMPONENT_MM3][:2]
    mid_x = np.argwhere(head)[:, 0].mean() if head.any() else mask.shape[0] / 2
    out = np.zeros(mask.shape, np.uint8)
    for k in order:
        cx = np.argwhere(comp == k)[:, 0].mean()
        out[comp == k] = 2 if cx > mid_x else 1
    return out


class VSModel:
    def __init__(self, ckpt: str):
        self.device = get_device()
        c = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model = UNet3D(in_ch=1, n_classes=2, base=c["base"]).to(self.device).eval()
        self.model.load_state_dict(c["state_dict"])
        self.meta = c

    def segment_path(self, path: str) -> Dict:
        """Returns resampled image (uint8), per-side label map, volumes (cm^3) and affine at SPACING."""
        img, affine, zooms = load_nifti(path)
        rs = resample(img, zooms, order=1)
        u8 = normalise_u8(rs)
        prob = sliding_window(self.model, u8_to_float(u8), self.device)
        sides = split_sides(prob > 0.5, foreground(rs), float(np.prod(SPACING)))
        return pack_result(u8, sides, affine, zooms)


def pack_result(u8: np.ndarray, sides: np.ndarray, affine: np.ndarray, zooms: Sequence[float]) -> Dict:
    new_aff = affine.copy()
    new_aff[:3, :3] = affine[:3, :3] @ np.diag([t / z for t, z in zip(SPACING, zooms)])
    v = float(np.prod(SPACING)) / 1000.0
    vols = {"left": round(float((sides == 1).sum() * v), 3), "right": round(float((sides == 2).sum() * v), 3)}
    vols["total"] = round(vols["left"] + vols["right"], 3)
    return {"image": u8, "sides": sides, "affine": new_aff, "volumes": vols}


# =============================================================================
# 5. TRAINING
# =============================================================================
def dice_bin(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


def train(args) -> None:
    set_seed(args.seed)
    prep = Path(args.prep)
    meta = json.loads((prep / "prep.json").read_text())
    tr, va, te = split_patients(meta["cases"], args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va, "test": te}, indent=2))
    LOG.info("Patients: %d train / %d val / %d test  (scans %d / %d / %d)",
             len({c['patient'] for c in tr}), len({c['patient'] for c in va}), len({c['patient'] for c in te}),
             len(tr), len(va), len(te))
    device = get_device()
    model = UNet3D(in_ch=1, n_classes=2, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * args.iters
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    dl = torch.utils.data.DataLoader(PatchDataset(prep, tr, args.iters * args.batch), batch_size=args.batch,
                                     num_workers=args.workers)
    val_cases = va[: args.val_scans]
    best, stale, hist = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = dice_ce_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
            sched.step()
            losses.append(loss.item())
        row = {"epoch": ep, "train_loss": float(np.mean(losses))}
        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            dices = []
            for c in val_cases:
                d = np.load(prep / c["file"])
                prob = sliding_window(model, u8_to_float(d["image"]), device)
                sides = split_sides(prob > 0.5, d["image"] > 0, float(np.prod(SPACING)))
                dices.append(dice_bin(sides > 0, d["label"] > 0))
            row["val_dice"] = float(np.mean(dices))
            if row["val_dice"] > best:
                best, stale = row["val_dice"], 0
                torch.save({"state_dict": model.state_dict(), "base": args.base, "spacing": SPACING, "patch": PATCH,
                            "epoch": ep, "val_dice": best}, out / "best_vs.pt")
                row["saved"] = True
            else:
                stale += 1
        hist.append(row)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        LOG.info("Epoch %03d/%03d  loss %.3f%s | %.0fs", ep, args.epochs, row["train_loss"],
                 f"  val Dice {row['val_dice']:.3f}{'  (saved)' if row.get('saved') else ''}" if "val_dice" in row else "",
                 time.time() - t0)
        if args.patience and stale >= args.patience:
            LOG.info("Early stopping.")
            break


# =============================================================================
# 6. EVALUATION: Dice + volume agreement + minimum detectable change by size
# =============================================================================
def size_bin(cm3: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= cm3 < hi:
            return f"{lo:g}-{hi:g} cm3" if hi < 1e8 else f">{lo:g} cm3"
    return "?"


def evaluate(args) -> Dict:
    prep = Path(args.prep)
    split = json.loads((Path(args.ckpt).parent / "split.json").read_text())
    m = VSModel(args.ckpt)
    rows = []
    for c in split["test"]:
        d = np.load(prep / c["file"])
        prob = sliding_window(m.model, u8_to_float(d["image"]), m.device)
        sides = split_sides(prob > 0.5, d["image"] > 0, float(np.prod(SPACING)))
        v = float(np.prod(SPACING)) / 1000
        gt_cm3, pred_cm3 = float((d["label"] > 0).sum() * v), float((sides > 0).sum() * v)
        rows.append({"patient": c["patient"], "date": c["date"], "dice": dice_bin(sides > 0, d["label"] > 0),
                     "gt_cm3": gt_cm3, "pred_cm3": pred_cm3,
                     "rel_diff": (pred_cm3 - gt_cm3) / gt_cm3 if gt_cm3 > 0 else float("nan")})
    out_dir = Path(args.ckpt).parent
    res = {"n_test_scans": len(rows), "n_test_patients": len({r["patient"] for r in rows}),
           "dice_mean": float(np.mean([r["dice"] for r in rows])),
           "dice_median": float(np.median([r["dice"] for r in rows])), "by_size": {}}
    for lo, hi in SIZE_BINS:
        sel = [r["rel_diff"] for r in rows if lo <= r["gt_cm3"] < hi and np.isfinite(r["rel_diff"])]
        if len(sel) < 3:
            continue
        sd = float(np.std(sel, ddof=1))
        res["by_size"][size_bin(lo)] = {
            "n": len(sel), "bias_pct": round(100 * float(np.mean(sel)), 1),
            "loa95_pct": [round(100 * (np.mean(sel) - 1.96 * sd), 1), round(100 * (np.mean(sel) + 1.96 * sd), 1)],
            # a change between two scans carries the error of both measurements -> sqrt(2)
            "mdc95_pct": round(100 * 1.96 * np.sqrt(2) * sd, 1)}
    (out_dir / "vs_eval.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / "vs_eval_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_agreement(rows, res, out_dir / "vs_volume_agreement.png")
    print(json.dumps(res, indent=2))
    return res


def plot_agreement(rows: List[Dict], res: Dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.info("matplotlib not installed - skipping plot (pip install matplotlib)")
        return
    ok = [r for r in rows if r["gt_cm3"] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    g = np.array([r["gt_cm3"] for r in ok]); p = np.array([r["pred_cm3"] for r in ok])
    ax[0].scatter(g, p, s=14, alpha=.7)
    lim = [min(g.min(), p.min()) * 0.8, max(g.max(), p.max()) * 1.2]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Expert volume (cm³)"); ax[0].set_ylabel("Model volume (cm³)")
    ax[0].set_title(f"Volume agreement · Dice {res['dice_mean']:.3f}")
    ax[1].scatter(g, 100 * (p - g) / g, s=14, alpha=.7); ax[1].axhline(0, color="k", lw=1)
    ax[1].axhline(20, color="tab:red", ls=":", lw=1); ax[1].axhline(-20, color="tab:red", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Expert volume (cm³)"); ax[1].set_ylabel("Model − expert (%)")
    ax[1].set_title("Relative error vs tumour size (dotted = ±20 %)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def growth_threshold(volume_cm3: float, eval_res: Optional[Dict]) -> float:
    """Relative change needed to call growth: max(20 %, MDC95 for this tumour size)."""
    thr = GROWTH_THRESHOLD
    if eval_res:
        b = eval_res.get("by_size", {}).get(size_bin(volume_cm3))
        if b:
            thr = max(thr, b["mdc95_pct"] / 100)
    return thr


# =============================================================================
# 7. LONGITUDINAL TRACKING
# =============================================================================
def build_timeline(results: List[Tuple[str, Dict]], eval_res: Optional[Dict]) -> List[Dict]:
    """results: [(date, pack_result dict)] sorted by date -> timeline rows with change + growth flags."""
    rows, prev = [], None
    base = results[0][1]["volumes"] if results else None
    for date, r in results:
        v = r["volumes"]
        row = {"date": date, **{f"{k}_cm3": v[k] for k in ("left", "right", "total")}, "flags": {}}
        for side in ("left", "right"):
            if prev is None or prev[side] <= 0:
                continue
            ch = (v[side] - prev[side]) / prev[side]
            thr = growth_threshold(prev[side], eval_res)
            row[f"{side}_change_pct"] = round(100 * ch, 1)
            row[f"{side}_threshold_pct"] = round(100 * thr, 1)
            if ch > thr:
                row["flags"][side] = "growth"
            elif ch > GROWTH_THRESHOLD:
                row["flags"][side] = "possible growth (within measurement noise)"
            elif ch < -thr:
                row["flags"][side] = "shrinkage"
        if base:
            for side in ("left", "right"):
                if base[side] > 0:
                    row[f"{side}_vs_baseline_pct"] = round(100 * (v[side] - base[side]) / base[side], 1)
        rows.append(row)
        prev = v
    return rows


def track(args) -> None:
    from mri_studio import build_tracker_html          # viewer lives with the other viewers
    m = VSModel(args.ckpt)
    rows = [r for r in read_index(Path(args.index)) if r["patient"] == args.patient]
    if not rows:
        raise SystemExit(f"No scans for patient {args.patient} in {args.index}")
    eval_res = json.loads(Path(args.eval).read_text()) if args.eval and Path(args.eval).exists() else None
    results = []
    for r in sorted(rows, key=lambda r: r["date"]):
        LOG.info("Segmenting %s %s", r["patient"], r["date"])
        res = m.segment_path(r["image"])
        if args.use_expert and r["label"]:
            lab, _, zooms = load_nifti(r["label"])
            lab_rs = resample((lab > 0).astype(np.float32), zooms, order=0) > 0.5
            res = pack_result(res["image"], split_sides(lab_rs, res["image"] > 0, float(np.prod(SPACING))),
                              load_nifti(r["image"])[1], zooms)
        results.append((r["date"], res))
    timeline = build_timeline(results, eval_res)
    for t in timeline:
        print(t)
    html_doc = build_tracker_html(results, timeline, eval_res, patient=args.patient,
                                  source="expert outlines" if args.use_expert else "model")
    Path(args.out).write_text(html_doc)
    LOG.info("Wrote %s", args.out)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Longitudinal vestibular schwannoma volume tracker.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index"); s.add_argument("--data", required=True); s.add_argument("--out", default="data/vs_index.csv")
    s = sub.add_parser("prepare"); s.add_argument("--index", default="data/vs_index.csv"); s.add_argument("--out", default="data/vs_prep")
    s = sub.add_parser("train")
    s.add_argument("--prep", default="data/vs_prep"); s.add_argument("--out", default="runs/vs")
    s.add_argument("--epochs", type=int, default=50); s.add_argument("--iters", type=int, default=150, help="steps per epoch")
    s.add_argument("--batch", type=int, default=2); s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--base", type=int, default=16); s.add_argument("--val-every", type=int, default=2)
    s.add_argument("--val-scans", type=int, default=30); s.add_argument("--patience", type=int, default=8)
    s.add_argument("--workers", type=int, default=0); s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("evaluate"); s.add_argument("--prep", default="data/vs_prep"); s.add_argument("--ckpt", default="runs/vs/best_vs.pt")
    s = sub.add_parser("track")
    s.add_argument("--ckpt", default="runs/vs/best_vs.pt"); s.add_argument("--index", default="data/vs_index.csv")
    s.add_argument("--patient", required=True); s.add_argument("--eval", default="runs/vs/vs_eval.json")
    s.add_argument("--use-expert", action="store_true", help="use expert outlines instead of the model (for comparison)")
    s.add_argument("--out", default="tracker.html")
    a = ap.parse_args()
    if a.cmd == "index":
        index(Path(a.data).expanduser(), Path(a.out))
    elif a.cmd == "prepare":
        prepare(Path(a.index), Path(a.out))
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "evaluate":
        evaluate(a)
    else:
        track(a)


if __name__ == "__main__":
    main()
