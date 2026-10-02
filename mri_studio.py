#!/usr/bin/env python3
"""
mri_studio.py - interactive, explainable front end for brain_mri_classifier.py
=============================================================================

Three tabs in one local web app:

1. Explain a slice   - upload a 2D slice (PNG/JPG) or a NIfTI volume; see class
                       probabilities plus a Grad-CAM heatmap of where the model looked.
2. 3D brain viewer   - upload a 3D NIfTI volume; the model scores every axial slice,
                       builds a 3D heatmap and opens it in an interactive NiiVue viewer
                       (rotate, slice, cut away, jump to hotspot). Exportable as a
                       single standalone HTML file you can open anywhere.
3. Patient view      - the same viewer in a guided, plain-language walkthrough mode
                       for explaining a finding to a patient.

Run:
    python mri_studio.py --checkpoint ./runs/kaggle/best_model.pt
then open http://127.0.0.1:7860 in your browser.

Requirements (in addition to brain_mri_classifier.py's):
    pip install gradio grad-cam
and the file  static/niivue.umd.js  next to this script (shipped with it).

RESEARCH / EDUCATION PROTOTYPE - NOT A MEDICAL DEVICE. Outputs are not a diagnosis.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from brain_mri_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    is_nifti,
    load_2d_image,
    load_model_for_inference,
    nib,
    normalize_intensity,
    pad_and_resize,
)

HERE = Path(__file__).resolve().parent
NIIVUE_JS = HERE / "static" / "niivue.umd.js"
VIEWER_TEMPLATE = HERE / "static" / "viewer_template.html"

DISCLAIMER = ("Research and education prototype - not a medical device and not a diagnosis. "
              "Only a qualified clinician can interpret a scan.")

# Plain-language descriptions used by the patient view. Deliberately general and cautious.
CLASS_INFO: Dict[str, Dict[str, str]] = {
    "glioma": {
        "title": "Glioma",
        "plain": "A glioma is a growth that starts in the glial cells - the cells that support and "
                 "protect nerve cells in the brain. Gliomas range from slow-growing to fast-growing. "
                 "The exact type and grade can only be confirmed by specialists, usually with a biopsy.",
        "next": "Doctors typically review the full scan, may order more imaging, and discuss whether a "
                "biopsy or treatment such as surgery, radiation or medication is appropriate.",
    },
    "meningioma": {
        "title": "Meningioma",
        "plain": "A meningioma grows from the meninges - the thin layers of tissue that wrap around the "
                 "brain, like a protective lining. Most meningiomas are benign (not cancer) and grow slowly.",
        "next": "Small ones are often just watched with repeat scans. Larger ones or those causing "
                "symptoms may be treated with surgery or focused radiation.",
    },
    "pituitary": {
        "title": "Pituitary tumour",
        "plain": "The pituitary is a pea-sized gland at the base of the brain that controls many hormones. "
                 "Most pituitary tumours are benign (not cancer). Depending on size and type they can "
                 "affect hormone levels or press on the nerves for vision.",
        "next": "Doctors often check hormone levels and vision. Treatment, if needed, can include "
                "medication, surgery (often through the nose) or radiation.",
    },
    "notumor": {
        "title": "No tumour pattern found",
        "plain": "The model did not find the patterns of the three tumour types it was trained to "
                 "recognise. That does not rule out other conditions - it only knows these four categories.",
        "next": "Your doctor's reading of the full scan, together with your symptoms and history, is "
                "what matters.",
    },
}
QUESTIONS = [
    "What exactly does the scan show, and how sure are you?",
    "Do I need more tests or scans to confirm this?",
    "What are my treatment options, and what are the risks of each?",
    "What happens if we wait and watch?",
    "How will this affect my daily life, work or driving?",
    "Who will be part of my care team, and who do I call with questions?",
]


def info_for(name: str) -> Dict[str, str]:
    key = name.lower().replace(" ", "").replace("_", "")
    for k, v in CLASS_INFO.items():
        if k.replace("_", "") == key or key.startswith(k):
            return v
    return {"title": name, "plain": f"The model's top category was '{name}'.",
            "next": "Discuss the result with your doctor."}


def is_normal_class(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in ("notumor", "no_tumor", "normal", "healthy"))


# =============================================================================
# Model + Grad-CAM
# =============================================================================
class Engine:
    """Holds the trained model and produces predictions + Grad-CAM maps."""

    def __init__(self, checkpoint: str):
        from pytorch_grad_cam import GradCAM

        self.model, self.meta, self.device = load_model_for_inference(checkpoint)
        self.names: List[str] = self.meta["class_names"]
        self.size: int = self.meta["img_size"]
        self.clip = tuple(self.meta["clip_percentiles"])
        self.norm = build_transforms(train=False)
        arch = self.meta["arch"]
        layer = self.model.layer4[-1] if arch == "resnet50" else self.model.features[-1]
        self.cam = GradCAM(model=self.model, target_layers=[layer])
        normal = [i for i, n in enumerate(self.names) if is_normal_class(n)]
        self.normal_idx: Optional[int] = normal[0] if normal else None

    # ---- core batch inference ------------------------------------------------
    def run(self, x01: torch.Tensor, target: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """x01: (B,3,S,S) in [0,1]. Returns probs (B,C) and CAM maps (B,S,S) in [0,1]."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        x = torch.stack([self.norm(t) for t in x01]).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x).float(), dim=1).cpu().numpy()
        tgt = probs.argmax(1) if target is None else np.full(len(probs), target)
        cams = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(int(t)) for t in tgt])
        return probs, cams

    # ---- 2D ------------------------------------------------------------------
    def explain_2d(self, path: str):
        if is_nifti(path):
            vol = self._load_volume(path)[0]
            z = vol.shape[2] // 2
            arr = normalize_intensity(np.rot90(vol[:, :, z]), self.clip)
            img = np.repeat(arr[None], 3, 0)
        else:
            img = load_2d_image(path, self.clip)
        x01 = pad_and_resize(torch.from_numpy(np.ascontiguousarray(img)), self.size)
        probs, cams = self.run(x01[None])
        base = x01.permute(1, 2, 0).numpy()
        return probs[0], base, cams[0]

    # ---- 3D ------------------------------------------------------------------
    def _load_volume(self, path: str):
        if nib is None:
            raise RuntimeError("pip install nibabel to read NIfTI files")
        img = nib.as_closest_canonical(nib.load(path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[..., 0]
        return np.nan_to_num(vol), img.affine

    def explain_3d(self, path: str, progress=None, batch: int = 16):
        """Score every axial slice, return data for the 3D viewer."""
        vol, affine = self._load_volume(path)
        X, Y, Z = vol.shape
        vol01 = normalize_intensity(vol, self.clip)          # for display
        thr = np.percentile(vol01, 50) if vol01.max() > 0 else 0
        brain_frac = [(vol01[:, :, z] > max(thr, 0.05)).mean() for z in range(Z)]
        zs = [z for z in range(Z) if brain_frac[z] > 0.05]    # skip empty slices
        if not zs:
            raise ValueError("Volume looks empty.")

        def slice_tensor(z):
            s = np.rot90(vol[:, :, z])                         # same orientation as training
            s = normalize_intensity(s, self.clip)              # per-slice, like 2D training images
            return pad_and_resize(torch.from_numpy(np.repeat(s[None], 3, 0).copy()), self.size)

        # pass 1: probabilities for every slice
        probs = np.full((Z, len(self.names)), np.nan, dtype=np.float32)
        for i in range(0, len(zs), batch):
            chunk = zs[i:i + batch]
            x = torch.stack([self.norm(slice_tensor(z)) for z in chunk]).to(self.device)
            with torch.no_grad():
                probs[chunk] = torch.softmax(self.model(x).float(), 1).cpu().numpy()
            if progress:
                progress(0.05 + 0.4 * (i + len(chunk)) / len(zs), desc="Scoring slices")

        valid = probs[zs]
        if self.normal_idx is not None:
            abn = 1.0 - valid[:, self.normal_idx]
            tumour_cls = [c for c in range(len(self.names)) if c != self.normal_idx]
            # top finding = class with highest mean probability over the 5 most suspicious slices
            top_slices = np.argsort(abn)[-5:]
            cls_scores = valid[top_slices][:, tumour_cls].mean(0)
            target = tumour_cls[int(cls_scores.argmax())]
            peak = float(np.sort(abn)[-5:].mean())
            finding = self.names[target] if peak >= 0.5 else self.names[self.normal_idx]
        else:
            abn = valid.max(1)
            target = int(valid.mean(0).argmax())
            finding, peak = self.names[target], float(valid[:, target].max())

        # volume-level summary: mean over the 5 most suspicious slices
        top_slices = np.argsort(abn)[-5:]
        summary = valid[top_slices].mean(0)

        # pass 2: Grad-CAM for the target class, weighted by that slice's probability
        heat = np.zeros((X, Y, Z), dtype=np.float32)
        for i in range(0, len(zs), batch):
            chunk = zs[i:i + batch]
            x01 = torch.stack([slice_tensor(z) for z in chunk])
            p, cams = self.run(x01, target=target)
            for z, cam, pz in zip(chunk, cams, p[:, target]):
                heat[:, :, z] = self._cam_to_slice(cam, X, Y) * float(pz)
            if progress:
                progress(0.45 + 0.5 * (i + len(chunk)) / len(zs), desc="Building 3D heatmap")
        from scipy.ndimage import gaussian_filter
        heat = gaussian_filter(heat, sigma=(1.5, 1.5, 2.5))    # smooth slice-to-slice jumps
        heat *= (vol01 > 0.02)                                 # keep inside the head
        if heat.max() > 0:
            heat /= heat.max()
        hotspot = [int(v) for v in np.unravel_index(int(heat.argmax()), heat.shape)]

        curve = [None if np.isnan(probs[z, 0]) else round(float(
            1 - probs[z, self.normal_idx] if self.normal_idx is not None else probs[z].max()), 4)
            for z in range(Z)]
        return {
            "vol01": vol01, "heat": heat, "affine": affine, "hotspot": hotspot,
            "finding": finding, "target": self.names[target], "peak": peak,
            "summary": {n: round(float(v), 4) for n, v in zip(self.names, summary)},
            "curve": curve, "shape": [X, Y, Z],
        }

    def _cam_to_slice(self, cam: np.ndarray, X: int, Y: int) -> np.ndarray:
        """Undo pad_and_resize + rot90 so a CAM lines up with the original (X, Y) slice."""
        h, w = Y, X                                            # rot90 of (X,Y) is (Y,X)
        s = max(h, w)
        up = F.interpolate(torch.from_numpy(cam)[None, None], size=(s, s), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
        top, left = (s - h) // 2, (s - w) // 2
        crop = up[top:top + h, left:left + w]
        return np.rot90(crop, k=-1)                            # back to (X, Y)


# =============================================================================
# Viewer HTML (standalone, NiiVue inlined)
# =============================================================================
def _nifti_b64(data: np.ndarray, affine: np.ndarray) -> str:
    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    img.header.set_data_dtype(data.dtype)
    raw = img.to_bytes()
    return base64.b64encode(gzip.compress(raw, compresslevel=6)).decode()


def build_viewer_html(res: Dict, mode: str = "clinician", source_name: str = "scan") -> str:
    vol8 = (res["vol01"] * 255).astype(np.uint8)
    heat8 = (res["heat"] * 255).astype(np.uint8)
    info = info_for(res["finding"])
    payload = {
        "mode": mode,
        "source": source_name,
        "volume": _nifti_b64(vol8, res["affine"]),
        "heat": _nifti_b64(heat8, res["affine"]),
        "hotspot": res["hotspot"],
        "calMin": int(np.percentile(heat8[heat8 > 0], 92)) if (heat8 > 0).any() else 60,
        "shape": res["shape"],
        "finding": res["finding"],
        "findingTitle": info["title"],
        "normal": is_normal_class(res["finding"]),
        "target": res["target"],
        "peak": res["peak"],
        "summary": res["summary"],
        "curve": res["curve"],
        "info": info,
        "questions": QUESTIONS,
        "disclaimer": DISCLAIMER,
    }
    tpl = VIEWER_TEMPLATE.read_text()
    js = NIIVUE_JS.read_text()
    return (tpl.replace("/*__NIIVUE__*/", js.replace("</script>", "<\\/script>"))
               .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))


def iframe(doc: str, height: int = 820) -> str:
    return (f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
            f'style="width:100%;height:{height}px;border:0;border-radius:12px;background:#0b0f14" '
            f'allow="fullscreen; xr-spatial-tracking"></iframe>')


# =============================================================================
# 2D overlay helpers
# =============================================================================
def overlay_image(base: np.ndarray, cam: np.ndarray) -> np.ndarray:
    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(np.clip(base, 0, 1).astype(np.float32), cam, use_rgb=True, image_weight=0.55)


def side_by_side(base: np.ndarray, over: np.ndarray) -> Image.Image:
    a = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    b = Image.fromarray(over)
    out = Image.new("RGB", (a.width * 2 + 8, a.height), (11, 15, 20))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 8, 0))
    return out.resize((out.width * 2, out.height * 2), Image.BICUBIC)


def patient_markdown(name: str, p: float) -> str:
    info = info_for(name)
    return (f"### What the model found: {info['title']}  ({p:.0%} confidence)\n\n"
            f"**In plain language:** {info['plain']}\n\n"
            f"**What usually happens next:** {info['next']}\n\n"
            f"_{DISCLAIMER}_")


# =============================================================================
# Tumour segmentation (brain_tumour_seg.py model)
# =============================================================================
SEG_TEMPLATE = HERE / "static" / "seg_viewer_template.html"


def describe_location(seg: np.ndarray, brain: np.ndarray, affine: np.ndarray) -> Tuple[str, List[int]]:
    """Plain-language position of the tumour relative to the brain's centre (RAS space)."""
    tum = np.argwhere(seg > 0)
    if len(tum) == 0:
        return "", [s // 2 for s in seg.shape]
    c_vox = tum.mean(0)
    to_mm = lambda v: affine[:3, :3] @ v + affine[:3, 3]
    d = to_mm(c_vox) - to_mm(np.argwhere(brain).mean(0))
    side = "right" if d[0] > 0 else "left"
    fb = "towards the front" if d[1] > 15 else "towards the back" if d[1] < -15 else "near the middle, front to back"
    ud = "upper part" if d[2] > 15 else "lower part" if d[2] < -15 else "middle height"
    text = f"It sits mainly on the {side} side of the brain, {fb}, in the {ud}."
    return text, [int(round(v)) for v in c_vox]


class SegEngine:
    def __init__(self, checkpoint: str):
        from brain_tumour_seg import load_seg_model
        self.model, self.meta, self.device = load_seg_model(checkpoint)

    def analyse(self, image: str, gt: Optional[str] = None, extra: Optional[List[str]] = None) -> Dict:
        from brain_tumour_seg import LABELS, load_case, region_dice, segment
        arr, affine, _ = load_case(image, extra)
        seg = segment(arr, self.model, self.meta, self.device)
        vox_cm3 = abs(np.linalg.det(affine[:3, :3])) / 1000.0
        vols = {k: round(float((seg == k).sum() * vox_cm3), 2) for k in (1, 2, 3)}
        vols["WT"] = round(sum(vols[k] for k in (1, 2, 3)), 2)
        brain = (arr != 0).any(0)
        location, centroid = describe_location(seg, brain, affine)
        t1gd = arr[2]
        nz = t1gd[brain]
        lo, hi = (np.percentile(nz, 0.5), np.percentile(nz, 99.5)) if nz.size else (0, 1)
        base = np.where(brain, np.clip((t1gd - lo) / (hi - lo + 1e-6), 0, 1), 0)
        gt_arr, dice = None, None
        if gt:
            gt_arr = np.asarray(nib.as_closest_canonical(nib.load(gt)).dataobj).astype(np.uint8)
            if gt_arr.shape == seg.shape:
                dice = region_dice(seg, gt_arr)
            else:
                gt_arr = None
        return {"base": base, "seg": seg, "gt": gt_arr, "affine": affine, "volumes": vols,
                "location": location, "centroid": centroid, "dice": dice, "labels": LABELS}


def build_seg_viewer_html(res: Dict, mode: str = "clinician", source_name: str = "scan") -> str:
    payload = {
        "mode": mode, "source": source_name, "disclaimer": DISCLAIMER, "questions": QUESTIONS,
        "volume": _nifti_b64((res["base"] * 255).astype(np.uint8), res["affine"]),
        "seg": _nifti_b64(res["seg"].astype(np.uint8), res["affine"]),
        "gt": _nifti_b64((res["gt"] > 0).astype(np.uint8), res["affine"]) if res["gt"] is not None else None,
        "volumes": {str(k): v for k, v in res["volumes"].items()} | {"WT": res["volumes"]["WT"]},
        "location": res["location"], "centroid": res["centroid"], "dice": res["dice"],
    }
    tpl = SEG_TEMPLATE.read_text()
    js = NIIVUE_JS.read_text()
    return (tpl.replace("/*__NIIVUE__*/", js.replace("</script>", "<\\/script>"))
               .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))


# =============================================================================
# Longitudinal growth tracker (vs_tracker.py)
# =============================================================================
TRACKER_TEMPLATE = HERE / "static" / "tracker_template.html"
DISPLAY_MM = 1.2          # display resolution for embedded scans (keeps the HTML small)


def _display_pair(image_u8: np.ndarray, sides: np.ndarray, affine: np.ndarray, spacing: Sequence[float]):
    from scipy import ndimage
    f = [s / DISPLAY_MM for s in spacing]
    img = ndimage.zoom(image_u8, f, order=1)
    seg = ndimage.zoom(sides, f, order=0)
    aff = affine.copy()
    aff[:3, :3] = affine[:3, :3] @ np.diag([1 / x for x in f])
    pts = np.argwhere(seg > 0)
    centroid = [int(v) for v in (pts.mean(0) if len(pts) else np.array(seg.shape) / 2)]
    return img.astype(np.uint8), seg.astype(np.uint8), aff, centroid


def build_tracker_html(results, timeline, eval_res, patient: str, source: str = "model", mode: str = "clinician") -> str:
    from vs_tracker import SPACING
    tps = []
    for (date, r), row in zip(results, timeline):
        img, seg, aff, cen = _display_pair(r["image"], r["sides"], r["affine"], SPACING)
        tps.append({"date": date, "image": _nifti_b64(img, aff), "seg": _nifti_b64(seg, aff), "centroid": cen,
                    "volumes": r["volumes"], "flags": row["flags"],
                    **{k: row[k] for k in row if k.endswith("_pct")}})
    payload = {"mode": mode, "patient": patient, "source": source, "timepoints": tps, "eval": eval_res,
               "questions": [
                   "Has the tumour grown, and by how much compared with normal measurement differences?",
                   "How often should I have scans from now on?",
                   "What would make us change from watching to treating?",
                   "What are the treatment options and their effects on hearing and balance?",
                   "Are there clinical trials I could join?"],
               "disclaimer": DISCLAIMER}
    tpl = TRACKER_TEMPLATE.read_text()
    js = NIIVUE_JS.read_text()
    return (tpl.replace("/*__NIIVUE__*/", js.replace("</script>", "<\\/script>"))
               .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))


def track_uploaded(files: List[str], vs_model, eval_res, use_expert: bool, mode: str):
    """Group uploaded NIfTIs by date (from VS_MC_RC2-style names) and build the tracker."""
    from vs_tracker import (SPACING, build_timeline, foreground, load_nifti, normalise_u8, pack_result,
                            parse_name, resample, split_sides)
    by_date: Dict[str, Dict[str, str]] = {}
    patient = "uploaded"
    for f in files:
        info = parse_name(Path(f))
        if not info or info["seq"] != "T1C":
            continue
        patient = info["pid"]
        by_date.setdefault(info["date"], {})["label" if info["seg"] else "image"] = f
    dates = sorted(d for d, v in by_date.items() if "image" in v)
    if not dates:
        raise ValueError("No contrast T1 scans found. Files must be named like "
                         "VS_MC_RC2_001_2019-04-28_T1C.nii.gz (outlines: ..._T1C_seg.nii.gz).")
    results = []
    for d in dates:
        v = by_date[d]
        if use_expert and "label" in v:
            img, aff, zooms = load_nifti(v["image"])
            rs = resample(img, zooms)
            lab, _, _ = load_nifti(v["label"])
            sides = split_sides(resample((lab > 0).astype(np.float32), zooms, order=0) > 0.5, foreground(rs),
                                float(np.prod(SPACING)))
            results.append((d, pack_result(normalise_u8(rs), sides, aff, zooms)))
        elif vs_model is not None:
            results.append((d, vs_model.segment_path(v["image"])))
        else:
            raise ValueError(f"No outline for {d} and no tracker model loaded (--vs-checkpoint).")
    timeline = build_timeline(results, eval_res)
    src = "expert outlines" if use_expert else "model"
    return build_tracker_html(results, timeline, eval_res, patient=patient, source=src, mode=mode), timeline, patient


# =============================================================================
# Gradio app
# =============================================================================
def make_app(engine: Optional[Engine], seg_engine: Optional[SegEngine] = None, vs_model=None, vs_eval=None):
    import gradio as gr

    out_dir = Path(tempfile.mkdtemp(prefix="mri_studio_"))

    def do_2d(file):
        if file is None:
            raise gr.Error("Upload a PNG/JPG slice or a NIfTI volume first.")
        names = engine.names
        probs, base, cam = engine.explain_2d(file)
        k = int(probs.argmax())
        label = {n: float(p) for n, p in zip(names, probs)}
        return label, side_by_side(base, overlay_image(base, cam)), patient_markdown(names[k], float(probs[k]))

    def do_3d(file, mode, progress=gr.Progress()):
        if file is None or not is_nifti(file):
            raise gr.Error("Upload a 3D NIfTI volume (.nii or .nii.gz).")
        progress(0.02, desc="Loading volume")
        res = engine.explain_3d(file, progress=progress)
        progress(0.97, desc="Rendering viewer")
        doc = build_viewer_html(res, mode=mode, source_name=Path(file).name)
        out = out_dir / f"{Path(file).name.split('.')[0]}_{mode}_viewer.html"
        out.write_text(doc)
        return iframe(doc), dict(res["summary"]), str(out)

    def do_seg(image, gt, mode, progress=gr.Progress()):
        if image is None or not is_nifti(image):
            raise gr.Error("Upload a 4-sequence NIfTI (FLAIR, T1, T1 + contrast, T2), e.g. a file from "
                           "Task01_BrainTumour/imagesTr or imagesTs.")
        progress(0.1, desc="Segmenting tumour in 3D")
        try:
            res = seg_engine.analyse(image, gt)
        except ValueError as e:
            raise gr.Error(str(e))
        progress(0.9, desc="Rendering viewer")
        doc = build_seg_viewer_html(res, mode=mode, source_name=Path(image).name)
        out = out_dir / f"{Path(image).name.split('.')[0]}_tumour3d_{mode}.html"
        out.write_text(doc)
        v = res["volumes"]
        md = (f"**Whole tumour:** {v['WT']:.1f} cm³  \n**Edema:** {v[1]:.1f} cm³ · **Non-enhancing core:** {v[2]:.1f} cm³ · "
              f"**Enhancing:** {v[3]:.1f} cm³  \n{res['location']}")
        if res["dice"]:
            d = res["dice"]
            md += f"  \n**Dice vs expert:** WT {d['WT']:.3f} · TC {d['TC']:.3f} · ET {d['ET']:.3f}"
        return iframe(doc), md, str(out)

    def do_track(files, use_expert, mode, progress=gr.Progress()):
        if not files:
            raise gr.Error("Upload the contrast T1 scans for one patient (and optionally their _seg outlines).")
        progress(0.1, desc="Measuring each scan")
        try:
            doc, timeline, pid = track_uploaded([str(f) for f in files], vs_model, vs_eval, use_expert, mode)
        except ValueError as e:
            raise gr.Error(str(e))
        out = out_dir / f"tracker_{pid}_{mode}.html"
        out.write_text(doc)
        lines = ["| Date | Left cm³ | Right cm³ | Flags |", "|---|---|---|---|"]
        for t in timeline:
            fl = ", ".join(f"{k}: {v}" for k, v in t["flags"].items()) or "–"
            lines.append(f"| {t['date']} | {t['left_cm3']:.2f} | {t['right_cm3']:.2f} | {fl} |")
        return iframe(doc, 860), "\n".join(lines), str(out)

    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")
    with gr.Blocks(title="MRI Studio") as app:
        gr.Markdown("# MRI Studio\nExplainable brain-MRI analysis in 2D and 3D. " + DISCLAIMER)
        with gr.Tab("Tumour growth tracker"):
            with gr.Row():
                with gr.Column(scale=1):
                    ft = gr.File(label="One patient's scans over time (…_YYYY-MM-DD_T1C.nii.gz, optional …_T1C_seg.nii.gz)",
                                 file_count="multiple", type="filepath")
                    ue = gr.Checkbox(value=vs_model is None, label="Use expert outlines when provided",
                                     interactive=vs_model is not None)
                    mt = gr.Radio(["clinician", "patient"], value="clinician", label="Start in")
                    bt = gr.Button("Build growth timeline", variant="primary")
                    mdt = gr.Markdown()
                    dlt = gr.File(label="Download standalone tracker (.html)")
                with gr.Column(scale=3):
                    vt = gr.HTML("<div style='padding:40px;opacity:.6'>Upload every contrast T1 scan for one "
                                 "patient, then press <b>Build growth timeline</b>.</div>")
            bt.click(do_track, [ft, ue, mt], [vt, mdt, dlt])
        if seg_engine is not None:
            with gr.Tab("Tumour segmentation 3D"):
                with gr.Row():
                    with gr.Column(scale=1):
                        fs = gr.File(label="MRI case: 4D NIfTI with FLAIR, T1, T1+contrast, T2", type="filepath")
                        fg = gr.File(label="Optional: expert label file (shows Dice + outline)", type="filepath")
                        ms = gr.Radio(["clinician", "patient"], value="clinician", label="Start in")
                        bs = gr.Button("Segment and view in 3D", variant="primary")
                        mds = gr.Markdown()
                        dls = gr.File(label="Download standalone viewer (.html)")
                    with gr.Column(scale=3):
                        vs = gr.HTML("<div style='padding:40px;opacity:.6'>Upload a case and press "
                                     "<b>Segment and view in 3D</b>. Takes about 5-30 s.</div>")
                bs.click(do_seg, [fs, fg, ms], [vs, mds, dls])
        if engine is not None:
            names = engine.names
            with gr.Tab("Explain a slice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        f2 = gr.File(label="MRI slice (PNG/JPG) or NIfTI (.nii/.nii.gz)", type="filepath")
                        b2 = gr.Button("Analyse", variant="primary")
                        lab2 = gr.Label(label="Class probabilities", num_top_classes=len(names))
                    with gr.Column(scale=2):
                        img2 = gr.Image(label="Input  |  Grad-CAM: where the model looked (red = most influential)")
                        md2 = gr.Markdown()
                b2.click(do_2d, f2, [lab2, img2, md2])
            for tab_name, mode in (("Classifier heatmap 3D", "clinician"), ("Classifier patient view", "patient")):
                with gr.Tab(tab_name):
                    with gr.Row():
                        with gr.Column(scale=1):
                            f3 = gr.File(label="3D NIfTI volume (.nii / .nii.gz)", type="filepath")
                            b3 = gr.Button("Build 3D view", variant="primary")
                            lab3 = gr.Label(label="Scan-level probabilities", num_top_classes=len(names))
                            dl3 = gr.File(label="Download standalone viewer (.html - opens in any browser)")
                        with gr.Column(scale=3):
                            view3 = gr.HTML("<div style='padding:40px;opacity:.6'>Upload a volume and press "
                                            "<b>Build 3D view</b>. Scoring every slice takes ~10-60 s.</div>")
                    m = gr.State(mode)
                    b3.click(do_3d, [f3, m], [view3, lab3, dl3])
    app._mri_theme = theme
    return app


def main():
    ap = argparse.ArgumentParser(description="MRI Studio - explainable 2D/3D brain MRI viewer.")
    ap.add_argument("--checkpoint", default=None, help="classifier from brain_mri_classifier.py (optional)")
    ap.add_argument("--seg-checkpoint", default=None, help="3D tumour segmenter from brain_tumour_seg.py (optional)")
    ap.add_argument("--vs-checkpoint", default=None, help="growth-tracker model from vs_tracker.py (optional)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a temporary public link (careful with patient data).")
    args = ap.parse_args()
    if not NIIVUE_JS.exists():
        raise SystemExit(f"Missing {NIIVUE_JS}. Keep the 'static' folder next to mri_studio.py.")
    # pick up trained models from their default locations when not given explicitly
    for attr, default in (("checkpoint", "runs/kaggle/best_model.pt"), ("seg_checkpoint", "runs/seg/best_seg.pt"),
                          ("vs_checkpoint", "runs/vs/best_vs.pt")):
        if not getattr(args, attr) and Path(default).exists():
            setattr(args, attr, default)
    engine = Engine(args.checkpoint) if args.checkpoint else None
    seg_engine = SegEngine(args.seg_checkpoint) if args.seg_checkpoint else None
    if engine:
        print(f"Classifier: {engine.meta['arch']} | classes: {engine.names} | device: {engine.device}")
    if seg_engine:
        print(f"Segmenter: 3D U-Net | grid {seg_engine.meta['size']} | device: {seg_engine.device}")
    vs_model, vs_eval = None, None
    if args.vs_checkpoint:
        from vs_tracker import VSModel
        vs_model = VSModel(args.vs_checkpoint)
        ev = Path(args.vs_checkpoint).parent / "vs_eval.json"
        vs_eval = json.loads(ev.read_text()) if ev.exists() else None
        print(f"Growth tracker: patch 3D U-Net | reliability limits: {'yes' if vs_eval else 'not yet (run evaluate)'}")
    app = make_app(engine, seg_engine, vs_model, vs_eval)
    app.launch(server_port=args.port, share=args.share, inbrowser=True, theme=app._mri_theme)


if __name__ == "__main__":
    main()
