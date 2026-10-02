"""Core tests for MRI Studio: model shapes, loss, metrics, preprocessing, patient-level splits and
growth flags. CPU only, no datasets or trained weights needed (about 20 s)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain_tumour_seg as seg  # noqa: E402
import vs_tracker as vs  # noqa: E402


def test_unet_output_shape_and_parameter_count():
    model = seg.UNet3D(base=8)
    x = torch.randn(1, 4, 32, 32, 16)
    assert model(x).shape == (1, 4, 32, 32, 16)
    full = seg.UNet3D()                       # the configuration used in training
    n = sum(p.numel() for p in full.parameters())
    assert 5.0e6 < n < 6.2e6, n               # README: about 5.6M parameters


def test_loss_falls_when_overfitting_one_case():
    torch.manual_seed(0)
    model = seg.UNet3D(base=4, depth=2)
    opt = torch.optim.AdamW(model.parameters(), 1e-2)
    x = torch.randn(1, 4, 16, 16, 16)
    y = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    y[:, 4:10, 4:10, 4:10] = 1
    y[:, 6:8, 6:8, 6:8] = 3
    first = None
    for _ in range(25):
        loss = seg.dice_ce_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.6 * first


def test_region_dice():
    gt = np.zeros((10, 10, 10), np.uint8); gt[2:6, 2:6, 2:6] = 1; gt[3:5, 3:5, 3:5] = 3
    assert all(v == pytest.approx(1.0) for v in seg.region_dice(gt, gt).values())
    empty = np.zeros_like(gt)
    d = seg.region_dice(empty, gt)
    assert all(v == 0.0 for v in d.values())
    assert all(v == 1.0 for v in seg.region_dice(empty, empty).values())   # nothing to find, nothing found


def test_preprocessing_crops_to_brain_and_normalises():
    arr = np.zeros((4, 40, 40, 30), np.float32)
    arr[:, 10:30, 12:28, 5:25] = np.random.default_rng(0).normal(500, 80, (4, 20, 16, 20))
    bb = seg.brain_bbox(arr, margin=2)
    assert [(s.start, s.stop) for s in bb] == [(8, 32), (10, 30), (3, 27)]
    z = seg.zscore(arr)
    brain = arr[0] != 0
    assert abs(z[0][brain].mean()) < 1e-3 and abs(z[0][brain].std() - 1) < 1e-3
    assert (z[0][~brain] == 0).all()
    x, _, _ = seg.preprocess(arr, (32, 32, 16))
    assert tuple(x.shape) == (4, 32, 32, 16)


def test_case_split_has_no_overlap():
    files = [Path(f"BRATS_{i:03d}.nii.gz") for i in range(100)]
    tr, va, te = seg.split_cases(files, seed=0)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert seg.split_cases(files, seed=0) == (tr, va, te)          # reproducible


def test_patient_split_keeps_every_scan_of_a_patient_together():
    cases = [{"patient": f"VS_{p:03d}", "date": f"2020-0{t + 1}-01"} for p in range(40) for t in range(3)]
    splits = vs.split_patients(cases, seed=1)
    owners = [{c["patient"] for c in s} for s in splits]
    assert not (owners[0] & owners[1] or owners[0] & owners[2] or owners[1] & owners[2])
    assert sum(len(s) for s in splits) == len(cases)


def test_left_and_right_tumours_are_labelled():
    mask = np.zeros((40, 20, 20), bool); mask[5:10, 8:12, 8:12] = True; mask[30:36, 8:12, 8:12] = True
    head = np.ones_like(mask)
    sides = vs.split_sides(mask, head, voxel_mm3=1.0)
    assert set(np.unique(sides[5:10, 8:12, 8:12])) == {1}       # low x = left (RAS)
    assert set(np.unique(sides[30:36, 8:12, 8:12])) == {2}


def _result(left, right=0.0):
    return {"volumes": {"left": left, "right": right, "total": left + right}}


def test_growth_needs_both_20_percent_and_the_measurement_noise_limit():
    eval_res = {"by_size": {vs.size_bin(1.0): {"mdc95_pct": 35.0}}}
    assert vs.growth_threshold(1.0, None) == pytest.approx(0.20)
    assert vs.growth_threshold(1.0, eval_res) == pytest.approx(0.35)
    rows = vs.build_timeline([("2020-01-01", _result(1.0)), ("2021-01-01", _result(1.25)),
                              ("2022-01-01", _result(1.8))], eval_res)
    assert rows[0]["flags"] == {}
    assert rows[1]["flags"]["left"] == "possible growth (within measurement noise)"   # +25 %: over 20 %, under MDC
    assert rows[2]["flags"]["left"] == "growth"                                        # +44 %
    assert rows[2]["left_vs_baseline_pct"] == pytest.approx(80.0)


def test_scan_names_are_parsed():
    info = vs.parse_name(Path("VS_MC_RC2_001_2019-05-14_T1C.nii.gz"))
    assert info is not None and info.get("date") == "2019-05-14"
