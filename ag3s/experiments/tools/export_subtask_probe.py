"""Re-train SUBTASK-b's kv_L4 phase probe and export it as a numpy asset (SUBTASK-c A1).

SUBTASK-b trained the probe but did not save it. This re-fits it on the same data with the same
settings and writes what `benchmark.ag3s.stages.subtask_probe.SubtaskProbe` reads:

    benchmark/ag3s/asset/subtask_probe/kv_L4_v1.npz   mean · scale · coef · intercept · classes · name
    benchmark/ag3s/asset/subtask_probe/kv_L4_v1.json  feature definition · C · training files · parity

Same data / settings as `outputs/verify/SUBTASK-b/probe.py`:
  * train = `feat/train/*.npz` (200 episodes), feature `normal__kv_L4`, labels from `phase_index`
    (0-3 pick, 4-8 place, 9-13 home; anything else dropped);
  * StandardScaler + LogisticRegression(C, max_iter=3000, tol=1e-4) (lbfgs, multinomial);
  * C = the value SUBTASK-b chose for kv_L4 (`probe_results.json`, GroupKFold(5) over episodes).

Parity: the numpy application (the asset, through `SubtaskProbe.proba`) on `feat/val` must reach
the SUBTASK-b val normal accuracy (0.981) within ±0.002, or the script exits non-zero and writes
nothing. sklearn is needed here only — run with `.venv-curobo` (has sklearn; `.venv-ag3s` hangs
on NFS):

    /mnt/dev/work/.venv-curobo/bin/python benchmark/ag3s/experiments/tools/export_subtask_probe.py
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]  # /mnt/dev/work
SRC = ROOT / "outputs" / "verify" / "SUBTASK-b"
OUT = ROOT / "benchmark" / "ag3s" / "asset" / "subtask_probe"
FEATURE = "kv_L4"
NAME = "kv_L4_v1"
CLASSES = ["pick", "place", "home"]
TOL = 0.002


def _probe_module():
    """`stages/subtask_probe.py` by path — this venv need not import the whole `benchmark.ag3s`."""
    path = ROOT / "benchmark" / "ag3s" / "stages" / "subtask_probe.py"
    spec = importlib.util.spec_from_file_location("_subtask_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def phase_to_class(p):
    p = np.asarray(p)
    c = np.full(p.shape, -1)
    c[(p >= 0) & (p <= 3)] = 0
    c[(p >= 4) & (p <= 8)] = 1
    c[(p >= 9) & (p <= 13)] = 2
    return c


def load(split):
    files = sorted(glob.glob(str(SRC / "feat" / split / "*.npz")))
    X, y = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        X.append(z[f"normal__{FEATURE}"])
        y.append(phase_to_class(z["phase_index"]))
    X, y = np.concatenate(X), np.concatenate(y)
    keep = y >= 0
    return X[keep], y[keep], files


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from sklearn import __version__ as skv
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ref = json.load(open(SRC / "probe_results.json"))["features"][FEATURE]
    C = float(ref["train_normal"]["C"])
    ref_acc = float(ref["val__normal"]["accuracy"])
    verify = json.load(open(ROOT / "benchmark/ag3s/docs/handoff/SUBTASK-b.verify.json"))
    verify_acc = float(verify["numbers"]["M1_probe_normal"][FEATURE]["val__normal"]["acc"])

    Xtr, ytr, train_files = load("train")
    Xva, yva, val_files = load("val")
    print(f"train {Xtr.shape} per class {np.bincount(ytr).tolist()} | val {Xva.shape} "
          f"per class {np.bincount(yva).tolist()} | C={C}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, tol=1e-4).fit(scaler.transform(Xtr), ytr)
    assert list(clf.classes_) == [0, 1, 2], clf.classes_

    sk_prob = clf.predict_proba(scaler.transform(Xva))
    sk_acc = float((sk_prob.argmax(1) == yva).mean())

    mod = _probe_module()
    probe = mod.SubtaskProbe(scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_,
                             classes=CLASSES, name=NAME)
    np_prob = probe.proba(Xva)
    np_pred = np_prob.argmax(1)
    np_acc = float((np_pred == yva).mean())
    max_dp = float(np.abs(np_prob - sk_prob).max())
    cm = np.zeros((3, 3), int)
    for t, p in zip(yva, np_pred):
        cm[t, p] += 1
    print(f"val acc: numpy {np_acc:.4f} | sklearn {sk_acc:.4f} | SUBTASK-b probe_results "
          f"{ref_acc:.4f} | verify.json {verify_acc:.4f} | max |p_np - p_sk| {max_dp:.2e}",
          flush=True)
    print("confusion (rows true pick/place/home):", cm.tolist(), flush=True)
    if abs(np_acc - verify_acc) > TOL:
        print(f"PARITY FAIL: |{np_acc:.4f} - {verify_acc:.4f}| > {TOL} — nothing written",
              flush=True)
        sys.exit(2)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{NAME}.npz"
    np.savez(npz, mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64),
             coef=clf.coef_.astype(np.float64), intercept=clf.intercept_.astype(np.float64),
             classes=np.asarray(CLASSES), name=np.asarray(NAME))
    # Re-load what was written and re-check, so the json's numbers are the asset's numbers.
    re_acc = float((mod.SubtaskProbe.load(npz).proba(Xva).argmax(1) == yva).mean())
    assert re_acc == np_acc, (re_acc, np_acc)

    meta = {
        "name": NAME,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "writer": "benchmark/ag3s/experiments/tools/export_subtask_probe.py (SUBTASK-c A1)",
        "classes": CLASSES,
        "class_from_phase_index": {"pick": "0-3", "place": "4-8", "home": "9-13"},
        "feature": {
            "key": FEATURE,
            "dim": int(Xtr.shape[1]),
            "dtype": "float32",
            "definition": ("prefix KV cache V, layer 4, batch 0, head 0 -> (S, 256); "
                           "[mean over valid text tokens (prefix mask[768:]) | mean over the 768 "
                           "image tokens] = 512. Same as outputs/verify/SUBTASK-b/extract.py:86-106"),
            "condition": "normal (state as recorded; gripper not masked)",
            "runtime_source": ("benchmark/ag3s/experiments/sources/pi05_attention.py "
                               "AttentionSampler prefix pass (no extra forward)"),
        },
        "model": {
            "pipeline": "StandardScaler + LogisticRegression (L2, lbfgs, multinomial)",
            "C": C,
            "max_iter": 3000,
            "tol": 1e-4,
            "C_source": "outputs/verify/SUBTASK-b/probe_results.json features.kv_L4.train_normal.C "
                        "(GroupKFold(5) over train episodes)",
            "apply": "p = softmax(((x - mean) / scale) @ coef.T + intercept)",
            "sklearn": skv,
            "numpy": np.__version__,
        },
        "checkpoint": verify["setup"]["checkpoint"],
        "train": {
            "dir": str(SRC / "feat" / "train"),
            "n_files": len(train_files),
            "files": [pathlib.Path(f).name for f in train_files],
            "n_frames": int(len(ytr)),
            "per_class": {c: int((ytr == i).sum()) for i, c in enumerate(CLASSES)},
        },
        "parity": {
            "val_dir": str(SRC / "feat" / "val"),
            "val_n_files": len(val_files),
            "val_n_frames": int(len(yva)),
            "val_acc_numpy": np_acc,
            "val_acc_sklearn": sk_acc,
            "val_acc_reference_verify_json": verify_acc,
            "val_acc_reference_probe_results": ref_acc,
            "tolerance": TOL,
            "abs_diff_vs_verify_json": abs(np_acc - verify_acc),
            "max_abs_dp_numpy_vs_sklearn": max_dp,
            "val_confusion_numpy": cm.tolist(),
            "val_confusion_reference": verify["numbers"]["M1_probe_normal"][FEATURE]
                                      ["val__normal"]["confusion"],
            "pass": True,
        },
        "npz_sha256": sha256(npz),
    }
    (out / f"{NAME}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {npz} and {npz.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
