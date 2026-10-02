"""Subtask label (pick / place / home) from the policy's own prefix KV cache (SUBTASK-c).

**What this is.** SUBTASK-b showed that the 16D checkpoint's hidden state carries the manipulation
phase *linearly*: a StandardScaler + multinomial logistic regression on `kv_L4` reads
pick / place / home at 0.981 on the held-out validation episodes and 0.93-0.97 on closed-loop runs
(`docs/handoff/SUBTASK-b.verify.json`, `SUBTASK.audit.md` §8). This module applies that probe with
numpy alone — no sklearn at run time — from an asset exported by
`experiments/tools/export_subtask_probe.py` (`asset/subtask_probe/kv_L4_v1.{npz,json}`).

**The feature (`kv_L4`).** Prefix KV cache **V**, layer 4, batch 0, head 0 → `(S, 256)`; the mean
over the *valid* text tokens (positions 768.. where the prefix mask is set) concatenated with the
mean over the 768 image tokens → 512 float32. The definition is SUBTASK-b's `extract.py:86-106`
and must stay identical to it; the jitted version lives next to the prefix pass that already
builds the cache (`experiments/sources/pi05_attention.AttentionSampler`), so it costs no forward.
`kv_feature` here is the numpy reference of the same reduction.

**What the label is used for (user spec, 2026-10-02).** Only to decide whether the attention
target may *become* the manipulated object (and so be carved out of the ESDF): `pick` — yes, as
before; `place` / `home` — no (`TargetConfirm` neither adopts a first target nor counts a
challenger, `clustering.subtask_gate`). The held object's carve → attach transition is the grasp
latch's, never the label's. `SubtaskDebounce` is the "N consecutive frames" rule the label has to
pass before it counts.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "CLASSES",
    "GATED_LABELS",
    "KV_HEAD",
    "KV_LAYER",
    "N_IMAGE_TOKENS",
    "PROBE_NAME",
    "SubtaskDebounce",
    "SubtaskProbe",
    "argmax_label",
    "default_probe_path",
    "kv_feature",
]

#: Class order of the exported probe (`classes` in the asset). SUBTASK-b `probe.py:CLASSES`.
CLASSES = ("pick", "place", "home")
#: Labels under which `clustering.subtask_gate` blocks a new manipulated object (spec §0).
GATED_LABELS = frozenset({"place", "home"})
#: The asset this module ships with; also what `AttentionPolicy` reports as `result["subtask"]["probe"]`.
PROBE_NAME = "kv_L4_v1"
#: Prefix layout: 3 cameras x 16 x 16 SigLIP tokens first, then the text (prompt + state) tokens.
N_IMAGE_TOKENS = 768
KV_LAYER = 4
KV_HEAD = 0


def default_probe_path() -> pathlib.Path:
    """`benchmark/ag3s/asset/subtask_probe/kv_L4_v1.npz`, located from this file (no symlinks)."""
    return pathlib.Path(__file__).resolve().parents[1] / "asset" / "subtask_probe" / f"{PROBE_NAME}.npz"


def kv_feature(v_layer: np.ndarray, prefix_mask: np.ndarray) -> np.ndarray:
    """numpy reference of the `kv_L4` reduction for **one** sample.

    `v_layer`: `(S, H)` — the V cache of the chosen layer, batch 0, head 0 (any float dtype).
    `prefix_mask`: `(S,)` bool — the prefix token mask. Returns `[mean valid text | mean image]`
    as float32 `(2H,)`, computed in float32 like SUBTASK-b's jitted reduction.
    """
    v = np.asarray(v_layer, np.float32)
    m = np.asarray(prefix_mask).reshape(-1)
    tm = m[N_IMAGE_TOKENS:].astype(np.float32)
    txt = (v[N_IMAGE_TOKENS:] * tm[:, None]).sum(0) / tm.sum()
    img = v[:N_IMAGE_TOKENS].mean(0)
    return np.concatenate([txt, img]).astype(np.float32)


class SubtaskProbe:
    """The exported linear probe: `p = softmax(((x - mean) / scale) @ coef.T + intercept)`."""

    def __init__(self, mean, scale, coef, intercept, classes=CLASSES, name: str = PROBE_NAME):
        self.mean = np.asarray(mean, np.float64).reshape(-1)
        self.scale = np.asarray(scale, np.float64).reshape(-1)
        self.coef = np.asarray(coef, np.float64)
        self.intercept = np.asarray(intercept, np.float64).reshape(-1)
        self.classes = tuple(str(c) for c in classes)
        self.name = str(name)
        d = self.mean.shape[0]
        if (self.scale.shape != (d,) or self.coef.shape != (len(self.classes), d)
                or self.intercept.shape != (len(self.classes),)):
            raise ValueError(
                f"subtask probe {self.name}: inconsistent shapes mean {self.mean.shape} scale "
                f"{self.scale.shape} coef {self.coef.shape} intercept {self.intercept.shape} "
                f"classes {self.classes}")

    @classmethod
    def load(cls, path: Optional[str | pathlib.Path] = None) -> "SubtaskProbe":
        """Read the asset npz (`None` = `default_probe_path()`)."""
        p = default_probe_path() if path is None else pathlib.Path(path)
        with np.load(p, allow_pickle=False) as z:
            name = str(z["name"]) if "name" in z.files else p.stem
            return cls(z["mean"], z["scale"], z["coef"], z["intercept"],
                       classes=[str(c) for c in z["classes"]], name=name)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def proba(self, x: np.ndarray) -> np.ndarray:
        """`(N, D)` or `(D,)` features → `(N, K)` / `(K,)` class probabilities (float64)."""
        x = np.asarray(x, np.float64)
        single = x.ndim == 1
        X = x.reshape(1, -1) if single else x
        z = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        return p[0] if single else p

    def predict(self, x: np.ndarray) -> dict[str, Any]:
        """One sample → `{"p": {class: float}, "argmax": str, "probe": name}` (`result["subtask"]`)."""
        p = self.proba(np.asarray(x).reshape(-1))
        return {"p": {c: float(v) for c, v in zip(self.classes, p)},
                "argmax": self.classes[int(np.argmax(p))],
                "probe": self.name}

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "classes": list(self.classes), "dim": self.dim}


def argmax_label(p: Optional[Mapping[str, float]]) -> Optional[str]:
    """The most probable class of a `{class: prob}` mapping; None for None / empty / non-finite."""
    if not p:
        return None
    items = [(str(k), float(v)) for k, v in dict(p).items()]
    if not all(np.isfinite(v) for _, v in items):
        return None
    # Ties go to the earlier class in `CLASSES` order, then the mapping's own order.
    order = {c: i for i, c in enumerate(CLASSES)}
    best = max(items, key=lambda kv: (kv[1], -order.get(kv[0], len(order))))
    return best[0]


def json_probe_record(path: Optional[str | pathlib.Path] = None) -> Optional[dict]:
    """The asset's json side-car (`<name>.json`), or None when absent."""
    p = (default_probe_path() if path is None else pathlib.Path(path)).with_suffix(".json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


class SubtaskDebounce:
    """The confirmed subtask label: it changes only after `frames` consecutive identical argmaxes.

    Fed once per request (`update`). An input of None (no label for that request) counts as its own
    "class": `frames` consecutive Nones bring the confirmed label back to None — a probe that stops
    answering falls back to the ungated behaviour on the same schedule a real label change would
    take, and a single missing frame neither confirms nor clears anything. Before the first
    confirmation (episode start, after `reset()`), the label is None.
    """

    _NONE = object()

    def __init__(self, frames: int = 3):
        if int(frames) < 1:
            raise ValueError(f"SubtaskDebounce frames must be >= 1, got {frames}")
        self.frames = int(frames)
        self.reset()

    def reset(self) -> None:
        self.label: Optional[str] = None
        self.p: Optional[dict[str, float]] = None
        self.argmax: Optional[str] = None
        self._candidate: Any = None
        self.streak = 0
        self.n_updates = 0

    def update(self, p: Optional[Mapping[str, float]]) -> Optional[str]:
        """This request's probabilities (or None). Returns the confirmed label after it."""
        self.n_updates += 1
        self.p = None if p is None else {str(k): float(v) for k, v in dict(p).items()}
        self.argmax = argmax_label(self.p)
        key = self._NONE if self.argmax is None else self.argmax
        if self._candidate is not None and key == self._candidate:
            self.streak += 1
        else:
            self._candidate, self.streak = key, 1
        if self.streak >= self.frames:
            self.label = None if key is self._NONE else str(key)
        return self.label

    @property
    def gated(self) -> bool:
        """True when the confirmed label is one under which no new manipulated object is taken."""
        return self.label in GATED_LABELS

    def record(self) -> dict[str, Any]:
        return {"p": None if self.p is None else dict(self.p), "argmax": self.argmax,
                "label": self.label, "streak": int(self.streak), "frames": int(self.frames)}
