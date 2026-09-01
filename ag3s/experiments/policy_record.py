"""Recording a real pi0.5 rollout in the form every AG3S verification step needs.

`gaussian_attention` in `mujoco_source` exists because, when AG3S was written, the only pi0.5
checkpoint available was trained on LIBERO's agent view and could not be pointed at an RB-Y1 ZED
frame. A fine-tuned `rby1_transport_14d` now exists, so the stand-in can be retired -- but only if
the *exact* observations that checkpoint saw are available offline, together with the ground truth
to score them against. That is what this module records.

**Only qpos is stored, never depth or segmentation.** MuJoCo is deterministic: `qpos` fixes the
robot, the crate, and every fruit, so depth, segmentation, intrinsics and extrinsics can all be
regenerated later by `TransportScene` -- the same code path AG3S is already tested through. Storing
them instead would multiply the file size by ~50x and, worse, would let the recorded rendering and
the analysis rendering drift apart without anything noticing.

The policy images *are* stored verbatim, because those cannot be regenerated safely: they are the
tensors that actually entered the network, after the 299x224 render and the squash to 224x224, and
a verification of attention has to run on the pixels the model saw rather than on a re-render that
is merely supposed to match.

Geometry note, needed by everything downstream:
    `render_cam(..., match_rby1_dataset=True)` renders 299x224 (4:3) and resizes to 224x224 with no
    crop and no pad. That is a pure horizontal squash, so *normalized* image coordinates survive it
    unchanged: a patch at policy column u_p maps to normalized column u_p / 224, which is the same
    normalized column in the 640x480 (also 4:3) frame AG3S reconstructs from. Attention therefore
    lifts onto the point cloud by plain normalized scaling, with no offset to get wrong.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Iterator, Optional, Sequence

import numpy as np

#: Policy image key -> (MuJoCo camera, AG3S CameraID, prefix-token slice).
#:
#: The token slice is not a guess. `AlohaInputs` builds its image dict in the fixed order
#: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, and `Pi0.embed_prefix` concatenates image tokens
#: in that dict's iteration order. SigLIP So400m/14 on a 224x224 input emits a 16x16 grid, so each
#: camera occupies exactly 256 consecutive prefix positions before the language tokens begin.
CAMERA_BINDINGS: dict[str, tuple[str, str, tuple[int, int]]] = {
    "cam_high": ("zed_left", "head", (0, 256)),
    "cam_left_wrist": ("wrist_cam_l", "left_wrist", (256, 512)),
    "cam_right_wrist": ("wrist_cam_r", "right_wrist", (512, 768)),
}

POLICY_CAMERA_NAMES = tuple(CAMERA_BINDINGS)
ATTENTION_GRID = 16
LANGUAGE_TOKEN_START = 768


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One inference step: what the policy was shown, and what it answered."""

    t_step: int
    qpos: np.ndarray  # (nq,) -- regenerates the whole scene
    qvel: np.ndarray  # (nv,)
    state: np.ndarray  # (14,) the policy's own state input
    actions: np.ndarray  # (H, 14) the chunk it returned
    infer_ms: float
    images: dict[str, np.ndarray]  # policy key -> (224, 224, 3) uint8, HWC

    @property
    def index(self) -> int:
        return int(self.t_step)


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """A whole rollout, plus the metadata needed to rebuild its scene."""

    path: pathlib.Path
    meta: dict[str, Any]
    steps: list[StepRecord]

    @property
    def prompt(self) -> str:
        return str(self.meta["prompt"])

    @property
    def model_xml(self) -> str:
        return str(self.meta["model_xml"])

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.steps)


# ------------------------------------------------------------------------------------ writing


class PolicyRecordWriter:
    """Appends one compressed `.npz` per inference step under a fresh run directory.

    Kept deliberately dumb: it copies arrays and writes them. Anything that needs interpretation --
    which camera is which, how a patch maps to a pixel -- lives in this module's constants so that
    the recorder cannot encode an assumption the analysis does not share.
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        *,
        model,
        model_xml: str,
        prompt: str,
        extra: Optional[dict[str, Any]] = None,
    ):
        import mujoco

        root = pathlib.Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for run_index in range(1_000_000):
            run_dir = root / f"run_{run_index:04d}"
            try:
                run_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover -- a million runs in one directory
            raise RuntimeError(f"no available run directory under {root}")

        self.run_dir = run_dir
        self._count = 0
        joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)
        ]
        self.meta: dict[str, Any] = {
            "prompt": prompt,
            "model_xml": str(model_xml),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "joint_names": joint_names,
            "camera_bindings": {k: list(v) for k, v in CAMERA_BINDINGS.items()},
            "policy_image_size": 224,
            "attention_grid": ATTENTION_GRID,
            **(extra or {}),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        print(f"[ag3s] recording policy observations to {self.run_dir}")

    def record(self, *, t_step: int, obs: dict, data, chunk: np.ndarray, infer_ms: float) -> None:
        payload: dict[str, Any] = {
            "t_step": np.int64(t_step),
            "qpos": np.asarray(data.qpos, np.float64).copy(),
            "qvel": np.asarray(data.qvel, np.float64).copy(),
            "state": np.asarray(obs["state"], np.float64).copy(),
            "actions": np.asarray(chunk, np.float32).copy(),
            "infer_ms": np.float64(infer_ms),
        }
        for key in POLICY_CAMERA_NAMES:
            image = np.asarray(obs["images"][key])
            # The observation carries CHW (AlohaInputs' convention); store HWC so the file can be
            # looked at without knowing that.
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            payload[f"image_{key}"] = np.ascontiguousarray(image[..., :3]).astype(np.uint8)
        np.savez_compressed(self.run_dir / f"step_{self._count:05d}.npz", **payload)
        self._count += 1

    def close(self) -> None:
        self.meta["n_steps"] = self._count
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        print(f"[ag3s] wrote {self._count} policy observations to {self.run_dir}")


# ------------------------------------------------------------------------------------ reading


def load_run(run_dir: str | pathlib.Path, *, limit: Optional[int] = None) -> RunRecord:
    """Read a recorded run back. `limit` truncates for a quick look."""
    path = pathlib.Path(run_dir)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{path} is not a run directory (no meta.json). Point at a run_XXXX/ directory."
        )
    meta = json.loads(meta_path.read_text())
    files = sorted(path.glob("step_*.npz"))
    if limit is not None:
        files = files[: int(limit)]
    steps = []
    for file in files:
        with np.load(file) as d:
            steps.append(
                StepRecord(
                    t_step=int(d["t_step"]),
                    qpos=np.asarray(d["qpos"], np.float64),
                    qvel=np.asarray(d["qvel"], np.float64),
                    state=np.asarray(d["state"], np.float64),
                    actions=np.asarray(d["actions"], np.float32),
                    infer_ms=float(d["infer_ms"]),
                    images={k: np.asarray(d[f"image_{k}"]) for k in POLICY_CAMERA_NAMES},
                )
            )
    if not steps:
        raise ValueError(f"no step_*.npz files in {path}")
    return RunRecord(path=path, meta=meta, steps=steps)


def replay_scene(record: RunRecord, *, height: int = 480, width: int = 640):
    """A `TransportScene` posed to reproduce the recorded rollout.

    The joint-name check is not ceremony. `qpos` is a flat vector whose meaning is entirely
    positional: if the analysis loads a different XML than the rollout did, every index still
    resolves and the scene silently comes out wrong -- a crate a few centimetres off, or an arm in
    another configuration -- and every downstream number would be quietly measured against fiction.
    """
    from benchmark.ag3s.experiments.mujoco_source import TransportScene

    scene = TransportScene(record.model_xml, height=height, width=width, settle_steps=0)
    if int(scene.model.nq) != int(record.meta["nq"]):
        raise ValueError(
            f"recorded rollout has nq={record.meta['nq']} but {record.model_xml} has "
            f"nq={scene.model.nq}; the analysis would be replaying a different scene"
        )
    recorded_names = list(record.meta["joint_names"])
    scene_names = [
        scene.mujoco.mj_id2name(scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        for i in range(scene.model.njnt)
    ]
    if recorded_names != scene_names:
        differing = [
            (i, a, b) for i, (a, b) in enumerate(zip(recorded_names, scene_names)) if a != b
        ]
        raise ValueError(f"joint order differs between rollout and analysis model: {differing[:5]}")
    return scene


def pose_scene(scene, step: StepRecord) -> None:
    """Put the scene exactly where it was when this observation was captured."""
    scene.data.qpos[:] = step.qpos
    scene.data.qvel[:] = step.qvel
    scene.mujoco.mj_forward(scene.model, scene.data)


# -------------------------------------------------------------------- attention <-> pixels


def patch_to_uv(row: int, col: int, height: int, width: int, *, grid: int = ATTENTION_GRID):
    """Centre pixel of attention patch `(row, col)` in an `(height, width)` frame.

    Valid for any 4:3 frame -- see the module docstring on why the 299x224 -> 224x224 squash leaves
    normalized coordinates alone.
    """
    return (
        (col + 0.5) / grid * width,
        (row + 0.5) / grid * height,
    )


def patch_of_uv(u: float, v: float, height: int, width: int, *, grid: int = ATTENTION_GRID):
    """Which attention patch pixel `(u, v)` falls in."""
    col = min(grid - 1, max(0, int(u / width * grid)))
    row = min(grid - 1, max(0, int(v / height * grid)))
    return row, col


def patch_coverage(labels: np.ndarray, body_ids: Sequence[int], *, grid: int = ATTENTION_GRID):
    """Per-body patch coverage `c_i(r, c)` and unoccluded pixel area, from a body-id label map.

    This is KNOWS Eq. (2) applied to MuJoCo segmentation instead of LIBERO's: the fraction of each
    patch covered by body i, which is what makes an attention argmax scoreable against ground truth
    without pretending a patch belongs to exactly one object.
    """
    labels = np.asarray(labels)
    height, width = labels.shape
    if height % grid or width % grid:
        # Crop rather than interpolate: a resampled id map invents ids that were never rendered.
        height, width = height - height % grid, width - width % grid
        labels = labels[:height, :width]
    ph, pw = height // grid, width // grid
    coverage = np.zeros((len(body_ids), grid, grid), np.float32)
    area = np.zeros(len(body_ids), np.float32)
    for k, body in enumerate(body_ids):
        mask = (labels == int(body)).astype(np.float32)
        area[k] = mask.sum()
        coverage[k] = mask.reshape(grid, ph, grid, pw).mean(axis=(1, 3))
    return coverage, area


__all__ = [
    "ATTENTION_GRID",
    "CAMERA_BINDINGS",
    "LANGUAGE_TOKEN_START",
    "POLICY_CAMERA_NAMES",
    "PolicyRecordWriter",
    "RunRecord",
    "StepRecord",
    "load_run",
    "patch_coverage",
    "patch_of_uv",
    "patch_to_uv",
    "pose_scene",
    "replay_scene",
]
