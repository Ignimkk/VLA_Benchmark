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
    #: Optional captured depth, only present when the rollout was recorded with `--record-depth`.
    #: MuJoCo camera name -> (H, W) float64 metres, already converted back from the stored uint16.
    depth: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    #: MuJoCo camera name -> (3, 3) intrinsics and (4, 4) camera-to-base, as captured.
    camera_intrinsics: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    T_base_cam: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def index(self) -> int:
        return int(self.t_step)

    @property
    def has_depth(self) -> bool:
        return bool(self.depth)


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
        depth_cameras: Sequence[str] = (),
        depth_hw: tuple[int, int] = (480, 640),
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
        self._depth_cameras = tuple(depth_cameras)
        self._depth_hw = (int(depth_hw[0]), int(depth_hw[1]))
        self._renderer = None
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
            "depth_cameras": list(self._depth_cameras),
            "depth_hw": list(self._depth_hw),
            "depth_encoding": "uint16 millimetres" if self._depth_cameras else None,
            **(extra or {}),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        print(f"[ag3s] recording policy observations to {self.run_dir}")

    def _capture_depth(self, model, data, camera: str):
        """Depth, intrinsics and extrinsics for one camera, in AG3S's conventions."""
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, self._depth_hw[0], self._depth_hw[1])
        r = self._renderer
        r.disable_depth_rendering()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = np.asarray(r.render(), np.float64)
        r.disable_depth_rendering()

        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy = float(model.cam_fovy[cid])
        f = (self._depth_hw[0] / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        K = np.array([[f, 0.0, self._depth_hw[1] / 2.0],
                      [0.0, f, self._depth_hw[0] / 2.0], [0.0, 0.0, 1.0]], np.float64)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = data.xmat[bid].reshape(3, 3)
        T_world_base[:3, 3] = data.xpos[bid]
        T_world_cam = np.eye(4)
        # MuJoCo: -z forward, +y up. OpenCV: +z forward, +y down.
        T_world_cam[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T_world_cam[:3, 3] = data.cam_xpos[cid]
        return depth, K, np.linalg.inv(T_world_base) @ T_world_cam

    def record(self, *, t_step: int, obs: dict, data, chunk: np.ndarray, infer_ms: float,
               model=None) -> None:
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
        if self._depth_cameras and model is not None:
            for camera in self._depth_cameras:
                depth, K, T = self._capture_depth(model, data, camera)
                # uint16 millimetres is what a real depth camera delivers, and what
                # `PointCloudConfig.depth_scale = 0.001` exists to decode. Storing the float64
                # renderer output instead would halve the file for no gain in realism and would
                # quietly hide the 1 mm quantisation every real sensor has.
                payload[f"depth_{camera}"] = np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)
                payload[f"K_{camera}"] = K
                payload[f"T_base_cam_{camera}"] = T
        np.savez_compressed(self.run_dir / f"step_{self._count:05d}.npz", **payload)
        self._count += 1

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
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
            cams = [k[len("depth_"):] for k in d.files if k.startswith("depth_")]
            steps.append(
                StepRecord(
                    t_step=int(d["t_step"]),
                    qpos=np.asarray(d["qpos"], np.float64),
                    qvel=np.asarray(d["qvel"], np.float64),
                    state=np.asarray(d["state"], np.float64),
                    actions=np.asarray(d["actions"], np.float32),
                    infer_ms=float(d["infer_ms"]),
                    images={k: np.asarray(d[f"image_{k}"]) for k in POLICY_CAMERA_NAMES},
                    depth={c: np.asarray(d[f"depth_{c}"], np.float64) / 1000.0 for c in cams},
                    camera_intrinsics={c: np.asarray(d[f"K_{c}"], np.float64) for c in cams},
                    T_base_cam={c: np.asarray(d[f"T_base_cam_{c}"], np.float64) for c in cams},
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
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

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


# ---------------------------------------------------- 어느 기록에서 나왔는가 (출처 도장)

#: 중간 산출물(`--dump-frames` npz, cuRobo 필드 npz)에 찍는 출처 도장의 키 이름.
#: 하나로 모아 두는 이유는 생산자와 소비자가 **문자열을 따로 적으면 짝 검사가 조용히
#: 아무것도 안 하기** 때문이다 — 키 이름이 어긋나면 `.get()` 이 `None` 을 돌려주고 검사는
#: 통과한다.
PROVENANCE_KEY = "source_record"
PROVENANCE_FINGERPRINT_KEY = "source_record_fingerprint"
PROVENANCE_FRAMES_KEY = "source_record_frames"


def record_fingerprint(record: RunRecord) -> str:
    """이 기록을 가리키는 짧은 지문. 같은 기록이면 같고, 다른 기록이면 거의 확실히 다르다.

    경로 문자열만으로는 부족하다 — `/tmp/rollout_frames.npz` 가 어느 `run_0000` 에서 나온
    것인지 경로가 말해 주지 않고, 두 기록이 같은 이름을 쓰는 일이 실제로 있다
    (`run_0004` 와 `…/20260924_long16d/run_0000` 은 둘 다 `run_XXXX` 다).

    **2026-09-25 에 이것이 없어서 난 일**: R4 가 자세는 16D 기록에서, depth 와 필드는 09-12 자
    14D npz 에서 가져온 혼합 실행이었는데 스크립트가 그것을 못 잡았다. 로봇은 16D 자세로
    서 있고 거리장은 14D 씬을 담은 채로 "낙관 오차" 를 쟀다.
    """
    import hashlib

    # **`--frames`/`limit` 에 흔들리면 안 된다.** 중간 산출물은 보통 앞 15 프레임만 굽고
    # 소비자는 `load_run(..., limit=n)` 으로 읽으므로, 지문이 프레임 수에 따라 달라지면
    # 짝이 맞는 짝을 짝이 아니라고 말한다. 그래서 `meta.json` 전체(프레임 수와 무관하다)와
    # **첫** 자세만 쓴다 — 끝 자세는 쓰지 않는다.
    meta = json.dumps(record.meta, sort_keys=True, ensure_ascii=False, default=str)
    first = record.steps[0]
    first_qpos = np.asarray(first.qpos, np.float64).tobytes().hex()
    # **액션 폭을 지문에 넣는다.** 보통은 `meta` 의 `policy_model` 이 14D 와 16D 를 이미
    # 가르지만, 그것은 문자열이라 녹화 스크립트가 같은 이름을 쓰면 같아진다. 이 STEP 이
    # 통째로 다루는 구분이 바로 그 폭이므로 지문이 직접 들고 있는 편이 낫다.
    # (`limit` 과 무관하다 — 첫 스텝에서만 읽는다.)
    width = int(np.asarray(first.state).reshape(-1).shape[0])
    return hashlib.sha256(
        f"{meta}|{width}|{first_qpos}".encode()).hexdigest()[:16]


def provenance_arrays(record: RunRecord, n_frames: int) -> dict:
    """`np.savez` 에 그대로 넘길 출처 도장. 생산자는 이것을 **반드시** 섞어 넣는다."""
    return {
        PROVENANCE_KEY: np.asarray(str(record.path)),
        PROVENANCE_FINGERPRINT_KEY: np.asarray(record_fingerprint(record)),
        PROVENANCE_FRAMES_KEY: np.asarray(int(n_frames)),
    }


def read_provenance(blob, *, where: str) -> dict:
    """npz 에서 출처 도장을 읽는다. 도장이 없으면 **어느 기록인지 모른다는 뜻**이다."""
    def _get(key):
        if key not in getattr(blob, "files", ()):
            return None
        value = blob[key]
        return value.item() if getattr(value, "shape", ()) == () else value

    fingerprint = _get(PROVENANCE_FINGERPRINT_KEY)
    return {
        "where": where,
        "record": _get(PROVENANCE_KEY),
        "fingerprint": None if fingerprint is None else str(fingerprint),
        "frames": _get(PROVENANCE_FRAMES_KEY),
        "stamped": fingerprint is not None,
    }


def require_same_record(record: RunRecord, *stamps: dict) -> None:
    """모든 중간 산출물이 `record` 에서 나왔는지 확인하고, 아니면 **즉시 멈춘다**.

    조용히 섞이는 것이 이 검토가 반복해서 만난 실패라서, 경고가 아니라 예외다. 도장이
    아예 없는(옛 형식) 산출물도 통과시키지 않는다 — "모른다" 를 "맞다" 로 읽는 것이
    정확히 2026-09-25 에 난 일이다.
    """
    want = record_fingerprint(record)
    problems = []
    for stamp in stamps:
        if not stamp.get("stamped"):
            problems.append(
                f"  - {stamp['where']}: 출처 도장이 없다 (옛 형식). 이 파일이 "
                f"{record.path} 에서 나온 것인지 확인할 방법이 없다 — 다시 구워라")
        elif stamp["fingerprint"] != want:
            problems.append(
                f"  - {stamp['where']}: 다른 기록에서 나왔다. 도장 "
                f"{stamp['fingerprint']} (기록 {stamp.get('record')!r}) "
                f"≠ 지금 읽는 {want} ({record.path})")
    if problems:
        raise SystemExit(
            "중간 산출물이 지금 읽는 기록과 짝이 맞지 않는다:\n"
            + "\n".join(problems)
            + f"\n\n기록 {record.path} (도장 {want}) 에 맞춰 다시 만들어라:\n"
              "  1) esdf_rollout --records <기록> --dump-frames <frames.npz>\n"
              "  2) curobo/build_rollout_fields.py --frames <frames.npz> --out <fields.npz>\n"
              "자세만 새 기록이고 depth·필드가 옛 npz 인 혼합 실행을 막으려는 검사다.")


# ------------------------------------------------------- what the record says about time

#: `t_step` 이 세는 단위와 기록 한 장의 간격을 헷갈리면 지연·단계 수치가 통째로 틀어진다.
#: 한 번만 여기서 풀어 둔다.
#:
#: * `t_step` 은 **제어 스텝** 번호다 (`pi05_infer.py:1387` 의 루프 변수).
#: * 제어 스텝 하나는 `1/ctrl_hz` 초다 (`pi05_infer.py:1110`,
#:   `steps_per_action = round(1/(CTRL_HZ * timestep))` 가 그만큼 `mj_step` 을 돈다).
#: * 기록은 **청크를 새로 받을 때만** 한 장 남으므로 (`pi05_infer.py:1500`), 기록 한 장의
#:   간격은 `open_loop_horizon` 제어 스텝이다 — 그래서 `t_step` 이 8 씩 뛴다.
#:
#: 2026-09-25 에 `step7_state_lag.STEP_MS = 16.0` 이 이 셋을 섞은 것이 드러났다: sim
#: timestep(2 ms)에 8 을 곱해 16 ms 라고 했는데, 8 이 곱해질 상대는 제어 주기(66.7 ms)였다.


def record_step_interval_ms(record: RunRecord) -> float:
    """기록 한 장에서 다음 한 장까지 몇 밀리초인가. **상수로 박지 않는다.**

    `meta` 의 `ctrl_hz` · `open_loop_horizon` 과 실제 `t_step` 간격을 **둘 다** 보고,
    어긋나면 예외를 낸다. 한쪽만 믿으면 기록 형식이 바뀐 날 조용히 틀린 단위가 나온다 —
    그것이 바로 이 함수가 생긴 이유다.
    """
    meta = record.meta
    ctrl_hz = float(meta.get("ctrl_hz") or 0.0)
    if ctrl_hz <= 0:
        raise ValueError(
            f"{record.path} 의 meta.json 에 쓸 수 있는 ctrl_hz 가 없다 "
            f"(ctrl_hz={meta.get('ctrl_hz')!r}). 제어 주기를 모르면 지연을 ms 로 못 바꾼다")

    strides = {int(b.t_step) - int(a.t_step)
               for a, b in zip(record.steps, record.steps[1:])}
    if len(record.steps) < 2:
        stride = int(meta.get("open_loop_horizon") or 0)
    elif len(strides) != 1:
        raise ValueError(
            f"{record.path} 의 t_step 간격이 일정하지 않다: {sorted(strides)}. "
            f"기록 한 장이 몇 제어 스텝을 덮는지 정할 수 없다")
    else:
        stride = strides.pop()
    if stride <= 0:
        raise ValueError(f"{record.path} 의 t_step 간격을 정할 수 없다 (stride={stride})")

    declared = int(meta.get("open_loop_horizon") or 0)
    if declared and declared != stride:
        raise ValueError(
            f"{record.path}: meta 의 open_loop_horizon={declared} 인데 실제 t_step 간격은 "
            f"{stride} 다. 둘이 다르면 어느 쪽이 참인지 모르므로 멈춘다")
    return 1000.0 * stride / ctrl_hz


def control_step_ms(record: RunRecord) -> float:
    """제어 스텝 하나의 밀리초. `1000 / ctrl_hz`."""
    ctrl_hz = float(record.meta.get("ctrl_hz") or 0.0)
    if ctrl_hz <= 0:
        raise ValueError(f"{record.path} 의 meta.json 에 쓸 수 있는 ctrl_hz 가 없다")
    return 1000.0 / ctrl_hz


# ------------------------------------------------------ 단계 경계를 기록에서 뽑는다

#: 파지 시작 앞의 두 구간이 몇 제어 스텝인가. 14D 시절 `(24, 56, 72)` 를 그대로 되살리는
#: 값이다 — 아래 `phase_boundaries_from_record` 의 docstring 에 검산이 있다.
PRE_GRASP_CONTROL_STEPS = 16
APPROACH_CONTROL_STEPS = 32


def gripper_columns_for_state(width: int) -> tuple[int, int]:
    """`state`/`actions` 의 폭에서 그리퍼 두 열을 낸다. 14D 는 `(6, 13)`, 16D 는 `(7, 15)`.

    폭에서 파생시키는 이유는 `trajopt.wire.gripper_columns` 와 같다: 상수로 박으면 차원을
    바꿀 때 **엉뚱한 열을 그리퍼로 읽고** 그것이 조용히 지나간다.
    """
    width = int(width)
    if width < 4 or width % 2:
        raise ValueError(f"액션 폭이 2*(N+1) 꼴이어야 한다: {width}")
    n = width // 2 - 1
    return (n, 2 * n + 1)


def grasp_onset(record: RunRecord, *, open_tolerance: float = 0.02):
    """파지가 시작되는 지점. `(기록 인덱스, t_step, 어느 손, 그리퍼 값)` 또는 `None`.

    **관측된 그리퍼가 열린 값에서 처음 벗어나는 곳**을 찾는다. `state` 는 그 호출 *시점의*
    관측이고 손을 닫으라는 지령은 **한 호출 앞의 청크**에 있었으므로, 단계 경계로 쓸 때는
    한 칸 앞당겨야 한다 — `phase_boundaries_from_record` 가 그 몫을 한다.
    """
    if not record.steps:
        return None
    width = int(np.asarray(record.steps[0].state).reshape(-1).shape[0])
    cols = gripper_columns_for_state(width)
    states = np.stack([np.asarray(s.state, np.float64).reshape(-1) for s in record.steps])
    for hand, col in zip(("left", "right"), cols):
        trace = states[:, col]
        open_value = float(trace[0])
        hit = np.flatnonzero(trace < open_value - float(open_tolerance))
        if hit.size:
            i = int(hit[0])
            return i, int(record.steps[i].t_step), hand, float(trace[i])
    return None


def phase_boundaries_from_record(record: RunRecord, *, fallback=(24, 56, 72)):
    """`(transit, approach, pre_grasp)` 제어 스텝 경계와, 그것을 어떻게 얻었는지.

    반환은 `(boundaries, evidence)` 이고 `evidence` 는 로그·JSON 에 그대로 실을 dict 다.
    **어떻게 얻었는지를 같이 돌려주는 것이 이 함수의 요점이다** — 경계는 판정이라서, 숫자만
    돌려주면 읽는 쪽이 그것을 측정으로 오해한다.

    규칙: 파지 시작 `g` 를 그리퍼 관측이 처음 벗어나는 t_step 에서 **한 기록 간격 앞당긴**
    값으로 잡고, 그 앞에 `pre_grasp` 16 · `approach` 32 제어 스텝을 둔다.

    **이 규칙은 새로 지어낸 것이 아니라 옛 상수를 만든 규칙이다.** `run_0004`(14D)에서
    왼 그리퍼는 기록 인덱스 10, `t_step=80` 에서 1.0 → 0.732 로 벗어난다. 간격이 8 이므로
    `g = 80 - 8 = 72`, `pre_grasp = 72 - 16 = 56`, `approach = 72 - 48 = 24` —
    하드코딩되어 있던 `(24, 56, 72)` 와 **세 값이 모두 일치한다.**

    긴 16D 기록에서는 벗어남이 `t_step=264`(인덱스 33) 이라 `(208, 240, 256)` 이 된다.
    그래서 옛 상수를 그대로 쓰면 호출 9~49 가 전부 `grasp` 으로 잘못 라벨됐던 것이다.
    """
    onset = grasp_onset(record)
    if onset is None:
        return tuple(int(x) for x in fallback), {
            "source": "fallback",
            "why": "그리퍼가 기록 내내 열린 값에서 벗어나지 않는다 — 파지가 없는 기록이다",
            "boundaries": [int(x) for x in fallback],
        }
    index, t_step, hand, value = onset
    stride = int(round(record_step_interval_ms(record)
                       / control_step_ms(record)))
    grasp = int(t_step) - stride
    pre_grasp = grasp - PRE_GRASP_CONTROL_STEPS
    approach = pre_grasp - APPROACH_CONTROL_STEPS
    boundaries = (approach, pre_grasp, grasp)
    if not (0 <= approach < pre_grasp < grasp):
        return tuple(int(x) for x in fallback), {
            "source": "fallback",
            "why": f"파지가 너무 일러 앞 구간이 안 들어간다 (파지 t_step={grasp}) — "
                   f"경계 {boundaries} 가 순서를 깬다",
            "grasp_onset": {"index": index, "t_step": int(t_step), "hand": hand,
                            "gripper": value},
            "boundaries": [int(x) for x in fallback],
        }
    return boundaries, {
        "source": "gripper",
        "grasp_onset": {"index": index, "t_step": int(t_step), "hand": hand,
                        "gripper": value},
        "record_stride_control_steps": stride,
        "rule": "파지 = 그리퍼가 벗어난 t_step − 기록 간격 1 칸 (지령은 한 호출 앞의 "
                "청크에 있었다). 그 앞에 pre_grasp 16 · approach 32 제어 스텝",
        "boundaries": [int(x) for x in boundaries],
    }


def phase_boundaries_for_path(run_dir, *, fallback=(24, 56, 72)):
    """`phase_boundaries_from_record` 와 같은데 **기록 전체**를 본다. 경로를 받는다.

    `--frames N` 으로 앞 몇 장만 읽은 `RunRecord` 에 경계를 물으면, 파지가 그 뒤에 있을 때
    "파지가 없다" 는 답이 나와 조용히 fallback 으로 떨어진다. 그러면 잘라 읽은 실행과 전부
    읽은 실행이 **다른 단계 정의**를 쓰게 된다 — 이 STEP 이 고치려던 바로 그 실패다.

    그래서 경계는 언제나 기록 전체에서 뽑는다. `t_step` 과 `state` 만 읽으므로 (npz 는 지연
    로딩이라 이미지가 풀리지 않는다) 50 장짜리도 싸다.
    """
    path = pathlib.Path(run_dir)
    meta = json.loads((path / "meta.json").read_text())
    files = sorted(path.glob("step_*.npz"))
    if not files:
        raise ValueError(f"no step_*.npz files in {path}")
    steps = []
    for file in files:
        with np.load(file) as d:
            steps.append(StepRecord(
                t_step=int(d["t_step"]), qpos=np.asarray(d["qpos"], np.float64),
                qvel=np.zeros(0), state=np.asarray(d["state"], np.float64),
                actions=np.zeros((0, 0), np.float32), infer_ms=0.0, images={}))
    return phase_boundaries_from_record(
        RunRecord(path=path, meta=meta, steps=steps), fallback=fallback)


def phase_for(t_step: int, boundaries) -> str:
    """제어 스텝 -> 단계. 접촉 허가와 target 파냄이 여기에 달려 있다.

    단계는 원래 과제 계층이 정하는 것이고 (`AG3S.attach()` 가 *"AG3S never calls this
    itself"* 라고 못박는 것과 같은 이유다), 재생에서는 경계로 흉내 낸다. 그 경계를
    **한 군데서만** 만들어 쓰려고 여기 둔다 — 예전에는 `esdf_rollout` 과
    `curobo/ground_truth` 가 같은 상수를 따로 들고 있었고, 한쪽만 고치면 두 수치가
    다른 단계 정의 위에서 나왔다.
    """
    transit, approach, pre_grasp = boundaries
    if t_step < transit:
        return "transit"
    if t_step < approach:
        return "approach"
    if t_step < pre_grasp:
        return "pre_grasp"
    return "grasp"


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
    "APPROACH_CONTROL_STEPS",
    "PROVENANCE_FINGERPRINT_KEY",
    "PROVENANCE_FRAMES_KEY",
    "PROVENANCE_KEY",
    "ATTENTION_GRID",
    "CAMERA_BINDINGS",
    "LANGUAGE_TOKEN_START",
    "POLICY_CAMERA_NAMES",
    "PRE_GRASP_CONTROL_STEPS",
    "PolicyRecordWriter",
    "RunRecord",
    "StepRecord",
    "control_step_ms",
    "grasp_onset",
    "gripper_columns_for_state",
    "load_run",
    "patch_coverage",
    "patch_of_uv",
    "patch_to_uv",
    "phase_boundaries_for_path",
    "phase_boundaries_from_record",
    "phase_for",
    "pose_scene",
    "provenance_arrays",
    "read_provenance",
    "record_fingerprint",
    "require_same_record",
    "record_step_interval_ms",
    "replay_scene",
]
