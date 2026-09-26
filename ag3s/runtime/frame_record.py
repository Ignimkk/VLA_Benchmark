"""프레임별 기록과 manifest — T0 이 요구하는 증거를 남기는 곳.

`AG3S_TOTAL_TEST_Prompt.md` 가 요구하는 것 셋을 한 곳에서 낸다.

1. **manifest** — 실행 중 **변하지 않는** 것. 한 번 쓰고 프레임들이 참조한다.
   버전 · GPU · 모델 경로 · 설정 · 좌표계 · seed · 선택된 backend · 렌더 주기.
2. **프레임별 기록** — observation / planning / control 마다 한 줄 JSONL.
3. **completeness 표** — 기대한 수와 실제 기록된 수, 중복 일련번호, timestamp 역전,
   설명 안 되는 `carried`/`stale`.

## 왜 세 종류로 나누나

| 프레임 | 무엇이 한 번 일어나나 | 기간 |
|---|---|---|
| observation | 카메라 캡처 + AG3S 한 바퀴 | 정책 호출 하나와 1:1 |
| planning | SQP 한 바퀴 (= 청크 하나) | `open_loop_horizon / ctrl_hz` = 533 ms |
| control | 개별 제어 스텝 | `1 / ctrl_hz` = 66.7 ms |

이 씬에서 observation 과 planning 은 1:1 이고 control 이 그 8 배다. 그래서 **거리장의
`carried` 는 control 에서만 생긴다** — planning frame 은 매번 새 필드를 받는다.

## 값을 만들어내지 않는다

프롬프트 공통 원칙 4 다. 갱신되지 않은 프레임의 값을 새로 계산한 것처럼 적지 않는다. 그래서
control frame 은 자기 필드의 상태를 `FieldProvenance.applied_by_client()` 로 채우고,
`carried`/`stale` 이면 마지막 실제 갱신의 `sequence` · `observed_at` · `age_ms` 를 함께 싣는다.

## 무엇이 없어서 세 번 막혔나 (T6a, 2026-09-26)

`T5c` 는 *"refined 가 reference 보다 여유거리가 좋은가"* 를 **미측정으로 닫았다** — 기록에
청크가 없었다. `T6` 는 75 chunk 중 42 번 멈춘 자리를 못 찾았다 — 로봇 자세도 물체 자세도
없었다. 그래서 planning frame 에 `actions` · `actions_reference` · `qpos` · `object_poses` ·
`max_violation_pair` 를, control frame 에 `qpos` 를 더했다.

**키를 더하는 것이고 바꾸는 것이 아니다.** 옛 키는 하나도 움직이지 않았고 completeness 표의
항목도 그대로다 (`plot_t0.py` 가 `frames.jsonl` 을 그 형식으로 읽는다).

한도(`timing.max_field_age_sec`)가 `None` 이면 `stale` 판정을 하지 않고
`staleness_checked: false` 가 실린다 — 검사를 안 한 것이 통과한 것으로 읽히지 않게 하는 것이
그 필드의 목적이다.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Optional

__all__ = ["FrameRecorder", "collect_manifest", "environment_versions"]


def environment_versions() -> dict:
    """이 프로세스가 실제로 import 한 것의 버전. **선언이 아니라 실측이다.**"""
    import importlib

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "torch", "jax", "jaxlib", "mujoco", "casadi", "osqp",
                 "warp", "curobo"):
        try:
            mod = importlib.import_module(name)
            out[name] = (getattr(mod, "__version__", None)
                         or getattr(getattr(mod, "config", None), "version", "?"))
        except Exception as exc:  # noqa: BLE001 — 없는 것도 기록이다
            out[name] = f"MISSING ({type(exc).__name__})"
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["gpu"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        out["cuda_available"] = None
    return out


def _jsonable(value: Any) -> Any:
    """`ndarray` 와 numpy 스칼라를 **되읽을 수 있는** 형태로 내린다.

    `json.dumps(..., default=str)` 는 배열을 만나면 예외를 내지 않고 `str()` 을 적는다 —
    `"[[0.1 0.2 ... 0.9]]"` 처럼 줄임표가 박힌, 되읽을 수 없는 문자열이다. **조용히 통과하는
    것이 문제다**: 청크 `[50, 16]` 을 그렇게 적으면 기록은 있는데 값은 없다.

    T6a 가 청크와 `qpos` 를 싣기 시작하면서 그 경로가 실제로 열렸으므로, 호출부마다
    `.tolist()` 를 기억하는 대신 여기서 한 번 내린다. numpy 를 import 하지 않는다 —
    `tolist`/`item` 이 있으면 쓰고 없으면 그대로 두는 덕분에 이 모듈은 numpy 없이도 돈다.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bytes, bool, int, float)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        # numpy 배열은 리스트로, numpy 스칼라는 `tolist()` 가 파이썬 수를 준다 (0-d 포함).
        return tolist()
    return value


def _git_commit(path: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def collect_manifest(*, seed: Optional[int], scene: dict, policy: dict, esdf: dict,
                     timing: dict, cameras: Any, render: dict,
                     extra: Optional[dict] = None) -> dict:
    """불변 설정을 한 덩어리로. T0 의 manifest 항목이 여기 다 있어야 한다."""
    man = {
        "recorded_at_wall": time.time(),
        "recorded_at_monotonic": time.monotonic(),
        "seed": seed,
        "versions": environment_versions(),
        "curobo_commit": _git_commit("/mnt/dev/work/curobo_src"),
        "scene": dict(scene),
        "policy": dict(policy),
        "esdf": dict(esdf),
        "timing": dict(timing),
        "cameras": list(cameras),
        "render": dict(render),
        # 좌표계는 **명시한다.** 이 검토에서 좌표계 오해가 두 번 났다 (cuRobo 함정 1·2).
        "frames_of_reference": {
            "points": "robot base frame, metres",
            "sign_convention": "ESDF negative = inside a surface, positive = free space",
            "esdf_origin": "centre of voxel (0,0,0), not the grid corner",
            "depth_wire": "uint16 millimetres (DEPTH_SCALE_MM = 1000)",
            "T_base_cam": "camera -> base, at the capture instant",
        },
    }
    if extra:
        man["extra"] = dict(extra)
    return man


@dataclasses.dataclass
class _Counters:
    expected_observation: int = 0
    observation: int = 0
    planning: int = 0
    control: int = 0
    third_person: int = 0
    duplicate_sequence: int = 0
    timestamp_reversal: int = 0
    unexplained_carried: int = 0
    field_state: dict = dataclasses.field(default_factory=dict)
    ipc: dict = dataclasses.field(default_factory=dict)


class FrameRecorder:
    """manifest 하나 + `frames.jsonl` 한 줄씩. 닫을 때 completeness 표를 낸다.

    **쓰기를 매 줄 flush 한다.** 실행이 중간에 죽어도 그때까지의 기록이 남아야 실패 분석에
    쓸 수 있고, 프롬프트의 실패 처리 루프가 그 기록에서 시작한다.
    """

    def __init__(self, out_dir: str, *, manifest: dict, expected_observation_frames: int = 0):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
        self._fh = (self.dir / "frames.jsonl").open("w", encoding="utf-8")
        self.counts = _Counters(expected_observation=int(expected_observation_frames))
        self._seen_seq: set[int] = set()
        self._last_stamp: dict[str, float] = {}
        self._age_limit = manifest.get("timing", {}).get("max_field_age_sec")

    # -- 한 줄 --------------------------------------------------------------------------
    def _write(self, row: dict) -> None:
        self._fh.write(json.dumps(_jsonable(row), ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def _check_order(self, kind: str, stamp: Optional[float]) -> bool:
        """timestamp 가 되돌아갔는가. T0 의 즉시 실패 조건 하나다."""
        if stamp is None:
            return False
        prev = self._last_stamp.get(kind)
        self._last_stamp[kind] = float(stamp)
        if prev is not None and float(stamp) < prev:
            self.counts.timestamp_reversal += 1
            return True
        return False

    def observation(self, *, t_step: int, stamps: dict, ag3s_status: str = "",
                    grounding_status: str = "", validity: str = "",
                    n_points: Optional[int] = None, notes: Any = (),
                    extra: Optional[dict] = None) -> None:
        newest = max(stamps.values()) if stamps else None
        reversed_ = self._check_order("observation", newest)
        self.counts.observation += 1
        self._write({
            "kind": "observation", "t_step": int(t_step),
            "camera_stamps": {k: float(v) for k, v in (stamps or {}).items()},
            "newest_stamp": newest,
            "camera_skew_sec": (None if not stamps
                                else float(max(stamps.values()) - min(stamps.values()))),
            "ag3s_status": ag3s_status, "grounding_status": grounding_status,
            "validity": validity, "n_points": n_points,
            "timestamp_reversed": reversed_,
            "notes": list(notes or ()),
            **(extra or {}),
        })

    def planning(self, *, seq: int, t_step: int, field: Any, verdict: dict,
                 timing_ms: dict, ipc: str = "ok",
                 actions: Any = None, actions_reference: Any = None,
                 qpos: Any = None, object_poses: Optional[dict] = None,
                 max_violation_pair: Optional[dict] = None,
                 extra: Optional[dict] = None) -> None:
        """청크 하나. `field` 는 `FieldProvenance` 이거나 응답의 `field` dict 다.

        **T6a 가 더한 다섯 키.** `T5`·`T6` 가 세 번 연속 같은 자리에서 막힌 것이 이유다 —
        기록에 청크도 자세도 없어 *"수정이 여유거리를 좋게 했나"* 와 *"영구히 멈춘 자리가
        어디인가"* 를 측정할 방법이 없었다.

        | 인자 | 무엇 | 없을 때 |
        |---|---|---|
        | `actions` | 서버가 계산한 **refined** 청크 `[H, ACTION_WIDTH]` | `null` |
        | `actions_reference` | 정책 **원본** 청크 | **키가 아예 안 실린다** |
        | `qpos` | 그 planning frame 의 `data.qpos` 전체 | `null` |
        | `object_poses` | 과일 넷 + crate 의 MuJoCo 참값 | `null` |
        | `max_violation_pair` | `max_violation_m` 을 만든 **제약의 신원** | `null` |

        **`actions_reference` 만 규칙이 다르다.** 그 키의 있음/없음 자체가 *"이 실행은
        shadow"* 라는 신호이기 때문이다 — 응답 쪽(`wire.ACTIONS_REFERENCE`)의 규약과 같게
        둔다. 나머지 넷은 값이 없어도 키를 남긴다: A2 가 프레임을 세로로 세므로 있음/없음이
        섞이면 파서가 두 갈래로 갈린다.
        """
        prov = self._as_prov(field)
        duplicate = seq in self._seen_seq
        if duplicate:
            self.counts.duplicate_sequence += 1
        self._seen_seq.add(int(seq))
        self.counts.planning += 1
        self.counts.ipc[ipc] = self.counts.ipc.get(ipc, 0) + 1
        state = prov.state if prov is not None else "unavailable"
        self.counts.field_state[state] = self.counts.field_state.get(state, 0) + 1
        self._write({
            "kind": "planning", "seq": int(seq), "t_step": int(t_step),
            "duplicate_sequence": duplicate,
            "ipc": ipc,
            "field": None if prov is None else prov.to_dict(),
            "verdict": dict(verdict or {}),
            "timing_ms": {k: float(v) for k, v in (timing_ms or {}).items()},
            "actions": actions,
            **({} if actions_reference is None
               else {"actions_reference": actions_reference}),
            "qpos": qpos,
            "object_poses": object_poses,
            "max_violation_pair": max_violation_pair,
            **(extra or {}),
        })

    def control(self, *, t_step: int, chunk_seq: int, step_in_chunk: int, now: float,
                field: Any, executed: bool, qpos: Any = None,
                extra: Optional[dict] = None) -> None:
        """개별 제어 스텝. **여기서 `carried`/`stale` 이 생긴다.**

        `step_in_chunk > 0` 이면 이 스텝의 기하는 이번 프레임에 갱신된 것이 아니다. 그것을
        `carried` 로 적고 마지막 실제 갱신을 함께 싣는 것이 프롬프트 공통 원칙 4 다.

        **`qpos` 는 T6a 가 더했다** (없으면 `null`). `executed` 는 *"실행하기로 했나"* 이고
        `qpos` 는 *"그래서 팔이 어디 있었나"* 다. T6 에서 `seq 38` 부터 38 chunk 가 연속으로
        멈췄는데 그 자리가 어디인지 알 방법이 없었던 것이 이 키가 없었기 때문이다.

        **`apply_action` 직후, `mj_step` 전의 값이다** — 즉 이 action 이 지령된 순간의 자세다.
        스텝 뒤 값을 적으면 "지령"과 "결과"가 한 프레임 밀려 기록된다.
        """
        prov = self._as_prov(field)
        applied = (None if prov is None else
                   prov.applied_by_client(now, age_limit_sec=self._age_limit,
                                          carried=step_in_chunk > 0))
        state = applied.state if applied is not None else "unavailable"
        self.counts.control += 1
        self.counts.field_state[state] = self.counts.field_state.get(state, 0) + 1
        # `carried`/`stale` 인데 마지막 갱신을 못 가리키면 설명이 안 되는 것이다.
        if state in ("carried", "stale") and (
                applied is None or applied.sequence < 0 or applied.observed_at is None):
            self.counts.unexplained_carried += 1
        self._write({
            "kind": "control", "t_step": int(t_step), "chunk_seq": int(chunk_seq),
            "step_in_chunk": int(step_in_chunk), "now": float(now),
            "executed": bool(executed),
            "field": None if applied is None else applied.to_dict(),
            # `carried`/`stale` 의 근거 — 마지막 실제 갱신이 무엇이었나.
            "last_field_update": (None if applied is None else {
                "sequence": applied.sequence, "backend": applied.backend,
                "frame_id": applied.frame_id, "frame_index": applied.frame_index,
                "observed_at": applied.observed_at, "age_ms": applied.age_ms}),
            "qpos": qpos,
            **(extra or {}),
        })

    def third_person(self, *, t_step: int, path: str) -> None:
        self.counts.third_person += 1
        self._write({"kind": "third_person", "t_step": int(t_step), "path": str(path)})

    # -- 닫기 ---------------------------------------------------------------------------
    @staticmethod
    def _as_prov(field: Any):
        from benchmark.ag3s.fields.provenance import FieldProvenance

        if field is None:
            return None
        if isinstance(field, FieldProvenance):
            return field
        if isinstance(field, dict):
            return FieldProvenance.from_dict(field)
        return None

    def completeness(self, *, ipc_stats: Optional[dict] = None) -> dict:
        """프롬프트 §5 의 표. **기대와 실제가 같은가**를 센다."""
        c = self.counts
        expected = c.expected_observation or c.observation
        table = {
            "expected_observation_frames": expected,
            "captured_observation_frames": c.observation,
            "planning_records": c.planning,
            "control_records": c.control,
            "third_person_frames": c.third_person,
            "planned_missing_frames": max(0, expected - c.observation),
            "duplicate_sequence_ids": c.duplicate_sequence,
            "timestamp_reversals": c.timestamp_reversal,
            "unexplained_carried_or_stale": c.unexplained_carried,
            "field_state_counts": dict(c.field_state),
            "ipc_outcomes": dict(c.ipc),
            "staleness_limit_sec": self._age_limit,
            "staleness_checked": self._age_limit is not None,
        }
        if ipc_stats:
            table["client_ipc_stats"] = dict(ipc_stats)
        # 통과 조건은 프롬프트가 정한 것 그대로다.
        table["pass"] = bool(
            table["captured_observation_frames"] == table["expected_observation_frames"]
            and table["planned_missing_frames"] == 0
            and table["duplicate_sequence_ids"] == 0
            and table["timestamp_reversals"] == 0
            and table["unexplained_carried_or_stale"] == 0)
        return table

    def close(self, *, ipc_stats: Optional[dict] = None) -> dict:
        table = self.completeness(ipc_stats=ipc_stats)
        (self.dir / "completeness.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=1, default=str))
        self._fh.close()
        return table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._fh.closed:
            self.close()
