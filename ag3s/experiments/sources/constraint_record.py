"""AG3S 가 충돌 제약을 만드는 과정을 청크마다 통째로 남긴다.

`policy_record.PolicyRecordWriter` 는 **정책의 입출력**(이미지·depth·state·chunk)을 남긴다.
여기 있는 것은 그 다음 층이다: 그 관측에서 AG3S 가 무엇을 뽑아냈는가 — attention 이 어디를
가리켰고, 어느 점이 target 이 됐고, 거리장이 무엇을 담았고, 제약 행이 로봇 구마다 얼마의 여유를
말했는가.

**왜 나중에 다시 계산하지 않고 저장하는가.** AG3S 는 결정적이지 않다. 클러스터링 시드, 정책이
돌려준 청크, 그리고 live 루프에서는 캡처 시각까지 실행마다 다르다. 재실행으로 재현하려 들면
"그때 그 프레임" 이 아니라 "비슷한 프레임" 을 보게 되고, 진단하려던 이상 프레임이 바로 그
차이 속으로 사라진다.

저장 형식은 `PolicyRecordWriter` 의 관례를 따른다 — 청크당 압축 npz 하나, 배열 키는 평평하게.
읽는 쪽이 이 모듈을 import 하지 않아도 `np.load` 만으로 열린다.
"""

from __future__ import annotations

import atexit
import json
import logging
import pathlib
import queue
import threading
import traceback
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["ConstraintRecordWriter", "load_constraint_run", "ESDF_MODES"]

#: `esdf_mode` 가 받는 값. 용량과 진단 가능성의 교환이다.
#:   none      — 격자 메타만. 청크당 수 KB.
#:   occupancy — 점유 상태(uint8). 기하가 어떻게 분리됐는지는 보이지만 거리장은 못 본다.
#:   full      — 거리장까지 float16. 청크당 약 1.1 MB (76x90x80, 20 mm 복셀 실측).
ESDF_MODES = ("none", "occupancy", "full")

#: 이 기록기가 **거리장을 실을 수 있는** backend. `esdf.backend` 의 값과 같은 문자열이다.
#:
#: 목록을 두는 이유는 하나다 — 서버가 **시작할 때** 거절할 수 있어야 한다. 2026-09-28 에
#: `--record-constraints --record-constraints-esdf occupancy` 로 75 chunk 를 돌렸는데 전부
#: `'CuroboEsdfField' object has no attribute 'max_distance'` 로 실패했고, 실패가 `print` 한
#: 줄이라 실행이 끝까지 갔다. 끝나고 **빈 디렉토리**를 본 것이 그 조용함의 대가다.
SUPPORTED_ESDF_BACKENDS = ("legacy", "curobo")


class _DeferredGrid:
    """GPU 에 있는 격자를 **쓰는 순간에** host 로 가져오는 자리표시 (T38 B3).

    cuRobo 계층(`DeviceEsdfField`)은 값 격자를 GPU 에 둔다. `--record-constraints-esdf full` 이
    그 격자를 실을 때, 요청 스레드에서 D2H 하지 않고 이것을 payload 에 넣는다 — background
    writer 는 worker 스레드에서, 동기 writer 는 `savez` 직전에 `resolve()` 한다.

    잡고 있는 것은 GPU tensor 하나뿐이고 (필드 객체가 아니다) `resolve` 는 **아무것도 바꾸지
    않는다** — 필드의 host 캐시를 채우지 않고, tensor 는 builder 가 계층마다 새로 만든 사본이라
    아무도 고치지 않는다. 기록 바이트는 예전과 같다: float32 host 배열 → `np.asarray(…, dtype)`.
    """

    __slots__ = ("_tensor", "_dtype")

    def __init__(self, tensor, dtype):
        self._tensor = tensor
        self._dtype = np.dtype(dtype)

    def resolve(self) -> np.ndarray:
        return np.asarray(self._tensor.detach().cpu().numpy(), self._dtype)


def _resolve_deferred(payload: dict) -> dict:
    return {k: (v.resolve() if isinstance(v, _DeferredGrid) else v) for k, v in payload.items()}


class ConstraintRecordWriter:
    """청크마다 AG3S 의 제약 생성 중간 산출물을 npz 로 쓴다.

    Args:
        output_dir: 루트. 하위에 `run_0000`, `run_0001`… 을 새로 만든다 (덮어쓰지 않는다).
        esdf_mode: `ESDF_MODES` 참고.
        max_points: 저장할 점군의 상한. 넘으면 **결정적으로** 솎아낸다(고정 스트라이드).
            무작위 표본이 아니라 스트라이드인 이유는, 같은 파일을 두 번 읽어 다른 그림이 나오면
            그림을 신뢰할 수 없기 때문이다. 0 이면 상한 없음.
        meta: 실행 조건. 나중에 두 실행을 비교할 때의 유일한 근거다.
        background: True 면 npz 쓰기(압축 · 파일 I/O)를 **별도 스레드**에서 한다 (T38 B5).
            서버 응답 경로에서 청크당 ≈ 245 ms 였고 그 대부분이 `zipfile.write` 였다
            (T38 Phase A, py-spy). 기록 **내용**은 동기 모드와 같다 — 무엇을 담을지는 여전히
            `record()` 안에서, 호출 스레드에서, 그 순간의 객체로 정한다 (`summary_json` 직렬화와
            배열 복사까지). worker 는 이미 정해진 바이트를 디스크에 옮길 뿐이다.

            - 순서: 파일 이름(`chunk_%05d`)은 `record()` 를 부른 순서로 그 자리에서 정하고, worker
              는 FIFO 하나로 쓴다.
            - 역압: 큐(`queue_size`)가 차면 `record()` 가 **기다린다** — 버리지 않는다.
            - 종료: `flush()` 가 큐를 비울 때까지 기다리고, `close()` 는 flush 뒤 worker 를
              멈춘다. 프로세스 종료(`atexit`)에도 flush 한다.
            - 실패: worker 의 예외는 기록을 **조용히 버리지 않는다** — 첫 실패는 크게, 이후는 한
              줄씩 로그에 남기고 `n_failed` · `failed_paths` 에 센다. 실패한 이름은 비어 있는
              번호로 남는다 (동기 모드는 같은 번호를 다음 청크가 다시 쓴다).
        queue_size: background 큐의 상한 (청크 수).
    """

    def __init__(self, output_dir, *, esdf_mode: str = "full", max_points: int = 200_000,
                 meta: Optional[dict[str, Any]] = None, background: bool = False,
                 queue_size: int = 8):
        if esdf_mode not in ESDF_MODES:
            raise ValueError(f"esdf_mode must be one of {ESDF_MODES}, got {esdf_mode!r}")
        self.esdf_mode = esdf_mode
        self.max_points = int(max_points)
        root = pathlib.Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for i in range(10_000):
            run_dir = root / f"run_{i:04d}"
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            self.run_dir = run_dir
            break
        else:
            raise RuntimeError(f"no available run directory under {root}")
        self.meta: dict[str, Any] = {"esdf_mode": esdf_mode, "max_points": self.max_points,
                                     **(meta or {})}
        #: 디스크에 **쓰인** 청크 수 (background 에서는 worker 가 올린다).
        self._count = 0
        #: background 에서 다음 청크가 받을 번호. 동기 모드는 `_count` 를 그대로 쓴다.
        self._next_index = 0
        #: background 에서 쓰기에 실패한 청크 수와 그 파일 이름.
        self.n_failed = 0
        self.failed_paths: list[str] = []
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        if background:
            self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
            self._worker = threading.Thread(target=self._drain, name="constraint-record-writer",
                                            daemon=True)
            self._worker.start()
            # daemon 스레드는 인터프리터가 끝날 때 그냥 죽는다. 그 전에 큐를 비운다 — atexit 은
            # daemon 스레드가 아직 살아 있을 때 돈다.
            atexit.register(self._atexit_flush)

    # ----------------------------------------------------------------------------------
    def record(self, *, t_step: int, chunk_index: int, constraint_set, debug: dict[str, Any],
               reference_chunk: Optional[np.ndarray] = None,
               refined_chunk: Optional[np.ndarray] = None,
               sphere_centres: Optional[np.ndarray] = None,
               sphere_radii: Optional[np.ndarray] = None,
               sphere_link_names: Optional[Sequence[str]] = None,
               clearance: Optional[np.ndarray] = None,
               to_result: Optional[Any] = None,
               attention_maps: Optional[dict[str, np.ndarray]] = None,
               occupancy: Optional[np.ndarray] = None,
               summary_extra: Optional[dict[str, Any]] = None) -> pathlib.Path:
        """한 청크. `debug` 는 `AG3S.process_multi_debug` 가 돌려준 두 번째 값 그대로.

        `occupancy` 는 따로 받는다. `EsdfField` 는 거리장만 들고 있고 세 상태 점유 배열은
        `EsdfBuilder` 안에 남기 때문이다 — 그 분리는 의도된 것이라(필드는 TO 가 읽는 것,
        점유는 그것을 만든 재료) 여기서 필드를 파고들어 꺼내오지 않는다.

        `summary_extra` 는 `summary_json` 에 **그대로 더할** JSON 값들이다 (T18 — `SafePolicy` 가
        `exec_feedback` · `exec_continuity` · `exec_latch_signal` 을 싣는다). 이 기록기가 모르는
        키를 여기서 해석하지 않는다. 이미 있는 키와 겹치면 기록기 쪽 값이 이긴다.
        """
        payload: dict[str, Any] = {
            "t_step": np.int64(t_step),
            "chunk_index": np.int64(chunk_index),
        }
        summary: dict[str, Any] = {
            "t_step": int(t_step),
            "chunk_index": int(chunk_index),
            "status": getattr(getattr(constraint_set, "status", None), "value", None),
            "validity": getattr(getattr(constraint_set, "validity", None), "value", None),
            "has_target": bool(getattr(constraint_set, "has_target", False)),
            "n_candidates": len(getattr(constraint_set, "candidates", ()) or ()),
            "notes": list(getattr(constraint_set, "notes", ()) or ()),
            "profile_ms": dict(getattr(constraint_set, "profile", {}) or {}),
        }

        # --- 지각 중간물 -------------------------------------------------------------
        # `debug` 의 클라우드는 `PointCloud` / `AttentionPointCloud` 객체다. 점 배열만 뽑되
        # attention 은 정규화본과 원본을 **둘 다** 남긴다 — 임계값은 정규화본을 보고 피크는
        # 원본을 본다는 것이 그 자료구조가 두 배열을 들고 있는 이유이고, 하나만 저장하면
        # 나중에 "왜 저 점이 시드가 됐나" 를 되짚을 수 없다.
        for key in ("raw_cloud", "filtered_cloud"):
            cloud = debug.get(key)
            if cloud is not None:
                payload[key] = self._thin(np.asarray(cloud.points, np.float32))
        attention_cloud = debug.get("attention_cloud")
        if attention_cloud is not None:
            payload["attention_cloud"] = self._thin(
                np.asarray(attention_cloud.cloud.points, np.float32))
            payload["attention_values"] = self._thin(
                np.asarray(attention_cloud.attention, np.float32))
            if attention_cloud.raw_attention is not None:
                payload["attention_raw"] = self._thin(
                    np.asarray(attention_cloud.raw_attention, np.float32))
        seeds = debug.get("seed_indices")
        if seeds is not None and len(seeds):
            payload["seed_indices"] = np.asarray(seeds, np.int32)
        mask = debug.get("support_mask")
        if mask is not None:
            payload["support_mask"] = self._thin(np.asarray(mask, bool))

        # --- attention: 카메라별 원본 맵 ---------------------------------------------
        # 역투영·grounding 을 나중에 다른 파라미터로 다시 돌려보려면 맵 자체가 있어야 한다.
        # 파생물(attention_cloud)만 남기면 "다른 임계값이었다면" 을 물어볼 수 없다.
        for cam, amap in (attention_maps or {}).items():
            # `CameraID` 는 enum 이라 f-string 이 "CameraID.HEAD" 를 낸다. npz 키는 나중에
            # 사람이 읽을 이름이어야 하므로 값을 쓴다.
            payload[f"attention_{getattr(cam, 'value', cam)}"] = np.asarray(amap, np.float16)

        # --- target grounding --------------------------------------------------------
        grounding = debug.get("grounding")
        if grounding is not None:
            summary["grounding_status"] = getattr(getattr(grounding, "status", None), "value", None)
            summary["grounding_best_score"] = float(getattr(grounding, "best_score", 0.0))
            summary["n_clusters"] = len(getattr(grounding, "clusters", ()) or ())
            summary["attention_peak_index"] = int(getattr(grounding, "attention_peak_index", -1))
        # T20: 조작 대상 정체 (`AG3S.manipulated.record()`, 파이프라인이 metrics 에 싣는다).
        # `state == "occluded"` 인 청크에서 접촉 허용이 살아 있었는지는 이것 없이 알 수 없다.
        summary["manipulated"] = (getattr(constraint_set, "metrics", None) or {}).get("manipulated")
        # T26: 제외 연속성 (`{active, source, mechanism, ...}`) · destination · 불변식 위반 ·
        # 이번 프레임 cluster 들의 admissibility — 기록 키 추가만 (T20 의 `manipulated` 와 같은 방식).
        # T29: `finger_joints` — 이 프레임의 FK 가 쓴 손가락 관절값과 그 출처.
        for key in ("exclusion", "destination", "invariant_violation", "admissibility",
                    "gripper_max_opening", "finger_joints"):
            summary[key] = (getattr(constraint_set, "metrics", None) or {}).get(key)
        target = getattr(constraint_set, "target", None)
        if target is not None:
            payload["target_points"] = np.asarray(target.points, np.float32)
            payload["target_point_indices"] = np.asarray(target.point_indices, np.int32)
            payload["target_centroid"] = np.asarray(target.centroid, np.float64)
            summary["target_attention_score"] = float(target.attention_score)
            summary["target_confidence"] = float(target.confidence)
            summary["target_n_points"] = int(len(target.points))

        # --- 거리장 -------------------------------------------------------------------
        field = getattr(constraint_set, "esdf", None)
        if field is not None:
            summary.update(self._write_field(field, payload, occupancy))

        # --- 제약 행 -------------------------------------------------------------------
        if sphere_centres is not None:
            payload["sphere_centres"] = np.asarray(sphere_centres, np.float32)
        if sphere_radii is not None:
            payload["sphere_radii"] = np.asarray(sphere_radii, np.float32)
        if sphere_link_names is not None:
            summary["sphere_link_names"] = list(sphere_link_names)
        if clearance is not None:
            clearance = np.asarray(clearance, np.float32)
            payload["clearance"] = clearance
            if clearance.size:
                summary["clearance_min_mm"] = float(clearance.min()) * 1000.0
                summary["n_violating_rows"] = int((clearance < 0).sum())
                # 지평 전체 (H, S) 로 들어오면 "어느 구가 한 번이라도 위반했나" 가 더 읽힌다.
                if clearance.ndim == 2:
                    summary["n_violating_spheres"] = int((clearance < 0).any(axis=0).sum())
                    summary["clearance_min_per_step_mm"] = [
                        float(v) * 1000.0 for v in clearance.min(axis=1)]

        # --- 청크 (원본과 TO 결과를 **둘 다**) -----------------------------------------
        # closed loop 에서는 TO 가 궤적을 바꾸므로 "원본이었다면" 을 같은 실행에서 볼 수 없다.
        # 원본 청크를 남기는 것이 그 대조군을 부분적으로나마 복원하는 유일한 방법이다.
        if reference_chunk is not None:
            payload["reference_chunk"] = np.asarray(reference_chunk, np.float32)
        if refined_chunk is not None:
            payload["refined_chunk"] = np.asarray(refined_chunk, np.float32)
        if to_result is not None:
            metrics = dict(getattr(to_result, "metrics", None) or {})
            summary["to"] = {
                "status": getattr(getattr(to_result, "status", None), "value", None),
                "iterations": int(getattr(to_result, "iterations", 0) or 0),
                "solve_ms": float(getattr(to_result, "solve_time_ms",
                                          getattr(to_result, "solve_ms", 0.0)) or 0.0),
                # **구조화된 값** (T15). `notes` 의 문장으로는 "몇 번 돌았나" 를 셀 수 없었다.
                "sqp_iterations": int(metrics.get("sqp_iterations", 0) or 0),
                "qp_iterations": int(metrics.get("qp_iterations", 0) or 0),
                "max_iterations": int(metrics.get("max_iterations", 0) or 0),
                "time_budget_ms": float(metrics.get("time_budget_ms", 0.0) or 0.0),
                "time_budget_hit": bool(metrics.get("time_budget_hit", False)),
                "max_iterations_hit": bool(metrics.get("max_iterations_hit", False)),
                # 충돌이 꺼진 기록이 "위반 0" 으로 읽히면 안 된다.
                "collision_enabled": bool(metrics.get("collision_enabled", True)),
                # 노트가 상태를 해석하는 유일한 근거다. `VIOLATED` 에는 두 갈래가 있다 —
                # 실제로 관통이 남은 경우와, 제약은 전부 통과했지만 AG3S 가 기하를 인증하지
                # 못한 경우. 둘은 완전히 다른 문제이고 상태값만으로는 구분되지 않는다.
                "notes": list(getattr(to_result, "notes", ()) or ()),
                "metrics": {k: _scalar(v) for k, v in (getattr(to_result, "metrics", {}) or {}).items()},
            }

        for key, value in (summary_extra or {}).items():
            summary.setdefault(key, value)
        payload["summary_json"] = np.asarray(json.dumps(summary, ensure_ascii=False))
        if self._queue is None:
            out = self.run_dir / f"chunk_{self._count:05d}.npz"
            np.savez_compressed(out, **_resolve_deferred(payload))
            self._count += 1
            return out
        # background (T38 B5): 이름은 지금 정하고, 배열은 **지금 복사한다**. `np.asarray` 는
        # dtype 이 같으면 원본을 그대로 돌려주고 `_thin` 은 view 를 돌려주므로, 복사하지 않으면
        # 다음 청크가 그 버퍼를 고칠 때 아직 안 쓰인 기록이 바뀐다.
        if not self._worker.is_alive():
            # 살아 있지 않은 worker 의 큐에 넣으면 아무도 안 쓰고, 큐가 차면 여기서 영원히 선다.
            # 예외로 올린다 — `SafePolicy._record` 가 세고 크게 말한다.
            raise RuntimeError(f"constraint record writer thread is not running "
                               f"({self.pending} chunk(s) unwritten in {self.run_dir})")
        out = self.run_dir / f"chunk_{self._next_index:05d}.npz"
        self._next_index += 1
        # GPU 격자(`_DeferredGrid`)는 복사하지 않는다 — worker 가 쓰기 직전에 가져온다. 그 tensor 는
        # builder 가 계층마다 새로 만든 사본이라 다음 청크가 고치지 않는다.
        snapshot = {k: (v if isinstance(v, _DeferredGrid) else np.array(v, copy=True))
                    for k, v in payload.items()}
        self._queue.put((out, snapshot))  # 차 있으면 기다린다 — 버리지 않는다
        return out

    # ----------------------------------------------------------------------------------
    def _drain(self) -> None:
        """background worker. 큐에서 하나씩 꺼내 쓴다. 어떤 예외도 루프를 죽이지 않는다."""
        q = self._queue
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                out, payload = item
                try:
                    np.savez_compressed(out, **_resolve_deferred(payload))
                    self._count += 1
                except Exception as exc:  # noqa: BLE001 — 세고 크게 말한다; 루프는 산다
                    self.n_failed += 1
                    self.failed_paths.append(out.name)
                    log = logging.getLogger(__name__)
                    if self.n_failed == 1:
                        log.error(
                            "!!! CONSTRAINT RECORD WRITE FAILED (background writer) !!!\n"
                            "    %s: %s: %s\n    이 청크의 기록은 디스크에 없습니다. 이후 실패도 "
                            "세어 `n_failed` 로 남깁니다.\n%s",
                            out, type(exc).__name__, exc, traceback.format_exc())
                    else:
                        log.error("[ag3s] constraint record write failed (%d 번째): %s: %s: %s",
                                  self.n_failed, out.name, type(exc).__name__, exc)
            finally:
                # 다음 `get()` 까지 기다리는 동안 이미 쓴 청크의 배열을 쥐고 있지 않는다.
                item = out = payload = None
                q.task_done()

    @property
    def background(self) -> bool:
        return self._queue is not None

    @property
    def pending(self) -> int:
        """아직 디스크에 안 쓰인 청크 수 (background). 동기 모드는 0."""
        return 0 if self._queue is None else self._queue.unfinished_tasks

    def flush(self) -> None:
        """background 큐가 빌 때까지 (마지막 청크가 쓰이거나 실패로 세어질 때까지) 기다린다.

        `Queue.join()` 을 그대로 쓰지 않는 이유: worker 가 어떤 이유로든 죽어 있으면 영원히
        선다 (종료 시 `atexit` 에서 서면 프로세스가 안 끝난다). 그 경우 남은 수를 실패로 센다.
        """
        q = self._queue
        if q is None:
            return
        with q.all_tasks_done:
            while q.unfinished_tasks:
                if not self._worker.is_alive():
                    lost = q.unfinished_tasks
                    self.n_failed += lost
                    logging.getLogger(__name__).error(
                        "!!! constraint record writer thread died with %d chunk(s) unwritten "
                        "in %s !!!", lost, self.run_dir)
                    return
                q.all_tasks_done.wait(0.5)

    def _atexit_flush(self) -> None:
        if self._queue is None:
            return
        pending = self.pending
        self.flush()
        print(f"[ag3s] constraint recorder flushed at exit ({pending} pending): "
              f"{self._count} written, {self.n_failed} failed -> {self.run_dir}", flush=True)

    def _stop_worker(self) -> None:
        if self._queue is None:
            return
        self.flush()
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join()
        self._queue = None
        self._worker = None
        atexit.unregister(self._atexit_flush)

    # ----------------------------------------------------------------------------------
    @staticmethod
    def field_layers(field) -> list[tuple[str, Any]]:
        """`[(이름, 단일 계층 필드), ...]` — legacy 든 cuRobo 합성이든 같은 모양으로 본다.

        이름은 기록의 키 접두가 되므로 **규약**이다.

        | 이름 | 무엇 |
        |---|---|
        | `""` (빈 문자열) | 질의가 답하는 **주 격자**. legacy 는 그 자체, cuRobo 는 가장 미세한 계층 (`field.grid` 와 같은 격자) |
        | `"coarse"` … | cuRobo 의 나머지 계층. 거친 것부터 |
        | `"free"` | **target 이 빠진 계층** (T8b). 손끝이 실제로 묻는 쪽이므로 유령을 찾을 때 이것이 정본이다 |

        지원하지 않는 필드면 **여기서 죽는다.** 조용히 빈 목록을 돌려주면 격자 없는 기록이
        남고, 그것은 없는 기록보다 나쁘다 (읽는 사람이 "거리장이 없던 프레임" 으로 읽는다).
        """
        layers = getattr(field, "layers", None)
        if layers is None:
            if not hasattr(field, "distance_grid") or not hasattr(field, "max_distance"):
                raise TypeError(
                    f"{type(field).__name__} 은 이 기록기가 아는 거리장 인터페이스가 아닙니다 "
                    f"(단일 계층은 `distance_grid`·`max_distance`·`grid`, 합성은 `layers`). "
                    f"지원 backend: {list(SUPPORTED_ESDF_BACKENDS)}")
            return [("", field)]
        layers = list(layers)
        if not layers:
            raise TypeError(f"{type(field).__name__}.layers 가 비어 있습니다")
        # `field.grid` 는 **가장 미세한 계층**이다 (`CuroboEsdfField.grid`). 질의의 허용오차가
        # 그것을 쓰므로 주 격자도 그쪽이어야 한다 — 두 곳이 다르면 기록의 격자와 최적화가 본
        # 격자가 갈라진다.
        out: list[tuple[str, Any]] = [("", layers[-1])]
        for i, layer in enumerate(layers[:-1]):
            out.append((f"coarse{i}" if len(layers) > 2 else "coarse", layer))
        for i, layer in enumerate(getattr(field, "target_free_layers", ()) or ()):
            out.append((f"free{i}" if i else "free", layer))
        return out

    @classmethod
    def supports_field(cls, field) -> bool:
        """이 필드를 실을 수 있나. **시작할 때** 묻기 위한 것이다."""
        try:
            cls.field_layers(field)
        except TypeError:
            return False
        return True

    def _write_field(self, field, payload: dict[str, Any],
                     occupancy: Optional[np.ndarray]) -> dict[str, Any]:
        """거리장을 `esdf_mode` 에 맞춰 싣고, 어느 모드에서든 격자 메타는 남긴다.

        **계층이 여럿이면 여럿 다 싣는다** (cuRobo 는 coarse + fine, 그리고 정책이 켜져 있으면
        target 없는 계층 한 겹 더). 주 계층은 접두 없는 예전 키 이름을 그대로 쓰므로 옛 기록을
        읽는 코드가 그대로 돈다.
        """
        layers = self.field_layers(field)
        info: dict[str, Any] = {
            "esdf_stats": {k: _scalar(v) for k, v in dict(field.stats or {}).items()},
            "esdf_layers": [],
        }
        for name, layer in layers:
            prefix = "esdf_" if not name else f"esdf_{name}_"
            grid = layer.grid
            payload[prefix + "origin"] = np.asarray(grid.origin, np.float64)
            payload[prefix + "shape"] = np.asarray(grid.shape, np.int32)
            payload[prefix + "voxel_size"] = np.float64(grid.voxel_size)
            info["esdf_layers"].append({
                "name": name or "main",
                "voxel_size": float(grid.voxel_size),
                "shape": [int(v) for v in grid.shape],
                "origin": [float(v) for v in np.asarray(grid.origin, np.float64)],
                "max_distance": float(layer.max_distance),
            })
            if name == "":
                # 옛 기록과 같은 키. 읽는 코드가 이것 하나만 알고 있어도 돈다.
                info["esdf_max_distance"] = float(layer.max_distance)
            if self.esdf_mode == "none":
                continue
            if self.esdf_mode == "full":
                device_values = getattr(layer, "values_device", None)
                if device_values is not None:
                    # T38 B3: GPU 계층. 격자는 **쓸 때** 가져온다 (`_DeferredGrid`) — 응답 경로에서
                    # 128³ D2H 를 하지 않는다. 같은 float32 → float16 변환이라 바이트가 같다.
                    payload[prefix + "distance"] = _DeferredGrid(device_values, np.float16)
                    continue
                values = getattr(layer, "distance_grid", None)
                if values is not None:
                    # float16 은 여기서 안전하다. 이 배열은 진단·시각화용이고, 최적화가 읽는 값은
                    # 언제나 살아 있는 필드에서 float64 로 보간돼 나온다. 0.4 m 범위에서
                    # float16 의 해상도는 0.25 mm 미만이라 20 mm 복셀의 이산화 오차에 묻힌다.
                    payload[prefix + "distance"] = np.asarray(values, np.float16)
        if self.esdf_mode == "none":
            return info
        if occupancy is not None:
            payload["esdf_occupancy"] = np.asarray(occupancy, np.uint8)
        else:
            # **없다는 사실을 적는다.** `occupancy` 모드로 켰는데 배열이 없으면 읽는 사람은
            # "점유가 비어 있던 프레임" 으로 읽는다. cuRobo backend 에는 legacy 의 3 상태
            # (FREE/OCCUPIED/UNKNOWN) 에 대응하는 배열이 **없다** — block-sparse TSDF 에서
            # "보고 지나간 자유공간" 과 "한 번도 안 본 곳" 이 둘 다 미할당이다
            # (`CuroboEsdfField.unknown_fraction` 이 같은 이유로 `None` 을 낸다).
            info["esdf_occupancy_available"] = False
            info["esdf_occupancy_reason"] = (
                "이 backend 는 3 상태 점유 배열을 내놓지 않습니다 (block-sparse TSDF). "
                "거리장 자체를 보려면 --record-constraints-esdf full 을 쓰십시오")
        return info

    def _thin(self, arr: np.ndarray) -> np.ndarray:
        if self.max_points <= 0 or arr.shape[0] <= self.max_points:
            return arr
        stride = int(np.ceil(arr.shape[0] / self.max_points))
        return arr[::stride]

    def close(self) -> None:
        # background 면 남은 것을 다 쓰고 worker 를 멈춘다. 이후 `record()` 는 동기로 쓴다.
        was_background = self._queue is not None
        self._stop_worker()
        self.meta["n_chunks"] = self._count
        if was_background:
            self.meta["writer"] = {"mode": "background", "n_failed": self.n_failed,
                                   "failed_paths": list(self.failed_paths)}
        (self.run_dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False))
        print(f"[ag3s] wrote {self._count} constraint records to {self.run_dir}")


# ------------------------------------------------------------------------------------ 읽기


def load_constraint_run(run_dir, *, limit: Optional[int] = None) -> tuple[dict, list[dict]]:
    """`(meta, chunks)` — 각 청크는 npz 의 배열과 `summary` 를 합친 딕셔너리."""
    path = pathlib.Path(run_dir)
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    files = sorted(path.glob("chunk_*.npz"))
    if limit is not None:
        files = files[:limit]
    chunks = []
    for f in files:
        with np.load(f, allow_pickle=False) as blob:
            row = {k: blob[k] for k in blob.files if k != "summary_json"}
            if "summary_json" in blob.files:
                row["summary"] = json.loads(str(blob["summary_json"]))
        row["path"] = f
        chunks.append(row)
    return meta, chunks


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 16 else f"<ndarray {value.shape}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
