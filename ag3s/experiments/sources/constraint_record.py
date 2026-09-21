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

import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["ConstraintRecordWriter", "load_constraint_run", "ESDF_MODES"]

#: `esdf_mode` 가 받는 값. 용량과 진단 가능성의 교환이다.
#:   none      — 격자 메타만. 청크당 수 KB.
#:   occupancy — 점유 상태(uint8). 기하가 어떻게 분리됐는지는 보이지만 거리장은 못 본다.
#:   full      — 거리장까지 float16. 청크당 약 1.1 MB (76x90x80, 20 mm 복셀 실측).
ESDF_MODES = ("none", "occupancy", "full")


class ConstraintRecordWriter:
    """청크마다 AG3S 의 제약 생성 중간 산출물을 npz 로 쓴다.

    Args:
        output_dir: 루트. 하위에 `run_0000`, `run_0001`… 을 새로 만든다 (덮어쓰지 않는다).
        esdf_mode: `ESDF_MODES` 참고.
        max_points: 저장할 점군의 상한. 넘으면 **결정적으로** 솎아낸다(고정 스트라이드).
            무작위 표본이 아니라 스트라이드인 이유는, 같은 파일을 두 번 읽어 다른 그림이 나오면
            그림을 신뢰할 수 없기 때문이다. 0 이면 상한 없음.
        meta: 실행 조건. 나중에 두 실행을 비교할 때의 유일한 근거다.
    """

    def __init__(self, output_dir, *, esdf_mode: str = "full", max_points: int = 200_000,
                 meta: Optional[dict[str, Any]] = None):
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
        self._count = 0

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
               occupancy: Optional[np.ndarray] = None) -> pathlib.Path:
        """한 청크. `debug` 는 `AG3S.process_multi_debug` 가 돌려준 두 번째 값 그대로.

        `occupancy` 는 따로 받는다. `EsdfField` 는 거리장만 들고 있고 세 상태 점유 배열은
        `EsdfBuilder` 안에 남기 때문이다 — 그 분리는 의도된 것이라(필드는 TO 가 읽는 것,
        점유는 그것을 만든 재료) 여기서 필드를 파고들어 꺼내오지 않는다.
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
            summary["to"] = {
                "status": getattr(getattr(to_result, "status", None), "value", None),
                "iterations": int(getattr(to_result, "iterations", 0) or 0),
                "solve_ms": float(getattr(to_result, "solve_ms", 0.0) or 0.0),
                # 노트가 상태를 해석하는 유일한 근거다. `VIOLATED` 에는 두 갈래가 있다 —
                # 실제로 관통이 남은 경우와, 제약은 전부 통과했지만 AG3S 가 기하를 인증하지
                # 못한 경우. 둘은 완전히 다른 문제이고 상태값만으로는 구분되지 않는다.
                "notes": list(getattr(to_result, "notes", ()) or ()),
                "metrics": {k: _scalar(v) for k, v in (getattr(to_result, "metrics", {}) or {}).items()},
            }

        payload["summary_json"] = np.asarray(json.dumps(summary, ensure_ascii=False))
        out = self.run_dir / f"chunk_{self._count:05d}.npz"
        np.savez_compressed(out, **payload)
        self._count += 1
        return out

    # ----------------------------------------------------------------------------------
    def _write_field(self, field, payload: dict[str, Any],
                     occupancy: Optional[np.ndarray]) -> dict[str, Any]:
        """거리장을 `esdf_mode` 에 맞춰 싣고, 어느 모드에서든 격자 메타는 남긴다."""
        grid = field.grid
        payload["esdf_origin"] = np.asarray(grid.origin, np.float64)
        payload["esdf_shape"] = np.asarray(grid.shape, np.int32)
        payload["esdf_voxel_size"] = np.float64(grid.voxel_size)
        info: dict[str, Any] = {
            "esdf_max_distance": float(field.max_distance),
            "esdf_stats": {k: _scalar(v) for k, v in dict(field.stats or {}).items()},
        }
        if self.esdf_mode == "none":
            return info
        if occupancy is not None:
            payload["esdf_occupancy"] = np.asarray(occupancy, np.uint8)
        if self.esdf_mode == "full":
            grid_values = field.distance_grid
            if grid_values is not None:
                # float16 은 여기서 안전하다. 이 배열은 진단·시각화용이고, 최적화가 읽는 값은
                # 언제나 살아 있는 `EsdfField` 에서 float64 로 보간돼 나온다. 0.4 m 범위에서
                # float16 의 해상도는 0.25 mm 미만이라 20 mm 복셀의 이산화 오차에 묻힌다.
                payload["esdf_distance"] = np.asarray(grid_values, np.float16)
        return info

    def _thin(self, arr: np.ndarray) -> np.ndarray:
        if self.max_points <= 0 or arr.shape[0] <= self.max_points:
            return arr
        stride = int(np.ceil(arr.shape[0] / self.max_points))
        return arr[::stride]

    def close(self) -> None:
        self.meta["n_chunks"] = self._count
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
