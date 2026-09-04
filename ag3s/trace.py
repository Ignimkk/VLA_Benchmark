"""벽시계 기준 실행 추적 — 한 제어 루프가 실제로 어디에 시간을 쓰는지.

`profiler.StageProfiler` 와 역할이 다르다. 프로파일러는 **AG3S 한 프레임 내부**를 단계별로 쪼개
평균과 표준편차를 낸다. 여기 있는 것은 그 바깥이다: 관측 생성 → 정책 추론 → depth 캡처 → AG3S →
TO → 실행이 벽시계 위 어디에 놓이는지, 그리고 제어 루프가 마감을 지켰는지.

두 가지를 나눠 기록한다.

* **`monotonic`** 이 구간 길이의 근거다. 시스템 시계가 NTP 로 뒤로 점프해도 음수 구간이 나오지
  않는다.
* **`epoch`** 은 `t0` 하나만 메타에 남긴다. 다른 로그(서버 쪽 추론 로그, 비디오 프레임)와 맞출
  때 필요한데, 매 스팬마다 두 시계를 다 저장하면 파일만 두 배가 되고 둘이 미세하게 어긋나 어느
  쪽이 근거인지 흐려진다.

재생(replay)이 아니라 **실제 루프**를 재는 것이 이 모듈이 있는 이유다. 재생은 값이 실측이어도
지연을 증명하지 못한다 — 청크 예산 안에 들어가는지는 실제로 돌려봐야 안다.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import time
from typing import Any, Iterator, Optional

__all__ = ["RunTrace", "span_summary"]


class RunTrace:
    """제어 루프의 스팬을 JSONL 로 append 한다.

    한 줄이 한 스팬이다. 루프가 도는 중에 프로세스가 죽어도 그때까지의 줄은 온전히 남는다 —
    실행 후에 한 번에 쓰는 방식이었다면 초과 지연으로 루프를 죽였을 때 아무것도 못 건진다.

    Args:
        output_dir: `trace.jsonl` 과 `meta.json` 이 놓일 디렉터리. 없으면 만든다.
        meta: 실행 조건. 나중에 두 실행을 비교할 때 무엇이 달랐는지 여기서만 알 수 있으므로
            설정값을 아끼지 말고 넣는다.
        enabled: False 면 모든 메서드가 no-op 이고 파일도 만들지 않는다. 호출부에서 `if trace:`
            분기를 치지 않아도 되게 하려는 것이다.
    """

    def __init__(self, output_dir, *, meta: Optional[dict[str, Any]] = None,
                 enabled: bool = True):
        self.enabled = bool(enabled)
        self._t0_monotonic = time.monotonic()
        self._t0_epoch = time.time()
        self._chunk_index = -1
        self._step_index = -1
        self._fh = None
        self.path: Optional[pathlib.Path] = None
        if not self.enabled:
            return
        root = pathlib.Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "trace.jsonl"
        self._fh = self.path.open("w")
        (root / "meta.json").write_text(json.dumps({
            "t0_epoch": self._t0_epoch,
            "t0_monotonic": self._t0_monotonic,
            "clock": "monotonic; durations are monotonic deltas, t0_epoch anchors absolute time",
            **(meta or {}),
        }, indent=2, ensure_ascii=False))

    # --- 루프 위치 ------------------------------------------------------------------------
    def begin_chunk(self, index: int) -> None:
        """새 청크(재계획 경계). 이후 스팬은 이 인덱스를 달고 나간다."""
        self._chunk_index = int(index)

    def begin_step(self, index: int) -> None:
        """새 제어 스텝."""
        self._step_index = int(index)

    # --- 스팬 -----------------------------------------------------------------------------
    @contextlib.contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """`with trace.span("ag3s"):` — 빠져나올 때 한 줄이 나간다.

        yield 되는 딕셔너리에 필드를 더 넣으면 그대로 기록된다. 구간이 끝나야 알 수 있는 값
        (AG3S 의 status, TO 의 반복 수)을 담으라고 있는 것이다. 예외가 나도 줄은 나가고
        `ok: false` 가 붙는다 — 실패한 구간이 로그에서 사라지면 그 구간만 빠르게 끝난 것처럼
        보인다.
        """
        extra: dict[str, Any] = dict(fields)
        if not self.enabled:
            yield extra
            return
        start = time.monotonic()
        ok = True
        try:
            yield extra
        except BaseException:
            ok = False
            raise
        finally:
            self._write(name, start, time.monotonic(), ok, extra)

    def mark(self, name: str, **fields: Any) -> None:
        """길이가 0인 스팬. 시점만 남기고 싶을 때(마감 초과, 청크 교체)."""
        if not self.enabled:
            return
        now = time.monotonic()
        self._write(name, now, now, True, dict(fields))

    def _write(self, name: str, start: float, end: float, ok: bool,
               extra: dict[str, Any]) -> None:
        row = {
            "name": name,
            "chunk": self._chunk_index,
            "step": self._step_index,
            "t_start_ms": (start - self._t0_monotonic) * 1000.0,
            "duration_ms": (end - start) * 1000.0,
            "ok": ok,
        }
        row.update(_jsonable(extra))
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 초과 지연으로 루프를 중단시키는 것이 이 실험의 예상 결과 중 하나다. 버퍼에 남은 줄이
        # 그때 사라지면 정작 알고 싶은 마지막 청크를 잃는다.
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunTrace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(obj: Any) -> Any:
    """numpy 스칼라와 배열을 json 이 삼킬 수 있는 형태로. 큰 배열은 여기 오면 안 된다."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 16 else f"<ndarray {obj.shape} {obj.dtype}>"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def span_summary(path) -> dict[str, dict[str, float]]:
    """기록된 trace 를 스팬 이름별 통계로. 리포트 스크립트와 테스트가 함께 쓴다."""
    import numpy as np

    rows: dict[str, list[float]] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.setdefault(row["name"], []).append(float(row["duration_ms"]))
    out = {}
    for name, values in rows.items():
        arr = np.asarray(values, np.float64)
        out[name] = {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "total_ms": float(arr.sum()),
        }
    return out
