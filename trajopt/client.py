"""로컬 쪽 — 3카메라 관측을 싸서 보내고, 안전 판정을 읽고, 실패를 전부 hold 로 수렴시킨다.

**실패 모드가 네 가지인데 결과는 하나여야 한다.** 서버가 unsafe 라고 했든, 응답이 안 왔든,
늦게 온 응답이 지난 청크의 것이든, 예외가 났든 — 로봇이 해야 할 일은 같다: 현재 관절을 유지한다.
네 경우를 각각 다르게 처리하면 그중 하나는 반드시 빠뜨리고, 빠뜨린 경로는 "검증되지 않은 청크를
실행" 으로 끝난다. 그래서 이 클래스는 `last_safe` 를 **기본 False 로 두고**, 모든 것이 확인된
경우에만 True 로 올린다.

오래된 응답을 버리는 이유는 따로 적을 만하다. 서버가 늦으면 그 청크는 이미 지나간 자세를 위해
계획된 것이다. 8스텝(533 ms) 뒤의 팔은 다른 곳에 있고, 그 청크의 첫 action 은 절대 관절 목표라
관절이 순간적으로 튄다. `seq` 왕복이 그것을 잡는 유일한 장치다.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.trajopt import wire

__all__ = ["SafeRemoteClient"]


class SafeRemoteClient:
    """`policy.infer` 를 감싸 AG3S 가 필요한 관측을 붙이고 안전 판정을 해석한다.

    Args:
        policy: `WebsocketClientPolicy`. 이 클래스는 전송을 하지 않고 그 객체에 맡긴다.
        scene: `TransportScene.attach(m, d)` — 제어 루프가 돌리는 바로 그 모델/데이터.
        cameras: 보낼 카메라. 기본은 머리 하나 + 손목 둘.
        timeout_s: 응답 제한. 넘으면 hold.
        phase / active_manipulators: AG3S 에 주입한다. AG3S 는 절대 추론하지 않는다.
        trace_dir: 왕복 시간을 기록할 곳. None 이면 기록하지 않는다.
    """

    def __init__(self, *, policy, scene, cameras: Sequence[str] = wire.DEFAULT_CAMERAS,
                 timeout_s: float = 2.0, phase: str = "approach",
                 active_manipulators: Sequence[str] = (), trace_dir: Optional[str] = None):
        self._policy = policy
        self._scene = scene
        self.cameras = tuple(cameras)
        self.timeout_s = float(timeout_s)
        self.phase = phase
        self.active_manipulators = tuple(active_manipulators)

        self._seq = 0
        #: 확인된 것이 없으면 실행하지 않는다. 첫 청크가 오기 전에도 이 값이 읽힌다.
        self.last_safe = False
        self.last_reason = "no response yet"
        self.last_verdict: dict[str, Any] = {}
        self.stats = {"sent": 0, "safe": 0, "unsafe": 0, "timeout": 0, "stale": 0, "error": 0}

        self._trace = None
        if trace_dir:
            from benchmark.ag3s.trace import RunTrace

            self._trace = RunTrace(trace_dir, meta={
                "mode": "safe_remote", "cameras": list(self.cameras),
                "timeout_s": self.timeout_s, "phase": phase,
            })

    # ----------------------------------------------------------------------------------
    def infer(self, obs: dict[str, Any], *, reset: bool = False) -> dict[str, Any]:
        """한 청크. 무슨 일이 있어도 `{"actions": [H, 14]}` 를 돌려준다.

        예외를 밖으로 내지 않는 것은 refiner 와 같은 이유다 — 네트워크 한 번 끊긴 것으로 제어
        루프가 죽으면, 팔은 마지막 `d.ctrl` 에 매달린 채 아무도 hold 를 걸어주지 않는다.
        """
        self._seq += 1
        seq = self._seq
        self.stats["sent"] += 1
        if self._trace is not None:
            self._trace.begin_chunk(seq)

        try:
            request = self._pack(obs, reset=reset, seq=seq)
        except Exception as exc:  # noqa: BLE001 — 카메라 렌더 실패도 hold 로 간다
            return self._hold(f"could not capture the cameras ({exc})", "error")

        started = time.monotonic()
        try:
            with _maybe_span(self._trace, "roundtrip") as span:
                result = self._policy.infer(request)
                span["seq"] = seq
        except Exception as exc:  # noqa: BLE001
            return self._hold(f"server error ({exc})", "error")

        elapsed = time.monotonic() - started
        if elapsed > self.timeout_s:
            # 응답은 왔지만 늦었다. 이 청크는 이미 지나간 자세를 위한 것이다.
            return self._hold(f"response took {elapsed * 1000:.0f} ms "
                              f"(limit {self.timeout_s * 1000:.0f} ms)", "timeout", result)

        got = int(result.get("seq", -1))
        if got != seq:
            return self._hold(f"stale response: asked for seq {seq}, got {got}", "stale", result)

        actions = np.asarray(result.get("actions"))
        if actions.ndim != 2 or actions.shape[1] != 14:
            return self._hold(f"unexpected action shape {actions.shape}; expected [H, 14]",
                              "error", result)

        self.last_verdict = {k: result.get(k) for k in
                             ("ag3s_status", "geometry_certified", "trajopt_status",
                              "max_violation_m", "timing_ms", "notes")}
        if not bool(result.get("safe", False)):
            return self._hold(self._explain(result), "unsafe", result)

        self.last_safe = True
        self.last_reason = ""
        self.stats["safe"] += 1
        if self._trace is not None:
            self._trace.mark("verdict", safe=True, seq=seq,
                             **{k: v for k, v in self.last_verdict.items() if k != "timing_ms"})
        return result

    # ----------------------------------------------------------------------------------
    def _pack(self, obs: dict[str, Any], *, reset: bool, seq: int) -> dict[str, Any]:
        """세 카메라를 **지금** 찍어 요청에 싣는다.

        카메라마다 `robot_state` 와 촬영 시각을 따로 담는다. 손목 카메라는 팔과 함께 움직이므로
        하나의 `q_now` 로 세 대를 변환하면 손목 클라우드가 번지고, 더 나쁘게는 자기 필터가
        어긋나 로봇 점이 씬에 남는다.
        """
        depth, K, T, state, stamps = {}, {}, {}, {}, {}
        with _maybe_span(self._trace, "capture"):
            for cam in self.cameras:
                frame = self._scene.capture(cam)
                depth[cam] = np.asarray(frame.depth, np.float64)
                K[cam] = np.asarray(frame.camera_intrinsics, np.float64)
                T[cam] = np.asarray(frame.T_base_cam, np.float64)
                state[cam] = np.asarray(frame.robot_state, np.float64)
                stamps[cam] = time.monotonic()
        return wire.pack_request(
            obs, cameras=self.cameras, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=state, stamps=stamps, phase=self.phase,
            active_manipulators=self.active_manipulators, reset=reset, seq=seq)

    def _hold(self, reason: str, kind: str,
              result: Optional[dict] = None) -> dict[str, Any]:
        """실행하지 않기로 한다. 청크는 **그대로 돌려준다** — 무엇이 거부됐는지 보이도록."""
        self.last_safe = False
        self.last_reason = reason
        self.stats[kind] = self.stats.get(kind, 0) + 1
        if self._trace is not None:
            self._trace.mark("verdict", safe=False, kind=kind, reason=reason)
        if result is not None and isinstance(result.get("actions"), np.ndarray):
            return result
        # 서버가 아무것도 못 줬다. 현재 자세를 유지하는 청크를 만들어 형태 계약을 지킨다 —
        # 호출부가 `chunk[chunk_step]` 을 무조건 인덱싱하므로 None 을 돌려주면 거기서 죽는다.
        hold = np.tile(np.asarray(self._current_state(), np.float64), (50, 1))
        return {"actions": hold, "safe": False, "hold_reason": reason}

    def _current_state(self) -> np.ndarray:
        """14-D action 레이아웃의 현재 상태. hold 청크의 모든 행이 된다."""
        state = self._scene.robot_state()
        if len(state) == 14:
            return state
        # 씬이 12-D 팔 관절만 준다면 그리퍼 자리를 열림(0)으로 채운다. 그리퍼를 임의로 닫으면
        # 잡고 있던 것을 떨어뜨린다.
        arms = np.asarray(state, np.float64)[:12]
        return np.concatenate([arms[:6], [0.0], arms[6:12], [0.0]])

    @staticmethod
    def _explain(result: dict[str, Any]) -> str:
        """왜 거부됐는지를 한 줄로. 상태값만으로는 두 갈래가 구분되지 않는다."""
        violation = result.get("max_violation_m")
        if not result.get("geometry_certified", True):
            return (f"AG3S could not certify the geometry "
                    f"(status={result.get('ag3s_status')}); the trajectory may clear every "
                    f"constraint and still meet something nobody saw")
        return (f"TO says {result.get('trajopt_status')}"
                + (f", {violation * 1000:.1f} mm of penetration remains"
                   if isinstance(violation, (int, float)) and np.isfinite(violation) else ""))

    def close(self) -> None:
        if self._trace is not None:
            self._trace.close()
        s = self.stats
        if s["sent"]:
            print(f"[safe] {s['sent']} chunks: safe {s['safe']}, unsafe {s['unsafe']}, "
                  f"timeout {s['timeout']}, stale {s['stale']}, error {s['error']}")


class _NullSpan:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


def _maybe_span(trace, name: str):
    return trace.span(name) if trace is not None else _NullSpan()
