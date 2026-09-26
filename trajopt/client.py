"""로컬 쪽 — 3카메라 관측을 싸서 보내고, 안전 판정을 읽고, 실패를 전부 hold 로 수렴시킨다.

**실패 모드가 네 가지인데 결과는 하나여야 한다.** 서버가 unsafe 라고 했든, 응답이 안 왔든,
늦게 온 응답이 지난 청크의 것이든, 예외가 났든 — 로봇이 해야 할 일은 같다: 현재 관절을 유지한다.
네 경우를 각각 다르게 처리하면 그중 하나는 반드시 빠뜨리고, 빠뜨린 경로는 "검증되지 않은 청크를
실행" 으로 끝난다. 그래서 이 클래스는 `last_safe` 를 **기본 False 로 두고**, 모든 것이 확인된
경우에만 True 로 올린다.

**shadow (T5) 는 그 수렴에 구멍을 내지 않는다.** shadow 실행은 *"전부 계산하되 수정된 청크를
로봇에 보내지 않는"* 것이고, 그래서 바꾸는 것은 **네 실패 모드 중 하나도 아니다** — 판정이
`unsafe` 인 프레임에서 hold 대신 **정책 원본 청크**를 실행한다는 것뿐이다. timeout·오래된
응답·서버 오류·차원 불일치는 shadow 에서도 그대로 hold 다: 그 넷은 "이 청크가 위험하다" 가
아니라 **"이 청크를 신뢰할 근거가 없다"** 이고, 원본이든 수정본이든 똑같이 근거가 없다.

`last_safe` 는 shadow 에서도 **서버 판정 그대로**다. 실행 여부를 알고 싶으면
`should_execute` · `last_executed_chunk` 를 본다. 한 값이 둘을 겸하게 만들면 기록에
*"safe 인데 unsafe 였다"* 가 남고, 나중에 그 기록을 읽는 사람은 어느 쪽이 참인지 알 수 없다.

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
        shadow: **shadow 실행(T5)이면 True.** 서버는 전부 돌지만 로봇은 정책 원본 청크를
            실행하고, `unsafe` 판정에 멈추지 않는다 (판정과 사유는 프레임마다 그대로 남는다).
            **서버도 `--shadow` 로 떠 있어야 한다** — 짝이 안 맞으면 생성자나 첫 왕복에서
            즉시 죽는다. 조용히 refined 를 실행하면 shadow 가 아닌데 shadow 라고 기록된다.
    """

    def __init__(self, *, policy, scene, cameras: Sequence[str] = wire.DEFAULT_CAMERAS,
                 timeout_s: float = 2.0, phase: str = "approach",
                 active_manipulators: Sequence[str] = (), trace_dir: Optional[str] = None,
                 shadow: bool = False):
        self._policy = policy
        self._scene = scene
        self.cameras = tuple(cameras)
        self.timeout_s = float(timeout_s)
        self.phase = phase
        self.active_manipulators = tuple(active_manipulators)
        self.shadow = bool(shadow)

        self._seq = 0
        #: 확인된 것이 없으면 실행하지 않는다. 첫 청크가 오기 전에도 이 값이 읽힌다.
        self.last_safe = False
        self.last_reason = "no response yet"
        self.last_verdict: dict[str, Any] = {}
        #: 마지막 응답이 실어 온 거리장 출처 (`FieldProvenance`). **hold 일 때도 붙든다** —
        #: control frame 이 왜 멈췄는지를 적으려면 그 프레임의 기하가 무엇이었는지 알아야 한다.
        self.last_field: Any = None
        #: 마지막 왕복의 결과. `ok` | `timeout` | `stale` | `unsafe` | `error`.
        #: T0 이 프레임마다 요구하는 IPC 기록이 이 값과 아래 `stats` 다.
        self.last_ipc = "none"
        #: **실제로 찍은** 카메라별 촬영 시각 (단조시계). `_pack` 이 채운다.
        #:
        #: 관측 프레임을 기록하려면 촬영 시각이 필요한데, 캡처는 이 클래스 안(`_pack`)에서
        #: 일어나므로 호출부는 그 값을 볼 길이 없었다. 응답의 `observed_at` 은 **서버가 고른**
        #: 가장 최근 한 개뿐이라 카메라 간 시차(skew)를 복원할 수 없다 — 세 대가 순차 렌더라
        #: 그 시차가 손목 클라우드의 번짐과 직결된다. 그래서 왕복이 어떻게 끝나든(hold 포함)
        #: 남겨 둔다: 요청을 **보내기 전에** 채우므로 timeout 이어도 촬영 시각은 남는다.
        self.last_stamps: dict[str, float] = {}
        #: 마지막 응답의 `ag3s` 블록 (`{}` 면 인증됐거나 옛 서버다). **hold 일 때도 붙든다** —
        #: 왜 멈췄는지를 적으려면 그 프레임의 지각 사유가 무엇이었는지 알아야 한다.
        #: `last_verdict` 에 합치지 않는 것은 그 딕셔너리의 키 집합이 T0 기록의 `verdict` 이고,
        #: 여기에 키를 더하면 옛 기록과 모양이 갈라지기 때문이다.
        self.last_ag3s: dict[str, Any] = {}
        #: 이번 프레임에 로봇이 실행할 청크가 **무엇인가**: `refined` | `reference` | `none`.
        #: `none` 이 hold 다. 기본값이 `none` 인 것은 `last_safe` 가 False 로 시작하는 것과
        #: 같은 이유다 — 첫 응답이 오기 전에도 이 값이 읽히고, 확인된 것이 없으면 실행하지 않는다.
        #:
        #: **`last_safe` 와 겸하지 않는 이유**: shadow 에서는 `last_safe=False` 인 프레임도
        #: 실행된다. 한 값이 판정과 실행을 겸하면 기록에서 그 둘을 되살릴 수 없다.
        self.last_executed_chunk = "none"
        #: 서버가 **계산한** 청크 (= 응답의 `actions`, 언제나 refined), 또는 `None` — 읽을
        #: 응답이 없었다는 뜻이다.
        #:
        #: **`infer` 의 반환값과 겸하지 않는 이유**: shadow 에서 그 반환값의 `actions` 는
        #: reference 로 바뀌어 나간다 (로봇이 실행하는 것이 그쪽이므로). 두 청크를 기록에
        #: 나란히 남기려면 호출부가 *"이 프레임의 refined 는 무엇이었나"* 를 한 곳에서 물을
        #: 수 있어야 하고, `result` dict 의 키 이름이 모드마다 다르면 그 질문이 호출부에서
        #: 두 갈래로 갈린다. `T5c` 가 `refined` 대 `reference` 비교를 미측정으로 닫은 것이
        #: 바로 그 청크들이 기록에 없었기 때문이다.
        self.last_actions_refined = None
        #: 정책 **원본** 청크, 또는 `None` — shadow 가 아니라는 뜻이다 (응답의
        #: `actions_reference` 있음/없음 그대로).
        self.last_actions_reference = None
        #: `max_violation_m` 을 만든 행의 신원, 또는 `None` (`wire.VIOLATION_PAIR`).
        #: `last_verdict` 에 넣지 않는 것은 그 딕셔너리의 키 집합이 T0 기록의 `verdict` 이고,
        #: 거기에 키를 더하면 옛 기록과 모양이 갈라지기 때문이다 — `last_ag3s` 와 같은 이유다.
        self.last_violation_pair = None
        self.stats = {"sent": 0, "safe": 0, "unsafe": 0, "timeout": 0, "stale": 0, "error": 0}

        self._trace = None
        if trace_dir:
            from benchmark.ag3s.runtime.trace import RunTrace

            self._trace = RunTrace(trace_dir, meta={
                "mode": "safe_shadow" if self.shadow else "safe_remote",
                "cameras": list(self.cameras),
                "timeout_s": self.timeout_s, "phase": phase,
            })

        # **짝을 접속 시점에 본다.** 프레임마다 보는 검사(`_check_shadow_pairing`)가 보증이지만,
        # 그것은 첫 왕복이 끝난 뒤다. 메타데이터는 접속 즉시 와 있으므로 여기서 죽으면 카메라도
        # 한 번 안 찍고, 무엇보다 **로봇이 아직 아무것도 실행하지 않았다**.
        self._check_server_metadata()

    # ----------------------------------------------------------------------------------
    def infer(self, obs: dict[str, Any], *, reset: bool = False) -> dict[str, Any]:
        """한 청크. 무슨 일이 있어도 `{"actions": [H, 14]}` 를 돌려준다.

        예외를 밖으로 내지 않는 것은 refiner 와 같은 이유다 — 네트워크 한 번 끊긴 것으로 제어
        루프가 죽으면, 팔은 마지막 `d.ctrl` 에 매달린 채 아무도 hold 를 걸어주지 않는다.

        **예외가 하나 있다: shadow 모드의 짝이 안 맞을 때** (`_check_shadow_pairing`). 그것은
        런타임 실패가 아니라 설정 오류이고, hold 로 삼키면 *"shadow 라고 적힌 T6 실행"* 이
        남는다. 이 검사는 청크를 돌려주기 **전에** 하므로 로봇은 한 스텝도 실행하지 않는다.
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
        if actions.ndim != 2 or actions.shape[1] != wire.ACTION_WIDTH:
            # **차원이 다르면 hold 다.** 자르거나 채워서 통과시키면 관절이 한 칸씩 밀린
            # 청크가 실행된다 — 형태는 맞고 뜻은 틀린, 가장 위험한 실패다.
            return self._hold(
                f"unexpected action shape {actions.shape}; expected "
                f"[H, {wire.ACTION_WIDTH}] (arm_joint_dim={wire.ARM_JOINT_DIM}). "
                "정책과 이 클라이언트의 차원이 다르면 서버 config 를 확인하십시오",
                "error", result)

        reference = wire.unpack_actions_reference(result)
        # 모드가 어긋났으면 여기서 죽는다. 아래 어느 갈래로도 가지 않는다.
        self._check_shadow_pairing(reference)
        if self.shadow and reference.shape != actions.shape:
            # reference 와 refined 의 모양이 다르면 어느 쪽이 어느 스텝인지 알 수 없다.
            # `actions` 차원 검사와 같은 이유로 hold 다 — 형태는 맞고 뜻은 틀린 실패.
            return self._hold(
                f"{wire.ACTIONS_REFERENCE} has shape {reference.shape} but actions has "
                f"{actions.shape}; refusing to execute a chunk of unknown alignment",
                "error", result)

        self.last_verdict = {k: result.get(k) for k in
                             ("ag3s_status", "geometry_certified", "trajopt_status",
                              "max_violation_m", "timing_ms", "notes")}
        self.last_field = wire.unpack_field(result)
        self.last_ag3s = wire.unpack_ag3s(result)
        # **두 청크와 위반 행의 신원을 붙든다.** 여기서 붙드는 것이 아래 shadow 갈래에서
        # `actions` 키가 reference 로 바뀌기 **전**이라는 점이 중요하다.
        self.last_actions_refined = actions
        self.last_actions_reference = reference
        self.last_violation_pair = wire.unpack_violation_pair(result)
        safe = bool(result.get("safe", False))
        if not safe and not self.shadow:
            return self._hold(self._explain(result), "unsafe", result)

        if safe:
            self.last_safe = True
            self.last_reason = ""
            self.last_ipc = "ok"
            self.stats["safe"] += 1
        else:
            # **shadow 인데 unsafe 다.** 멈추지 않는다 — 멈추면 에피소드가 서고 볼 것이 없어진다.
            # 대신 판정과 사유를 **그대로** 남긴다: `last_safe` 는 False 이고 `last_ipc` 는
            # `unsafe` 다. 실행됐다는 사실은 `last_executed_chunk` 가 따로 말한다. 여기서
            # `last_safe` 를 True 로 올리면 "unsafe 여도 진행했다" 가 기록에서 사라진다.
            self.last_safe = False
            self.last_reason = self._explain(result)
            self.last_ipc = "unsafe"
            self.stats["unsafe"] += 1
            # 키를 **일어났을 때만** 만든다. shadow 가 아닌 실행의 `stats` 를 T0 과 같은
            # 키 집합으로 두려는 것이다 (`--trajectory-out` 의 `ipc_stats` 가 이것을 받는다).
            self.stats["shadow_override"] = self.stats.get("shadow_override", 0) + 1

        if not self.shadow:
            self.last_executed_chunk = "refined"
            if self._trace is not None:
                self._trace.mark("verdict", safe=True, seq=seq,
                                 **{k: v for k, v in self.last_verdict.items()
                                    if k != "timing_ms"})
            return result

        # shadow: 로봇이 실행하는 것은 **정책 원본**이다. refined 는 버리지 않고 다른 키로
        # 함께 돌려준다 — 조용히 덮으면 호출부가 무엇을 받았는지 알 수 없다.
        self.last_executed_chunk = "reference"
        if self._trace is not None:
            self._trace.mark("verdict", safe=self.last_safe, seq=seq, shadow=True,
                             executed="reference",
                             **{k: v for k, v in self.last_verdict.items() if k != "timing_ms"})
        out = dict(result)
        out["actions"] = reference
        out["actions_refined"] = actions
        out["executed_chunk"] = "reference"
        return out

    @property
    def should_execute(self) -> bool:
        """이 프레임의 청크를 실행해도 되는가. **hold 판단은 이 하나만 본다.**

        shadow 가 아니면 `last_safe` 와 정확히 같다 — 그래서 기본 동작이 바뀌지 않는다.
        shadow 면 판정이 unsafe 여도 참이 될 수 있고, 그때 무엇이 실행되는지는
        `last_executed_chunk` 가 말한다.
        """
        return self.last_executed_chunk != "none"

    def _check_server_metadata(self) -> None:
        """접속 즉시 서버가 shadow 인지 본다. 전송 계층이 메타데이터를 안 주면 넘어간다.

        여기서 못 잡는 경우(메타데이터 없는 전송)는 `_check_shadow_pairing` 이 첫 왕복에서
        잡는다. 그래서 이 검사는 **보증이 아니라 조기 경고**다 — 조기인 것이 중요한 이유는
        여기서 죽으면 카메라도 한 번 안 찍고 로봇이 아무것도 실행하지 않았기 때문이다.
        """
        get = getattr(self._policy, "get_server_metadata", None)
        if not callable(get):
            return
        try:
            meta = dict(get() or {})
        except Exception:  # noqa: BLE001 — 메타데이터를 못 읽는 것으로 죽지 않는다
            return
        if bool(meta.get("shadow", False)) == self.shadow:
            return
        raise RuntimeError(self._pairing_message(server_shadow=not self.shadow))

    def _check_shadow_pairing(self, reference) -> None:
        """`--safe-shadow`(로컬)와 `--shadow`(서버)의 짝. **안 맞으면 즉시 죽는다.**

        응답에 `actions_reference` 가 있고 없는 것이 서버 모드의 유일한 신호다. 두 방향 모두
        예외인 이유는 **둘 다 기록을 거짓으로 만들기** 때문이다:

        * 로컬만 shadow → reference 가 없다. 조용히 refined 를 실행하면 shadow 가 아닌데
          shadow 라고 기록된다.
        * 서버만 shadow → 로컬이 reference 를 무시하고 refined 를 실행한다. 로봇은 닫힌
          고리로 도는데 서버 기록은 shadow 라고 말한다.

        hold 로 수렴시키지 않는 이유: hold 는 *"이 청크를 실행하지 않는다"* 이고 그래도 실행은
        계속된다. 설정이 어긋난 채로 계속 도는 실행은 결과가 무슨 뜻인지 아무도 모른다.
        """
        if self.shadow and reference is None:
            raise RuntimeError(self._pairing_message(server_shadow=False))
        if not self.shadow and reference is not None:
            raise RuntimeError(self._pairing_message(server_shadow=True))

    def _pairing_message(self, *, server_shadow: bool) -> str:
        """두 곳(생성자·프레임)에서 같은 문장을 쓴다. 그래서 *"응답이 왔는데"* 처럼 한쪽에서만
        참인 말을 쓰지 않는다 — 근거는 `actions_reference` 의 있음/없음 하나다."""
        if server_shadow:
            return (
                f"the server runs with --shadow (it carries {wire.ACTIONS_REFERENCE!r}) but "
                "this client does not: --safe-shadow is off. Refusing to continue — the robot "
                "would execute the refined chunk, a closed loop, while the server-side record "
                "says shadow. Either add --safe-shadow locally or restart the server without "
                "--shadow.")
        return (
            f"--safe-shadow is on but the server carries no {wire.ACTIONS_REFERENCE!r}, so it "
            "is not running with --shadow. Refusing to continue — executing the refined chunk "
            "here would record a shadow run that actually closed the loop, the one mistake this "
            "mode exists to prevent. Restart the server with --shadow.")

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
        self.last_stamps = dict(stamps)
        return wire.pack_request(
            obs, cameras=self.cameras, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=state, stamps=stamps, phase=self.phase,
            active_manipulators=self.active_manipulators, reset=reset, seq=seq)

    def _hold(self, reason: str, kind: str,
              result: Optional[dict] = None) -> dict[str, Any]:
        """실행하지 않기로 한다. 청크는 **그대로 돌려준다** — 무엇이 거부됐는지 보이도록."""
        self.last_safe = False
        self.last_reason = reason
        self.last_ipc = kind
        # **hold 는 shadow 에서도 hold 다.** timeout·stale·서버 오류·차원 불일치는 "이 청크가
        # 위험하다" 가 아니라 "신뢰할 근거가 없다" 이고, 원본이든 수정본이든 근거가 없다.
        self.last_executed_chunk = "none"
        # **hold 일 때도 필드 출처를 붙든다.** 응답이 아예 없으면 `unavailable` 을 이유와
        # 함께 남긴다 — control frame 이 왜 멈췄는지를 적으려면 그 프레임의 기하가 무엇이었는지
        # 알아야 하고, 비워 두면 "기록이 없다" 와 "필드가 없었다" 가 구별되지 않는다.
        if result is not None:
            self.last_field = wire.unpack_field(result)
            self.last_ag3s = wire.unpack_ag3s(result)
            # 응답이 왔으니 청크도 신원도 **그 응답이 말한 그대로** 남긴다. hold 여도 서버가
            # 무엇을 계산했는지는 기록에 남아야 한다 — 왜 거부됐는지는 그것 없이 못 읽는다.
            # `ipc` 가 `timeout`/`stale` 이면 그 청크가 **지나간 자세를 위한 것**이라는 사실은
            # 그 값이 따로 말한다.
            blob = result.get("actions")
            self.last_actions_refined = (None if blob is None
                                         else np.asarray(blob))
            self.last_actions_reference = wire.unpack_actions_reference(result)
            self.last_violation_pair = wire.unpack_violation_pair(result)
        else:
            # 응답이 아예 없으면 지각 사유도 없다. **지난 프레임 것을 남겨 두지 않는다** —
            # 남기면 이 프레임이 그 사유로 멈춘 것처럼 읽힌다. 청크와 위반 행의 신원도 같다.
            self.last_ag3s = {}
            self.last_actions_refined = None
            self.last_actions_reference = None
            self.last_violation_pair = None
            from benchmark.ag3s.fields.provenance import FieldProvenance
            self.last_field = FieldProvenance.unavailable(
                f"no response to read a field from ({kind}): {reason}")
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
        """action 레이아웃의 현재 상태. hold 청크의 모든 행이 된다.

        레이아웃은 `[왼팔 N, 왼 그리퍼, 오른팔 N, 오른 그리퍼]` 이고 `N = wire.ARM_JOINT_DIM`
        이다. **6 을 박지 않는다** — 박아 두면 16D 에서 그리퍼가 손목 자리에 들어가고 hold
        청크가 엉뚱한 관절을 지령한다.
        """
        state = np.asarray(self._scene.robot_state(), np.float64)
        if len(state) == wire.ACTION_WIDTH:
            return state
        n = wire.ARM_JOINT_DIM
        if len(state) < 2 * n:
            raise ValueError(
                f"씬이 준 상태가 {len(state)}-D 인데 레이아웃은 팔 관절 {2 * n} 개를 "
                f"필요로 합니다 (arm_joint_dim={n}). 무엇으로 채워야 할지 추측하지 않습니다")
        # 씬이 팔 관절만 준다면 그리퍼 자리를 열림(0)으로 채운다. 그리퍼를 임의로 닫으면
        # 잡고 있던 것을 떨어뜨린다.
        arms = state[:2 * n]
        return np.concatenate([arms[:n], [0.0], arms[n:2 * n], [0.0]])

    @staticmethod
    def _explain(result: dict[str, Any]) -> str:
        """왜 거부됐는지를 한 줄로. 상태값만으로는 두 갈래가 구분되지 않는다.

        **인증 실패에는 사유를 붙인다.** 2026-09-25 첫 live smoke 에서 이 문장이
        `"AG3S could not certify the geometry (status=degraded)"` 까지만 나왔고, 거기서 멈추면
        읽는 사람이 서버에 다시 들어가야 한다 — 그런데 서버 로그에도 없었다. 응답의 `ag3s`
        블록이 이제 사유를 싣고 있으므로 그것을 이 한 줄에 넣는다.
        """
        from benchmark.ag3s.runtime import degradation

        violation = result.get("max_violation_m")
        if not result.get("geometry_certified", True):
            block = wire.unpack_ag3s(result)
            # 제어 루프의 한 줄이다 — 프레임마다 찍히므로 전문을 넣으면 다른 것을 밀어낸다.
            # 전문은 `ag3s` 블록의 `notes` 와 프레임 기록에 그대로 있다.
            why = degradation.explain(block.get("notes") or (), detail_chars=64)
            if not why:
                # 코드 달린 사유가 없다 = 옛 서버이거나 제약 집합 자체가 없었다. 그 사실을
                # 말한다 — 빈칸으로 두면 "사유가 없는 degraded" 와 구별되지 않는다.
                why = ("; ".join(str(n) for n in (block.get("notes") or ())[:2])
                       or "no reason on the wire (the server predates the `ag3s` block)")
            return (f"AG3S could not certify the geometry "
                    f"(status={result.get('ag3s_status')}, "
                    f"grounding={block.get('grounding_status', 'unknown')}); the trajectory may "
                    f"clear every constraint and still meet something nobody saw — {why}")
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
            if self.shadow:
                print(f"[shadow] the robot executed the policy reference chunk on every "
                      f"executed chunk; {s.get('shadow_override', 0)} of them carried an "
                      f"unsafe verdict and ran anyway (nothing was held for the verdict)")


class _NullSpan:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


def _maybe_span(trace, name: str):
    return trace.span(name) if trace is not None else _NullSpan()
