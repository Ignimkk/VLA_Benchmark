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

**T23 — 판정은 사유별이고, HOLD 는 명령을 삼키지 않는다.** 서버가 `verdict_reasons` 를 싣고
(`wire` 머리말), 이 클래스는 서버 `safe` 와 사유 표(`wire.GATE_DEFAULT`)가 둘 다 실행이라고 할 때만
실행한다 — 권한 link ↔ 조작 대상 접촉(`allowed_contact`) 은 실행, 진짜 충돌은 HOLD. 통신 실패 넷은
`comms` 사유 하나로 모인다 (위의 수렴은 그대로다). HOLD 동안 로봇에 무엇을 줄지는 `HoldController`
가 정한다: 팔은 HOLD 진입 때 한 번 잡은 **명령** 목표 `q_hold`, gripper 는 마지막 명령 — 매 스텝
측정값이 아니다 (지침 §8.2 · §8.5). "현재 관절을 유지한다" 의 뜻이 그렇게 바뀌었다.

**T27 — 게이트 끄기 (`gate="off"`, 사용자 사다리 실험).** 서버 판정·`verdict_reasons` 와 무관하게
**refined 청크(`actions`)를 실행한다.** 판정은 계산해 **기록만** 한다 — `last_gate` 에
`{mode: off, would_hold, kinds}` 가 실려 "게이트가 켜져 있었다면 무엇이 막았나" 를 같은 실행에서 볼 수
있다. 예외는 하나, 청크가 **도착하지 않은** 경우(통신 실패 넷)다 — 실행할 청크가 없으므로 T23 의 fixed
HOLD 로 가고, `comms_holds` 로 세고 크게 찍는다 (실험에서 이것은 0 이어야 한다). shadow 와 다르다: shadow
는 **reference** 를 실행하고 gate off 는 **refined** 를 실행한다 — 둘은 같이 쓸 수 없다.

오래된 응답을 버리는 이유는 따로 적을 만하다. 서버가 늦으면 그 청크는 이미 지나간 자세를 위해
계획된 것이다. 8스텝(533 ms) 뒤의 팔은 다른 곳에 있고, 그 청크의 첫 action 은 절대 관절 목표라
관절이 순간적으로 튄다. `seq` 왕복이 그것을 잡는 유일한 장치다.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.trajopt import wire

__all__ = ["SafeRemoteClient", "HOLD_GRIPPER_NORM", "ExecutionLog", "HoldController",
           "HOLD_MODES", "GATE_MODES", "action_row", "gate_banner"]

#: HOLD 동안 로봇에 주는 목표 (T23, 지침 §8.5). 첫 항목이 기본이다.
#:
#: | 값 | 팔 | gripper |
#: |---|---|---|
#: | `fixed` | HOLD 에 **들어갈 때 한 번** 잡은 `q_hold` = 직전 스텝에 **명령한** 팔 목표 (`d.ctrl`). HOLD 가 이어지는 동안 그대로 | 마지막으로 **명령한** gripper — 열지도 닫지도 않는다 |
#: | `legacy` | 매 스텝 측정 `qpos` (T23 전) | 매 스텝 측정 개도 (T23 전) |
HOLD_MODES = ("fixed", "legacy")

#: 로컬 게이트 (T23). 첫 항목이 기본이다.
#:
#: * `reasons` — 서버 `safe` **와** `verdict_reasons` 의 처리(`wire.GATE_DEFAULT`)가 둘 다 실행일 때만
#:   실행. 로컬은 서버보다 엄격해질 수만 있다.
#: * `legacy` — T23 전의 판정: `trajopt_status ∈ {optimal, feasible}` (= `TrajOptStatus.safe`) 그리고 `safe`.
#:   `allowed_contact` · `occluded_target` 도 HOLD 다.
#: * `off` — **판정은 기록만, 실행에 쓰지 않는다** (T27). 도착한 청크는 언제나 refined 를 실행한다.
#:   `last_gate.would_hold` 가 `reasons` 게이트였다면의 결정이다. 도착하지 않은 청크만 HOLD (`comms`).
GATE_MODES = ("reasons", "legacy", "off")


def gate_banner(gate: str) -> Optional[str]:
    """시작 로그에 **크게** 찍을 문장, 또는 `None` (게이트가 켜져 있으면 조용하다)."""
    if gate != "off":
        return None
    return ("[safe] !!! GATE OFF (--safe-gate off) — 판정은 기록만, 실행에 쓰지 않는다 !!!\n"
            "[safe]     도착한 청크는 서버 safe·verdict_reasons 와 무관하게 **refined(actions)** 를 "
            "실행한다. HOLD 는 청크가 도착하지 않은 경우(timeout·stale·error·dimension = comms)뿐이고 "
            "그때마다 크게 찍는다 — 실험에서 0 이어야 한다.\n"
            "[safe]     planning 기록의 gate.would_hold / gate.kinds 가 '게이트가 켜져 있었다면' 의 "
            "결정이다. shadow 와 다르다 (shadow 는 reference 를 실행한다)")

#: hold 청크의 그리퍼 열에 넣을 **정규화** 값 — 씬이 실제 상태를 내놓지 않을 때만 쓴다.
#:
#: 로컬이 `ctrl = norm × RBY1_GRIPPER_OPEN(−0.045)` 로 적용하므로 (`pi05_infer.py:535`)
#: **`1.0` = 열림, `0.0` = 닫힘**이다. 예전 코드는 `0.0` 을 넣고 주석에 "열림" 이라고 적었다 —
#: 값과 주석이 반대였고, 그 값은 실제로 그리퍼를 **닫는다.**
#:
#: **물건을 쥔 채 hold 가 걸리면 이 값은 그것을 놓는다.** 그래서 정상 경로는 씬의
#: `gripper_norm()` 을 읽는 쪽이고, 이 상수는 읽을 수 없을 때의 마지막 수단이다. 열림을 고른
#: 이유는 "무엇을 쥐고 있는지 모르는 채 닫는 것" 이 사람·물체 양쪽에 더 나쁜 방향이기 때문이다.
HOLD_GRIPPER_NORM = 1.0


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
                 shadow: bool = False, gate: str = "reasons",
                 gate_table: Optional[dict[str, str]] = None):
        if gate not in GATE_MODES:
            raise ValueError(f"gate must be one of {GATE_MODES}, got {gate!r}")
        if gate == "off" and shadow:
            # **둘은 다른 청크를 실행한다** (shadow = reference, gate off = refined). 같이 주면
            # 무엇이 실행됐는지 기록이 말할 수 없다.
            raise ValueError(
                "gate='off' executes the REFINED chunk; shadow=True executes the policy "
                "REFERENCE chunk. They are different experiments — pick one")
        self._policy = policy
        #: 로컬 게이트 방식 (`GATE_MODES`) 과 사유 → 처리 표 (없으면 `wire.GATE_DEFAULT`).
        self.gate = gate
        self.gate_table = dict(gate_table) if gate_table is not None else dict(wire.GATE_DEFAULT)
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
        #: 카메라별 **렌더가 끝난** 벽시계 순간 (단조시계) — 진단용 (T30c). `last_stamps` 와 따로다.
        #:
        #: 시뮬레이션 관측이면 `last_stamps` 는 세 카메라가 같은 한 순간(`capture_time`)이고, 렌더
        #: 시간의 퍼짐(osmesa 순차 렌더, head ↔ wrist 102–142 ms)은 여기에만 남는다. T30c 전의
        #: `last_stamps` 값이 바로 이것이다.
        self.last_render_stamps: dict[str, float] = {}
        #: `last_stamps` 가 무엇인가 (`wire.STAMP_MODES`): `sim_frozen` | `render_end`. 요청 전 `none`.
        self.last_stamp_mode = "none"
        #: 마지막 요청에 실은 정책 RNG seed (T39). 싣지 않았으면 `None`.
        self.last_policy_seed: Optional[int] = None
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
        #: 정책 **원본** 청크, 또는 `None`.
        #:
        #: **T9 부터 closed loop 에서도 채워진다.** 그 전에는 shadow 에서만 왔고, 그래서
        #: *"TO 가 청크를 얼마나 바꿨나"* 를 닫힌 고리에서 잴 수 없었다 — 같은 결핍에 다섯 번
        #: 막혔다. `None` 은 이제 "shadow 가 아니다" 가 아니라 **그 프레임에 원본이 없었다**
        #: (hold · 옛 서버) 를 뜻한다. 모드는 `self.shadow` 와 응답의 `shadow` 키가 말한다.
        self.last_actions_reference = None
        #: `max_violation_m` 을 만든 행의 신원, 또는 `None` (`wire.VIOLATION_PAIR`).
        #: `last_verdict` 에 넣지 않는 것은 그 딕셔너리의 키 집합이 T0 기록의 `verdict` 이고,
        #: 거기에 키를 더하면 옛 기록과 모양이 갈라지기 때문이다 — `last_ag3s` 와 같은 이유다.
        self.last_violation_pair = None
        #: 응답의 `to` 블록 (`wire.unpack_to`) — SQP 반복 수 · 시간 예산에 걸렸나 · 충돌 on/off.
        #: 빈 dict 면 이 블록을 모르는 옛 서버다. `last_verdict` 에 넣지 않는 이유는 위와 같다.
        self.last_to: dict = {}
        #: 마지막 요청에 **실어 보낸** 실행 피드백 (`wire.make_exec_feedback` / `no_exec_feedback`).
        #: 로컬 planning 기록이 *"서버가 이 청크를 계획할 때 무엇을 알았나"* 를 적으려면 이 값이
        #: 필요하다 — 요청은 보내고 나면 사라지므로 여기 붙든다.
        self.last_exec_feedback: dict = wire.no_exec_feedback("no request sent yet")
        #: 이번 청크의 판정 사유 (T23, `wire.verdict_reasons` 형식). 통신 실패면 클라이언트가 만든
        #: `comms` 사유 하나다. `last_verdict` 에 넣지 않는 이유는 `last_ag3s` 와 같다 (T0 키 집합).
        self.last_reasons: list[dict[str, Any]] = []
        #: 그 사유의 출처: `server` (응답에 실렸다) · `derived` (옛 서버 — `safe` 에서 유도) ·
        #: `client` (통신 실패 — 응답을 못 읽었다) · `none` (아직 없음).
        self.last_reasons_source = "none"
        #: 게이트의 결정 `{mode, action, hold_kinds, source, server_safe}`. 기록용.
        self.last_gate: dict[str, Any] = {}
        #: 제어 루프가 **마지막으로 명령한** action 행 (`note_command`). `_hold` 가 응답 없이 만드는
        #: 대체 청크의 행이 이것이다 (T23: 팔·gripper = 마지막 명령). 없으면 현재 상태.
        self.last_command: Optional[np.ndarray] = None
        #: 사유 kind 별 청크 수 (T23). `stats` 와 따로 둔다 — `stats` 의 키 집합은 T0 기록의 것이다.
        self.reason_stats: dict[str, int] = {}
        self.stats = {"sent": 0, "safe": 0, "unsafe": 0, "timeout": 0, "stale": 0, "error": 0}
        #: `gate="off"` 에서 청크가 **도착하지 않아** HOLD 한 수 (T27). 실험에서 0 이어야 한다.
        #: `stats` 와 따로 둔다 — `stats` 의 키 집합은 T0 기록의 것이다.
        self.comms_holds = 0
        #: `gate="off"` 에서 **게이트가 켜져 있었다면 HOLD 였을** 청크 수와 그 사유 kind 별 수.
        self.would_hold = 0
        self.would_hold_kinds: dict[str, int] = {}

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
    def infer(self, obs: dict[str, Any], *, reset: bool = False,
              exec_feedback: Optional[dict[str, Any]] = None,
              capture_time: Optional[float] = None,
              policy_seed: Optional[int] = None) -> dict[str, Any]:
        """한 청크 — `_infer_once` 를 부르고 판정 사유를 센다 (T23). 계약은 `_infer_once` 에.

        `capture_time` (T30c): **시뮬레이션이 이 관측을 위해 멈춘 순간** (`time.monotonic()`).
        주면 세 카메라의 이미지 · `robot_state` · 외부 파라미터가 모두 이 한 순간으로 찍힌다
        (`stamp_mode = "sim_frozen"`). 호출자는 그 순간부터 `infer` 가 돌아올 때까지 `mj_step` 을
        하지 않았음을 **보증**한다. `None` 이면 예전처럼 카메라별 렌더 끝 순간이다 (`render_end`) —
        실제 카메라처럼 순간이 정말로 다른 경우를 한 순간으로 뭉개지 않으려는 기본값이다.

        `policy_seed` (T39): 주면 이 요청에 `ag3s/policy_seed` 를 싣고, 서버는 이 요청의 `policy.infer`
        직전에 정책 RNG 를 `jax.random.key(seed)` 로 놓는다. 에피소드 첫 요청에만 준다. 서버가 회신하지
        않으면 (`policy_seed` 를 모르는 옛 서버) **즉시 죽는다** — seed 가 조용히 무시된 run 을 짝지은
        비교에 넣으면 안 된다. `None` 이면 요청이 T39 전과 같다.
        """
        self.last_reasons = []
        self.last_reasons_source = "none"
        self.last_gate = {}
        if reset:
            # 지난 에피소드의 명령으로 HOLD 청크를 만들지 않는다.
            self.last_command = None
        out = self._infer_once(obs, reset=reset, exec_feedback=exec_feedback,
                               capture_time=capture_time, policy_seed=policy_seed)
        for kind in {str(r.get("kind")) for r in self.last_reasons}:
            self.reason_stats[kind] = self.reason_stats.get(kind, 0) + 1
        return out

    def note_command(self, applied: Optional[dict[str, Any]]) -> None:
        """제어 루프가 이번 스텝에 **실제로 명령한** 값 (`{"arm": [2N], "gripper": [2]}`, T18 의
        `applied_ctrl` 형식). `_hold` 의 대체 청크가 이것을 쓴다. `None` 이면 아무 일도 없다."""
        if applied is None:
            return
        self.last_command = action_row(applied["arm"], applied["gripper"])

    def _infer_once(self, obs: dict[str, Any], *, reset: bool = False,
                    exec_feedback: Optional[dict[str, Any]] = None,
                    capture_time: Optional[float] = None,
                    policy_seed: Optional[int] = None) -> dict[str, Any]:
        """한 청크. 무슨 일이 있어도 `{"actions": [H, 14]}` 를 돌려준다.

        `exec_feedback` 은 **직전 청크의 실행 사실**이다 (T18, `ExecutionLog.close()`). 이
        클래스는 청크를 **고르기만** 하고 실제로 `d.ctrl` 에 무엇이 들어갔는지는 모르므로, 그것을
        아는 제어 루프가 넘긴다. `None` 이면 `available=False` 로 싣는다 — 키를 빼면 서버가
        옛 클라이언트로 읽는다.

        예외를 밖으로 내지 않는 것은 refiner 와 같은 이유다 — 네트워크 한 번 끊긴 것으로 제어
        루프가 죽으면, 팔은 마지막 `d.ctrl` 에 매달린 채 아무도 hold 를 걸어주지 않는다.

        **예외가 하나 있다: shadow 모드의 짝이 안 맞을 때** (`_check_shadow_pairing`). 그것은
        런타임 실패가 아니라 설정 오류이고, hold 로 삼키면 *"shadow 라고 적힌 T6 실행"* 이
        남는다. 이 검사는 청크를 돌려주기 **전에** 하므로 로봇은 한 스텝도 실행하지 않는다.
        """
        self._seq += 1
        seq = self._seq
        self.stats["sent"] += 1
        # T39 — 보내기 전에 검사한다 (틀린 seed 는 hold 가 아니라 호출자의 버그다).
        self.last_policy_seed = (None if policy_seed is None
                                 else wire.check_policy_seed(policy_seed))
        if self._trace is not None:
            self._trace.begin_chunk(seq)

        try:
            request = self._pack(obs, reset=reset, seq=seq, exec_feedback=exec_feedback,
                                 capture_time=capture_time, policy_seed=self.last_policy_seed)
        except Exception as exc:  # noqa: BLE001 — 카메라 렌더 실패도 hold 로 간다
            return self._hold(f"could not capture the cameras ({exc})", "error", comms="error")

        started = time.monotonic()
        try:
            with _maybe_span(self._trace, "roundtrip") as span:
                result = self._policy.infer(request)
                span["seq"] = seq
        except Exception as exc:  # noqa: BLE001
            return self._hold(f"server error ({exc})", "error", comms="error")

        elapsed = time.monotonic() - started
        if elapsed > self.timeout_s:
            # 응답은 왔지만 늦었다. 이 청크는 이미 지나간 자세를 위한 것이다.
            return self._hold(f"response took {elapsed * 1000:.0f} ms "
                              f"(limit {self.timeout_s * 1000:.0f} ms)", "timeout", result)

        got = int(result.get("seq", -1))
        if got != seq:
            return self._hold(f"stale response: asked for seq {seq}, got {got}", "stale", result)

        # T39 — seed 를 보냈으면 회신이 와야 한다. 설정 오류라 hold 가 아니라 죽는다 (shadow 짝과 같다).
        self._check_policy_seed_echo(self.last_policy_seed, result)

        actions = np.asarray(result.get("actions"))
        if actions.ndim != 2 or actions.shape[1] != wire.ACTION_WIDTH:
            # **차원이 다르면 hold 다.** 자르거나 채워서 통과시키면 관절이 한 칸씩 밀린
            # 청크가 실행된다 — 형태는 맞고 뜻은 틀린, 가장 위험한 실패다.
            return self._hold(
                f"unexpected action shape {actions.shape}; expected "
                f"[H, {wire.ACTION_WIDTH}] (arm_joint_dim={wire.ARM_JOINT_DIM}). "
                "정책과 이 클라이언트의 차원이 다르면 서버 config 를 확인하십시오",
                "error", result, comms="dimension")

        reference = wire.unpack_actions_reference(result)
        # 모드가 어긋났으면 여기서 죽는다. 아래 어느 갈래로도 가지 않는다.
        self._check_shadow_pairing(reference, wire.unpack_shadow(result))
        if self.shadow and reference.shape != actions.shape:
            # reference 와 refined 의 모양이 다르면 어느 쪽이 어느 스텝인지 알 수 없다.
            # `actions` 차원 검사와 같은 이유로 hold 다 — 형태는 맞고 뜻은 틀린 실패.
            return self._hold(
                f"{wire.ACTIONS_REFERENCE} has shape {reference.shape} but actions has "
                f"{actions.shape}; refusing to execute a chunk of unknown alignment",
                "error", result, comms="dimension")

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
        # **최적화기가 몇 번 돌았나** (T15). 빈 dict 는 이 블록을 모르는 옛 서버다.
        self.last_to = wire.unpack_to(result)
        # **사유별 게이트** (T23). 서버 `safe` 와 로컬 표가 둘 다 실행이라고 할 때만 실행이다 —
        # 로컬은 서버보다 엄격해질 수만 있다. 옛 서버면 사유를 `safe` 에서 유도하므로 결정이 같다.
        server_safe = bool(result.get("safe", False))
        reasons, source = wire.unpack_verdict_reasons(result)
        self.last_reasons, self.last_reasons_source = reasons, source
        if self.gate == "off":
            return self._execute_ungated(result, reasons, source, server_safe, seq)
        if self.gate == "legacy":
            action = ("execute" if server_safe and result.get("trajopt_status") in
                      ("optimal", "feasible") else "hold")
            holding = [] if action == "execute" else (
                wire.gate_decision(reasons, self.gate_table)[1]
                or [{"kind": "legacy", "detail": f"trajopt_status="
                                                 f"{result.get('trajopt_status')}"}])
        else:
            action, holding = wire.gate_decision(reasons, self.gate_table)
            if not server_safe and action == "execute":
                # 서버가 HOLD 라고 했는데 로컬 표로는 실행이다 — 서버가 `legacy` 판정이거나 로컬 표가
                # 더 너그럽다. 서버가 이긴다.
                action = "hold"
                holding = [{"kind": "server_unsafe", "detail": "the server said safe=False"}]
        self.last_gate = {"mode": self.gate, "action": action, "source": source,
                          "server_safe": server_safe,
                          "hold_kinds": [str(r.get("kind")) for r in holding]}
        safe = action == "execute"
        if not safe and not self.shadow:
            return self._hold(self._explain(result, holding), "unsafe", result)

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
            self.last_reason = self._explain(result, holding)
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

    def _execute_ungated(self, result: dict[str, Any], reasons: list, source: str,
                         server_safe: bool, seq: int) -> dict[str, Any]:
        """`gate="off"` (T27): **refined 를 실행한다.** 판정은 `reasons` 게이트로 계산해 기록만 한다.

        `last_safe` · `last_ipc` · `stats` 는 **판정 그대로**다 (shadow 와 같은 규율) — 게이트가
        HOLD 였을 청크는 `last_safe=False` · `last_ipc="unsafe"` 이고, 실행됐다는 사실은
        `last_executed_chunk="refined"` 가 따로 말한다. 한 값이 판정과 실행을 겸하면 기록에서 둘을
        되살릴 수 없다.
        """
        would, holding = wire.gate_decision(reasons, self.gate_table)
        if not server_safe and would == "execute":
            would = "hold"
            holding = [{"kind": "server_unsafe", "detail": "the server said safe=False"}]
        would_hold = would == "hold"
        hold_kinds = [str(r.get("kind")) for r in holding]
        self.last_gate = {"mode": "off", "action": "execute", "source": source,
                          "server_safe": server_safe, "would_hold": would_hold,
                          "kinds": [str(r.get("kind")) for r in reasons],
                          "would_hold_kinds": hold_kinds,
                          # 실제로 HOLD 를 만든 사유 — 게이트가 꺼져 있으므로 언제나 비어 있다.
                          # 키를 두는 것은 `pi05_infer.py` 가 이 키로 연속 HOLD 를 세기 때문이다.
                          "hold_kinds": []}
        if would_hold:
            self.last_safe = False
            self.last_reason = self._explain(result, holding)
            self.last_ipc = "unsafe"
            self.stats["unsafe"] += 1
            # 일어났을 때만 키를 만든다 (`shadow_override` 와 같은 규약 — T0 `stats` 키 집합).
            self.stats["gate_off_override"] = self.stats.get("gate_off_override", 0) + 1
            self.would_hold += 1
            for kind in dict.fromkeys(hold_kinds):
                self.would_hold_kinds[kind] = self.would_hold_kinds.get(kind, 0) + 1
        else:
            self.last_safe = True
            self.last_reason = ""
            self.last_ipc = "ok"
            self.stats["safe"] += 1
        self.last_executed_chunk = "refined"
        if self._trace is not None:
            self._trace.mark("verdict", safe=self.last_safe, seq=seq, gate="off",
                             executed="refined", would_hold=would_hold,
                             **{k: v for k, v in self.last_verdict.items() if k != "timing_ms"})
        return result

    @property
    def should_execute(self) -> bool:
        """이 프레임의 청크를 실행해도 되는가. **hold 판단은 이 하나만 본다.**

        shadow 가 아니면 `last_safe` 와 정확히 같다 — 그래서 기본 동작이 바뀌지 않는다.
        shadow 면 판정이 unsafe 여도 참이 될 수 있고, 그때 무엇이 실행되는지는
        `last_executed_chunk` 가 말한다.
        """
        return self.last_executed_chunk != "none"

    @staticmethod
    def _check_policy_seed_echo(sent: Optional[int], result: dict[str, Any]) -> None:
        """보낸 seed 와 서버의 회신이 같은가 (T39). 보내지 않았으면 검사하지 않는다."""
        if sent is None:
            return
        echo = wire.unpack_policy_seed_echo(result)
        if echo != sent:
            raise RuntimeError(
                f"sent {wire.POLICY_SEED}={sent} but the server replied {echo!r}. The server does "
                "not apply policy seeds (pre-T39 benchmark code?) — its RNG kept running in "
                "request order, so this run is NOT determined by (episode, seed). Restart the "
                "server from code that has benchmark/trajopt/policy_seed.py")

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

    def _check_shadow_pairing(self, reference, server_shadow=None) -> None:
        """`--safe-shadow`(로컬)와 `--shadow`(서버)의 짝. **안 맞으면 즉시 죽는다.**

        **근거가 T9 에서 옮겨갔다.** 예전에는 `actions_reference` 의 있음/없음이 서버 모드의
        유일한 신호였고, 그래서 closed loop 이 원본 청크를 실을 수 없었다 (실으면 로컬이
        "서버가 shadow 다" 로 읽고 즉시 죽는다). 지금은 응답의 `shadow` 키가 모드를 **명시**
        하고, 이 검사는 그것을 본다. `actions_reference` 는 데이터일 뿐이다.

        `server_shadow is None` 이면 그 키를 모르는 **옛 서버**이므로 예전 규칙으로 물러난다 —
        그때는 `actions_reference` 의 있음/없음이 여전히 유일한 신호다.

        **거절은 한 줄도 느슨해지지 않았다.** 두 방향 모두 예외인 이유는 **둘 다 기록을 거짓으로
        만들기** 때문이다:

        * 로컬만 shadow → reference 가 없다. 조용히 refined 를 실행하면 shadow 가 아닌데
          shadow 라고 기록된다.
        * 서버만 shadow → 로컬이 reference 를 무시하고 refined 를 실행한다. 로봇은 닫힌
          고리로 도는데 서버 기록은 shadow 라고 말한다.

        hold 로 수렴시키지 않는 이유: hold 는 *"이 청크를 실행하지 않는다"* 이고 그래도 실행은
        계속된다. 설정이 어긋난 채로 계속 도는 실행은 결과가 무슨 뜻인지 아무도 모른다.
        """
        if server_shadow is None:
            # 옛 서버 — 추론으로 물러난다.
            if self.shadow and reference is None:
                raise RuntimeError(self._pairing_message(server_shadow=False))
            if not self.shadow and reference is not None:
                raise RuntimeError(self._pairing_message(server_shadow=True))
            return
        if bool(server_shadow) != self.shadow:
            raise RuntimeError(self._pairing_message(server_shadow=bool(server_shadow)))
        if self.shadow and reference is None:
            # 서버가 shadow 라고 말했는데 실행할 청크를 안 보냈다. 모드 불일치가 아니라
            # **서버 쪽 결함**이지만, 결과는 같다 — 실행할 것이 없으므로 여기서 죽는다.
            raise RuntimeError(
                f"the server says {wire.SHADOW}=True but sent no {wire.ACTIONS_REFERENCE}: "
                "there is nothing for the robot to execute in shadow mode. This is a server-side "
                "fault, not a mode mismatch — check the server log for a hold on this chunk.")

    def _pairing_message(self, *, server_shadow: bool) -> str:
        """두 곳(생성자·프레임)에서 같은 문장을 쓴다. 그래서 *"응답이 왔는데"* 처럼 한쪽에서만
        참인 말을 쓰지 않는다.

        **근거를 문장에 적지 않는다** (T9). 이 메시지는 메타데이터(`shadow`) · 응답의 `shadow`
        키 · 옛 서버의 `actions_reference` 추론 세 경로에서 다 쓰이므로, 한 경로의 근거를 적으면
        나머지 두 경로에서 거짓말이 된다. 예전 문장은 *"it carries 'actions_reference'"* 라고
        적었는데, 그것은 closed loop 도 그 키를 싣는 지금 틀린 말이다.
        """
        if server_shadow:
            return (
                "the server runs with --shadow but this client does not: --safe-shadow is off. "
                "Refusing to continue — the robot would execute the refined chunk, a closed "
                "loop, while the server-side record says shadow. Either add --safe-shadow "
                "locally or restart the server without --shadow.")
        return (
            "--safe-shadow is on but the server is not running with --shadow. Refusing to "
            "continue — executing the refined chunk here would record a shadow run that "
            "actually closed the loop, the one mistake this mode exists to prevent. Restart "
            f"the server with --shadow. (The server may also be an old build that does not "
            f"report {wire.SHADOW!r}; then it is telling us by not carrying "
            f"{wire.ACTIONS_REFERENCE!r}.)")

    # ----------------------------------------------------------------------------------
    def _pack(self, obs: dict[str, Any], *, reset: bool, seq: int,
              exec_feedback: Optional[dict[str, Any]] = None,
              capture_time: Optional[float] = None,
              policy_seed: Optional[int] = None) -> dict[str, Any]:
        """세 카메라를 **지금** 찍어 요청에 싣는다.

        카메라마다 `robot_state` 와 촬영 시각을 따로 담는다. 손목 카메라는 팔과 함께 움직이므로
        하나의 `q_now` 로 세 대를 변환하면 손목 클라우드가 번지고, 더 나쁘게는 자기 필터가
        어긋나 로봇 점이 씬에 남는다.

        **실행 피드백도 여기서 싣는다** (T18). 없으면 `available=False` 와 이유를 싣는다 —
        키를 빼면 서버가 이 클라이언트를 옛 버전으로 읽는다. 카메라를 찍기 **전에** 붙드는 것은
        캡처가 실패해 hold 로 가도 *"무엇을 보내려 했나"* 가 기록에 남게 하려는 것이다.

        **촬영 시각 (T30c).** 카메라마다 렌더가 끝난 순간을 `render_stamps` 로 잰다 (진단용).
        `capture_time` 을 받았으면 판정용 `stamps` 는 세 카메라 모두 그 한 순간이다 — 이
        메서드는 `m`/`d` 를 읽기만 하고 `mj_step` 을 하지 않으므로, 순차 렌더의 시차는 씬이
        아니라 렌더의 것이다. 받지 않았으면 `stamps = render_stamps` (T30c 전 동작).
        """
        # 이번 요청의 값만 남긴다 — 캡처 전에 실패하면 지난 청크의 시각이 이번 기록에 실리지 않게.
        self.last_stamps, self.last_render_stamps, self.last_stamp_mode = {}, {}, "none"
        if exec_feedback is None:
            exec_feedback = wire.no_exec_feedback(
                "first chunk of the episode (reset)" if reset else
                "the control loop passed no exec_feedback to SafeRemoteClient.infer")
        self.last_exec_feedback = dict(exec_feedback)
        if capture_time is not None:
            capture_time = float(capture_time)
            # 미래의 순간은 호출자의 버그다 (다른 시계 · 렌더 뒤에 잰 값). 그런 순간을 실으면 서버가
            # 보는 나이가 음수가 된다 — 조용히 싣지 않고 hold 로 간다 (`_infer_once` 가 잡는다).
            if not np.isfinite(capture_time) or capture_time > time.monotonic():
                raise ValueError(
                    f"capture_time {capture_time!r} is not a past time.monotonic() instant")
        depth, K, T, state, render_stamps = {}, {}, {}, {}, {}
        with _maybe_span(self._trace, "capture"):
            for cam in self.cameras:
                frame = self._scene.capture(cam)
                depth[cam] = np.asarray(frame.depth, np.float64)
                K[cam] = np.asarray(frame.camera_intrinsics, np.float64)
                T[cam] = np.asarray(frame.T_base_cam, np.float64)
                state[cam] = np.asarray(frame.robot_state, np.float64)
                render_stamps[cam] = time.monotonic()
        if capture_time is None:
            stamps, mode = dict(render_stamps), "render_end"
        else:
            stamps, mode = {cam: capture_time for cam in self.cameras}, "sim_frozen"
        self.last_stamps = dict(stamps)
        self.last_render_stamps = dict(render_stamps)
        self.last_stamp_mode = mode
        return wire.pack_request(
            obs, cameras=self.cameras, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=state, stamps=stamps, phase=self.phase,
            active_manipulators=self.active_manipulators, reset=reset, seq=seq,
            exec_feedback=self.last_exec_feedback,
            render_stamps=render_stamps, stamp_mode=mode, policy_seed=policy_seed)

    def _hold(self, reason: str, kind: str,
              result: Optional[dict] = None, *, comms: Optional[str] = None) -> dict[str, Any]:
        """실행하지 않기로 한다. 청크는 **그대로 돌려준다** — 무엇이 거부됐는지 보이도록.

        `comms` 는 통신 실패의 하위 종류 (`wire.COMMS_KINDS`) — 주면 사유를 `comms` 하나로 둔다
        (T23). `unsafe` HOLD 는 호출부가 이미 서버의 사유를 `last_reasons` 에 넣었다.
        """
        self.last_safe = False
        self.last_reason = reason
        self.last_ipc = kind
        if self.gate == "off":
            # **게이트가 꺼져 있어도 도착하지 않은 청크는 실행할 수 없다** (T27). 그것을 크게 찍고
            # 센다 — 사다리 실험에서 이 수는 0 이어야 하고, 0 이 아니면 그 실행은 "HOLD 없음" 이
            # 아니다. `gate="off"` 에서 `unsafe` 로 여기 오는 길은 없다 (`_execute_ungated`).
            self.comms_holds += 1
            print(f"[safe] !!! GATE OFF but chunk seq {self._seq} did NOT arrive ({kind}"
                  f"{'/' + comms if comms else ''}) — fixed HOLD #{self.comms_holds}: {reason}")
        if kind != "unsafe":
            sub = comms if comms in wire.COMMS_KINDS else (
                kind if kind in wire.COMMS_KINDS else "error")
            self.last_reasons = [wire.make_reason("comms", reason, ipc=sub)]
            self.last_reasons_source = "client"
            self.last_gate = {"mode": self.gate, "action": "hold", "source": "client",
                              "server_safe": (None if result is None
                                              else bool(result.get("safe", False))),
                              "hold_kinds": ["comms"]}
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
            self.last_to = wire.unpack_to(result)
        else:
            # 응답이 아예 없으면 지각 사유도 없다. **지난 프레임 것을 남겨 두지 않는다** —
            # 남기면 이 프레임이 그 사유로 멈춘 것처럼 읽힌다. 청크와 위반 행의 신원도 같다.
            self.last_ag3s = {}
            self.last_actions_refined = None
            self.last_actions_reference = None
            self.last_violation_pair = None
            self.last_to = {}
            from benchmark.ag3s.fields.provenance import FieldProvenance
            self.last_field = FieldProvenance.unavailable(
                f"no response to read a field from ({kind}): {reason}")
        self.stats[kind] = self.stats.get(kind, 0) + 1
        if self._trace is not None:
            self._trace.mark("verdict", safe=False, kind=kind, reason=reason)
        if result is not None and isinstance(result.get("actions"), np.ndarray):
            return result
        # 서버가 아무것도 못 줬다. 유지 청크를 만들어 형태 계약을 지킨다 — 호출부가
        # `chunk[chunk_step]` 을 무조건 인덱싱하므로 None 을 돌려주면 거기서 죽는다.
        #
        # **T23: 행 = 마지막으로 명령한 팔·gripper** (`note_command`). 측정값이 아니다 — 매 스텝
        # 측정값을 목표로 주면 servo 의 정착 오차가 목표에 쌓이고(T16 드리프트), 닫힘 명령을 열린
        # 측정 개도로 덮는다 (지침 §8.2). 명령이 아직 없을 때만(에피소드 첫 청크) 현재 상태다.
        row = self._hold_row()
        hold = np.tile(row, (50, 1))
        return {"actions": hold, "safe": False, "hold_reason": reason}

    def _hold_row(self) -> np.ndarray:
        """대체 청크의 한 행 — 마지막 명령, 없으면 현재 상태 (`_current_state`)."""
        if self.last_command is not None and len(self.last_command) == wire.ACTION_WIDTH:
            return np.asarray(self.last_command, np.float64)
        return np.asarray(self._current_state(), np.float64)

    def _current_state(self) -> np.ndarray:
        """action 레이아웃의 현재 상태. hold 청크의 모든 행이 된다.

        레이아웃은 `[왼팔 N, 왼 그리퍼, 오른팔 N, 오른 그리퍼]` 이고 `N = wire.ARM_JOINT_DIM`
        이다. **6 을 박지 않는다** — 박아 두면 16D 에서 그리퍼가 손목 자리에 들어가고 hold
        청크가 엉뚱한 관절을 지령한다.

        **매핑은 이름으로 한다** (2026-09-28 에 고쳤다). 그 전에는 `state[:2N]` 로 **위치를
        잘랐고**, 씬이 주는 것은 `DEFAULT_RBY1_JOINTS` 순서의 20-vector (torso 6 · 오른팔 7 ·
        왼팔 7) 이므로 **14 개 열 전부 엉뚱한 관절**이 들어갔다:

        | chunk 열 | 위치로 자르면 들어가던 값 |
        |---|---|
        | 왼팔 0~6 | `torso_0..5` + `right_arm_0` |
        | 오른팔 0~6 | `right_arm_1..6` + `left_arm_0` |

        `ChunkLayout.rby1` 이 이미 이름으로 매핑하므로 **같은 길**을 쓴다 — 매핑이 두 곳에 따로
        있으면 한쪽이 조용히 한 칸 밀린다.

        **그리퍼는 정규화 값(0~1)이다.** 로컬이 `ctrl = norm × RBY1_GRIPPER_OPEN(−0.045)` 로
        적용하므로 (`pi05_infer.py:535`) **`1.0` 이 열림이고 `0.0` 이 닫힘**이다. 예전 코드는
        `0.0` 을 넣으면서 주석에 "열림" 이라고 적었다 — 값과 주석이 반대였다. 씬이 그리퍼 상태를
        내놓으면 그것을 쓰고(`gripper_norm()`), 못 내놓으면 `HOLD_GRIPPER_NORM` 을 쓴다.

        **읽는 사람이 알아야 할 것**: 물건을 쥔 채 hold 가 걸리면 열림 값은 그것을 **놓는다.**
        그래서 씬이 실제 상태를 내놓는 쪽이 정상 경로이고, 상수는 마지막 수단이다. 그리고 이
        경로는 **지금 로봇이 쓰는 경로가 아니다** — `pi05_infer.py:1645` 가 hold 프레임에서 자기
        `rby1_state()` 를 쓴다. 여기 고친 것은 잠복 버그이고, `{"actions": ...}` 로 나가므로
        다른 호출부가 실행하면 실제로 팔이 엉뚱한 각도로 지령된다.
        """
        state = np.asarray(self._scene.robot_state(), np.float64)
        if len(state) == wire.ACTION_WIDTH:
            return state
        n = wire.ARM_JOINT_DIM
        grippers = self._gripper_norm()
        layout = self._chunk_layout()
        if layout is not None and len(state) == layout.nq_model:
            out = np.empty(wire.ACTION_WIDTH, np.float64)
            for column, q in layout.action_to_q:
                out[column] = state[q]
            for column, value in zip(wire.gripper_columns(n), grippers):
                out[column] = value
            return out
        if len(state) == 2 * n:
            # 씬이 **팔 관절만** 준 경우. 그때만 위치로 읽는 것이 맞다 — 그 이상을 받으면
            # 무엇이 어느 열인지 이름 없이는 알 수 없으므로 추측하지 않는다.
            return np.concatenate(
                [state[:n], [grippers[0]], state[n:2 * n], [grippers[1]]])
        raise ValueError(
            f"씬이 준 상태가 {len(state)}-D 인데 이 레이아웃이 아는 것은 "
            f"{wire.ACTION_WIDTH}-D(action) · {2 * n}-D(팔만) · "
            f"{'모델 q' if layout is None else f'{layout.nq_model}-D(모델 q)'} 입니다. "
            "무엇이 어느 열인지 추측하지 않습니다 — 추측하면 팔이 엉뚱한 관절로 지령됩니다")

    def _chunk_layout(self):
        """이 차원의 RB-Y1 레이아웃, 또는 `None` (만들 수 없으면).

        `ChunkLayout.rby1` 을 호출할 뿐이다. **여기서 매핑을 다시 쓰지 않는 것**이 요점이다 —
        refiner 가 쓰는 것과 같은 표를 써야 hold 청크의 열과 refined 청크의 열이 같은 관절을
        가리킨다.
        """
        try:
            from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
            from benchmark.trajopt.types import ChunkLayout

            return ChunkLayout.rby1(DEFAULT_RBY1_JOINTS, action_dim=wire.ACTION_WIDTH,
                                    arm_joint_dim=wire.ARM_JOINT_DIM)
        except Exception:  # noqa: BLE001 — 레이아웃을 못 만들면 아래 길이 검사가 말한다
            return None

    def _gripper_norm(self) -> tuple[float, float]:
        """`(왼, 오른)` 정규화 그리퍼 값. 씬이 내놓으면 그것, 아니면 `HOLD_GRIPPER_NORM`."""
        read = getattr(self._scene, "gripper_norm", None)
        if callable(read):
            try:
                left, right = read()
                return (float(left), float(right))
            except Exception:  # noqa: BLE001 — 읽기 실패로 hold 를 못 만들면 안 된다
                pass
        return (HOLD_GRIPPER_NORM, HOLD_GRIPPER_NORM)

    @staticmethod
    def _explain(result: dict[str, Any], holding: Optional[Sequence[dict]] = None) -> str:
        """`_explain_verdict` 앞에 HOLD 를 만든 사유 kind 를 붙인다 (T23).

        `collision` · `unverified` 는 서버가 적은 행의 신원(`detail`)을 함께 싣는다 — 어느 link 가
        무엇과 몇 mm 인지가 한 줄에 있어야 제어 루프 로그만 보고도 읽힌다.
        """
        base = SafeRemoteClient._explain_verdict(result)
        holding = [r for r in (holding or ()) if isinstance(r, dict)]
        if not holding:
            return base
        kinds = ", ".join(dict.fromkeys(str(r.get("kind")) for r in holding))
        detail = next((str(r.get("detail")) for r in holding
                       if r.get("kind") in ("collision", "unverified") and r.get("detail")), "")
        return f"[{kinds}] {base}" + (f" — {detail}" if detail else "")

    @staticmethod
    def _explain_verdict(result: dict[str, Any]) -> str:
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
            if self.reason_stats:
                print("[safe] verdict reasons (chunks): " + ", ".join(
                    f"{k} {v}" for k, v in sorted(self.reason_stats.items())))
            if self.gate == "off":
                print(f"[safe] GATE OFF: executed the refined chunk on {s['sent'] - self.comms_holds}"
                      f"/{s['sent']} chunks; a reasons gate would have held {self.would_hold} "
                      f"({', '.join(f'{k} {v}' for k, v in sorted(self.would_hold_kinds.items())) or '-'}); "
                      f"comms HOLD {self.comms_holds}"
                      + (" !!! (must be 0 for a no-HOLD run)" if self.comms_holds else ""))
            if self.shadow:
                print(f"[shadow] the robot executed the policy reference chunk on every "
                      f"executed chunk; {s.get('shadow_override', 0)} of them carried an "
                      f"unsafe verdict and ran anyway (nothing was held for the verdict)")


class ExecutionLog:
    """제어 루프가 청크 하나 동안 **실제로 적용한 것**을 모은다 (T18).

    `SafeRemoteClient` 는 청크를 고를 뿐이고, `d.ctrl` 에 무엇이 들어갔는지는 제어 루프만 안다 —
    HOLD 프레임은 청크가 아니라 `rby1_state()` 를 적용하기 때문이다 (지침 §8.1). 그 사실이 서버에
    가지 않아서 T16 의 닫힘 제안(t=176 · 208)이 **실행되지 않았다는 것**을 서버가 몰랐다.

    쓰는 법 (`pi05_infer.py` 의 제어 루프)::

        feedback = log.close(measured_gripper=...)      # 직전 청크의 사실, 새 요청 직전에
        result = client.infer(obs, exec_feedback=feedback)
        log.open_from(client, t_step=t_step)            # 이번 청크의 결정
        ...
        apply_action(...)
        log.step(executed=..., planned_row=chunk[i], applied_arm=..., applied_gripper=...)

    **값을 지어내지 않는다.** 열린 청크가 없거나 한 스텝도 돌지 않았으면 `available=False` 와
    이유를 낸다.
    """

    def __init__(self):
        self._chunk: Optional[dict[str, Any]] = None

    def reset(self) -> None:
        self._chunk = None

    def open(self, *, seq: int, t_step: int, executed_chunk: str, ipc: str,
             reason: str = "") -> None:
        """청크 하나를 연다. **HOLD 사유는 여기서 정해진다** — 로컬의 결정이 청크 단위이므로."""
        hold = executed_chunk == "none"
        self._chunk = {
            "seq": int(seq), "t_step_start": int(t_step),
            "executed_chunk": str(executed_chunk), "ipc": str(ipc),
            # `ipc` 가 `ok` 인데 청크를 안 골랐다면 설명이 안 되는 것이다 — `error` 로 적는다.
            "hold_kind": ((ipc if ipc in wire.HOLD_KINDS else "error") if hold else None),
            "hold_reason": (str(reason) if hold else None),
            "executed": [], "planned": [], "applied_gripper": [], "applied_arm": [],
        }

    def open_from(self, client: "SafeRemoteClient", *, t_step: int) -> None:
        """`client` 가 방금 내린 결정으로 연다. `infer()` **직후**에 부른다."""
        self.open(seq=client._seq, t_step=t_step, executed_chunk=client.last_executed_chunk,
                  ipc=client.last_ipc, reason=client.last_reason)

    def step(self, *, executed: bool, planned_row, applied_arm, applied_gripper) -> None:
        """한 제어 스텝. `apply_action` **직후** — `d.ctrl` 에 실제로 들어간 값을 준다.

        `planned_row` 는 그 스텝에 계획 청크가 말한 행 (`chunk[chunk_step]`) 이다. HOLD 여도
        준다 — *"제안됐으나 실행 안 됨"* 을 서버가 보려면 제안이 나란히 있어야 한다.
        """
        if self._chunk is None:
            return
        row = np.asarray(planned_row, np.float64).reshape(-1)
        n = (len(row) - 2) // 2
        left, right = wire.gripper_columns(n)
        self._chunk["executed"].append(bool(executed))
        self._chunk["planned"].append((float(row[left]), float(row[right])))
        grip = np.asarray(applied_gripper, np.float64).reshape(-1)
        self._chunk["applied_gripper"].append((float(grip[0]), float(grip[1])))
        self._chunk["applied_arm"].append(np.asarray(applied_arm, np.float64).reshape(-1))

    @property
    def n_steps(self) -> int:
        return 0 if self._chunk is None else len(self._chunk["executed"])

    def close(self, *, measured_gripper) -> dict[str, Any]:
        """열린 청크의 사실을 와이어 형식으로 내고 닫는다. `measured_gripper` 는 **지금**
        (= 다음 요청의 촬영 시점) 잰 (왼, 오른) 개도, 1 = 열림."""
        chunk, self._chunk = self._chunk, None
        if chunk is None:
            return wire.no_exec_feedback("no chunk has run yet in this episode")
        if not chunk["executed"]:
            return wire.no_exec_feedback(
                f"chunk seq {chunk['seq']} was opened but ran no control step")
        return wire.make_exec_feedback(
            seq=chunk["seq"], t_step_start=chunk["t_step_start"],
            executed=chunk["executed"], executed_chunk=chunk["executed_chunk"],
            ipc=chunk["ipc"], hold_kind=chunk["hold_kind"], hold_reason=chunk["hold_reason"],
            planned_gripper=np.asarray(chunk["planned"], np.float64),
            applied_gripper=np.asarray(chunk["applied_gripper"], np.float64),
            applied_arm=np.stack(chunk["applied_arm"]),
            measured_gripper=measured_gripper)


def action_row(arm, gripper) -> np.ndarray:
    """`[왼팔 N, 왼 gripper, 오른팔 N, 오른 gripper]` — `applied_ctrl` 형식(`arm=[왼 N, 오른 N]`,
    `gripper=[왼, 오른]` 정규화)을 action 한 행으로. `apply_action` 이 받는 순서다."""
    arm = np.asarray(arm, np.float64).reshape(-1)
    grip = np.asarray(gripper, np.float64).reshape(-1)
    if arm.shape[0] % 2 or grip.shape != (2,):
        raise ValueError(f"arm must be [2N] and gripper [2], got {arm.shape} and {grip.shape}")
    n = arm.shape[0] // 2
    return np.concatenate([arm[:n], grip[:1], arm[n:], grip[1:]])


class HoldController:
    """HOLD 동안 로봇에 **무엇을** 줄지와, 언제 **그만둘지** (T23, 지침 §8.4 · §8.5).

    T23 전에는 HOLD 스텝마다 `action = rby1_state()` — 그 스텝의 측정 관절과 측정 gripper 개도를
    목표로 다시 줬다. 그래서 (1) HOLD 청크의 닫힘 명령이 사라지고 (T16 t=176 · 208), (2) servo 의
    정착 오차가 목표에 쌓여 팔이 0.10–0.17°/step 흘렀다 (T16 t=72–92, 손가락 23 mm) — 지침 §8.2.

    `fixed` (기본) 는 이렇게 한다:

    * **`q_hold` 를 HOLD 에 들어갈 때 한 번 잡는다** — 직전 스텝에 **명령한** 팔 목표 (`d.ctrl`),
      측정 `qpos` 가 아니다. HOLD 가 청크를 넘어 이어지는 동안 같은 값이다. 청크가 한 번 실행되면
      풀린다.
    * **gripper 는 팔과 따로** — 마지막으로 명령한 gripper 를 그대로 준다. 열지도 닫지도 않는다
      (지침 §8.4: HOLD 중 VLA 대로 닫지 않는다). 쥔 물체가 있으면(T22 `held`) 그 명령이 곧 파지 유지다.
    * 직전 명령을 모르면 (에피소드 첫 스텝) 측정값으로 한 번 잡고 `source="measured"` 로 남긴다.

    `legacy` 는 T23 전 그대로 매 스텝 측정값이다.

    **복구 한도**: 연속 HOLD 청크 수가 `max_hold_chunks` 를 넘으면 `begin_chunk` 가 중단 사유를
    돌려준다 (`None` = 한도 없음). 멈추는 것은 제어 루프다.

    제어 루프에서 쓰는 법 (`pi05_infer.py`)::

        abort = hold.begin_chunk(executed=client.should_execute, reason=..., kinds=...)
        ...
        if HOLD: action = hold.hold_action(last_command=applied_ctrl(), measured=rby1_state())
        else:    hold.release()
    """

    def __init__(self, *, mode: str = "fixed", max_hold_chunks: Optional[int] = None):
        if mode not in HOLD_MODES:
            raise ValueError(f"mode must be one of {HOLD_MODES}, got {mode!r}")
        if max_hold_chunks is not None and int(max_hold_chunks) < 0:
            raise ValueError(f"max_hold_chunks must be >= 0 or None, got {max_hold_chunks}")
        self.mode = mode
        self.max_hold_chunks = None if max_hold_chunks is None else int(max_hold_chunks)
        self.reset()

    def reset(self) -> None:
        #: 이번 HOLD 의 팔 기준 `[왼 N, 오른 N]` 과 gripper `[왼, 오른]`. HOLD 가 아니면 `None`.
        self.q_hold: Optional[np.ndarray] = None
        self.gripper_hold: Optional[np.ndarray] = None
        #: `q_hold` 를 어디서 잡았나: `command` (직전 `d.ctrl`) · `measured` (명령을 몰랐다).
        self.source: Optional[str] = None
        #: 이번 HOLD 에 들어간 뒤 적용한 HOLD 스텝 수.
        self.age = 0
        #: 연속 HOLD 청크 수 (청크가 실행되면 0).
        self.consecutive = 0
        self.entry_reason = ""
        self.entry_kinds: list[str] = []
        self.reason = ""
        self.kinds: list[str] = []
        self.stats = {"hold_chunks": 0, "hold_steps": 0, "holds": 0, "max_consecutive": 0}

    # --- 청크 경계 -------------------------------------------------------------------------
    def begin_chunk(self, *, executed: bool, reason: str = "",
                    kinds: Sequence[str] = ()) -> Optional[dict[str, Any]]:
        """새 청크의 결정을 알린다. 한도를 넘으면 **중단 사유** dict, 아니면 `None`."""
        if executed:
            self.consecutive = 0
            return None
        self.consecutive += 1
        self.stats["hold_chunks"] += 1
        self.stats["max_consecutive"] = max(self.stats["max_consecutive"], self.consecutive)
        self.reason, self.kinds = str(reason), [str(k) for k in kinds]
        if self.max_hold_chunks is not None and self.consecutive > self.max_hold_chunks:
            return {"hold_chunks": self.consecutive, "limit": self.max_hold_chunks,
                    "reason": self.reason, "kinds": list(self.kinds),
                    "entry_reason": self.entry_reason, "entry_kinds": list(self.entry_kinds)}
        return None

    # --- 제어 스텝 -------------------------------------------------------------------------
    def hold_action(self, *, last_command: Optional[dict[str, Any]],
                    measured: np.ndarray) -> np.ndarray:
        """HOLD 한 스텝의 action 행.

        `last_command` 는 **이번 `apply_action` 전의** `d.ctrl` (`applied_ctrl()` 형식) — 즉 직전
        스텝의 명령이다. `measured` 는 `rby1_state()` (action 레이아웃). `legacy` 면 `measured` 그대로.
        """
        measured = np.asarray(measured, np.float64).reshape(-1)
        if self.q_hold is None:
            self.stats["holds"] += 1
            self.entry_reason, self.entry_kinds = self.reason, list(self.kinds)
            self.age = 0
            n = (measured.shape[0] - 2) // 2
            if self.mode == "fixed" and last_command is not None:
                self.q_hold = np.asarray(last_command["arm"], np.float64).reshape(-1).copy()
                self.gripper_hold = np.asarray(last_command["gripper"],
                                               np.float64).reshape(-1).copy()
                self.source = "command"
            else:
                self.q_hold = np.concatenate([measured[:n], measured[n + 1:2 * n + 1]])
                self.gripper_hold = np.asarray([measured[n], measured[2 * n + 1]])
                self.source = "measured"
        else:
            self.age += 1
        self.stats["hold_steps"] += 1
        if self.mode == "legacy":
            return measured
        return action_row(self.q_hold, self.gripper_hold)

    def release(self) -> None:
        """청크가 실행되는 스텝. HOLD 기준을 푼다 (다음 HOLD 는 새로 잡는다)."""
        self.q_hold = None
        self.gripper_hold = None
        self.source = None
        self.age = 0

    @property
    def holding(self) -> bool:
        return self.q_hold is not None

    def record(self, measured: Optional[np.ndarray] = None) -> Optional[dict[str, Any]]:
        """control frame 의 `hold` 키. HOLD 가 아니면 `None`.

        `measured` (`rby1_state()`) 를 주면 `q_hold` 대비 측정 편차의 최대값(rad)을 함께 싣는다 —
        드리프트를 기록만 보고 셀 수 있게.
        """
        if self.q_hold is None:
            return None
        out = {"mode": self.mode, "source": self.source,
               "entry_reason": self.entry_reason, "entry_kinds": list(self.entry_kinds),
               "reason": self.reason, "kinds": list(self.kinds),
               "q_hold": [float(v) for v in self.q_hold],
               "gripper_hold": [float(v) for v in self.gripper_hold],
               "age": int(self.age), "hold_chunks": int(self.consecutive)}
        if measured is not None:
            m = np.asarray(measured, np.float64).reshape(-1)
            n = (m.shape[0] - 2) // 2
            arm = np.concatenate([m[:n], m[n + 1:2 * n + 1]])
            if arm.shape == self.q_hold.shape:
                out["max_abs_dev_rad"] = float(np.max(np.abs(arm - self.q_hold)))
        return out


class _NullSpan:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


def _maybe_span(trace, name: str):
    return trace.span(name) if trace is not None else _NullSpan()
