"""클라이언트와 서버가 주고받는 것의 **유일한 정의**.

포장과 해체를 같은 파일에 둔 이유는 하나다. 둘이 갈라지면 증상이 조용하다 — 키 이름 하나가
달라도 서버는 "카메라가 안 왔다" 며 기하 없이 계속 돌고, 로봇은 제약 없는 궤적을 안전하다고
믿으며 실행한다. 예외가 아니라 **더 위험한 정상 동작**으로 실패하는 종류의 버그다. 그래서
`pack_*` 와 `unpack_*` 은 짝을 이뤄 여기 있고, 테스트가 왕복을 검사한다.

## 요청

정책이 이미 쓰는 키(`state`, `images`, `prompt`)에 `ag3s/` 접두 키를 **더한다**. 접두사를
붙이는 것은 정책 입력 변환이 모르는 키를 만나 깨지지 않게 하려는 것이고, 서버에서 벗겨낸다.

| 키 | 내용 |
|---|---|
| `ag3s/cameras` | 카메라 이름 목록. 이 순서가 나머지 배열의 순서다 |
| `ag3s/depth/<cam>` | (H, W) uint16 밀리미터 — 실제 depth 카메라가 주는 것 |
| `ag3s/K/<cam>` | (3, 3) intrinsics |
| `ag3s/T_base_cam/<cam>` | (4, 4) extrinsics, **촬영 시점** |
| `ag3s/robot_state/<cam>` | (nq,) **촬영 시점** 관절. 카메라마다 다르다 |
| `ag3s/stamp/<cam>` | 촬영 시각 (초, 단조시계). 이미지 · `robot_state` · `T_base_cam` 이 **이 한 순간**의 것이다. 시뮬레이션 관측이면 세 카메라가 같은 값 (T30c, 아래) |
| `ag3s/render_stamp/<cam>` | **진단용** — 그 카메라의 렌더가 끝난 벽시계 순간 (초, 단조시계). 서버는 판정에 쓰지 않는다 (T30c) |
| `ag3s/stamp_mode` | `stamp/<cam>` 이 무엇인가: `sim_frozen` (관측 하나에 한 순간) · `render_end` (카메라마다 렌더 끝) (T30c) |
| `ag3s/phase` | 조작 단계. AG3S 는 절대 추론하지 않고 주입받는다 |
| `ag3s/active_manipulators` | 접촉이 허용된 매니퓰레이터 |
| `ag3s/reset` | True 면 SEAM·AG3S·TO 의 내부 상태와 warm-start 를 모두 버린다 |
| `ag3s/seq` | 요청 일련번호. 응답에 그대로 돌아오고, 오래된 응답을 버리는 근거가 된다 |
| `ag3s/exec_feedback` | **직전 청크의 실행 사실** (T18). 아래 |
| `ag3s/policy_seed` | **선택.** 정책 RNG seed (int, T39). 서버가 이 요청의 `policy.infer` 직전에 정책 RNG 를 `jax.random.key(seed)` 로 다시 놓는다. 아래 |

카메라마다 `robot_state` 를 따로 싣는 것이 이 형식의 핵심이다. 손목 카메라는 팔과 함께
움직이므로 80 ms 전 프레임은 80 ms 전 자세에 놓여야 한다. 하나의 `q_now` 로 세 대를 변환하면
손목 클라우드가 번지고, 더 나쁘게는 자기 필터가 어긋나 로봇 점이 씬에 남아 그리퍼에 용접된
유령 장애물로 뭉친다.

### `ag3s/exec_feedback` — 서버가 "제안" 이 아니라 "실행된 것" 을 안다 (T18)

T16 에서 정책이 t=176 · t=208 에 gripper 닫힘을 제안했지만 그 청크는 HOLD 로 실행되지 않았고,
**서버는 그것을 몰랐다.** 서버가 가진 것은 자기가 **보낸** 청크뿐이었다 — latch 는 계획 청크의 첫
행을 gripper 신호로 읽었고, 연속성 참조는 이전 청크의 앞 K 스텝이 실행됐다고 **가정**했다.
HOLD·부분 실행이면 셋 다 틀린다 (지침 §6.4).

그래서 로컬이 다음 요청에 **직전 청크의 사실**을 싣는다. 하나의 딕셔너리이고, 소유자는 로컬
제어 루프다 (`client.ExecutionLog` 가 `apply_action` 직후에 모은다). 갱신 시점은 **청크 경계**다
— 요청 하나에 청크 하나의 사실이 실린다.

| 안쪽 키 | 형 | 뜻 |
|---|---|---|
| `available` | bool | 사실이 실렸나. `False` 면 `reason` 만 있다 (에피소드 첫 청크 등) |
| `reason` | str | `available=False` 의 이유 |
| `seq` | int | **어느 계획에 대한 사실인가** — 그 청크를 받은 요청의 `ag3s/seq` |
| `t_step_start` | int | 그 청크의 첫 제어 스텝 (로컬 `t_step`) |
| `n_steps` | int | 그 계획이 걸려 있는 동안 돈 제어 스텝 수 (보통 8) |
| `n_exec` | int | 그중 **계획 청크의 행**을 적용한 스텝 수 |
| `n_hold` | int | `n_steps - n_exec` — 계획 행 대신 HOLD 목표를 적용한 스텝 수 (T23: `q_hold` 고정, `legacy` 면 현재 자세) |
| `executed` | list[bool] `[n_steps]` | 스텝별 실행 여부. 부분 실행의 **모양**까지 남긴다 |
| `executed_chunk` | str | 로컬이 고른 청크: `refined` · `reference` · `none`(= HOLD) |
| `ipc` | str | 그 왕복의 결과: `ok` · `unsafe` · `timeout` · `stale` · `error` |
| `hold_kind` | str \| None | HOLD 사유 종류 (`HOLD_KINDS`). 실행했으면 `None` |
| `hold_reason` | str \| None | HOLD 사유 문장 (`SafeRemoteClient.last_reason`) |
| `planned_gripper` | float32 `[n_steps, 2]` | 그 스텝에 **계획 청크가 말한** (왼, 오른) gripper, 정규화 |
| `applied_gripper` | float32 `[n_steps, 2]` | **실제 `d.ctrl` 에 들어간** (왼, 오른) gripper, `ctrl / RBY1_GRIPPER_OPEN` 로 정규화 (1 = 열림). HOLD 면 HOLD 목표 — T23 기본(`fixed`)은 **마지막으로 명령한** gripper, `legacy` 는 `rby1_state()` 의 측정값 |
| `applied_arm` | float32 `[n_steps, 2N]` | 실제 `d.ctrl` 의 팔 목표 `[왼 N, 오른 N]` — drift 평가용 (지침 §8.2) |
| `measured_gripper` | float32 `[2]` | **이번 요청의 촬영 시점** `qpos` 로 잰 (왼, 오른) 개도. `build_obs` 와 같은 규약 (`|q| / |RBY1_GRIPPER_OPEN|`, 1 = 열림) |

**키가 없음과 `available=False` 는 다르다.** 키가 없으면 이 계약을 모르는 **옛 클라이언트**이고,
`available=False` 는 사실을 낼 수 없었던 새 클라이언트다 (`unpack_exec_feedback` 이 둘을 `reason`
으로 가른다). `ag3s` 블록 · `shadow` 와 같은 규약이다 — 있음/없음 자체가 신호다.

**크기.** 8 스텝이면 배열 넷이 합쳐 1 KB 미만이다 (`[8, 2] + [8, 2] + [8, 14] + [2]` float32 =
584 B). T9 가 감수한 응답 17 KB 증가에 비하면 무시할 만해서 스텝별 `applied_arm` 까지 와이어에 싣는다.

### `ag3s/stamp/<cam>` — 시뮬레이션 관측 하나에는 **촬영 순간이 하나** (T30c, 2026-09-29)

T28 E3b 에서 `uncertified` HOLD 125 청크가 `camera_transform_stale` 이었다. 서버의 신선도 검사
(`ag3s/runtime/multiview.py:check_freshness`)는 한 관측 안에서 가장 최신 카메라보다
`timing.max_transform_age_sec` (100 ms) 넘게 뒤진 이미지를 "외부 파라미터가 오래됐다" 로 본다. 그런데
옛 `_pack` 은 카메라마다 **렌더가 끝난 순간**을 찍었고, headless osmesa 는 세 카메라를 **차례로**
렌더하므로 head 가 wrist 보다 102–142 ms (중앙값 121) 앞섰다. 그동안 시뮬레이션은 **정지**해
있다 — 세 이미지 · 자세 · 외부 파라미터는 같은 순간의 씬이다. 늦게 찍힌 것은 씬이 아니라 렌더다.

그래서 시뮬레이션 클라이언트는 **시뮬레이션이 그 관측을 위해 멈춘 순간 하나**를 모든 카메라의
`stamp/<cam>` 에 싣는다 (`stamp_mode = "sim_frozen"`, `SafeRemoteClient.infer(capture_time=)`).
렌더 벽시계는 버리지 않고 `render_stamp/<cam>` 에 따로 싣는다 — 옛 `stamp/<cam>` 과 같은 값이다.

**서버의 검사는 바꾸지 않는다.** 실기에서는 카메라가 정말로 서로 다른 순간을 찍고, 그때는 그 검사가
맞다. `capture_time` 을 주지 않은 호출자는 예전처럼 카메라별 렌더 끝 순간을 싣는다
(`stamp_mode = "render_end"`) — 한 순간을 **주장**하는 것은 시뮬레이션이 멈춰 있음을 아는 호출자뿐이다.
틀린 한 순간은 진짜 지연을 숨기고(위험한 쪽), 틀린 카메라별 순간은 HOLD 를 낳는다(안전한 쪽).

### `ag3s/policy_seed` — `(episode, seed)` 가 run 을 정한다 (T39, 2026-10-02)

π0.5 정책의 RNG 는 서버 시작 때 `jax.random.key(0)` 이고 요청마다 split 될 뿐 **다시 놓이지 않는다**
(`openpi/policies/policy.py:65, 75`). 그래서 run 의 noise 는 그 run 이 서버 요청 순서의 몇 번째인가가
정했다 — 같은 서버를 새로 띄우면 조건이 달라도 파지 전까지 같은 궤적이 나온 이유다 (T30 · T37).
O1 은 조건 간에 **같은 noise 로 짝짓기** 위해 seed 를 명시한다.

- **선택 키다.** 없으면 요청 바이트가 T39 전과 같고 서버 동작도 같다 (RNG 는 이어서 split).
- 클라이언트는 에피소드의 **첫 요청에만** 싣는다 (`pi05_infer.py --policy-seed`). 서버는 그 요청의
  `policy.infer` 직전에 정책 RNG 를 `jax.random.key(seed)` 로 놓고, 이후 요청은 거기서 이어 split 한다.
  그러면 요청 n 의 key 가 `(seed, n)` 만의 함수다.
- **소비하는 것은 서버다** — 정책 관측에는 들어가지 않는다 (`ag3s/` 접두는 `strip_request` 가 벗기고,
  `--no-safe` 서버는 맨 키 `policy_seed` 를 `trajopt.policy_seed.SeededPolicy` 가 벗긴다).
- 응답에 **같은 이름의 키로 되돌려준다** (`policy_seed`, 적용한 요청에만). 클라이언트는 보낸 seed 가
  돌아오지 않으면 죽는다 — 이 키를 모르는 옛 서버는 seed 를 조용히 무시하고, 그러면 짝지은 비교가
  짝이 아닌 채로 기록된다.
- 범위: `0 ≤ seed < 2**32` 정수 (bool 거절). 정책의 `_rng` 를 못 찾으면 서버가 크게 실패한다.

## 응답

`actions` 외에 안전 판정을 싣는다. 로컬은 이 판정이 유효할 때만 실행한다.

| 키 | 내용 |
|---|---|
| `actions` | `[H, ACTION_WIDTH]` (16D 기본). 안전하지 않아도 실린다 — 왜 멈췄는지 보려면 무엇이 제안됐는지 알아야 한다 |
| `actions_reference` | 정책의 **원본** 청크. **2026-09-26 부터 closed loop 에서도 실린다.** 아래 |
| `shadow` | **모드**. `actions_reference` 가 아니라 **이 키**가 shadow 서버의 신호다. 아래 |
| `seq` | 요청의 일련번호를 그대로 돌려준다. 오래된 응답을 버리는 근거 |
| `policy_seed` | **선택** (T39). 그 요청에 정책 RNG seed 를 적용했으면 그 값. 위 `ag3s/policy_seed` 절 |
| `timing_ms` | 단계별 시간 (서버 시계) |
| `ag3s_status` · `geometry_certified` · `trajopt_status` · `max_violation_m` · `safe` · `notes` | 안전 판정 |
| `field` | **거리장의 출처** — `sequence` · `backend` · `observed_at` · `state` · 계층. 아래 |
| `ag3s` | **선택 키.** `ag3s_status` 가 `ok` 가 아닐 때만 실린다 — **왜** 인증이 안 됐나. 아래 |
| `max_violation_pair` | **선택 키.** `max_violation_m` 을 만든 **행의 신원**. 아래 |
| `verdict_reasons` | **선택 키** (T23). `safe` 의 **사유 목록** `[{kind, action, detail, evidence}]`. 아래 |

### `verdict_reasons` — `safe` 를 bool 하나로 축약하지 않는다 (T23, 지침 §8.3 · §9)

T16 에서 `--no-collision` 인데도 34 청크 중 25 개가 HOLD 였다. 이유는 하나로 뭉쳐 있었다 —
`TrajOptStatus.safe` 가 아니면 HOLD. 그 안에는 **파지에 필요한 접촉**(손가락이 사과를 파고든다),
**가려진 target**(`no_seed` 인데 manipulated 는 살아 있다), 진짜 충돌이 같은 `violated` 로 섞여
있었다. 그래서 서버가 사유를 나눠 싣고, 로컬 게이트가 사유별로 처리한다.

| kind | 뜻 | 기본 처리 (`GATE_DEFAULT`) |
|---|---|---|
| `allowed_contact` | 위반 행이 전부 **권한 있는 link ↔ manipulated 물체** 다 (T20 manipulated, `contact.contact_links`) | 실행 |
| `budget_only` | 예산/반복 상한으로 끝났지만 최종 검사 통과 | 실행 |
| `occluded_target` | 미인증의 이유가 target 없음(`no_target`) 하나뿐이고 manipulated 가 `occluded` 다 (T20) | 실행 |
| `uncertified_waived` | 기하 미인증이지만 서버가 `require_certified_geometry=False` 로 떠 있다 | 실행 |
| `subtask_no_target` | target 이 없는 이유가 **subtask gate** 다 (`grounding_status=subtask_gated` · `validity=valid` · carve 중인 대상 없음, `ag3s.types.subtask_no_target`). 이 기하는 **인증된 것으로 본다** (`geometry_certified=True`) — 이 사유는 "target 이 없는데 왜 실행했나" 를 남기는 기록이다 (SUBTASK-e) | 실행 |
| `collision` | 권한 없는 link, 또는 manipulated 가 아닌 물체와의 관통 (> tolerance) | HOLD |
| `uncertified` | `geometry_certified=False` (위 `occluded_target` 이 아닌 것 전부. `subtask_no_target` 은 미인증이 아니다) | HOLD |
| `unverified` | 최적화기가 검사하지 못했다 (`solver_failed` · `unconstrained`) | HOLD |
| `comms` | **클라이언트가 만든다** — `timeout` · `stale` · `dimension` · `error` (`evidence.ipc`) | HOLD |
| `no_perception` | 서버가 `--no-perception` 으로 떠 있다 (T27, `to_only_policy`) — AG3S·ESDF 가 없고 **아무것도 검사하지 않았다** | HOLD |

**모르는 kind 는 HOLD 다** (fail closed).

**`no_perception` 은 `safe=True` 와 함께 온다 — 아래 `safe` 규칙의 유일한 예외다** (T27). 서버는 검사를
하지 않았으므로 막을 근거도 없고 (`safe=True`), 로컬 사유 표는 "검사되지 않은 청크" 를 실행하지 않는다
(`unverified` 와 같은 처리). 그래서 이 서버의 청크는 **로컬이 `--safe-gate off` 로 명시할 때만** 실행된다 —
shadow 의 짝 검사와 같은 규율이다: 양쪽이 같은 실험이라고 말해야 로봇이 움직인다. `action` 은 서버가 이 표로 붙인 값이고 읽기 편의용이다 —
로컬 게이트는 자기 표(`gate_decision`)로 다시 정한다.

**`safe` 는 그대로 실린다: `safe = (HOLD 처리인 사유가 없다)`.** 옛 클라이언트는 `safe` 만 읽으므로
그대로 동작한다. 새 클라이언트는 `safe` **와** 자기 게이트가 둘 다 실행이라고 할 때만 실행한다 —
로컬은 서버보다 **엄격해질 수만** 있다 (서버가 `legacy` 판정으로 떠 있으면 그것이 이긴다).
`trajopt_status` 는 최적화기의 상태 그대로다: `violated` 인데 `safe=True` 인 청크(= `allowed_contact`)
가 있다.

**키가 없으면** 옛 서버이거나, 서버가 씬을 못 얻어 최적화기가 돌지 않은 청크다(그 응답의 모양은
T23 전과 같다). `unpack_verdict_reasons` 가 그때 `safe` · `geometry_certified` 로 사유를 **유도**하고
`source="derived"` 로 표시한다 — 유도한 사유는 언제나 옛 `safe` 와 같은 결정을 낸다.

### `max_violation_pair` — `violated` 가 **어느 제약**인가

`T5f` 가 여기서 막혔다: 판정에는 `max_violation_m` 한 숫자만 있고, 그 숫자를 만든 행이 어느
link 대 어느 obstacle 인지는 **최적화기 안에만** 있었다. 숫자만 보고는 고칠 수 없다 — 팔꿈치가
탁자를 스친 것과 손끝이 사과를 파고든 것은 전혀 다른 일이고, 대응도 다르다.

`CollisionLinearizer.worst_row` 가 그 신원을 만든다: `block`(candidate · plane · esdf) ·
`step` · `link` · `slot` · `candidate_id` · `obstacle` · `point_m`. 값은 `clearance_m` 이고
`max_violation_m` 과 부호만 반대다 (`violation = max(0, -clearance)`).

**활성 제약이 하나도 없던 프레임에는 키가 실리지 않는다.** 없는 신원을 `null` 로 싣는 것보다
키를 빼는 것이 낫다 — 그래야 *"신원을 낼 수 있는 서버"* 와 옛 서버가 구별된다.
`actions_reference` · `ag3s` 와 같은 규약이다.

### `shadow` — **모드는 이제 명시된다** (2026-09-26, T9)

`actions_reference` 의 **있음/없음**이 shadow 서버의 신호였다. 그 신호는 값이 하나뿐인 자리에
두 가지 뜻(무엇을 실행할지 고르는 데 필요한 청크 · 서버가 어느 모드인지)을 실었고, 둘째 뜻
때문에 **closed loop 에서는 원본 청크를 실을 수 없었다.**

그 결핍에 다섯 번 막혔다. 마지막이 결정적이다 — 충돌 제약이 하나도 활성이 아닌 판에서도 TO 가
청크를 고친다 (실행되는 8 step 안에서 중앙값 2.6°, 최대 8.8°, 손이 다가갈수록 커진다). 그러면
남은 변형은 전부 **목적함수**가 만든 것인데, closed loop 의 원본 청크가 기록에 없으면 그 크기를
잴 수 없다. shadow 는 성공하고(245.2 mm 들어 올린다) closed loop 은 헛잡는 차이가 거기 있다.

그래서 두 뜻을 갈랐다.

| 키 | 뜻 | 언제 |
|---|---|---|
| `actions_reference` | 정책 원본 청크 — **데이터** | 언제나 (closed loop 포함) |
| `shadow` | 서버가 reference 를 **실행하라고 내보내는 모드인가** — **모드** | 언제나 (`True`/`False`) |

**`shadow` 는 `False` 여도 실린다.** 이 키만이 짝 검사의 근거이므로, 없음을 "closed loop" 으로
읽으면 **옛 서버**(키를 모르는 서버)와 구별할 수 없다. 옛 서버에는 `actions_reference` 의 있음/
없음으로 물러나 추론한다 (`client._check_shadow_pairing`).

### `actions_reference` — 서버가 계산에 쓴 **입력** 청크

**shadow 실행(T5)은 전부 계산하되 수정된 청크를 로봇에 보내지 않는다.** 수정이 여유거리를 나쁘게
만드는지를 로봇을 움직이기 전에 보려는 것이다. 그러려면 로컬이 **정책의 원본 청크**를 알아야 하는데,
지금까지 응답에는 `actions`(= refined) 하나뿐이라 로컬이 그것을 볼 길이 없었다.

**`actions` 는 shadow 에서도 여전히 refined 다.** 서버는 자기가 계산한 것을 그대로 말하고
(거짓말하지 않는다), 무엇을 실행할지는 로컬이 고른다 — `SafetyVerdict` 가 판정이지 명령이
아닌 것과 같은 계약이다.

**키는 이제 언제나 실린다** (T9). *"TO 가 청크를 얼마나 바꿨나"* 는 closed loop 에서 가장
알아야 하는 값이고, 그것을 재려면 `actions`(refined) 옆에 원본이 있어야 한다.

**대가는 응답 크기다.** 청크 하나가 `[H, 16] float32` 이므로 planning 기록 한 줄이 실측
19,837 B → shadow 수준(36,704 B)으로 커진다. 그 값을 아는 채로 고른 것이다 — 15 Hz 에서
한 프레임에 17 KB 가 더 흐르는 것보다, 같은 결핍에 여섯 번째로 막히는 것이 비싸다.

**무엇을 실행할지는 여전히 `shadow` 키가 정한다.** 로컬이 그 짝을 검사하고
(`client.SafeRemoteClient`), 한쪽만 켜져 있으면 즉시 실패한다 — 조용히 refined 를 실행하면
shadow 가 아닌데 shadow 라고 기록된다. **그 거절은 T9 에서 한 줄도 느슨해지지 않았다**: 근거가
"reference 가 왔나" 에서 "서버가 shadow 라고 말했나" 로 옮겨간 것뿐이다.

### `ag3s` — `degraded` 의 **사유**

`ag3s_status` 는 *"인증했나"* 만 말하고 *"왜 못 했나"* 는 말하지 않았다. 2026-09-25 첫 live
smoke 에서 그것이 바로 막혔다: 2 청크 중 1 개가 `max_violation_m = 0.0` (궤적은 모든 제약을
통과)인데 `ag3s_status = degraded` · `geometry_certified = False` 로 HOLD 됐고, **왜 degraded
인지가 와이어에도 서버 로그에도 없었다.**

사유는 서버 안에 이미 있었다 — `CollisionConstraintSet.notes` 다. 나가는 길이 없었을 뿐이다
(응답의 `notes` 는 **최적화기**쪽 `TrajOptResult.notes` 이고 지각 쪽이 아니다).

| 안쪽 키 | 내용 |
|---|---|
| `status` | `CollisionConstraintSet.status` (= `ag3s_status` 와 같은 값. 블록만 보고도 짝을 확인할 수 있게 둔다) |
| `validity` | `ConstraintValidity` — `status` 가 그것에서 파생된다 |
| `grounding_status` | target 을 왜 못 잡았나. **T5b 까지 로컬은 이 값을 `unavailable-on-client` 로 적고 있었다** |
| `reasons` | `[{"code", "detail"}]` — **기계가 읽는 사유.** 코드는 `ag3s/runtime/degradation.py:CODES` 에 등록된 것뿐이고, 코드 하나가 소스의 한 분기에 대응한다 |
| `notes` | 지각 쪽 노트 전부 (코드 없는 산문 노트까지). 사람이 읽는 쪽 |

**`status` 가 `ok` 일 때는 키가 아예 없다.** 회귀 기준선과 T0 기록이 정상 프레임의 응답에
달려 있으므로 그쪽은 한 바이트도 건드리지 않는다. `actions_reference` 와 같은 규약이고, 이유도
같다 — 키의 있음/없음 자체가 신호다.

**`reasons` 가 비는 경우는 없다.** `degradation.ensure_reason` 이 마지막 관문에서 `degraded`
인데 코드가 하나도 없으면 `degraded_without_reason` 을 달아 보낸다. 그 코드가 보이면 씬의 성질이
아니라 **AG3S 의 배선 결함**이다.

### `field` — 거리장이 언제 무엇으로 만들어졌는가

`actions` 만 보면 이 청크의 기하가 방금 관측된 것인지 낡은 것인지 알 방법이 없다. 2026-09-18
의 통합에서 `attach()` 뒤 **14 프레임 동안 지각이 한 번도 안 돌았는데 상태는 `ok`** 로
나갔고, 그것이 안 보인 이유가 이 블록이 없었기 때문이다.

`state` 는 `new` | `carried` | `stale` | `unavailable` 이다. **서버는 `new` 나 `unavailable`
만 찍는다** — planning frame 마다 AG3S 를 돌리므로. `carried` 와 `stale` 은 **control frame**
에서 생기고 (청크 하나가 8 스텝 = 533 ms 를 덮는다) 클라이언트가
`FieldProvenance.applied_by_client()` 로 채운다.

**age 의 기준은 `observed_at`(클라이언트 시계의 촬영 시각)이다.** 서버의 `built_at` 은
monotonic 원점이 달라 클라이언트가 자기 시계와 견줄 수 없으므로 단계 시간에만 쓴다.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "PREFIX", "DEFAULT_CAMERAS", "GRIPPER_COLUMNS", "gripper_columns",
    "ARM_JOINT_DIM", "ACTION_WIDTH", "ACTIONS_REFERENCE", "SHADOW", "TO_BLOCK",
    "pack_request", "strip_request", "unpack_camera_observations",
    "pack_response", "unpack_field", "unpack_actions_reference", "unpack_ag3s",
    "unpack_violation_pair", "unpack_shadow", "unpack_to",
    "SafetyVerdict", "AG3S_BLOCK", "VIOLATION_PAIR",
    "EXEC_FEEDBACK", "HOLD_KINDS", "EXECUTED_CHUNKS",
    "make_exec_feedback", "no_exec_feedback", "unpack_exec_feedback",
    "exec_feedback_jsonable",
    "VERDICT_REASONS", "REASON_KINDS", "GATE_DEFAULT", "GATE_ACTIONS", "COMMS_KINDS",
    "make_reason", "gate_decision", "unpack_verdict_reasons", "derive_verdict_reasons",
    "POLICY_SEED", "POLICY_SEED_LIMIT", "check_policy_seed", "unpack_policy_seed",
    "unpack_policy_seed_echo",
]

PREFIX = "ag3s/"

#: 응답 키 — 정책의 원본 청크. **T9 부터 closed loop 에서도 실린다** (위 머리말).
#: 이름을 상수로 두는 이유는 서버·클라이언트·테스트 세 곳이 같은 문자열을 써야 하고, 오타가
#: 나면 로컬이 "reference 가 안 왔다" 로 읽어 즉시 실패하기 때문이다 — 조용히는 안 지나가지만
#: 원인을 찾는 데 시간이 든다.
ACTIONS_REFERENCE = "actions_reference"

#: 응답 키 — **모드.** `True` 면 서버는 로컬이 reference 를 실행하기를 기대한다 (shadow).
#: `False` 여도 실린다: 이 키가 짝 검사의 유일한 근거이므로, 없음은 "closed loop" 이 아니라
#: **"이 키를 모르는 옛 서버"** 를 뜻해야 한다. 그 구분이 없으면 옛 서버에 shadow 로컬을
#: 붙였을 때 조용히 refined 가 실행된다.
SHADOW = "shadow"

#: 응답의 **선택 키** — `ag3s_status` 가 `ok` 가 아닐 때의 사유 블록 (위 머리말).
#: `ag3s/` 접두(요청 쪽)와 글자가 겹치지만 충돌하지 않는다: `strip_request` 는 **요청**만
#: 가르고 `"ag3s/"`(슬래시 포함)로 시작하는 키만 본다. 응답은 애초에 stripping 을 안 지난다.
AG3S_BLOCK = "ag3s"

#: 응답 키 — **최적화기가 실제로 몇 번 돌았나** (T15). `ag3s` 블록과 같은 규약이다.
#:
#: 예전에는 이 값이 `notes` 의 문장 (`"time budget 50 ms reached after 1 iteration(s)"`) 뿐이라
#: **셀 수 없었다.** *"편차가 0 이 아니면 몇 번 돌았나"* 가 언제나 다음 질문인데, 그때마다 기록이
#: 없어 실행을 다시 해야 했다. 구조화된 값으로 싣는다.
TO_BLOCK = "to"

#: 응답의 **선택 키** — `max_violation_m` 을 만든 행의 신원 (`SafetyVerdict.max_violation_pair`).
#: `T5f` 가 *"`violated` 가 어느 제약인가"* 에서 막힌 것을 여는 키다. 활성 제약이 하나도 없던
#: 프레임에는 실리지 않는다 — 없는 신원을 `null` 로 싣는 것과 키를 빼는 것 중, 뒤쪽이
#: *"이 서버는 신원을 낼 수 있다"* 를 잃지 않는다 (옛 서버와 구별된다).
VIOLATION_PAIR = "max_violation_pair"

#: 요청 키 (`ag3s/` 접두 뒤) — **직전 청크의 실행 사실** (T18). 위 머리말의 표가 안쪽 키의 정의다.
EXEC_FEEDBACK = "exec_feedback"

#: 요청 키 (`ag3s/` 접두 뒤) — `stamp/<cam>` 의 뜻 (T30c). 머리말의 `ag3s/stamp/<cam>` 절.
STAMP_MODE = "stamp_mode"
#: `sim_frozen`: 시뮬레이션이 관측을 위해 멈춘 한 순간 (모든 카메라 같은 값).
#: `render_end`: 카메라마다 렌더가 끝난 순간 (T30c 전 동작, 호출자가 순간을 주지 않을 때).
STAMP_MODES = ("sim_frozen", "render_end")

#: 요청 키 (`ag3s/` 접두 뒤, `--no-safe` 요청은 맨 키) 이자 응답 키 — **정책 RNG seed** (T39).
#: 머리말의 `ag3s/policy_seed` 절. 응답의 같은 이름 키는 서버가 그 seed 를 **적용했다** 는 회신이다.
POLICY_SEED = "policy_seed"
#: seed 의 상한 (배타). `jax.random.key` 는 64 비트도 받지만 x64 가 꺼진 JAX 에서 상위 비트가 어떻게
#: 접히는지에 기대지 않는다 — O1 의 seed 는 `10·ep + rep` (≈ 2·10⁴) 이다.
POLICY_SEED_LIMIT = 2 ** 32

#: HOLD 사유의 종류. `SafeRemoteClient.last_ipc` 의 값 중 `ok` 를 뺀 것과 같다 — 사유를 새로
#: 지어내지 않고 로컬이 이미 쓰는 이름을 그대로 싣는다.
HOLD_KINDS = ("unsafe", "timeout", "stale", "error")

#: 로컬이 고른 청크 (`SafeRemoteClient.last_executed_chunk`). `none` 이 HOLD 다.
EXECUTED_CHUNKS = ("refined", "reference", "none")

#: 응답의 **선택 키** — `safe` 의 사유 목록 (T23). 머리말의 `verdict_reasons` 절.
VERDICT_REASONS = "verdict_reasons"

#: 사유의 종류. 머리말 표의 순서 그대로 — 앞 다섯이 실행, 뒤 다섯이 HOLD 다 (`GATE_DEFAULT`).
#: `subtask_no_target` (SUBTASK-e) 는 끝이 아니라 실행 쪽 자리에 넣었다 — 이 튜플의 순서를 읽는
#: 곳은 없고 (`GATE_DEFAULT` 가 처리를 정한다), 표와 같은 순서가 읽기 쉽다.
REASON_KINDS = ("allowed_contact", "budget_only", "occluded_target", "uncertified_waived",
                "subtask_no_target",
                "collision", "uncertified", "unverified", "comms", "no_perception")

#: 게이트의 처리 두 가지. **정지(abort)는 여기 없다** — 한 청크의 사유가 아니라 연속 HOLD 수가
#: 정하는 것이고 (`--safe-max-hold-chunks`), 그것은 제어 루프의 일이다.
GATE_ACTIONS = ("execute", "hold")

#: 사유 → 기본 처리 (지침 §8.4). **여기 없는 kind 는 HOLD 다** (`gate_decision`).
GATE_DEFAULT: dict[str, str] = {
    "allowed_contact": "execute",
    "budget_only": "execute",
    "occluded_target": "execute",
    "uncertified_waived": "execute",
    # SUBTASK-e — gate 가 일부러 비운 target. 기하는 인증됐다 (carve 없는 가장 보수적인 field).
    # 이 kind 를 모르는 옛 클라이언트는 HOLD 한다 (fail closed) — 서버와 같은 wire 를 써야 실행된다.
    "subtask_no_target": "execute",
    "collision": "hold",
    "uncertified": "hold",
    "unverified": "hold",
    "comms": "hold",
    # T27 — `--no-perception` 서버. 검사되지 않은 청크다. 실행하려면 로컬 `--safe-gate off`.
    "no_perception": "hold",
}

#: `comms` 사유의 하위 종류 (`evidence.ipc`). `dimension` 은 `last_ipc` 로는 `error` 로 남는다 —
#: T18 의 `HOLD_KINDS` 를 바꾸지 않으려는 것이고, 구별은 이 사유가 한다.
COMMS_KINDS = ("timeout", "stale", "dimension", "error")

#: 머리 하나 + 손목 둘. `CameraID` 에 HEAD 가 하나뿐이라 `zed_right` 는 `zed_left` 와 겹친다.
DEFAULT_CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")

#: 팔당 관절 수. 청크 레이아웃 전체가 이 값에서 나온다 — openpi 가 delta mask 를
#: ``make_bool_mask(N, -1, N, -1)`` 로 만들므로 (`training/config.py:278-281`) 레이아웃이
#: ``[왼팔 N, 왼 그리퍼, 오른팔 N, 오른 그리퍼]`` 이고 총 차원이 ``2*(N+1)`` 이다.
#:
#: **7 이 기본이다** — 2026-09-24 에 16D(`pi05_rby1_randomized_pick_place_16d_lora`)로 전환하고
#: 14D 를 버리기로 판정했다. 16D 는 `arm_6` 손목을 정책이 지령한다 (14D 에서는 고정이었다).
ARM_JOINT_DIM = 7

#: 액션의 총 차원. `2*(ARM_JOINT_DIM+1)`.
ACTION_WIDTH = 2 * (ARM_JOINT_DIM + 1)


def gripper_columns(arm_joint_dim: int = ARM_JOINT_DIM) -> tuple[int, int]:
    """그리퍼가 앉은 두 열. TO 는 이 열을 **절대 건드리지 않는다** — 결정 변수가 아니고,
    손을 여닫는 것은 충돌 회피가 판단할 일이 아니다.

    상수로 박지 않는 이유: 14D 는 `(6, 13)`, 16D 는 `(7, 15)` 다. 박아 두면 차원을 바꿀 때
    **엉뚱한 열을 그리퍼로 보호하고** 그것이 조용히 지나간다 — 손목 관절이 보호되고 그리퍼가
    최적화되는, 증상이 안 보이는 종류의 실패다.
    """
    n = int(arm_joint_dim)
    return (n, 2 * n + 1)


#: 기본 차원의 그리퍼 열. 편의용이고, 차원이 다른 호출자는 `gripper_columns(N)` 을 쓴다.
GRIPPER_COLUMNS = gripper_columns()

#: depth 를 uint16 밀리미터로 보내는 배율. 실제 depth 카메라가 주는 형식이고
#: `PointCloudConfig.depth_scale = 0.001` 이 그것을 되돌린다. float64 로 보내면 payload 가
#: 네 배가 되면서 모든 실제 센서가 갖는 1 mm 양자화를 조용히 숨긴다.
DEPTH_SCALE_MM = 1000.0


def pack_request(obs: dict[str, Any], *, cameras: Sequence[str],
                 depth: dict[str, np.ndarray], intrinsics: dict[str, np.ndarray],
                 extrinsics: dict[str, np.ndarray], robot_state: dict[str, np.ndarray],
                 stamps: dict[str, float], phase: str,
                 active_manipulators: Sequence[str] = (),
                 reset: bool = False, seq: int = 0,
                 exec_feedback: Optional[dict[str, Any]] = None,
                 render_stamps: Optional[dict[str, float]] = None,
                 stamp_mode: Optional[str] = None,
                 policy_seed: Optional[int] = None) -> dict[str, Any]:
    """정책 관측에 AG3S 가 필요한 것을 더한다. `obs` 는 제자리에서 바뀌지 않는다.

    `exec_feedback` 은 `make_exec_feedback` / `no_exec_feedback` 의 결과다. **`None` 이면 키를
    싣지 않는다** — 서버는 그것을 "이 계약을 모르는 옛 클라이언트" 로 읽는다. 새 클라이언트는
    낼 사실이 없어도 `no_exec_feedback(reason)` 을 실어 그 둘을 구별되게 한다.

    `render_stamps` · `stamp_mode` (T30c) 도 **주었을 때만** 키가 생긴다 — 옛 호출자의 요청은 키
    집합이 그대로다. `stamps` 가 판정에 쓰이는 촬영 순간이고, `render_stamps` 는 진단용이다.

    `policy_seed` (T39) 도 **주었을 때만** `ag3s/policy_seed` 가 생긴다 — `None` 이면 요청이 T39 전과
    바이트까지 같다. 값은 `check_policy_seed` 로 검사한다 (보내기 전에 죽는다).
    """
    if stamp_mode is not None and stamp_mode not in STAMP_MODES:
        raise ValueError(f"stamp_mode must be one of {STAMP_MODES}, got {stamp_mode!r}")
    out = dict(obs)
    if exec_feedback is not None:
        out[PREFIX + EXEC_FEEDBACK] = dict(exec_feedback)
    if stamp_mode is not None:
        out[PREFIX + STAMP_MODE] = str(stamp_mode)
    out[PREFIX + "cameras"] = list(cameras)
    out[PREFIX + "phase"] = str(phase)
    out[PREFIX + "active_manipulators"] = list(active_manipulators)
    out[PREFIX + "reset"] = bool(reset)
    out[PREFIX + "seq"] = int(seq)
    if policy_seed is not None:
        out[PREFIX + POLICY_SEED] = check_policy_seed(policy_seed)
    for cam in cameras:
        d = np.asarray(depth[cam], np.float64)
        out[f"{PREFIX}depth/{cam}"] = np.clip(
            np.rint(d * DEPTH_SCALE_MM), 0, 65535).astype(np.uint16)
        out[f"{PREFIX}K/{cam}"] = np.asarray(intrinsics[cam], np.float64)
        out[f"{PREFIX}T_base_cam/{cam}"] = np.asarray(extrinsics[cam], np.float64)
        out[f"{PREFIX}robot_state/{cam}"] = np.asarray(robot_state[cam], np.float64)
        out[f"{PREFIX}stamp/{cam}"] = float(stamps[cam])
        if render_stamps is not None:
            out[f"{PREFIX}render_stamp/{cam}"] = float(render_stamps[cam])
    return out


def check_policy_seed(value: Any) -> int:
    """정책 RNG seed 를 `int` 로. 정수가 아니거나 (`bool` 포함) `[0, POLICY_SEED_LIMIT)` 밖이면 예외.

    조용히 고치지 않는다 — `1.5` 를 `1` 로, `-1` 을 `2**32-1` 로 바꿔 쓰면 기록의 seed 와 실제 noise 가
    갈라지고, 짝지은 비교가 짝이 아닌 채로 남는다.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{POLICY_SEED} must be an int, got {type(value).__name__} {value!r}")
    seed = int(value)
    if not 0 <= seed < POLICY_SEED_LIMIT:
        raise ValueError(f"{POLICY_SEED} must be in [0, 2**32), got {seed}")
    return seed


def unpack_policy_seed(scene: dict[str, Any]) -> Optional[int]:
    """요청의 정책 RNG seed (T39). `scene` 은 `strip_request` 의 두 번째 값 (접두 없는 키).

    키가 없으면 `None` — seed 를 모르는 클라이언트이거나 에피소드의 첫 요청이 아니다. 값이 틀리면
    `check_policy_seed` 의 예외 그대로 (서버가 크게 실패한다).
    """
    if POLICY_SEED not in scene:
        return None
    return check_policy_seed(scene[POLICY_SEED])


def unpack_policy_seed_echo(response: dict[str, Any]) -> Optional[int]:
    """응답의 seed 회신 (T39). 서버가 그 요청에 seed 를 **적용했을 때만** 있다. 없으면 `None`."""
    value = response.get(POLICY_SEED)
    return None if value is None else int(value)


def unpack_stamps(scene: dict[str, Any]) -> dict[str, Any]:
    """요청의 촬영 시각 (T30c). `scene` 은 `strip_request` 의 두 번째 값 (접두 없는 키).

    `{"mode", "stamps", "render_stamps", "skew_sec", "render_spread_sec"}`. `mode` 가 `None` 이면
    `stamp_mode` 를 모르는 옛 클라이언트이고, `render_stamps` 가 빈 dict 면 그 진단을 안 실은
    것이다. skew/spread 는 카메라가 없으면 `None`. 서버의 판정은 이 함수를 쓰지 않는다 — 기록과
    검증용이다.
    """
    cams = [c for c in (scene.get("cameras", ()) or ()) if f"stamp/{c}" in scene]
    stamps = {c: float(scene[f"stamp/{c}"]) for c in cams}
    render = {c: float(scene[f"render_stamp/{c}"]) for c in cams
              if f"render_stamp/{c}" in scene}

    def _spread(values: dict[str, float]):
        return float(max(values.values()) - min(values.values())) if values else None

    mode = scene.get(STAMP_MODE)
    return {"mode": None if mode is None else str(mode), "stamps": stamps,
            "render_stamps": render, "skew_sec": _spread(stamps),
            "render_spread_sec": _spread(render)}


def strip_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(정책이 볼 관측, AG3S 가 볼 필드)` 로 가른다.

    정책 입력 변환은 자기가 모르는 키를 만나면 깨진다. 그래서 서버가 가장 먼저 하는 일이 이
    분리이고, 정책은 접두 키를 **한 번도 보지 않는다**.
    """
    policy_obs = {k: v for k, v in request.items() if not k.startswith(PREFIX)}
    scene = {k[len(PREFIX):]: v for k, v in request.items() if k.startswith(PREFIX)}
    return policy_obs, scene


def unpack_camera_observations(scene: dict[str, Any], *, robot_model,
                               attention: Optional[dict[str, np.ndarray]] = None) -> list:
    """`CameraObservation` 목록. 카메라가 하나도 없으면 빈 목록 — 예외가 아니다.

    카메라 누락은 정상적으로 일어난다(드라이버 한 프레임 드롭). 여기서 예외를 내면 지각 결함이
    정책을 통째로 죽인다. 빈 목록이면 AG3S 가 기하 없음을 **상태로** 보고하고, 그 상태를 보고
    실행할지 멈출지는 상위가 정한다 — AG3S 가 스스로 정하지 않는 것과 같은 이유다.
    """
    from benchmark.ag3s.types import CameraID, CameraObservation
    from benchmark.ag3s.experiments.sources.mujoco_source import CAMERA_MOUNTS

    attention = attention or {}
    out = []
    for cam in scene.get("cameras", ()) or ():
        depth_mm = scene.get(f"depth/{cam}")
        if depth_mm is None:
            continue
        mount_link, camera_id = CAMERA_MOUNTS.get(cam, (None, cam))
        out.append(CameraObservation(
            camera_id=CameraID.parse(camera_id),
            depth=np.asarray(depth_mm, np.float64) / DEPTH_SCALE_MM,
            camera_intrinsics=np.asarray(scene[f"K/{cam}"], np.float64),
            # 외부 파라미터가 왔으면 그대로 쓴다. 클라이언트가 촬영 시점에 이미 풀어서 보냈고,
            # 서버가 FK 로 다시 푸는 것보다 정확하다 — 서버는 그 순간의 자세를 모른다.
            T_base_cam=np.asarray(scene[f"T_base_cam/{cam}"], np.float64),
            robot_state=np.asarray(scene[f"robot_state/{cam}"], np.float64),
            timestamp=float(scene.get(f"stamp/{cam}", 0.0)),
            attention_map=attention.get(cam),
        ))
    return out


class SafetyVerdict:
    """서버가 내리는 판정. 로컬이 실행 여부를 결정할 때 읽는 전부.

    **판정이지 명령이 아니다.** 서버는 "이 청크를 안전하다고 인증할 수 있는가" 까지만 말하고,
    멈출지 실행할지는 로컬이 정한다. AG3S 가 기하 불확실성을 보고만 하고 결정하지 않는 것과
    같은 계약이다.
    """

    def __init__(self, *, ag3s_status: str, geometry_certified: bool, trajopt_status: str,
                 max_violation_m: float, safe: bool, notes: Sequence[str] = (),
                 max_violation_pair: Optional[dict[str, Any]] = None,
                 verdict_reasons: Optional[Sequence[dict[str, Any]]] = None):
        self.ag3s_status = ag3s_status
        self.geometry_certified = bool(geometry_certified)
        self.trajopt_status = trajopt_status
        self.max_violation_m = float(max_violation_m)
        self.safe = bool(safe)
        self.notes = list(notes)
        #: `max_violation_m` 을 만든 **행의 신원** (`CollisionLinearizer.worst_row`), 또는
        #: `None` — 활성 제약이 없었거나 서버가 이 키를 모르는 버전이다. 위 머리말 참조.
        self.max_violation_pair = (None if max_violation_pair is None
                                   else dict(max_violation_pair))
        #: `safe` 의 사유 목록 (T23), 또는 `None` — 최적화기가 돌지 않은 청크이거나 옛 서버.
        #: `[]` 은 "사유 없음 = 깨끗하다" 이고 `None` 과 다르다 (머리말 `verdict_reasons` 절).
        self.verdict_reasons = (None if verdict_reasons is None
                                else [dict(r) for r in verdict_reasons])

    def to_dict(self) -> dict[str, Any]:
        out = {
            "ag3s_status": self.ag3s_status,
            "geometry_certified": self.geometry_certified,
            "trajopt_status": self.trajopt_status,
            "max_violation_m": self.max_violation_m,
            "safe": self.safe,
            "notes": self.notes,
        }
        # **없으면 키를 싣지 않는다** — `ag3s` 블록·`actions_reference` 와 같은 규약이다.
        # 활성 제약이 하나도 없던 프레임의 응답이 T0 때와 한 바이트도 달라지지 않게 두려는 것이고,
        # 있음/없음이 *"서버가 이 신원을 낼 수 있는 버전인가"* 를 그대로 말해 준다.
        if self.max_violation_pair is not None:
            out[VIOLATION_PAIR] = dict(self.max_violation_pair)
        # T23 — 같은 규약. `None` 이면 키가 없다 (씬이 없어 풀지 못한 청크 · 옛 서버).
        if self.verdict_reasons is not None:
            out[VERDICT_REASONS] = [dict(r) for r in self.verdict_reasons]
        return out


def pack_response(actions: np.ndarray, verdict: SafetyVerdict, *, seq: int,
                  timing_ms: dict[str, float], field: Optional[Any] = None,
                  actions_reference: Optional[np.ndarray] = None,
                  shadow: bool = False,
                  ag3s: Optional[dict[str, Any]] = None,
                  to: Optional[dict[str, Any]] = None,
                  extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """응답. `actions` 는 `[H, ACTION_WIDTH]` 이고, 안전하지 않아도 실린다.

    안전하지 않을 때 청크를 **빼지 않는** 이유는, 로컬이 "왜 멈췄는지" 를 보려면 무엇이
    제안됐는지 알아야 하기 때문이다. 실행 여부는 `safe` 하나로 결정된다.

    `field` 는 `FieldProvenance` 다. **`None` 이면 `unavailable` 로 싣는다** — 키를 빼면
    읽는 쪽이 "필드가 없었다" 와 "서버가 옛 버전이라 안 보냈다" 를 구별할 수 없고, 전자는
    hold 해야 하고 후자는 배선 결함이라 대응이 다르다.

    `actions_reference` 는 **`None` 이면 키를 싣지 않는다** — 그러나 그것이 더 이상 "shadow 가
    아니다" 를 뜻하지 않는다 (T9). 모드는 `shadow` 키가 말하고, 이 키는 **데이터의 있음/없음**
    만 말한다: 원본 청크를 낼 수 없었던 프레임(정책이 청크를 못 낸 hold 등)이 그 경우다.

    `shadow` 는 **`False` 여도 싣는다.** 짝 검사의 근거가 이 키 하나이므로, 없음이 "closed
    loop" 이 아니라 "이 키를 모르는 옛 서버" 를 뜻해야 한다.

    `ag3s` 도 **`None` 이면 키를 싣지 않는다** — 정상 프레임(`ag3s_status == "ok"`)의 응답을
    T0 때와 같게 두기 위해서다. 호출부(`SafePolicy._ag3s_block`)가 `ok` 일 때 `None` 을 준다.

    Raises:
        ValueError: `actions_reference` 의 모양이 `actions` 와 다를 때. 로컬이 둘 중 하나를
            골라 실행하므로 모양이 어긋난 채 나가면 **한 칸씩 밀린 청크가 실행된다** — 형태는
            맞고 뜻은 틀린, 가장 위험한 실패다. 여기서 크게 죽는 편이 낫다.
    """
    from benchmark.ag3s.fields.provenance import FieldProvenance

    prov = field if field is not None else FieldProvenance.unavailable(
        "the server produced no collision field for this chunk")
    packed = np.asarray(actions, np.float32)
    out = {
        "actions": packed,
        "seq": int(seq),
        "timing_ms": {k: float(v) for k, v in timing_ms.items()},
        "field": prov.to_dict(),
        # **모드는 명시한다** (T9). `actions_reference` 의 있음/없음으로 추론하던 자리다.
        SHADOW: bool(shadow),
        **verdict.to_dict(),
    }
    if actions_reference is not None:
        reference = np.asarray(actions_reference, np.float32)
        if reference.shape != packed.shape:
            raise ValueError(
                f"actions_reference has shape {reference.shape} but actions has "
                f"{packed.shape}. The client picks one of the two to execute, so a mismatch "
                "would put a shifted chunk on the robot")
        out[ACTIONS_REFERENCE] = reference
    if ag3s:
        out[AG3S_BLOCK] = dict(ag3s)
    if to:
        out[TO_BLOCK] = dict(to)
    if extra:
        out.update(extra)
    return out


def unpack_actions_reference(response: dict[str, Any]) -> Optional[np.ndarray]:
    """응답의 정책 원본 청크, 또는 **`None`** — 그 프레임에 원본이 없었다는 뜻이다.

    **T9 부터 "없음" 은 모드가 아니라 데이터의 부재다.** 모드는 `unpack_shadow` 가 답한다.
    `unpack_field` 와 달리 없음을 상태 객체로 감싸지 않는 것은 그대로다 — 이 값을 어떻게 다룰지
    (실행할지, 기록만 할지)는 `SafeRemoteClient` 한 곳에서 정한다.
    """
    blob = response.get(ACTIONS_REFERENCE)
    return None if blob is None else np.asarray(blob)


def unpack_to(response: dict[str, Any]) -> dict[str, Any]:
    """응답의 `to` 블록, 없으면 **빈 딕셔너리** (이 블록을 모르는 옛 서버).

    `unpack_ag3s` 와 같은 규약이다 — 없음을 상태 객체로 감싸지 않는다. 판정을 바꾸는 값이 아니고
    **셀 수 있게 하는** 값이기 때문이다.
    """
    block = response.get(TO_BLOCK)
    return dict(block) if isinstance(block, dict) else {}


def unpack_shadow(response: dict[str, Any]) -> Optional[bool]:
    """서버가 shadow 모드인가. **`None` 은 "말하지 않았다"** = 이 키를 모르는 옛 서버.

    세 값을 가르는 것이 요점이다. `True`/`False` 는 서버의 선언이고 `None` 은 선언이 없는
    것이다. 없음을 `False` 로 접으면 옛 서버에 shadow 로컬을 붙였을 때 조용히 refined 가
    실행된다 — 그래서 호출부는 `None` 일 때 `actions_reference` 의 있음/없음으로 물러나
    추론한다 (`client._check_shadow_pairing`).
    """
    if SHADOW not in response:
        return None
    return bool(response[SHADOW])


def unpack_ag3s(response: dict[str, Any]) -> dict[str, Any]:
    """응답의 `ag3s` 블록, 없으면 **빈 딕셔너리**.

    키 없음은 두 가지를 뜻할 수 있다 — 기하가 인증됐다(`ok`) 거나, 서버가 이 블록을 모르는
    버전이다. `unpack_field` 처럼 그 둘을 구분해 주지 않는 이유는 **`ag3s_status` 가 이미 같은
    응답에 있기** 때문이다: `ag3s_status != "ok"` 인데 이 블록이 비어 있으면 그것이 곧 옛 서버다.
    읽는 쪽이 그 조합을 보고 판단할 수 있으므로 여기서 상태 객체를 만들지 않는다.
    """
    blob = response.get(AG3S_BLOCK)
    return dict(blob) if isinstance(blob, dict) else {}


def unpack_violation_pair(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    """응답의 `max_violation_pair`, 없으면 `None`.

    `unpack_field` 처럼 "없음" 을 상태 객체로 감싸지 않는 이유는 `unpack_ag3s` 와 같다 —
    같은 응답의 `max_violation_m` 이 이미 숫자를 말하고 있으므로, 이 키의 없음은 판정을
    바꾸지 않는다. 신원을 못 얻었다는 사실 자체가 기록에 `null` 로 남으면 그것으로 충분하다.
    """
    blob = response.get(VIOLATION_PAIR)
    return dict(blob) if isinstance(blob, dict) else None


def unpack_field(response: dict[str, Any]):
    """응답의 `field` 블록 -> `FieldProvenance`. 키가 없으면 그것도 상태로 답한다.

    키 없음은 **서버가 이 블록을 모르는 버전**이라는 뜻이고, 필드가 없는 것과 다르다.
    조용히 같게 취급하면 배선 결함이 안전 판정처럼 보인다.
    """
    from benchmark.ag3s.fields.provenance import FieldProvenance

    blob = response.get("field")
    if blob is None:
        return FieldProvenance.unavailable(
            "the response carried no `field` block — the server predates field provenance; "
            "staleness cannot be judged for this chunk")
    return FieldProvenance.from_dict(blob)


# --- 실행 피드백 (T18) -------------------------------------------------------------------------


def no_exec_feedback(reason: str) -> dict[str, Any]:
    """낼 사실이 없는 요청의 피드백. **키를 빼는 대신 이것을 싣는다** — 옛 클라이언트와 구별된다."""
    return {"available": False, "reason": str(reason)}


def make_exec_feedback(*, seq: int, t_step_start: int, executed: Sequence[bool],
                       executed_chunk: str, ipc: str, hold_kind: Optional[str],
                       hold_reason: Optional[str], planned_gripper, applied_gripper,
                       applied_arm, measured_gripper) -> dict[str, Any]:
    """직전 청크의 실행 사실을 와이어 형식으로. 표는 머리말의 `ag3s/exec_feedback` 절.

    **여기서 모양을 검사한다** — 스텝 수가 배열마다 다르면 "몇 번째 스텝에 무엇이 들어갔나" 가
    어긋난 채 서버 기록에 남고, 그것은 틀린 사실이다. 없는 것보다 나쁘다.

    Raises:
        ValueError: 배열의 스텝 수가 `executed` 와 다르거나, `executed_chunk` · `hold_kind` 가
            등록되지 않은 값일 때.
    """
    flags = [bool(v) for v in executed]
    n = len(flags)
    planned = _rows(planned_gripper, n, 2, "planned_gripper")
    applied = _rows(applied_gripper, n, 2, "applied_gripper")
    arm = np.asarray(applied_arm, np.float32)
    if arm.ndim != 2 or arm.shape[0] != n:
        raise ValueError(f"applied_arm must be [n_steps={n}, 2N], got {arm.shape}")
    measured = np.asarray(measured_gripper, np.float32).reshape(-1)
    if measured.shape != (2,):
        raise ValueError(f"measured_gripper must be (left, right), got {measured.shape}")
    if executed_chunk not in EXECUTED_CHUNKS:
        raise ValueError(f"executed_chunk {executed_chunk!r} is not one of {EXECUTED_CHUNKS}")
    if hold_kind is not None and hold_kind not in HOLD_KINDS:
        raise ValueError(f"hold_kind {hold_kind!r} is not one of {HOLD_KINDS} (or None)")
    n_exec = int(sum(flags))
    return {
        "available": True,
        "seq": int(seq),
        "t_step_start": int(t_step_start),
        "n_steps": n,
        "n_exec": n_exec,
        "n_hold": n - n_exec,
        "executed": flags,
        "executed_chunk": str(executed_chunk),
        "ipc": str(ipc),
        "hold_kind": None if hold_kind is None else str(hold_kind),
        "hold_reason": None if hold_reason is None else str(hold_reason),
        "planned_gripper": planned,
        "applied_gripper": applied,
        "applied_arm": arm,
        "measured_gripper": measured,
    }


def unpack_exec_feedback(scene: dict[str, Any]) -> dict[str, Any]:
    """`strip_request` 의 `scene` 에서 실행 피드백. **언제나 `available` 키가 있다.**

    세 경우를 `reason` 으로 가른다:

    * 키 없음 → 옛 클라이언트 (`"the request carried no exec_feedback ..."`)
    * `available=False` → 새 클라이언트가 낼 사실이 없었다 (그 `reason` 을 그대로)
    * 모양이 틀림 → **예외를 내지 않고** `available=False` + `malformed ...` — 피드백이 틀렸다고
      정책을 죽이면 안 된다. 틀린 사실을 쓰는 것보다 "없다" 로 물러나는 것이 낫다.

    배열은 `np.float64` 로 돌려준다. 전송(msgpack_numpy)을 지난 배열은 읽기 전용이다.
    """
    blob = scene.get(EXEC_FEEDBACK)
    if blob is None:
        return no_exec_feedback(
            "the request carried no exec_feedback — the client predates T18")
    if not isinstance(blob, dict):
        return no_exec_feedback(f"malformed exec_feedback: expected a dict, got "
                                f"{type(blob).__name__}")
    if not blob.get("available", False):
        return no_exec_feedback(str(blob.get("reason") or "the client reported no facts"))
    try:
        flags = [bool(v) for v in blob["executed"]]
        n = len(flags)
        out = {
            "available": True,
            "seq": int(blob["seq"]),
            "t_step_start": int(blob.get("t_step_start", -1)),
            "n_steps": n,
            "n_exec": int(sum(flags)),
            "n_hold": n - int(sum(flags)),
            "executed": flags,
            "executed_chunk": str(blob.get("executed_chunk", "none")),
            "ipc": str(blob.get("ipc", "")),
            "hold_kind": (None if blob.get("hold_kind") is None
                          else str(blob["hold_kind"])),
            "hold_reason": (None if blob.get("hold_reason") is None
                            else str(blob["hold_reason"])),
            "planned_gripper": np.asarray(
                _rows(blob["planned_gripper"], n, 2, "planned_gripper"), np.float64),
            "applied_gripper": np.asarray(
                _rows(blob["applied_gripper"], n, 2, "applied_gripper"), np.float64),
            "applied_arm": np.asarray(blob.get("applied_arm", np.zeros((n, 0))), np.float64),
            "measured_gripper": np.asarray(blob["measured_gripper"], np.float64).reshape(-1),
        }
        if out["measured_gripper"].shape != (2,):
            raise ValueError(f"measured_gripper has shape {out['measured_gripper'].shape}")
        if out["applied_arm"].ndim != 2 or out["applied_arm"].shape[0] != n:
            raise ValueError(f"applied_arm has shape {out['applied_arm'].shape}")
        # 로컬이 센 값과 `executed` 에서 다시 센 값이 다르면 둘 중 하나가 거짓이다.
        for key in ("n_exec", "n_hold", "n_steps"):
            if key in blob and int(blob[key]) != out[key]:
                raise ValueError(f"{key}={blob[key]} disagrees with `executed` ({out[key]})")
    except (KeyError, TypeError, ValueError) as exc:
        return no_exec_feedback(f"malformed exec_feedback: {type(exc).__name__}: {exc}")
    return out


def exec_feedback_jsonable(feedback: dict[str, Any]) -> dict[str, Any]:
    """기록(`summary_json`)용 — 배열을 리스트로. 값은 바꾸지 않는다."""
    out = {}
    for key, value in feedback.items():
        tolist = getattr(value, "tolist", None)
        out[key] = tolist() if callable(tolist) else value
    return out


def _rows(value, n: int, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, np.float32)
    if n == 0 and arr.size == 0:
        return np.zeros((0, width), np.float32)
    if arr.shape != (n, width):
        raise ValueError(f"{name} must be [n_steps={n}, {width}], got {arr.shape}")
    return arr


# --- 판정 사유 (T23) ---------------------------------------------------------------------------


def make_reason(kind: str, detail: str = "", **evidence: Any) -> dict[str, Any]:
    """사유 하나. `action` 은 `GATE_DEFAULT` 로 붙인다 (모르는 kind 는 `hold`).

    `evidence` 는 기계가 읽는 근거다 (행의 신원, 거리, tier …). 값은 JSON 으로 옮길 수 있어야
    한다 — 기록(`summary_json` · `frames.jsonl`)에 그대로 들어간다.

    Raises:
        ValueError: 등록되지 않은 kind. 서버·클라이언트가 같은 문자열을 써야 하므로 오타는 여기서
            죽는다 — 조용히 지나가면 게이트가 그 사유를 "모르는 것" 으로 HOLD 한다.
    """
    if kind not in REASON_KINDS:
        raise ValueError(f"reason kind {kind!r} is not one of {REASON_KINDS}")
    out: dict[str, Any] = {"kind": str(kind), "action": GATE_DEFAULT.get(kind, "hold"),
                           "detail": str(detail)}
    if evidence:
        out["evidence"] = dict(evidence)
    return out


def gate_decision(reasons: Sequence[dict[str, Any]],
                  table: Optional[dict[str, str]] = None) -> tuple[str, list[dict[str, Any]]]:
    """`(처리, HOLD 로 만든 사유들)`. 처리는 `execute` 또는 `hold`.

    **사유가 하나라도 HOLD 면 HOLD 다.** 모르는 kind · kind 가 없는 항목도 HOLD 다 — 서버가 새
    사유를 더했는데 로컬이 모르면, 그것을 실행 쪽으로 읽는 것은 fail open 이다.
    """
    table = GATE_DEFAULT if table is None else table
    holding = []
    for r in reasons:
        if not isinstance(r, dict):
            holding.append({"kind": "malformed", "action": "hold", "detail": repr(r)})
        elif table.get(str(r.get("kind")), "hold") != "execute":
            holding.append(dict(r))
    return ("hold" if holding else "execute"), holding


def derive_verdict_reasons(response: dict[str, Any]) -> list[dict[str, Any]]:
    """`verdict_reasons` 가 없는 응답 (옛 서버 · 씬 없는 청크) 에서 사유를 **유도**한다.

    유도 규칙은 옛 `safe` 와 같은 결정을 내도록 고른다: `safe` 면 사유 없음, 아니면
    기하 미인증 → `uncertified`, 그 밖 → `collision`. 어느 쪽이든 HOLD 다.
    """
    if bool(response.get("safe", False)):
        return []
    if not bool(response.get("geometry_certified", False)):
        return [make_reason("uncertified",
                            f"derived: geometry not certified (ag3s_status="
                            f"{response.get('ag3s_status')})", derived=True)]
    return [make_reason("collision",
                        f"derived: trajopt_status={response.get('trajopt_status')}, "
                        f"max_violation_m={response.get('max_violation_m')}", derived=True)]


def unpack_verdict_reasons(response: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """`(사유 목록, 출처)`. 출처는 `server` (키가 있었다) 또는 `derived` (`derive_verdict_reasons`).

    키가 있는데 목록이 아니면 **HOLD 사유 하나로** 읽는다 — 틀린 모양을 "사유 없음" 으로 읽으면
    fail open 이다.
    """
    if VERDICT_REASONS not in response:
        return derive_verdict_reasons(response), "derived"
    blob = response.get(VERDICT_REASONS)
    if not isinstance(blob, (list, tuple)):
        return [{"kind": "malformed", "action": "hold",
                 "detail": f"verdict_reasons is {type(blob).__name__}, not a list"}], "server"
    return [dict(r) if isinstance(r, dict) else {"kind": "malformed", "action": "hold",
                                                 "detail": repr(r)} for r in blob], "server"
