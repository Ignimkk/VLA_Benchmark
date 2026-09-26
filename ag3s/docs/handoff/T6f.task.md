# T6f — TO 가 실행 창에만 걸리게 하고, 팔을 실제 굵기로 모델링한다

> writer: lead (A0) · 2026-09-26 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T6f.impl.md`
> 사용자 판정·지시 2026-09-26.

## 확정된 원인 (`T6d` 실측)

closed-loop 네 실행이 전부 사과를 **0.0 mm** 도 못 움직였고, shadow(정책 원본 chunk) 하나만
**245.2 mm** 들어 바구니에 넣었다. 원인은 **AG3S 도 cuRobo 도 아니다**:

- `manipulated_link_margin`(권한 link 만 margin 완화)은 **동작한다** — 120 구 중 22 구가
  실제로 완화를 받는다 (approach 50→20 mm, grasp 0 mm).
- `esdf.unknown_policy = free` 이고, 참값으로 **225 프레임 중 220 프레임에서 100 mm 안에 씬
  물체가 없다.** field 가 없는 것을 만들어 내지 않았다.
- **TO 는 오히려 잘 하고 있다** — `refined` 가 `reference` 보다 참값 clearance 를 나쁘게 만든
  chunk 가 **0/75**, 전역 최소가 −24.61 → **+25.61 mm** 로 좋아진다.

**진짜 원인은 실행 창이다.**

| 왼손끝 → 사과 표면 최소거리 | **실행되는 앞 8 스텝** | 전체 50 스텝 |
|---|---|---|
| shadow, 원본 chunk | **−17.96 mm** (닿는다) | −18.17 mm |
| closed-loop, refined | **+89.86 mm** | **−17.61 mm** |

**refined chunk 도 50 스텝 안에서는 사과에 닿는다. 그런데 실행되는 것은 앞 8 스텝뿐이고
그 창에서는 90 mm 떨어져 있다.** TO 가 *"먼저 비켜 돌아간 뒤 나중에 접근"* 하는 궤적을 만드는데,
회피 부분만 반복 실행되고 접근 부분은 매번 버려진다. `step_in_chunk` 가 **0..7 뿐**인 것이
증거다 (600 제어 스텝 / 75 chunk, 세 실행 모두).

## 고칠 것 1 — **TO 를 실행 창에만 건다** (사용자 판정)

50 스텝 전체가 아니라 **실제로 실행될 스텝만** 다듬는다. 그러면 회피를 뒤로 미룰 곳이 없다.

- 실행 길이는 `OPEN_LOOP_HORIZON`(8)이다. **코드에서 그 값을 읽어라** — 상수를 다시 박지 마라.
- **계획 지평을 줄이는 것이 아니라 다듬는 구간을 줄이는 것**인지, 둘을 같이 줄이는 것인지
  코드를 읽고 정해라. 근거를 `impl.md` 에 적어라.
- 계산량이 줄어 실시간성에도 도움이 된다 (서버 total p50 3445.6 ms, chunk 예산 533 ms).
  **얼마나 줄지는 예상만 적고 사실로 쓰지 마라** — 측정은 A2 다.

## 고칠 것 2 — **팔을 실제 굵기로** (사용자 지시)

lead 가 MJCF mesh 로 실측했다 (`bounding_capsules`, 3 분할):

| link | **MJCF mesh 실측** | 지금 제약 구 |
|---|---|---|
| `link_*_arm_4` | 39.4~53.0 mm | 38.4 mm |
| **`link_*_arm_5`** | **65.4~68.4 mm** | **81.2 mm** |
| `link_*_arm_6` | 23.7~27.3 mm | 26.0~29.2 mm |

`arm_5` 는 근사에서 **약 15 mm** 가 덧붙었다. `urdf_sphere_chain.py:443` 의 `_effective_radius`
가 `sqrt(capsule 반지름² + (간격/2)²)` 이라 **간격을 좁히면 반지름이 내려간다.**

**사용자 판정: "굳이 모든 link 를 다 덮지 않아도 된다. capsule 반지름을 줄여라."**

- `sphere_spacing`(지금 기본 1.0, `urdf_sphere_chain.py:365` — *"1.0 guarantees coverage;
  smaller is finer and slower"*)을 **설정으로 조절 가능**하게 하고 기본값을 낮춰라.
- capsule 반지름 자체에 **배율 또는 상한**을 둘 수 있게 하라. **기본값은 지금과 같게 두고**,
  서버 flag 나 config 로 바꿀 수 있게 한다 — 되돌릴 수 있어야 한다.
- **덮지 못하는 부분이 생기면 그 사실을 시작 로그에 크게 찍어라.** *"이 설정은 link 를 다 덮지
  않는다"* 를 조용히 넘기면 안 된다. `--exclude-links` 가 그렇게 하고 있다.
- **self-filter 모델은 건드리지 마라.** 거기서 가늘어지면 로봇 점이 장애물로 샌다.

## 고칠 것 3 — `esdf_margin` 을 줄일 수 있게

지금 **50 mm** 다 (`serve_safe.py --esdf-margin`, 기본 0.05). 구 반지름 81 mm 와 합쳐
중심에서 **131 mm** 자유공간을 요구했다. **flag 는 이미 있으니 기본값을 바꿀지만 정하면 된다** —
**바꾸지 말고 `impl.md` 에 "flag 로 조절 가능하다" 만 적어라.** 값은 사용자가 정한다.

## 확인할 것 4 — **관절 매핑** (사용자 지시)

*"TO 에서 chunk 를 받을 때와 출력할 때 관절 매핑이 잘 되어 있는지 확인해 달라."*

client 쪽은 이미 검증됐다 — A2 가 `(chunk_seq, step_in_chunk)` 행과 다음 control `qpos` 를
599 쌍 대조해 **mae ≤ 0.011 rad** 로 확인했고, 열 배치는
`left_arm_0..6 = col 0..6 · left_gripper = col 7 · right_arm_0..6 = col 8..14 · right_gripper = col 15` 다.

**검증되지 않은 것은 TO 안쪽이다.** chunk 의 열 → optimizer 의 결정변수 → 다시 chunk 의 열로
돌아오는 길이 맞는가. **읽고 확인하고, 테스트로 고정해라.** 이 프로젝트는 같은 종류의
index 오류를 이미 밟았다 — `safe_policy.py:411` 이 left gripper 열을 **6**(14D 배치)으로 박아
두었고 16D 는 `wire.gripper_columns()` 가 `(7, 15)` 를 준다. **그 별건도 이번에 같이 고쳐라.**

## 보고만 할 것 5 — 타원체

사용자 제안: *"구 말고 타원을 쓰는 것도 좋은 방법일 것이다."*

팔뚝처럼 **긴 link 는 구 사슬보다 타원체가 훨씬 잘 맞는다.** 다만 ESDF 질의가 점 기반
(구 중심 + 반지름)이라 타원체는 질의 방식이 달라진다. **이번에 구현하지 마라.**
**코드를 읽고 "무엇을 바꿔야 하는가 · 얼마나 큰 일인가"를 `impl.md` 에 한 절로 적어라.**
저장소에 선례가 있다 (`benchmark/knows_vla/visualize_ellipsoids.py`).

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 **791 passed**. 테스트를 더해라 — 기본 설정이면 지금과 같은 구 집합 · 실행 창만 다듬는다 ·
관절 매핑 왕복이 항등이다 · 덮지 못하면 경고가 난다.

**긴 rollout 을 돌리지 마라.** 재기동과 재실행은 lead 와 사용자가 한다.

## 규칙

- 네 소유: `benchmark/**/*.py` · `tests/**` · `handoff/T6f.impl.md`.
- **포트 8000 서버(PID 717165)를 죽이지 마라.** `pkill -f` 금지 — PID 로만.
- 커밋하지 않는다. 규칙 I — technical term 은 영어로.
- 박힌 수치(0.0 / 245.2 mm · 8 / 50 · +89.86 / −17.96 / −17.61 mm · 0/75 · −24.61 → +25.61 mm ·
  65.4~68.4 / 81.2 mm · 50 mm · 131 mm · 791) 밖의 값을 지어내지 마라.
