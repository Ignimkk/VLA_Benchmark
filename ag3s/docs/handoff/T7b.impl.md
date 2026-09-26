# T7b — 구현

> writer: ag3s-implementer (A1) · 2026-09-26 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

손잡이 셋을 더했다. **`--links gripper`** 는 제약을 손가락 넷(`ee_finger_l1/l2/r1/r2`)만 남기고
팔뚝·손목·몸통·반대팔을 제약에서 통째로 뺀다. **`--no-self-collision`** 은 쥔 물체 대 로봇 구
블록을 끈다 — 코드를 먼저 읽어 보니 끄는 길이 없었고, 그 블록이 이 repo 에 있는 자기 충돌
**전부**였다. **`--capsule-radius-scale-link gripper=0.35`** 는 손가락만 얇게 한다 — 전역
`--capsule-radius-scale` 은 굵기가 꼭 필요한 팔뚝·반대팔까지 같이 얇게 만들기 때문에 link 별
배율을 새로 만들었다.

**기본값은 셋 다 지금 그대로다.** flag 를 하나도 안 주면 제약 구 120 개, 자기 필터 구 218 개,
`covers every capsule`, `self_collision=True` — 시작 로그도 예전과 한 줄도 다르지 않다. 그것을
테스트로 박았다.

---

## 이것은 진단이지 해답이 아니다

이 설정에서 **팔뚝·몸통·반대팔은 무엇에 부딪혀도 아무도 막지 않고, 로봇이 자기 자신과 부딪혀도
아무도 모른다.** `build_constraint_robot_model` 의 주석이 적은 것과 같다 — *"빼는 것이 위험을
지우지는 않는다."* 세 flag 는 **무엇이 apple 을 0.0 mm 로 붙들고 있는지 가르는 도구**이고,
그것을 알아낸 다음에는 되돌려야 한다. 되돌리는 방법은 flag 를 빼는 것뿐이다 (아래 "되돌리기").

---

## 1. 손가락만 남기는 길 — `--links gripper`

`--exclude-links` 로 같은 일을 하려면 남길 것 넷을 빼고 **열넷을 손으로 적어야** 하고,
`link_filter` 는 집합 교집합이라 오타 하나가 조용히 사라진다. 그래서 `--links` 에 선택지를
더했다. 이름 목록의 정본은 `GRIPPER_LINKS` 하나이고, `ARM_LINKS` 의 손끝도 거기서 온다 —
두 곳에 적으면 갈라지고, 갈라지는 날 `gripper` 가 `arms` 의 부분집합이 아니게 된다.

| `--links` | 제약 link | 제약 구 |
|---|---|---|
| `arms` (기본) | 양팔 14 + 손가락 4 = 18 | **120** |
| `gripper` (새것) | 손가락 4 | **44** |
| `all` | 전신 | (전신) |

자기 필터 모델은 어느 선택지에서도 **218 구 전신 그대로**다.

## 2. self-collision 을 끄는 flag — `--no-self-collision`

**먼저 코드를 읽었고, 끄는 길은 없었다.** 그리고 읽는 동안 더 중요한 것이 나왔다.

- 이 repo 에 **로봇-로봇 쌍 검사는 아예 없다** (N1 study 가 같은 것을 적는다: *"self-collision
  제약이 `benchmark/trajopt/` 에 코드 0 줄로 아예 없다"*).
- 있는 자기 충돌은 하나뿐이다 — CasADi `ConstraintBuilder` 의 **쥔 물체 대 로봇 구** 블록
  (`attached_self_mask`, `constraint_builder.py:279`). §23 의 "반대팔 · 토르소 · 같은 팔의
  비접촉 link" 를 한 기계로 덮는 그 블록이다.
- **그리고 지금 서버 설정에서는 그 블록조차 만들어지지 않는다.** `esdf` backend +
  `use_support_planes: False` 에서 `build_spec = primitive_path or bool(surfaces)` 가 False 이고
  (`pipeline.py:574`), 그러면 `to_adapter` 가 CasADi 파라미터 벡터를 아예 안 쓴다
  (`to_adapter.py:141`). 실제 TO 경로(`linearize.py` → `sqp.py`)는 질의점(로봇 구 + 쥔 물체
  점)을 **ESDF 에만** 대고 보며, 쌍 검사가 없다.

→ **verifier 가 알아야 할 결론: T7b 실행에서 `--no-self-collision` 이 지우는 행은 0 개다.**
그래도 flag 를 만든 이유는 둘이다. (a) 사용자 지시가 "우선 꺼 놓자 (추후 다시 켤 수 있도록)"
이고, 의도가 코드와 기록에 남아야 한다. (b) primitive 경로나 support plane 을 켠 실행에서는
그 블록이 실제로 살아나고, 그때 이 flag 가 유일한 스위치다.

끄는 방식은 **행 삭제가 아니라 mask 0** 이다. 행은 상수 1 이 되고 기울기는 0 이므로 행 수 ·
파라미터 길이 · 희소성 · solver 객체가 양쪽에서 같다 — 다시 켜는 것이 같은 칸에 1 을 쓰는
파라미터 변경이라는 뜻이고, 그것이 "되돌릴 수 있다" 의 기계적 의미다.

## 3. 그리퍼 capsule 축소 — `--capsule-radius-scale-link gripper=0.35`

**전역 배율은 팔 전체에 걸리므로 손가락만 줄이는 길이 없었다. 만들었다** (link 별 배율).
`--capsule-radius-scale 0.35` 는 손가락을 6.8~8.8 mm 로 만들지만 `link_*_arm_5` 도 81.2 →
40.8 mm 로 같이 얇게 만든다. 굵기가 꼭 필요한 부분이 바로 거기다.

### 수치와 근거

손가락 capsule 은 URDF 가 아니라 MJCF mesh 실측에서 온다 (`gap_filling_capsules`): 손가락 하나에
capsule 3 개, `r = 12.97 / 14.43 / 17.62 mm`. 구 사슬 팽창까지 끝난 값이 아래다.

| 손가락 배율 | 손가락 구 | 구 + `esdf_margin` 10 mm = **테이블 위 최저 도달 높이** | 사과 반지름 38 mm 까지 여유 |
|---|---|---|---|
| 1.0 (URDF 그대로) | 13.9~18.7 mm | 23.9~**28.7** mm | 9.3 mm |
| 0.8 (T7a 가 쓴 값) | 11.5~15.4 mm | 21.5~**25.4** mm | 12.6 mm |
| **0.35 (권장)** | **6.8~8.8 mm** | 16.8~**18.8** mm | 19.2 mm |

**권장값을 0.35 로 고른 근거.** 제약은 `d_esdf(중심) ≥ 구반지름 + esdf_margin` 이고 테이블
위에서는 `d_esdf(중심) = 높이` 이므로, 손가락 구가 내려갈 수 있는 최저 높이가 그대로
`구반지름 + 10 mm` 다. 사과 **적도**(높이 38 mm)만 스치는 것으로는 집을 수 없고 사과의
**아래 절반**까지 내려가야 하므로 목표를 `38/2 = 19 mm` 로 두었다 — `0.35` 가 그 선 아래로
들어오는 값이다 (18.8 mm). 지시서에 박힌 수치(38 mm · 10 mm) 밖의 값은 쓰지 않았다.

### 읽는 사람이 알아야 할 것 — 손가락은 원래 범인이 아니었다

위 표의 1.0 행이 말한다: **URDF 그대로도 손가락은 28.7 mm 까지 내려간다** (사과 적도보다
9.3 mm 아래). 사과 적도에서 그리퍼를 붙들고 있던 것은 손가락이 아니다.

| 제약 link | 구 반지름 | 구 + 10 mm |
|---|---|---|
| `link_*_arm_5` (팔뚝) | 81.2 mm | 91.2 mm |
| `link_*_arm_6` (손목) | 25.9~29.2 mm | 35.9~39.2 mm |
| `ee_finger_*` (손가락) | 13.9~18.7 mm | 23.9~28.7 mm |

**`link_*_arm_6` 의 35.9~39.2 mm 는 사과 반지름 38 mm 와 겹친다.** `--links gripper` 는 팔뚝과
손목을 제약에서 통째로 빼므로, 그 사슬이 사라진다. capsule 축소는 그 위에 얹는 여유다.

---

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/experiments/reports/grounding_report.py:88` | `GRIPPER_LINKS` 신설, `ARM_LINKS` 가 그것을 뒤에 붙여 만들어진다 | `--links gripper` 의 이름 정본을 한 곳에 둔다. 두 곳에 적으면 `gripper ⊄ arms` 가 되는 날이 온다 |
| `benchmark/trajopt/serve_safe.py:81` | `LINK_GROUPS` · `constraint_link_filter(links)` | `--links` 해석을 한 곳으로. `build_ag3s` 에 박혀 있던 `None if links == "all" else ARM_LINKS` 를 대체 |
| `benchmark/trajopt/serve_safe.py:444` | `--links` 선택지에 `gripper` | 손가락 넷을 손으로 적는 오타 위험 제거 |
| `benchmark/trajopt/serve_safe.py:104` | `parse_link_scales(pairs)` | `LINK=배율` 파싱. 키에 집합 이름(`arms`·`gripper`) 허용, `all` 은 거절, 형식·부호 오류는 서버 시작 전에 죽는다 |
| `benchmark/trajopt/serve_safe.py:482` | `--capsule-radius-scale-link LINK=F` | link 별 배율. 기본 `()` = 예전과 같음 |
| `benchmark/trajopt/serve_safe.py:490` | `--no-self-collision` | 기본 off(=자기 충돌 켠 상태). `--no-safe` 와 함께 주면 시작 전에 거절 (`:560`) |
| `benchmark/trajopt/serve_safe.py:147` | `announce_diagnostic_scope(...)` | **기본이 아닌 설정을 시작 로그에 크게 찍는다.** 함수로 뺀 이유는 MuJoCo 없이 테스트가 그 메시지를 실제로 받아 볼 수 있어야 하기 때문 |
| `benchmark/trajopt/serve_safe.py:269` | `build_ag3s` 가 그것을 부른다 + `self_collision` 인자(기본 `True`) | 제약 모델을 만든 직후에 찍어 `--exclude-links` 경고·coverage report 와 같은 순서로 읽히게 |
| `benchmark/trajopt/serve_safe.py:308` | 껐을 때만 `{"constraint": {"self_collision": False}}` 를 `AG3SConfig` 에 넣는다 | 기본값을 여기 다시 적으면 스위치가 둘이 되고, 갈라진 쪽이 안전 판정이다 (`sphere_options` 와 같은 규약) |
| `benchmark/trajopt/serve_safe.py:196` | `sphere_options` 가 `capsule_radius_scale_by_link` 를 실어 보낸다 (빈 dict 면 키 없음) | 안 준 flag 는 키가 아예 없다 = 호출이 예전과 글자 그대로 같다 |
| `benchmark/trajopt/serve_safe.py:564` | `reject_bad_flag_combinations` 가 `sphere_options` 를 한 번 감싸 호출 | 파싱 오류를 argparse 오류로 낸다. 안 그러면 체크포인트 두 벌을 GPU 에 올린 뒤에 traceback 으로 죽는다 |
| `benchmark/trajopt/serve_safe.py:634` | recorder meta 에 `self_collision: False` (껐을 때만) | 자기 충돌이 꺼진 기록을 켜진 것과 나란히 읽는 것이 이 flag 의 가장 나쁜 실패다 |
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:442` | `capsule_radius_scale_by_link` 인자 (기본 `None`) + `scale_for_link()` (`:547`) | link 별 배율. 배율을 읽는 곳이 하나여야 구를 만드는 쪽과 로그가 안 갈라진다 |
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:471`, `:493` | 없는 이름 · 이 모델에 capsule 이 없는 이름을 **거절** | 조용히 버리면 아무 link 도 얇아지지 않은 채 얇게 했다고 믿는다 (`--exclude-links` 가 이미 밟은 함정) |
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:605` | `coverage_report()` 가 link 별 배율을 함께 찍는다 | 전역 배율만 보이면 손가락이 다른 값으로 얇아진 것을 로그에서 알 방법이 없다 |
| `benchmark/ag3s/config.py:757` | `ConstraintConfig.self_collision: bool = True` | 자기 충돌 스위치의 정본 |
| `benchmark/ag3s/constraints/attached.py:293` | `self_collision_mask(..., enabled=True)` — `False` 면 전부 0 | 행을 지우지 않는다. 같은 칸에 1 을 다시 쓰면 돌아온다 |
| `benchmark/ag3s/constraints/constraint_builder.py:169` | 생성자에서 자기 충돌 off 를 **크게 경고** | `serve_safe` 를 거치지 않는 호출자(실험 스크립트·테스트)도 반드시 듣는다 — `UrdfSphereChain` 의 덮개 경고와 같은 자리, 같은 이유 |
| `benchmark/ag3s/constraints/constraint_builder.py:426` | mask 가 config 를 읽는다 | 기본값을 여기 다시 적으면 스위치가 둘 |
| `tests/trajopt/test_gripper_only_constraints.py` | 새 파일, 38 test | 아래 |

**self-filter 모델은 손대지 않았다.** `build_robot_model` 에는 `sphere_options` 통로가 여전히
없고 (테스트가 그것을 확인한다), `--links` 도 `--capsule-radius-scale-link` 도 거기 닿지 않는다.
218 구 전신 그대로다 — 거기서 얇아지면 로봇 점이 필터를 통과해 장애물로 샌다.

---

## 되돌리기

| 켜고 싶은 것 | 하는 일 |
|---|---|
| 제약 범위 | `--links` 를 빼거나 `--links arms` — 120 구로 돌아온다 |
| self-collision | `--no-self-collision` 을 뺀다 — mask 가 1 로 돌아오고 행 수·희소성은 애초에 안 움직였다 |
| 손가락 굵기 | `--capsule-radius-scale-link` 를 뺀다 — `covers every capsule` 로 돌아온다 |

세 flag 는 서로 독립이다. 하나만 켜서 돌릴 수 있다.

---

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**879 passed** = 직전 **841** + 새 파일 **38**. 실패 0. **기존 테스트 파일은 한 줄도 안 고쳤다** —
기본값을 안 건드렸다는 가장 직접적인 증거다.

새 테스트가 지키는 것:

- **기본값** — parser 기본값 셋 · `ConstraintConfig().self_collision is True` ·
  `UrdfSphereChain` 의 `capsule_radius_scale_by_link` 기본 `None` · 빈 dict 를 명시해도 같은 모델 ·
  `sphere_options({기본}) == {}`.
- **손가락만 남는다** — `GRIPPER_LINKS` 글자 고정 · `ARM_LINKS` 의 꼬리가 그것 ·
  gripper 모델의 구가 arms 모델의 손가락 구와 **개수도 굵기도** 같다 · `--links grippers` 오타는
  parser 와 resolver 양쪽에서 거절.
- **자기 충돌** — mask 가 전부 0 · 물체가 팔을 삼킨 형상에서 켜면 위반 행이 나오고 끄면
  **전부 상수 1** · 그때 `n_constraints` · `n_attached_rows` · 파라미터 길이 · 경계가 양쪽에서
  같다 (= 되돌릴 수 있다) · 달라지는 것은 `attached_self_mask` 블록뿐.
- **link 별 배율** — 적은 link 만 얇아지고 나머지는 `pytest.approx` 로 같다 · 구 수 불변 ·
  전역 배율이라면 팔뚝까지 얇아졌을 것이라는 대조 · 전역과 link 별을 함께 주면 적은 쪽이 이긴다 ·
  오타 · capsule 없는 link · 0 이하 배율 전부 거절 · `coverage_shortfall` 과
  `coverage_report()` 에 그 사실이 남는다.
- **시작 로그** — 기본 실행은 **아무 말도 안 한다** (`caplog.text == ""`) · gripper·자기충돌 off ·
  link 별 배율 각각에서 `!!! ... !!!` 가 나간다 · `ConstraintBuilder` 생성자도 따로 외친다 ·
  켜져 있으면 조용하다.
- **`--no-safe` 와의 조합** — `--no-self-collision` · `--capsule-radius-scale-link` 둘 다 거절.

추가로 `build_ag3s` 를 체크포인트 없이 두 번 띄워 로그를 눈으로 확인했다 (rollout 아님):

- 기본: `constraints 120 spheres (arms)` · `covers every capsule` · 새 경고 0 줄.
- `--links gripper --no-self-collision --capsule-radius-scale-link gripper=0.35`:
  `CONSTRAINT SCOPE IS GRIPPER` · `SELF-COLLISION IS OFF` · `PER-LINK CAPSULE RADIUS SCALE IS
  ACTIVE` · `constraints 44 spheres (gripper)` · 손가락 12 capsule 의 덮개 부족 9.9 mm 까지
  줄줄이 · `self-filter 218 spheres` 그대로.

---

## verifier 가 알아야 할 것

- **새 flag · 기본값 변경**: flag 3 개 신설 (`--links gripper` 선택지 · `--no-self-collision` ·
  `--capsule-radius-scale-link`). **기본값 변경은 없다.** flag 를 안 주면 제약 구 120 · 자기 필터
  구 218 · `self_collision=True` · `covers every capsule` — 회귀 기준선이 그대로여야 한다.
- 권장 실행 인자: `--links gripper --no-self-collision --capsule-radius-scale-link gripper=0.35`.
  (`--capsule-radius-scale 0.8` 은 **같이 주지 않는다** — 그러면 손가락은 0.35, 나머지는 0.8 이
  되는데 `--links gripper` 에서 나머지는 제약에 없으므로 의미가 없고 로그만 시끄러워진다.)
- **`--no-self-collision` 이 이 실행에서 지우는 행은 0 개다** (위 2절). 자기 충돌이 꺼졌다는
  것이 최악 위반 행에 나타나기를 기대하면 안 된다 — 나타날 행이 애초에 없다.
- **재생산이 필요한 산출물**: 없음. TSDF/ESDF 생산 경로(`.venv-curobo`)는 건드리지 않았다.
- **옛 기록과 호환이 깨지는가**: 아니다. recorder meta 는 껐을 때만 `self_collision` 키가
  늘고, `sphere_options` 안에 `capsule_radius_scale_by_link` 가 들어간다 — 둘 다 준 경우에만.
  기본 실행의 meta 키 집합은 T0 때와 같다.
- 측정은 내가 하지 않았다. 긴 rollout 을 돌리지 않았고, 포트 8000 서버(PID 1435228)도
  건드리지 않았다.

---

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.**

`--links gripper` 로 최악 위반 link 는 `ee_finger_*` 중 하나가 될 수밖에 없다 (남은 것이 그것
뿐이다). 그 값이 여전히 크면 미는 것은 팔 자세가 아니라 **손가락 자신이 테이블/사과 근처에서
받는 제약**이고, 작으면 (또는 위반이 0 에 가까우면) T6e~T7a 의 최악 행 사슬은 전부
"결정 변수가 아닌 곳에서 온 상수" 였다는 뜻이다. 후자면 apple 이 0.0 mm 를 벗어나는지가 다음
갈림길이고, 그래도 0.0 mm 면 원인은 제약 모델 밖(refiner 의 실행 창·grasp latch·shadow 와의
차이)에 있다.

capsule 축소 쪽에서 내가 기대하는 것은 `ee_finger_*` 행의 여유거리가 `0.8` 대비 약 6.6 mm
(15.4 → 8.8) 만큼 좋아지는 것뿐이다. 그 이상이 나오면 다른 것이 함께 바뀐 것이므로 의심해야
한다.
