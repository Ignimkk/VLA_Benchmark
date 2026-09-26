# T6e — 특정 link 를 constraint model 에서 뺄 수 있게 한다

> writer: lead (A0) · 2026-09-26 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T6e.impl.md`
> 사용자 요청 2026-09-26: *"`link_left_arm_5` / `link_right_arm_5` 에 collision 체크를 풀고
> hold 위반 없이 테스트해 보자."*

## 왜

로컬 세 실행에서 **최악 위반 link 가 늘 `link_left_arm_5` / `link_right_arm_5`(팔뚝)** 이고
손가락이 아니다. 유일한 HOLD 의 위반 좌표는 apple 에서 **153 mm** 떨어져 있고, **사용자가 그
자리를 직접 확인했는데 아무것도 없다.**

그 두 link 를 제약에서 빼고 돌리면 **팔뚝 때문에 팔 자세가 틀어지는 것인지**가 한 번에 갈린다.

## 지금 있는 것

`benchmark/ag3s/experiments/reports/grounding_report.py:93` —
`build_constraint_robot_model(scene, link_filter=ARM_LINKS)`. `ARM_LINKS` 는 양팔 7 관절 +
손가락 넷의 **이름 tuple** 이고, `link_filter=None` 이면 전신이다.

`benchmark/trajopt/serve_safe.py` 는 `--links {arms,all}` 둘만 준다 —
`link_filter=None if links == "all" else ARM_LINKS`. **특정 link 를 빼는 길이 없다.**

## 무엇을 하나

**서버에 link 를 빼는 flag 를 더한다.** 이름과 모양은 네가 정해라 — 예를 들어
`--exclude-links link_left_arm_5 link_right_arm_5` 처럼 `--links` 의 결과에서 빼는 식.

지킬 것:

- **기본 동작이 바뀌지 않는다.** flag 를 안 주면 지금과 같은 `ARM_LINKS` 여야 한다.
  테스트로 고정해라.
- **없는 이름을 주면 조용히 넘어가지 마라.** 오타 하나로 아무것도 안 빠진 채 "뺐다" 고 믿게
  된다 — 이 프로젝트가 여러 번 밟은 함정이다 (`EE_BODY_L` 대 `ee_left`,
  gripper 열 6 대 7). **주어진 이름이 모델에 없으면 크게 말해라.**
- **서버가 시작할 때 무엇을 뺐는지 로그에 찍어라.** `--esdf-backend` 가 그렇게 하고 있다
  (`serve_safe.py:71-82`) — 조용히 다른 설정으로 떠 있는 것이 가장 나쁘다.
- 제외된 link 가 **자기 필터(self-filter) 모델에는 그대로 남아야 한다.** 둘은 일부러 다른
  모델이다 — 자기 필터에서 빼면 그 link 의 점이 장애물로 샌다.

## 이것이 해답이 아니라 진단이라는 것을 적어라

팔뚝을 제약에서 빼면 **그 팔뚝이 무엇에 부딪히든 아무도 안 막는다.** `build_constraint_robot_model`
의 주석이 같은 것을 적고 있다 — *"빼는 것이 위험을 지우지는 않는다."* `impl.md` 에 한 줄로.

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 **777 passed**. 테스트를 더해라 — flag 없으면 옛 link 집합 그대로 · 준 이름이 빠진다 ·
없는 이름이면 실패하거나 크게 경고한다 · 자기 필터 모델은 안 바뀐다.

**긴 rollout 을 돌리지 마라.** 재기동과 재실행은 lead 와 사용자가 한다.

## 규칙

- 네 소유: `benchmark/**/*.py` · `tests/**` · `handoff/T6e.impl.md`.
- **포트 8000 서버(PID 413170)를 죽이지 마라.** `pkill -f` 금지 — PID 로만.
- 커밋하지 않는다. 규칙 I — technical term 은 영어로.
- 박힌 수치(153 mm · 25.18 mm · 777) 밖의 값을 지어내지 마라.
