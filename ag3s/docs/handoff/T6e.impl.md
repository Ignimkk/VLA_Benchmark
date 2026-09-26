# T6e — 구현

> writer: ag3s-implementer (A1) · 2026-09-26 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

`serve_safe.py` 에 **`--exclude-links LINK [LINK ...]`** 를 더했다. `--links` 가 고른 집합에서
준 이름만 빼고 제약 모델을 다시 만든다 — `--exclude-links link_left_arm_5 link_right_arm_5` 로
팔뚝 둘을 제약에서 뺄 수 있다. 기동 script 가 `"$@"` 를 그대로 넘기므로
`PORT=8000 bash pi05_TO_hybrid/logs/serve_safe_rby1_16d.sh --exclude-links link_left_arm_5 link_right_arm_5`
로 뜬다.

세 가지를 flag 자체보다 공들여 막았다: **(1)** flag 를 안 주면 제약 link 집합이 글자 그대로
예전과 같다. **(2)** 제약 모델에 없는 이름은 `ValueError` 로 서버를 시작 전에 죽인다 —
`link_filter` 는 집합 교집합이라(`urdf_sphere_chain.py:389`) 모르는 이름을 조용히 버리고,
그러면 오타 하나로 **아무것도 안 빠진 채 "뺐다" 고 믿는** 실행이 남는다. **(3)** 자기 필터
모델은 손대지 않는다 — 거기서 빼면 그 link 의 점이 필터를 통과해 장애물로 샌다.

**이것은 해답이 아니라 진단이다.** 팔뚝을 제약에서 빼는 동안 그 팔뚝이 무엇에 부딪혀도 아무도
막지 않는다 — `build_constraint_robot_model` 의 주석이 바퀴에 대해 쓴 것과 같은 말이다:
*"빼는 것이 위험을 지우지는 않는다."* 이 실행으로 가르는 것은 "최악 위반 link 가 늘 팔뚝인
것이 팔 자세를 틀고 있는가" 하나뿐이고, 답이 무엇이든 팔뚝 제약은 되돌려야 한다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/serve_safe.py:32-77` | `constraint_links_minus(present, exclude, known=…)` 새 함수 | 이름 검증과 필터 계산을 **순수 함수 하나**로 모았다. 없는 이름이면 여기서 `ValueError` — URDF 에 아예 없는 이름과 "URDF 에는 있으나 제약 모델에 구가 없는 link" 를 갈라 말한다 (후자도 빼도 아무 일이 안 일어나므로 거절한다). 전부 빼면 충돌 행이 0 이 되므로 그것도 거절한다 |
| `benchmark/trajopt/serve_safe.py:80-131` | `build_ag3s(..., exclude_links=())` · 제약 모델만 다시 만든다 · 시작 로그 | `filter_robot = build_robot_model(scene)` 는 손대지 않는다. 뺀 뒤에는 `logging.warning` 으로 **무엇을 뺐는지 · 구가 몇 개에서 몇 개로 줄었는지 · 자기 필터에는 남아 있다는 것 · 진단용이라는 것**을 찍는다 (`--esdf-backend` 가 legacy 경고를 찍는 것과 같은 이유 — 조용히 다른 설정으로 떠 있는 것이 가장 나쁘다). `AG3S: ... constraints N spheres (...)` 의 괄호도 `arms minus link_left_arm_5,link_right_arm_5` 로 바뀐다 |
| `benchmark/trajopt/serve_safe.py:252-323` | `main()` 의 parser 를 `build_parser()` 로 뽑았다 | 테스트가 **같은 parser** 를 봐야 "flag 를 안 주면 지금과 같다" 를 고정할 수 있다. 인자 정의·기본값은 한 줄도 바꾸지 않았다 (`--exclude-links` 만 추가) |
| `benchmark/trajopt/serve_safe.py:271-277` | `--exclude-links` (`nargs="+"`, 기본 `()`) | 이름과 모양은 지시서의 예를 그대로 따랐다 |
| `benchmark/trajopt/serve_safe.py:325-338` | `reject_bad_flag_combinations(ap, args)` | 옛 `--shadow` + `--no-safe` 검사를 그대로 옮기고 `--exclude-links` + `--no-safe` 를 더했다. `--no-safe` 면 제약 모델을 아예 안 만들므로 flag 가 아무 일도 안 하는데 뺐다고 믿게 된다 |
| `benchmark/trajopt/serve_safe.py:381-390` | 기록 meta 에 `exclude_links` | 준 경우에만 키를 더한다(shadow 와 같은 규약 — 기본 기록의 키 집합은 T0 때와 같다). 어느 link 를 뺀 실행인지 모르는 기록은 다른 실행과 비교할 수 없다 |
| `tests/trajopt/test_exclude_constraint_links.py` (새 파일, 14 tests) | 위 네 가지를 고정 | 기본 집합(`ARM_LINKS` 18 개)을 글자로 박았고, 오타·구 없는 link·전부 제외를 각각 거절하는지, 전신 모델에 팔뚝이 남는지, `--no-safe` 조합이 죽는지를 본다 |

`grounding_report.py`(`ARM_LINKS`·`build_constraint_robot_model`)와 `robot_models/` 는 **한 줄도
안 고쳤다.** 다른 소비자(`safe_replay`·`esdf_rollout`·`bringup`·studies)가 전부 그 기본값을
쓰고 있어서다.

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**791 passed** (직전 777 + 새 14). 실패 0.

flag 가 실제로 서버 경로에서 도는지는 **모델 구성만** 확인했다 (rollout 아님, GPU 아님):
`build_ag3s(..., links="arms")` 의 제약 구 **120** 개가 `exclude_links=["link_left_arm_5",
"link_right_arm_5"]` 에서 **110** 개로 줄고, 자기 필터는 양쪽 다 **218** 개로 같다.
`--exclude-links link_left_arm_55` 는 `ValueError` 로 죽는다. 포트 8000 서버(PID 413170)는
건드리지 않았고 살아 있다.

## verifier 가 알아야 할 것

- **새 flag**: `--exclude-links LINK [LINK ...]`, 기본 `()` = 지금과 같다. 기본값 변경 없음.
- 뺀 실행의 서버 로그에는 `constraint model: EXCLUDING ...` **WARNING** 한 줄과
  `constraints 110 spheres (arms minus ...)` 가 찍힌다. **그 두 줄이 없으면 flag 가 안 먹은
  실행이다** — 판정 전에 로그에서 확인해 달라.
- `--record-constraints` 로 남긴 기록의 `meta` 에 뺀 경우에만 `exclude_links` 키가 생긴다.
  기본 실행의 키 집합은 그대로다.
- 재생산이 필요한 산출물(npz 등): 없음.
- 옛 기록과 호환이 깨지는가: 없음.
- 회귀 기준선(`위반으로 시작 14/15` · `has_target 9/15` · `frame0 clearance_before +0.157 mm`)에
  닿는 코드는 아니다 — flag 를 안 주면 제약 모델이 비트 단위로 같다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다. -->

팔뚝을 빼면 남은 제약(팔 0-4 · 손끝)이 최악 위반 link 를 다시 정하므로, **위반이 사라지는지**가
아니라 **어디로 옮겨 가는지**가 정보다. 두 갈래 다 뜻이 있다: 위반이 손끝/다른 팔 link 로
옮겨 가면 apple 153 mm 밖의 그 좌표는 팔뚝이 원인이 아니었다는 뜻이고, HOLD 가 사라지면
팔뚝 구가 실제로 없는 것을 밀고 있었다는 뜻이다 — 그때는 **왜 그 자리에 점이 있는가**(점군·
자기 필터 잔여)가 다음 질문이고, 팔뚝을 계속 빼 두는 것은 답이 아니다.
