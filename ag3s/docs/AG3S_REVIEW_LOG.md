# AG3S 공동 코드 검토 기록

> **목적** — 씬(장애물) 쪽 primitive 근사를 걷어내고 **attention + TSDF/ESDF** 로 대체하기
> 위해, 현재 코드를 스텝별로 함께 읽고 기능적 오류를 판정한다. 로봇 모델 자체의 primitive
> (sphere chain) 는 유지된다.
>
> **규칙** — 사용자와 함께 판정하기 전까지 어떤 발견도 "확정" 으로 쓰지 않는다.
> 각 스텝은 `설명 → 사용자 질문 → 공동 판정 → 기록` 순서로 진행하고, 스텝을 합치지 않는다.

> **검토 계획서** — [AG3S_REVIEW_PLAN.md](AG3S_REVIEW_PLAN.md). 스텝 분해, 삭제 경계표,
> 사전 조사에서 찾은 발견 목록의 근거가 그쪽에 있다. 이 파일은 그 계획의 **실행 기록**이다.

---

## 진행 현황

| Step | 대상 | 상태 | 완료 |
|---|---|---|---|
| 0 | 로그 생성 + `ag3s` ↔ `trajopt` 경계 지도 | 진행 중 | — |
| 1 | `esdf.py` — TSDF/ESDF 코어 | 대기 | — |
| 2 | `pipeline.py` ESDF 조립부 | 대기 | — |
| 3 | `trajopt/linearize.py` ESDF 소비부 | 대기 | — |
| 4 | `reconstruction.py` + `robot_filter.py` | 대기 | — |
| 5 | `attention_lifting.py` | 대기 | — |
| 6 | `target_grounding.py` | 대기 | — |
| 7 | `multiview.py` | 대기 | — |
| 8 | `support_surface.py` + 평면 행 | 대기 | — |
| 9 | `collision_candidates.py` + `geometry.py` | 대기 | — |
| 10 | `clearance.py` + `constraint_builder.py` + `to_adapter.py` + `attached.py` | 대기 | — |
| 11 | `tests/ag3s/` + `tests/trajopt/` | 대기 | — |

---

## 누적 발견

상태는 `미확정` → (`확정` | `기각` | `보류`). 심각도는 전환 설계에 미치는 영향 기준.

| ID | 심각도 | 상태 | 요약 | 위치 | Step |
|----|--------|------|------|------|------|
| E1 | 심각 | 미확정 | GRASP 의 target carving 이 **전신에 대해** target 을 지운다 | `pipeline.py:684-696` → `esdf.py:339` | 2 |
| E2 | 심각 | 미확정 | 테이블이 무한 half-space. 홈 자세에서 26/61 sphere 위반, 최악 −0.677 m | `constraint_builder.py:216` | 8 |
| E3 | 심각 | 미확정 | attached object 가 실제 optimizer 에 도달하지 않는다 | `linearize.py:126` (읽는 코드 없음) | 10 |
| E4 | 중간 | 미확정 | ESDF 격자 밖은 무조건 자유 | `esdf.py:400`, `esdf.py:516` | 1 |
| E5 | 중간 | 미확정 | per-camera 통계가 세 대 모두 `"camera"` 로 붕괴 | `pipeline.py:658` ↔ `esdf.py:466` | 2 |
| E6 | 중간 | 미확정 | TSDF 적분이 전 복셀 중심을 매 카메라·매 프레임 투영 | `esdf.py:127-141` | 1 |
| E7 | 낮음 | 미확정 | 로봇 마스크 생성이 클라우드를 통째로 재구성 | `pipeline.py:597-625` | 2 |
| F2 | 중간 | 미확정 | attention 정규화가 카메라별 — 융합 `max` 가 스케일을 고른다 | `attention_lifting.py:280` | 5 |
| F8 | 중간 | 미확정 | 희소 attention 에서 p95 컷이 0 이 되어 0값 시드 허용 | `target_grounding.py:183` | 6 |
| F9 | 중간 | 미확정 | `lift()` 의 `image_hw` 기본값이 다운샘플 후 어긋난다 | `attention_lifting.py:265` | 5 |
| F10 | 낮음 | 미확정 | `DEGRADED` 가 `NO_TARGET` 을 가린다 | `to_adapter.py:176-179` | 10 |
| X1 | 정보 | 미확정 | primitive 폴백 순서가 문서와 반대 (삭제 대상) | `geometry.py:255` | 9 |
| X2 | 정보 | 미확정 | overflow 집합체가 빈 슬롯과 구분 불가 (삭제 대상) | `constraint_builder.py:466` | 9 |

---

## 모듈 처분

`필요` = ESDF 전환 후에도 그대로 / `축소` = 일부만 남음 / `삭제` = 제거.
각 스텝에서 확정한다.

| 모듈 | 처분 | 근거 |
|---|---|---|
| `esdf.py` | (Step 1) | — |
| `pipeline.py` | (Step 2) | — |
| `trajopt/linearize.py` | (Step 3) | — |
| `reconstruction.py` | (Step 4) | — |
| `robot_filter.py` | (Step 4) | — |
| `attention_lifting.py` | (Step 5) | — |
| `target_grounding.py` | (Step 6) | — |
| `multiview.py` | (Step 7) | — |
| `support_surface.py` | (Step 8) | — |
| `collision_candidates.py` | (Step 9) | — |
| `geometry.py` | (Step 9) | — |
| `clearance.py` | (Step 10) | — |
| `constraint_builder.py` | (Step 10) | — |
| `to_adapter.py` | (Step 10) | — |
| `attached.py` | (Step 10) | — |
| `robot_models/urdf_sphere_chain.py` | 필요 | 로봇 primitive 는 유지. 자체 캡슐→구 이산화, `geometry.py` 미의존 |

---

## Step 0 — 회귀 기준선 + 경계 지도

- 시작: 2026-09-05 23:53 KST / 완료: —
- 처분: —
- attention 의 역할: —

### 회귀 기준선

검토 시작 시점의 상태. 나중에 코드를 고칠 때 비교 대상이 된다.

```
$ src/openpi/.venv/bin/python -m pytest tests/ag3s/ tests/trajopt/ -q
560 passed, 217 warnings in 49.34s      # ag3s 445 + trajopt 115
```

```
$ src/openpi/.venv/bin/python -m benchmark.ag3s.pipeline --frames 12 --profile
points: 256884 raw -> 41333 voxel -> 40057 after self-filter
candidates: 2 object, 0 unknown, 1 target, 1 support

AG3S per-stage latency (11 frame(s) after 1 warm-up)
  stage                        mean ms     std    n
  -------------------------- --------- ------- ----
  scene_reconstruction           39.08    3.58   11
  robot_self_filter              11.75    2.16   11
  support_surface                46.77    3.34   11
  attention_lifting               5.28    0.44   11
  target_grounding               37.66    5.19   11
  collision_candidates           11.97    2.08   11
  primitive_fitting               0.24    0.02   11
  constraint_generation           0.18    0.02   11
  -------------------------- --------- ------- ----
  TOTAL                         152.94
```

이 표에서 이미 읽히는 것 하나 — **삭제 대상인 primitive 경로는 전체 시간의 8% 밖에 안 쓴다**
(`collision_candidates` 11.97 + `primitive_fitting` 0.24 + `constraint_generation` 0.18).
비용이 큰 것은 `support_surface` 46.8 ms 와 `target_grounding` 37.7 ms 이고, 둘 다 **남는** 쪽이다.
(단 이 fixture 는 합성 씬이고 ESDF 는 꺼져 있다. RB-Y1 실측에서는 ESDF 자체가 1213 ms 중
대부분을 쓴다 — Step 1 에서 확인한다.)

### 경계 지도 — 누가 무엇을 하는가

(설명은 대화에서 진행. 확정 후 여기에 채운다.)

### 사용자 질문과 답

(대기)

### 발견

(Step 0 에서는 위 "누적 발견" 표를 미확정 후보로 등재하는 것까지만 한다.)
