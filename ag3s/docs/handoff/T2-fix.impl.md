# T2-fix — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-09-25

## 무엇을 했나 (평이한 요약 먼저)

`target_grounding` 이 만드는 `metrics` dict 에 **`runner_up_score` 한 키를 더했다.** 후보 cluster
목록은 이미 점수 내림차순으로 정렬돼 있어서 1 등이 `best`(=`clusters[0]`)고 2 등은 `clusters[1]`
이다 — 그 `target_score` 를 그대로 넣었다. **후보가 하나면 `0.0`** 이다 (근거는 아래 별도 절).
그 밖에 동작을 바꾼 것은 없다. `LOW_SCORE` 경로는 `metrics` 를 만들지 않으므로 손대지 않았다.

이 한 키가 없어서 `safe_policy` 가 latch 에 넘기던 `runner_up` 이 전 프레임 0.0 이었고, 확신
문턱이 `1.3 × 1e-9` 로 무너져 **점수가 0 보다 크면 무조건 `confident`** 였다. 이제 2 등이 있으면
실제 격차로 판정한다.

판정 (a)(캡슐 0 개를 소리 나게)는 **이미 tree 에 들어가 있고 세 항목 모두 그대로였다 — 다시 쓰지
않았다.** 판정 (b)(손바닥을 constraint model 에 넣기)는 **하지 않았다 — 판정으로 유지.**

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/stages/target_grounding.py:402-417` | `metrics` dict 에 `"runner_up_score"` 추가 (키 7 개 → 8 개). 값은 `float(clusters[1].target_score) if len(clusters) > 1 else 0.0` | `trajopt/safe_policy.py:341` 이 읽는 키 이름이 바로 이것이다. 없어서 `.get(..., 0.0)` 이 항상 0.0 을 돌려주고 `grasp_latch.py:142` 의 격차 문턱이 무력화돼 있었다 |
| `tests/ag3s/test_target_grounding.py:498-568` | 테스트 3 개 추가 (파일 끝에 절로 붙였다) | 키의 존재 · 두 분기의 뜻 · 문턱이 다시 뜻을 갖는다는 것을 각각 고정 |

**`target_grounding.py` 저장 시각 (A2 가 겹침을 판정할 값):**

| | 값 |
|---|---|
| mtime (`stat -c %y`) | **`2026-09-25 14:48:13.122620000 +0000`** |
| 변경 **전** md5 | `caae04a1432fc52d00b1d9d37aa216e9` (= `benchmark/.git` 의 `HEAD:ag3s/stages/target_grounding.py` 와 동일 — 이 파일은 그때까지 uncommitted 변경이 없었다) |
| 변경 **후** md5 | `ee33a055a941811da6df094fbe985a94` |

A2 의 run 이 `caae04a1…` 을 찍었다면 변경 전 조건이고, `ee33a055…` 면 겹쳤다. 이 파일 말고는
`ag3s/**` 아래 어떤 `.py` 도 건드리지 않았다.

### 후보가 하나뿐일 때 `0.0` 을 고른 이유 (한 줄이 아니라 세 줄이 됐다)

1. **`0.0` 은 "2 등의 점수" 의 글자 그대로의 값이다.** 2 등이 없으면 2 등의 점수는 0 이다.
   sentinel 을 새로 만들지 않았다.
2. **latch 의 답으로도 옳다.** 경쟁자가 없는 프레임은 애매하지 않다 — 확신해도 되는 것이 맞다.
3. **신호로서 모호하지 않다.** cluster 의 점수는 **0.0 에 도달할 수 없다** — `_score_cluster`
   (`target_grounding.py:252-256`) 가 평균 attention 과 `exp(-d/scale)` 두 항을 섞고 후자는
   항상 양수다. 그래서 `runner_up_score == 0.0` 은 늘 "후보가 하나" 를 뜻하고 "2 등이 0 점" 과
   섞이지 않는다. **별도의 `n_candidates` 키를 더하지 않은 이유가 이것이다.**
4. 같은 규약을 이미 쓰는 코드가 있다 — `experiments/diagrams/ppt_target_grounding.py:75-78`
   `_scores()` 가 `(sc[1] if len(sc) > 1 else 0.0)` 로 1·2 등을 내놓는다. 파이프라인과 그림이
   같은 값을 말하게 됐다.

## 판정 (a) 확인 — 세 항목 모두 **이미 됐다**

| 확인 항목 | 결과 |
|---|---|
| `KeyError` 경로와 `bounding_capsules` 가 빈 list 를 내는 경로 **둘 다** 경고하는가 | **그렇다.** `mujoco_source.py:543-557` 에서 두 경로가 같은 `empty` 리스트로 모이고 그 뒤 `_warn_no_capsules` 를 부른다. 빈 list 경로는 `bounding_capsules` 의 `points.shape[0] < 8 → return []` (`:403-404`) 이고 `reason` 이 `"the body has no mesh geoms to measure"` 로 갈린다 |
| `tests/ag3s/test_gap_filling_capsules.py` 가 통과하는가 | **12 collected / 12 passed, skip 0.** (함수 8 개 + parametrize 2 개가 각각 2 개로 펼쳐진다. 그중 경고 guard 5 개가 uncommitted 분) |
| `UNCOVERED_LINKS` 전부가 capsule 을 내는가 — 경고가 하나도 안 떠야 정상 | **그렇다.** `test_every_uncovered_link_actually_produces_capsules` 와 `test_a_healthy_list_stays_quiet`(경고 record 0 건을 단언) 이 둘 다 통과. 모델 자산이 있어 skip 되지 않았다 |

관찰 하나만 남긴다(고치지 않았다 — 지시서가 "같으면 다시 쓰지 마라" 였다): **빈 list 경로는
코드로는 덮여 있지만 테스트가 없다.** 지금 5 개 guard 는 전부 `KeyError` 경로(`FT_sensor_L`,
`EE_BODY_TYPO`)를 탄다. mesh 없는 body 를 쓰는 case 를 하나 더할지는 lead 판단.

## 판정 (b) — 하지 않았다

`experiments/reports/rby1_transport.py:454` 는 `ee_finger_` 만 보는 채로 그대로 뒀다.
`config.py:367-370` 의 `DEFAULT_CONTACT_LINKS` 와 어긋나는 것은 **판정으로 유지되는 알려진
상태다.** 코드 변경 0 줄.

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**667 passed, 0 failed** (103.6 s). 경고 231 건은 osqp·pyparsing 의 기존 DeprecationWarning 으로
이 변경과 무관하다.

지시서의 직전 기준은 **659 passed** 였다. 차이 8 을 전부 셌다 — 지어낸 수가 없다:

| | 개수 |
|---|---|
| 직전 기준 | 659 |
| 이미 tree 에 있던 uncommitted 경고 guard (`test_gap_filling_capsules.py`, `git diff` 로 `+def test_` 5 개 확인) | +5 |
| 내가 더한 `runner_up_score` 테스트 | +3 |
| 합 | **667** |

즉 내 변경 **전**의 이 tree 는 664 였고, 그 +5 는 판정 (a) 로 이미 들어와 있던 것이다.
떨어진 테스트는 없다.

새 테스트 3 개 (`tests/ag3s/test_target_grounding.py`):

| 테스트 | 무엇을 고정하나 |
|---|---|
| `test_metrics_carry_the_runner_up_score_the_latch_reads` | 키가 있고 값이 `clusters[1].target_score` 다 · `> 1e-9` (무력화된 문턱의 회귀 가드) · 그리고 **압도적인 target 은 여전히 confident** 여야 한다(`confidence >= 1.3 × runner_up`) — 이 변경이 latch 를 영영 안 걸리게 만드는 쪽으로 기울지 않았다는 반대편 가드 |
| `test_a_single_candidate_reports_no_runner_up` | 2D ablation(`use_3d_connectivity=False`) 은 후보가 정확히 1 개다 → `0.0` |
| `test_two_equally_good_candidates_are_reported_as_ambiguous` | **이 수정이 사는 값.** 합성 씬(동일한 blob 둘 + 중간의 고립된 peak 점 하나)에서 두 점수가 거의 같고, `score < 1.3 × runner_up` 이므로 그 프레임은 이제 confident 가 **아니다**. 예전에는 `runner_up=0.0` 으로 완벽히 confident 였다 |

세 번째 테스트가 경계에 걸려 흔들리지 않는지 rng seed 0~5 로 확인했다 — 비가 1.0 근처에 모여
문턱 1.3 에서 멀다. 테스트 자체에는 이 비를 **숫자로 박지 않았다** (`LatchConfig().score_ratio`
를 읽어서 비교한다 — 문턱을 바꾸면 테스트가 따라간다).

## verifier 가 알아야 할 것

- **새로 생긴 플래그·기본값 변경: 없음.** config 를 건드리지 않았다. `score_ratio` 는 그대로 1.3.
- **재생산이 필요한 산출물(npz 등): 없음.** 거리장·TSDF 생산 경로에 영향이 없다.
- **옛 기록과 호환이 깨지는가: 깨지지 않는다 — 한 방향으로만.** `metrics` 는 키가 **늘어난다**.
  옛 기록을 읽는 쪽은 그 키가 없는 채로 돌아가고(`.get` 기본값), 새 기록을 읽는 옛 코드도 dict
  에 키가 더 있는 것을 무시한다. 기존 테스트도 `set(metrics) >= {...}` 상위집합 검사다
  (`tests/ag3s/test_target_grounding.py:130`).
- **행동이 바뀌는 자리는 정확히 한 곳이다** — `grasp_latch._Confirm.update` 의 `confident`.
  잠금 시점(`target_latched`)이 **늦어지거나 안 걸릴 수 있다.** attach/detach 가 그것에 달려
  있으니 회귀 기준선을 볼 때 이 사슬을 먼저 본다.
- **`ep1807` 을 다시 잰다면** 변경 후 md5 (`ee33a055…`) 를 run 전후로 찍어 두는 게 좋다 —
  T2 의 `all_75_frames_runner_up_score = 0.0` 과 비교되는 값이 될 것이다.

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.** 기대가 보이면 판정이 뒤집힌다 (C2·D2).

**전부 예상이다. 사실로 쓰지 마라. 측정한 숫자는 하나도 없다** — 단위 테스트 밖으로 나가지
않았다.

1. **회귀 기준선 셋은 안 움직일 것으로 예상한다.** `위반으로 시작 14/15` · `has_target 9/15` ·
   `frame0 clearance_before +0.157 mm` — 셋 다 latch 가 잠기기 **전/무관**하게 정해지는 값이라고
   읽었다. 특히 `has_target` 은 grounding 의 `status`(`target is not None`)이고 `runner_up_score`
   는 `status` 계산에 전혀 들어가지 않는다. `frame0 clearance_before` 는 attach 이전 프레임이다.
   **이 셋이 움직였다면 내가 읽은 것이 틀렸다는 뜻이므로, 고치기 전에 원인을 먼저 찾아야 한다.**
2. **`target_latched` 는 늦어질 수 있다.** ep1807 은 frame 2 에 걸렸는데, 그러려면
   `confirm_frames` 만큼의 프레임이 연속으로 confident 여야 한다. 이제 2 등이 1 등의 1/1.3 =
   0.77 배보다 큰 프레임은 세지 않는다(끊지도 않는다 — `grasp_latch.py:143-146`). **T2 가
   ep1807 에서 "애매한 2 위 후보 자체가 없었다"(`identity_vs_runner_up_relation_ep1807`) 고
   적었으므로 ep1807 에서는 잠금이 그대로일 가능성이 더 크다고 본다.** 후보가 계속 하나뿐이면
   `runner_up_score = 0.0` 이고 판정은 변경 전과 **글자 그대로 같다.**
3. **따라서 이 변경의 효과는 ep1807 에서는 잘 안 보일 것이라고 예상한다.** 효과가 보이려면
   후보가 둘 이상인 프레임이 필요하다. **A2 가 확인해 줬으면 하는 것은 딱 하나:
   ep1807 75 프레임에서 `len(clusters) > 1` 인 프레임이 몇 개인가**, 그리고 그 프레임들의
   `runner_up_score` 와 `target_score / runner_up_score` 비. 그 비가 1.3 을 넘는지 못 넘는지가
   이 수정이 그 기록에서 무엇을 바꿨는지의 전부다.
4. 그 다음으로 볼 것: `target_latched` 프레임 번호와 attach 프레임 번호가 변경 전과 같은가.
