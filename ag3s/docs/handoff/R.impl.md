# R-port / R-measure — 구현

> **14D 와 비교하지 않는다** (사용자 판정 2026-09-25). 아래 값은 전부 **16D 에서 얼마인가**
> 이고, 14D 수치는 판정 근거로 쓰지 않는다. 아래 X1 절에 14D 숫자가 한 번 나오는데, 그것은
> 비교가 아니라 **"왜 두 숫자를 나란히 놓을 수 없는가"** 를 적기 위한 것이다.

---

# X1 — 고정 구 분할 조사 (사용자 판정: 즉시)

## 결론 먼저

**분할은 옳다. 그리고 액션 폭(14→16)은 분할에 아무 영향이 없다 — 회귀 기준선도 영향받지 않는다.**

내가 올린 "14D 27 개 → 16D 42 개" 는 **모델 차원 때문이 아니었다.** 결정적 실험: 같은 기록,
같은 프레임, 같은 물체 점군에서 **`free_joints` 만** 12 개(14D 레이아웃) ↔ 14 개(16D 레이아웃)로
바꿔 봤다.

```
14D layout (arm_joint_dim=6, 자유 관절 12): 고정 구 42 / 120
16D layout (arm_joint_dim=7, 자유 관절 14): 고정 구 42 / 120
14D 에서만 고정: 0 개    16D 에서만 고정: 0 개
```

**집합이 비트 단위로 같다.** 액션 폭은 이 분할에 들어올 자리가 없다.

## 1. 분할이 어디서 정해지나

`benchmark/ag3s/constraints/attached.py:241-292` `rigid_spheres()`. **목록이 아니라
기구학에서 파생된다** — 자유 관절만 24 회 흔들어(`scale=0.30` rad) 쥔 물체와의 여유거리
변화폭이 `1e-6` m 미만인 구를 고정으로 표시한다.

## 2. 42 는 옳은가 — **옳다.** 그리고 왜 그런지 끝까지 확인했다

고정 42 개의 정체:

| 링크 | 구 | 왜 고정인가 |
|---|---|---|
| `ee_finger_l1` | 11 | 쥔 물체와 **같은 강체**로 움직인다 (물체가 `link_left_arm_6` 에 탄다) |
| `ee_finger_l2` | 11 | 같음 |
| `link_left_arm_6` | 15 | 같음 — 물체의 부모 링크 그 자체 |
| `link_left_arm_5` | 5 | **아래 별도** |
| 합 | **42** | |

`link_left_arm_5` 는 물체보다 **몸쪽**이라 "같이 움직인다" 로는 설명이 안 된다. 두 경로를
갈라 확인했다:

* **몸쪽 관절(`arm_0`~`arm_5`)을 흔들 때** — `link_left_arm_5`·`link_left_arm_6`·물체가
  하나의 강체로 움직인다 (`arm_6` 이 안 움직이므로). 상대 거리 보존.
* **`arm_6` 만 흔들 때** — `arm_6` 은 전완 축을 도는 **roll 관절**이고,
  `link_left_arm_5` 의 구 5 개가 **그 회전축 위에 있다**:

  ```
  left_arm_6 회전축: 원점 [0.292 0.331 1.043] 방향 [-0.737 0.145 0.660]
    link_left_arm_4   구  4  축까지 거리 37.835 ~ 51.602 mm
    link_left_arm_5   구  5  축까지 거리  0.194 ~   0.207 mm   <- 축 위
    link_left_arm_6   구 15  축까지 거리  0.965 ~  55.775 mm
  ```

  그리고 그 구들의 **최근접 물체 점도 축 위**에 있다 (손목 쪽 점). 축 위의 두 점 사이 거리는
  축 둘레 회전에 **정확히** 보존되므로, `arm_6` 을 1.0 rad 까지 돌려도 여유가 **0.000 nm**
  변한다 (float64 12 자리까지 동일). 같은 perturbation 에서 왼팔의 다른 링크는 4576 µm 움직인다.

즉 42 는 **기하적으로 정확한 답**이고 버그가 아니다.

> **왜 27 과 나란히 놓을 수 없나** (비교가 아니라 설명이다): 옛 측정은 **구 105 개**짜리
> 모델에서 나왔고(archive 가 *"구 105 중 27"* 로 적고 있다), 지금 제약 모델은 **120 개**다
> (`ARM_LINKS` 18 링크: 팔 14 + 손끝 4). 구 집합 자체가 다르므로 두 수는 같은 축의 값이
> 아니다. 사용자 판정대로 **16D 값만 쓴다: 120 구 중 고정 42 · 움직일 수 있는 구 78.**

## 3. 회귀 기준선이 영향을 받았나 — **아니다. 받을 수 없다**

세 겹으로 확인했다:

1. **`rigid_spheres` 에 소비자가 없다.** repo 전체 grep 결과 정의(`attached.py:241`), 내가
   이번에 쓴 `studies/n1_self_collision_clearance.py`, 그리고 `tests/ag3s/test_rigid_spheres.py`
   뿐이다. 파이프라인·trajopt 어디에서도 부르지 않는다.
2. **`self_collision_mask` 에 대한 내 앞선 주장은 틀렸다 — 아래에서 고쳐 적는다.**
   (요지는 바뀌지 않는다: 그 마스크는 회귀 기준선 경로에 영향을 주지 않는다. 그러나
   *"읽는 코드가 0 줄"* 은 사실이 아니었다.)
3. **회귀 기준선 경로는 attach 자체를 안 한다.** `esdf_rollout.py` 에 `attach` 문자열이
   0 건이라, attached 슬롯이 채워지지도 않는다.

**방향**: 없음. 분할은 어떤 측정 경로에도 올라가 있지 않고, 이번에 내가 만든 N1 스크립트가
**첫 소비자**다.

## 정정 — `attached_self_mask` 다섯 자리가 각각 무엇을 하나

내가 *"쓰이기만 하고 읽는 코드가 0 줄"* 이라고 쓴 것은 **틀렸다.** 원인은 내 grep 이
**문자열 `attached_self_mask` 만** 찾고, 그 슬롯 오프셋을 담은 **변수 `lo_am`** 을 안 찾은
것이다. `lo_am` 은 네 번 나오고 그중 하나가 실제 제약 행을 만든다.

| 파일:줄 | 무엇을 하나 |
|---|---|
| `constraint_builder.py:153` | **슬롯 선언** — `("attached_self_mask", a * np_ * s)`. 쥔 물체 슬롯마다 로봇 구 `s` 개분의 마스크 자리 |
| `constraint_builder.py:163-164` | **행 수에 포함된다** — `n_attached_rows = horizon * a * np_ * (m + k + s)`. 마지막 `s` 항이 바로 self-collision 블록이다. 자리만 잡은 것이 아니라 **세어진다** |
| `constraint_builder.py:245` | `_attached_rows()` 안에서 슬롯 오프셋을 `lo_am` 으로 **읽는다** |
| **`constraint_builder.py:280-289`** | **실제 제약 행을 만든다.** 로봇 구마다 `mask = p[lo_am + flat*s + sphere_index]` → `both = active * mask` → `h = ‖centre − robot_centre‖ − (r_obj + r_robot + margin)` → `out.append(both*h + (1−both)*1.0)`. **쥔 물체 대 로봇 자기 구의 여유거리 행이고, 마스크가 0 인 쌍만 `1.0`(항상 만족)으로 중화된다.** 이것이 self-collision 그 자체다 |
| `constraint_builder.py:397` · `:414` · `:423` | 프레임마다 마스크 **수치를 채운다** (`self_collision_mask(...)` → `values[lo_am + …]`) |

그리고 `:218` `expr_factory` 가 `ca.vertcat(rows + plane_rows + self._attached_rows(Q, p))`
로 그 행들을 **제약 벡터에 실어 내보낸다.**

### 그런데도 N1 이 성립하는 이유 — **실려 나가지만 아무도 평가하지 않는다**

* `expr_factory` 를 부르는 곳은 `ag3s/constraints/to_adapter.py:59` `to_casadi()` 하나와
  테스트들뿐이다.
* **`benchmark/trajopt/` 는 `to_casadi()` 를 쓰지 않는다.** `trajopt/linearize.py:3-20` 이
  그 이유를 직접 적고 있다 — *"Why this does not call AG3S's CasADi fragment"*. 단일 CasADi
  그래프는 빌드 19 s · 평가 19 ms 인데, 구 FK + 위치 야코비안으로 직접 세우면 0.066 s · 1.07 ms
  라서 **trajopt 는 자기 선형화를 따로 만든다.**
* trajopt 가 만드는 블록은 **질의점 대 씬**뿐이다: `linearize.py:416-434` `attached_states` 가
  쥔 물체를 추가 **질의점**으로 만들고, `sphere_states` 가 `Q = S + N`(로봇 구 + 쥔 점, 후자는
  반지름 0)으로 이어 붙인 뒤 전부 candidate·평면·ESDF 에 **질의**한다. 질의점끼리의 쌍은
  어디서도 계산하지 않는다 — 즉 **물체 대 씬은 있고, 물체 대 로봇 자신은 없다.**
* `ConstraintBuilder` 인스턴스는 `pipeline.py:128` 하나에서만 만들어지고
  `ConstraintSet.builder` 로 실려 나가는데, **`benchmark/trajopt/` 를 포함해 그것을 읽는
  소비자가 repo 에 0 개다** (grep: `.builder` 소비자 없음, `safe_policy.py:45` 는 docstring 언급).

**그래서 고쳐 적으면**: self-collision 제약은 **AG3S 쪽 심볼릭 빌더에 완전히 구현되어 있고
행 수까지 잡혀 있다.** 없는 것은 구현이 아니라 **배선**이다 — 실제로 도는 최적화기
(`benchmark/trajopt/`)가 그 제약 벡터를 아예 평가하지 않는다. N1 의 원문 *"self-collision 제약이
`benchmark/trajopt/` 에 코드 0 줄로 아예 없다"* 는 **경로를 `trajopt/` 로 못박은 그대로 정확하다.**
내 요약이 그것을 "마스크가 안 읽힌다" 로 옮기면서 틀렸다.

이 정정은 X1 의 결론을 바꾸지 않는다 — 어느 쪽이든 회귀 기준선 경로는 attach 를 하지 않고
(`esdf_rollout.py` 에 `attach` 0 건), trajopt 는 그 행들을 평가하지 않는다.

---

# X2 — 감사 목록 (목록만. 고치지 않았다)

## 축 0 — **`has_target` 이 틀린 대상을 성공이라고 말한다** (코디네이터 지시로 맨 위)

### 먼저 정정할 것 — **`LOW_SCORE` 는 `has_target` 을 제대로 막고 있다**

지시문의 전제(*"`LOW_SCORE` 가 19 프레임에 찍혔지만 `has_target` 은 여전히 참"*)는 코드와
맞지 않는다. 셋을 이으면 둘은 **같은 프레임에서 동시에 참일 수 없다**:

1. `benchmark/ag3s/stages/target_grounding.py:378-381` — 점수가 문턱 미만이면
   `GroundingResult(None, GroundingStatus.LOW_SCORE, ...)` 를 돌려준다. **`target` 이 `None` 이다.**
2. `benchmark/ag3s/runtime/pipeline.py:390` · `:400` · `:653/681` · `:664/684` — `target` 과
   `status` 는 **같은 `GroundingResult` 한 개**에서 나온다 (`:400` 의 `dataclasses.replace` 는
   `target is not None` 일 때만 돈다).
3. `benchmark/ag3s/types.py:913-914` — `has_target = (self.target is not None)`.

그러므로 `LOW_SCORE` ⟹ `target is None` ⟹ `has_target == False` 다. 31 + 19 = 50 으로
기록 길이와 정확히 맞는 것도 이 때문이다 — **두 집합은 겹치지 않고 서로를 채운다.**
게이트는 작동한다.

### 그래서 진짜 결함은 더 나쁘다 — **막을 수 있는 종류가 아니다**

남는 사실은 그대로다: 통과한 **31 프레임에서 중심이 사과에서 0.36~3.59 m 떨어져 있다.**
이유는 게이트가 새는 것이 아니라 **게이트가 재는 것이 "맞는 물체인가" 가 아니기 때문**이다.

`target_grounding.py:236-261` `_score_cluster`:

```
peak_proximity      = exp(-‖centroid − peak_point‖ / peak_scale)
spatial_consistency = sqrt(compactness · peak_proximity)
target_score        = (w_a·mean_attention + w_g·spatial_consistency) / (w_a + w_g)
```

세 항(`mean_attention` · `compactness` · `peak_proximity`)이 **전부 attention peak 를 기준으로
한 상대량**이다. 프롬프트가 무엇을 지목했는지, 그 물체가 어디 있는지는 이 식에 들어오지
않는다. 즉 `target_score` 는 **"attention 과 얼마나 잘 맞는가"** 이지 **"맞는 물체인가"** 가
아니다. attention 이 엉뚱한 곳에 peak 를 찍으면, 그 엉뚱한 peak 에 가장 잘 맞는 클러스터가
높은 점수로 통과한다 — 그리고 `target.confidence = best.target_score`
(`target_grounding.py:390`) 로 실려 나가 **읽는 쪽에는 정확도 점수처럼 보인다.**

이 기록에서 attention 이 엉뚱한 곳에 찍히는 이유는 축 C·축 E 다: grounding 이 읽는
`cam_high` 에 사과가 **65 프레임 내내 0 px** 이다. 볼 수 없는 것에 peak 를 찍을 수는 없으므로
peak 는 반드시 다른 물체 위에 선다.

**fail-closed 와의 관계**: AG3S 는 *"클러스터가 문턱을 넘었는가"* 에 대해서는 닫혀 있고
(`NO_CLUSTER`/`LOW_SCORE` 로 `None` 을 낸다), *"넘은 클러스터가 옳은가"* 에 대해서는
**닫힐 수단이 없다** — 그것을 판별할 정보가 파이프라인 안에 없다. `target_score_threshold`
기본값 `0.25` (`config.py:91`) 를 올려도 방향이 안 맞는 고신뢰 오답은 그대로 통과한다.

**고치지 않았다.** 이것은 문턱 조정이 아니라 **판정**이 필요한 자리다 (예: grounding 결과를
독립 신호로 교차검증할 것인가, 아니면 `confidence` 가 정확도가 아님을 계약에 명시할 것인가).

## 판정 8 — 오프라인 attention 배선을 live 와 같게 (**고쳤다**)

`benchmark/trajopt/experiments/esdf_rollout.py`:

| 파일:줄 | 무엇이 |
|---|---|
| `:133-164` | `ci = cameras.index("cam_high")` 한 장 → **`ATTENTION_ALIAS` 로 세 대의 인덱스를 각각** (`cam_high→zed_left` · `cam_left_wrist→wrist_cam_l` · `cam_right_wrist→wrist_cam_r`). live 의 `serve_safe.py:143-161` `ALIAS` 와 같은 표를 쓴다. npz 에 없는 카메라는 조용히 빠진다 — live 의 `attention.get(cam)` 과 같은 실패 방식(그 카메라만 없고 나머지는 산다) |
| `:276-286` | `att` 하나 → **`att_by_cam` dict**. 합성 블롭 경로도 카메라마다 만든다 (예전에는 head 프레임 하나로 만들어 세 대에 쓰려 했다) |
| `:297` | `attention_map=(att if c == "zed_left" else None)` → **`attention_map=att_by_cam.get(c)`** — `wire.py:171` 과 **같은 표현**이다 |
| `:392-393` | 결과 JSON 에 `attention_cameras` · `attention_wiring` 를 적는다 |

축 순서는 생산자에서 확인했다: `sources/pi05_attention.py:231-234` 가
`(len(run), len(denoise), len(aggs), n_layers, n_heads, len(cameras), GRID, GRID)` 로 할당하므로
**`(frame, denoise, agg, layer, head, camera, row, col)`** 이고, 소비자 인덱싱과 일치한다.

### **회귀 기준선이 움직일 수 있다 — A2 가 알아야 한다**

기준선 명령이 `--attention … --cameras all` 이라, 예전에는 attention 이 `zed_left` 에만
들어갔고 이제 세 대에 들어간다. AG3S 입력이 바뀌므로 개수가 움직일 수 있어 **CLAUDE.md 의
기준선 명령을 그대로 다시 돌렸다**:

```
attention: 3 대에 붙인다 ['wrist_cam_l', 'wrist_cam_r', 'zed_left'] (live 와 같은 배선)
15청크
  전체 구  : 위반으로 시작 15/15  해소 13  개선 15
  상태: feasible 13, violated 2
```

**개수는 넷 다 그대로다** (기준선: 위반 15/15 · 해소 13 · 개선 15 · feasible 13, violated 2).
회귀 없음.

**그런데 grounding 은 살아났다**: `has_target` 이 **0/15 → 9/15**.

이것은 CLAUDE.md 의 기준선 주석을 **낡게 만든다**. 거기에는

> 이 기준선은 attention → target grounding 을 재지 않는다. 두 카메라 구성 모두 `has_target`
> 이 15/15 프레임에서 false 다 (live 경로는 같은 에피소드에서 성공한다 — 차이는 오프라인이
> 고정된 attention 셀 선택을 쓰는 것이고 …)

라고 적혀 있는데, **원인 진단이 틀렸다.** 셀 선택이 아니라 배선이었고, 배선을 live 와 같게
하니 같은 셀·같은 기록에서 9/15 가 선다. 기준선이 재는 범위가 넓어졌다 (이제 grounding 도
포함한다). **문서 갱신은 A3 의 것이고, 기준선 주석을 고칠지는 A0 의 판정이다.**

### 같은 꼴이지만 **고치지 않은** 세 곳

| 파일:줄 | 왜 안 고쳤나 |
|---|---|
| `studies/step7_fusion_asymmetry.py:127` | **의도된 것으로 보인다.** 이 실험의 주제가 *"두 융합 방식이 갈리는 곳"* 이라 카메라 비대칭이 측정 대상이다. 바꾸면 실험이 묻는 것이 바뀐다 |
| `studies/a1_horizon_visibility.py:145` | 판정 8 의 범위 밖. live 대표성을 따지는 실험이면 같이 고쳐야 한다 — **A0 판정 필요** |
| `reports/observation_sdf_report.py:127` | 같음 |

## 축 E — **설계 항목: grounding 이 `cam_high` 한 대에 의존한다** (판정 9)

`cam_high`(=`zed_left`)가 두 기록 65 프레임 전부 사과 0 px 다. 배선(축 C)을 고쳐 세 카메라에
attention 을 붙여도, **grounding 이 어느 클라우드 위에서 도는가**는 따로 답해야 하는 물음이다.
손목 카메라만 대상을 보는 상황이 실기에서 흔하다면 이것은 구현 결함이 아니라 설계 판정이다.
축 0 과 맞물린다 — 대상이 안 보이는 카메라에서 peak 를 뽑으면 반드시 엉뚱한 물체가 뽑힌다.

# X2 (이어서) — 16D 정렬 감사

## 축 A — **이미 16D 에 맞아 있다** (확인한 것)

| 무엇 | 파일:줄 | 상태 |
|---|---|---|
| `ARM_JOINT_DIM` / `ACTION_WIDTH` | `benchmark/trajopt/wire.py:82,85` | 7 / 16 |
| `ChunkLayout.rby1` 기본 `arm_joint_dim` | `benchmark/trajopt/types.py:162` | 7. **6 을 박은 호출부가 repo 에 없다** |
| `DEFAULT_RBY1_JOINTS` | `robot_models/urdf_sphere_chain.py:329-332` | 팔당 `range(7)` — arm_6 포함 (20 DoF) |
| `ARM_LINKS` | `reports/grounding_report.py:87-90` | `range(7)` — `link_*_arm_6` 포함 |
| `client._current_state` 그리퍼 채우기 | `trajopt/client.py:196-205` | `wire.ARM_JOINT_DIM` 에서 파생. 주석이 *"6 을 박지 않는다"* 라고 명시 |
| 제약 모델 구 수 | (측정) | 120 = 팔 14 링크 + 손끝 4 |

## 축 B — **14D 서술이 남은 곳** (동작은 맞고 문장이 틀렸다)

| 파일:줄 | 무엇이 틀렸나 |
|---|---|
| `benchmark/trajopt/experiments/esdf_rollout.py:73` | *"결정 변수가 양팔 **12관절**뿐이므로"* — 16D 에서는 **14** 관절이다. `--constraint-links` 의 근거 문장 |
| `benchmark/ag3s/experiments/reports/grounding_report.py:97` | 같은 문장, 같은 오류 |
| `benchmark/trajopt/client.py:80` | *"`{"actions": [H, **14**]}` 를 돌려준다"* |
| `benchmark/trajopt/safe_policy.py:59` | *"`infer(obs) -> {"actions": [H, **14**], ...}`"* |

동작에는 영향이 없다(전부 docstring). 다만 **`--constraint-links arms` 를 정당화하는 논리가
"12 관절" 위에 서 있어서**, 그 근거를 다시 읽을 때 수를 고쳐야 한다.

## 축 C — **오프라인 경로가 live 와 다르게 배선된 자리** (코디네이터 추가 지시)

live 는 `wire.py:171` 에서 `attention_map=attention.get(cam)` 으로 **카메라마다** attention 을
붙인다 (`serve_safe.py:143-161` 의 `attention_extractor` 가 세 대를 다 꺼낸다).
오프라인은 **`zed_left` 한 대에만** 붙인다:

| 파일:줄 | 코드 |
|---|---|
| `benchmark/trajopt/experiments/esdf_rollout.py:267` | `attention_map=(att if c == "zed_left" else None)` |
| `benchmark/ag3s/experiments/studies/a1_horizon_visibility.py:145` | 같은 꼴 |
| `benchmark/ag3s/experiments/studies/step7_fusion_asymmetry.py:127` | 같은 꼴 |
| `benchmark/ag3s/experiments/reports/observation_sdf_report.py:127` | `attention if name == "zed_left" else None` |
| `benchmark/ag3s/experiments/reports/attention_report.py:110` | RGB 도 `cam_high` 만 |

그리고 attention npz 를 읽는 **10 군데가 전부 `cam_high` 한 장만 인덱싱**한다
(`esdf_rollout.py:141`, `grasp_damage.py:97`, `safe_replay.py:147`, `geometry_report.py:156`,
`separation_report.py:117`, `a7_episode_walkthrough.py:96`, `a1_horizon_visibility.py:101`,
`step7_fusion_asymmetry.py:98`, `step6_f8_seeds.py:82`, `step5_f9_image_hw.py:87`).
**npz 에는 세 대가 다 들어 있다** (`cameras: ['cam_high','cam_left_wrist','cam_right_wrist']`) —
배선이 셋 중 둘을 버린다.

T1-a 가 본 *"attention 최대 칸이 사과에 떨어진 프레임 0/15"* 는 이 배선과 맞물린다: 이 기록에서
사과는 `wrist_cam_l` 에만 보이는데(코디네이터 실측: `zed_left` 0/50, `wrist_cam_r` 0/50),
오프라인은 `zed_left` 에만 attention 을 붙인다. **셀 선택의 문제가 아니라 배선의 문제다.**

> 축 C 는 **16D 문제가 아니다** — 14D 에서도 같았다. 다만 "오프라인 수치가 live 를 대표하는가"
> 라는 물음의 답이 여기 걸려 있어서 같은 목록에 둔다.

## 축 D — 기록 형식

16D 기록에는 depth 가 없다(`depth_cameras: []`). 7 개 스크립트는 모두 qpos 로 재렌더하므로
영향이 없지만, **`step.depth` 를 기대하는 코드를 새로 쓰면 조용히 빈 dict 를 받는다**
(`policy_record.py:250` 이 `depth_*` 키 유무로 카메라를 만든다).

---

# R-port — 구현 (이식 7 건 완료, P5 는 이번에 이식)

> writer: ag3s-implementer (A1) · 2026-09-25 · 읽는 쪽: lead(A0), verifier(A2), scribe(A3)
>
> 1 차(예측)의 "깨질 곳 목록" 은 이 문서 아래쪽 **부록**으로 옮겼다. A2 의 실측과 맞춘
> 결과도 거기 있다.

## 무엇을 했나 (평이한 요약 먼저)

`R-port.task.md` 의 P1~P8 중 **P5 를 뺀 일곱을 고쳤다.** P5(R3 이 기록을 먹게)는
코디네이터 지시대로 판정이 날 때까지 손대지 않았다.

세 갈래다.

1. **기록이 스스로 말하게 했다.** 시간 단위(P7)와 단계 경계(P6)가 코드에 상수로 박혀
   있었고 **둘 다 틀렸다**. 이제 `meta.json` 의 `ctrl_hz`·`open_loop_horizon` 과 기록의
   그리퍼에서 파생한다.
2. **조용히 섞이는 것을 막았다**(P3). 중간 산출물 npz 에 **출처 도장**을 찍고 소비자가
   확인해 안 맞으면 즉시 멈춘다. 2026-09-25 의 혼합 실행이 지금 코드로는 통과하지 못한다.
3. **결함을 고쳤다.** 프로세스 분리(P1), 출력 경로 인자화(P2), 입력 경로 인자화(P4),
   그리고 없던 측정 스크립트 신규(P8).

### 가장 말할 만한 것 — 단계 경계 규칙은 **새로 지어낸 것이 아니다**

P6 을 고치면서 "그리퍼가 벗어난 t_step 에서 한 칸 앞당긴 곳이 파지 시작" 이라는 규칙을
세웠는데, 그것을 `run_0004`(14D)에 적용하면

```
왼 그리퍼가 1.0 -> 0.732 로 벗어남: 기록 인덱스 10, t_step 80
파지     = 80 - 8(기록 간격) = 72
pre_grasp = 72 - 16          = 56
approach  = 72 - 48          = 24
                    -> (24, 56, 72)
```

**하드코딩되어 있던 `(24, 56, 72)` 와 세 값이 모두 일치한다.** 즉 이 규칙은 옛 상수를
만든 바로 그 규칙이고, 옛 상수는 규칙이 아니라 **한 기록에서 나온 답을 굳힌 것**이었다.
같은 규칙을 16D 긴 기록에 적용하면 `(208, 240, 256)` 이 나온다.
`tests/ag3s/test_record_timing.py::test_derivation_reproduces_the_old_hardcoded_constants`
가 이것을 못박는다.

## 바뀐 파일

### 새 공용 모듈 — `sources/policy_record.py` (P3·P6·P7 의 공통 바탕)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/experiments/sources/policy_record.py:306-393` | **출처 도장** — `record_fingerprint` · `provenance_arrays` · `read_provenance` · `require_same_record` 신규 | 중간 npz 가 어느 기록에서 나왔는지 스스로 말하게 한다. 지문은 `meta.json` 전체 + 첫 qpos + **액션 폭**이고 **`limit`/`--frames` 에 흔들리지 않는다** (흔들리면 짝이 맞는 짝을 거부한다) |
| 같은 파일 `:396-451` | **시간 단위** — `record_step_interval_ms` · `control_step_ms` 신규 | `meta` 의 `ctrl_hz`·`open_loop_horizon` 과 실제 `t_step` 간격을 **둘 다** 보고 어긋나면 예외 |
| 같은 파일 `:454-560` | **단계 경계** — `gripper_columns_for_state` · `grasp_onset` · `phase_boundaries_from_record` · `phase_boundaries_for_path` · `phase_for` 신규 | `phase_for` 를 여기로 모았다. 예전에는 `esdf_rollout` 과 `curobo/ground_truth` 가 같은 상수를 따로 들고 있어서, 한쪽만 고치면 두 수치가 **다른 단계 정의** 위에서 나왔다 |

`phase_boundaries_for_path` 는 **기록 전체**를 본다. `--frames 3` 으로 잘라 읽은
`RunRecord` 에 물으면 파지가 그 뒤에 있을 때 "파지 없음 → fallback" 으로 조용히 떨어지는
것을 구현 중에 실제로 관측해서 나눴다.

### P7 — `STEP_MS` 단위 확정 (먼저 코드로 확정한 뒤 고쳤다)

**확정한 것 먼저.** `step7_state_lag.py` 에서 `STEP_MS` 가 곱해지는 상대는 `--lags` 의
`k` 이고(`:173` `ms = k * step_ms`), 그 `k` 는 `built[i - k]` 로 **기록 한 장** 단위의
지연이다(`:148`). 그리고 기록 한 장의 간격은:

* `pi05_infer.py:1387` — `t_step` 은 제어 스텝 루프의 변수다
* `pi05_infer.py:1110` — `steps_per_action = round(1/(CTRL_HZ * m.opt.timestep))` = `round(1/(15*0.002))` = **33 sim 스텝**, 즉 제어 스텝 하나 = 66.7 ms
* `pi05_infer.py:141-142` — `CTRL_HZ = 15`, `OPEN_LOOP_HORIZON = 8`; 기록은 청크를 새로 받을 때만 남으므로(`:1500`) 한 장 = **8 제어 스텝 = 533.3 ms**

옛 `16.0` 은 `8 × 2 ms`(sim timestep)로, **8 이 곱해질 상대를 잘못 잡은 값**이다. 33 배 차이.

| 파일:줄 | 무엇이 |
|---|---|
| `benchmark/ag3s/experiments/studies/step7_state_lag.py:57-68` | `STEP_MS = 16.0` **삭제**. 그 자리에 왜 틀렸는지 주석 |
| 같은 파일 `:109-111` | `step_ms = record_step_interval_ms(run)` 으로 파생 |
| 같은 파일 `:173` | `ms = k * step_ms` |
| 같은 파일 `:190-199` | 한계 안에 드는 비영 지연이 **없을 때**의 문장을 따로 뒀다 (아래 참고) |
| 같은 파일 `:104-106` | `--out` 신규 (그림 디렉터리) |
| 같은 파일 `:274-283` | 파일명을 `args.records[-4:]`(경로 끝 4 글자) → **기록 디렉터리 이름**으로. 끝 슬래시에 깨지던 것도 같이 |
| 같은 파일 `:288-305` | 규칙 A 의 sidecar `.json` 신규 |
| 같은 파일 `:221-233` | 허용 범위 띠가 축의 15 % 보다 좁으면 라벨을 밖으로 빼 화살표로 가리킨다 (새 단위에서 100 ms 는 수천 ms 축의 띠 하나다) |

**판정 방향이 뒤집힌다.** 새 단위에서는 **0 이 아닌 어떤 지연도 설정 한계(100 ms) 밖**이라,
옛 출력의 *"한계 안에서 이미 새니 한계가 느슨하다"* 가 성립하지 않는다. 지금은 이렇게 찍는다:

> 설정 한계(100 ms) 안에 들어오는 0 아닌 지연이 **없다** — 기록 한 장이 533 ms 라 가장 작은
> 비영 지연조차 한계 밖이다.
> `>>>` 판정: 이 기록의 해상도로는 한계 안쪽을 못 잰다. 한계가 느슨한지 아닌지는 이 측정이
> 답하지 못한다 — 답하려면 제어 스텝 단위 기록이 필요하다.

### P6 — 단계 경계를 기록에서 파생

| 파일:줄 | 무엇이 |
|---|---|
| `benchmark/trajopt/experiments/esdf_rollout.py:44-46` | 자체 `phase_for` 삭제, 공용 것 임포트 |
| 같은 파일 `:82-88` | `--phase-boundaries` 기본값 `(24,56,72)` → **`None`(= 기록에서 뽑음)** |
| 같은 파일 `:120-131` | 파생 + **어떻게 얻었는지 출력에 찍는다** |
| 같은 파일 `:255` · `:355` | `phase_for(step.t_step, boundaries)`; 결과 JSON 에 `phase_boundaries` 증거 블록 |
| `benchmark/ag3s/experiments/curobo/ground_truth.py:335-340` | `_phase(t, b=(24,56,72))` **삭제** (플래그가 없어 바꿀 수 없었다) |
| 같은 파일 `:108-111` · `:145-152` · `:194` | `--phase-boundaries` 신규 + 파생 + 사용 |

경계는 **판정**이라 숫자만 돌려주지 않고 `evidence` dict 를 같이 돌려주고, 그것이 출력과
sidecar JSON 에 그대로 실린다.

### P3 — 자산 짝 검사 (**가장 중요하다고 했던 것**)

도장을 **생산자가 찍고 소비자가 본다**. 세 군데를 다 이어야 검사가 뜻을 갖는다:

| 파일:줄 | 무엇이 |
|---|---|
| `benchmark/trajopt/experiments/esdf_rollout.py:344-351` | `--dump-frames` npz 에 `provenance_arrays(run, …)` 를 섞어 저장 |
| `benchmark/ag3s/experiments/curobo/build_rollout_fields.py:78-92` · `:175` | 도장을 **그대로 흘려보낸다**. 이 스크립트는 cuRobo venv 에 있어 기록을 직접 못 읽으므로 새로 찍을 수 없다. 도장이 없으면 경고 |
| `benchmark/ag3s/experiments/curobo/export_frame.py:96-99` | `verify_two_tier` 가 읽을 프레임 npz 에 도장 |
| `benchmark/ag3s/experiments/curobo/ground_truth.py:128-143` | **세 인자의 짝을 확인하고 안 맞으면 `SystemExit`** |

**검증했다** — 2026-09-25 의 바로 그 조합이 이제 통과하지 못한다:

```
$ ... ground_truth --frames-npz /tmp/rollout_frames.npz --fields /tmp/rollout_fields.npz \
      --records outputs/live_test/20260924_long16d/run_0000
중간 산출물이 지금 읽는 기록과 짝이 맞지 않는다:
  - --frames-npz /tmp/rollout_frames.npz: 출처 도장이 없다 (옛 형식). ... 다시 구워라
  - --fields /tmp/rollout_fields.npz: 출처 도장이 없다 (옛 형식). ... 다시 구워라
기록 outputs/live_test/20260924_long16d/run_0000 (도장 53ca28d97e60a1f8) 에 맞춰 다시 만들어라: ...
```

**도장이 없는 옛 형식도 통과시키지 않는다.** "모른다" 를 "맞다" 로 읽는 것이 정확히 그때
난 일이다.

덧붙여 — **미세 계층이 없으면 명확히 멈춘다** (`ground_truth.py:145-160`). 예전에는
`two.layers[1]` 에서 맨 `IndexError` 가 나서 읽는 쪽이 "버그" 와 "잴 것이 없다" 를 구분할
수 없었다. 지금은 *"코드 결함이 아니라 기록의 성질이다"* 까지 말한다. A2 의
`rollout_fields_16d.npz`(미세 0/15)로 조건을 확인했다.

### P2 — 출력 경로 인자화

| 파일:줄 | 무엇이 |
|---|---|
| `benchmark/ag3s/experiments/curobo/ground_truth.py:32-35` | `OUT` 기본값을 `figures/` → **`figures/r-16d/`** |
| 같은 파일 `:105-107` | `--out` 신규 |
| 같은 파일 `:470-500` | 파일명에 기록 태그(`curobo-ground-truth-<기록>.png`), 이미 있으면 경고, sidecar `.json` 신규 |

`step7_state_lag.py` 도 같은 처리를 했다(위 P7 표).

### P4 — `verify_two_tier.py` 입력 인자화

`benchmark/ag3s/experiments/curobo/verify_two_tier.py` — argparse 가 아예 없던 48 줄
스크립트를 다시 썼다. `--frame` **필수**(기본값 없음), `--device`/`--coarse`/`--fine`/
`--tsdf-voxel`/`--table-top`/`--out-json`. 출처 도장을 읽어 찍고, `target_centroid` 가
유한하지 않으면 *"grounding 이 서지 않았다 — 코드 결함이 아니라 프레임의 성질"* 로 멈춘다.
`export_frame.py` 의 `--records`·`--out` 도 **필수로** 바꿨다 (기본값 `run_0004` ·
`/tmp/rby1_frame.npz` 가 혼합 실행의 입구였다). 죽은 venv 경로(`src/openpi/.venv`)도 고쳤다.

### P1 — R1 프로세스 분리 (사용자 판정)

`benchmark/ag3s/experiments/live/verify_backend.py` 를 다시 썼다. **T0 불변식
(`pipeline.py:906`)은 건드리지 않았다.**

```
dump  --backend legacy  --seed 101 --out R1_legacy.npz   # 프로세스 1
dump  --backend curobo  --seed 101 --out R1_curobo.npz   # 프로세스 2
compare --legacy … --curobo … --out R1_verify_backend.json
```

* `cmd_dump`(`:119-205`) — backend **하나**. 구 거리·중심·반지름·링크 이름과 메타를 npz 로.
* `scene_fingerprint`(`:101-113`) — 관측 depth·외부 파라미터·구 위치의 해시. **seed 만으로는
  부족하다** (MuJoCo 나 모델 XML 이 달라지면 같은 seed 가 다른 씬을 낸다).
* `cmd_compare`(`:217-330`) — npz 둘만 읽는다. **파이프라인을 임포트하지 않는다** — 그러면
  이 단계가 다시 한 프로세스에 두 backend 를 부르는 길이 열린다. 씬 지문이 다르면 멈춘다.
* 배선 검사가 **더 세졌다**: 예전 `builder_class != "EsdfBuilder"` 대신
  `legacy_instances_created == 0` 을 **cuRobo 프로세스 안에서 센 값**으로 본다.

### P8 — N1 측정 스크립트 신규

`benchmark/ag3s/experiments/studies/n1_self_collision_clearance.py` (신규 · 335 줄).
`outputs/verify/` 가 아니라 `benchmark/ag3s/experiments/` 아래에 뒀다.

파지 프레임마다: 쥔 물체 점을 **MuJoCo 분할에서** 얻고(grounding 이 이 기록에서 서지 않고,
그것은 N1 이 묻는 질문과 무관하다) → `attach_from_target` → `rigid_spheres()` 로 **고정 구**와
**움직일 수 있는 구**를 기구학에서 가르고 → 두 집합의 여유거리를 따로 낸다. 전환 신호
("움직일 수 있는 구가 마진 안으로")를 직접 판정해 찍는다.

## 단위 검증

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m pytest tests/ -q
```

**648 passed, 0 failed** — 기존 **625** 가 전부 그대로 통과하고 (내 변경 전에 돌려 확인한
수와 같다) 신규 `tests/ag3s/test_record_timing.py` **23** 이 더해진 값이다.

신규 테스트가 못박는 것:

* `test_derivation_reproduces_the_old_hardcoded_constants` — 그리퍼 파생이 `run_0004` 에서
  `(24, 56, 72)` 를 **그대로** 낸다. 규칙이 옛 상수를 만든 규칙이라는 근거.
* `test_long_16d_record_needs_different_boundaries` — 같은 규칙이 긴 기록에서 `(208,240,256)`.
* `test_record_step_interval_is_derived_not_16ms` — 533 ms 이고 옛 16 ms 의 30 배 이상.
* `test_fingerprint_is_independent_of_how_many_frames_were_loaded` — **깨지면 짝 검사가
  짝이 맞는 짝을 거부한다.**
* `test_require_same_record_rejects_unstamped` — 옛 형식을 통과시키지 않는다.

> 테스트를 쓰다가 지문의 구멍을 하나 찾았다: 처음에는 `meta` + 첫 qpos 만 해시해서,
> **meta 가 같고 액션 폭만 다른** 두 기록이 같은 지문을 냈다. 액션 폭을 지문에 넣어 고쳤다
> (`policy_record.py:340-347`). 이 STEP 이 통째로 다루는 구분이라 지문이 직접 들고 있는
> 편이 낫다.

## 기능 확인 (측정이 아니다 — 스크립트가 도는지만 본다)

> **전체 회귀 측정과 figure 는 A2 의 것이다.** 아래는 내가 고친 코드가 도는지 본 것이고,
> 수치는 그 부수물이다. `R.measure` 의 값으로 인용하지 말 것.

* **P6·P3** — `esdf_rollout --records <긴 기록> --frames 3 --dump-frames …`
  → `단계 경계: (208, 240, 256) — 출처 gripper, 파지 시작 {index 33, t_step 264, left, 0.638}`,
  덤프에 도장 `53ca28d97e60a1f8`. 같은 기록은 통과, `run_0004` 는 거부.
* **P7** — `step7_state_lag --records <긴 기록> --frames 4 --first 6 --lags 0 1 2 3`
  → `기록 스텝 = 533.3 ms`, 지연 0/533/1067/1600 ms, 누수 전부 0, 카메라 이동 0→12.9 mm,
  판정은 위의 "못 잰다" 문장.
* **P8** — `n1_self_collision_clearance --records <긴 기록> --frames 38`
  → 파지 프레임 4 개(호출 32~35). 고정 구 42 / 움직일 수 있는 구 78.
  고정 최소 여유 중앙 **−55.1 mm**(최악 −59.4), 움직일 수 있는 구 최소 여유 중앙
  **+185.1 mm**(최악 +176.0). 전환 신호 **아직 아니다** (마진 50 mm 의 3.5 배) → 잠복 유지.
* **미세 계층 guard** — A2 의 `rollout_fields_16d.npz` 는 미세 0/15 로 guard 에 걸리고,
  옛 14D `/tmp/rollout_fields.npz` 는 15/15 로 통과한다.

## verifier(A2) 가 알아야 할 것

**새 플래그·기본값 변경** (여기가 제일 중요하다):

| 스크립트 | 바뀐 것 |
|---|---|
| `curobo/ground_truth.py` | `--frames-npz` · `--fields` · `--records` 가 **전부 필수**가 됐다 (기본값 삭제). `--out` · `--phase-boundaries` 신규. 그림이 `figures/r-16d/curobo-ground-truth-<기록>.png` 로 간다 |
| `curobo/verify_two_tier.py` | **`python -m` 으로 부른다** (전에는 파일 직접 실행). `--frame` 필수 |
| `curobo/export_frame.py` | `--records` · `--out` **필수** |
| `live/verify_backend.py` | **하위명령이 생겼다**: `dump`(backend 하나) / `compare`. 옛 호출법은 안 먹는다 |
| `trajopt/experiments/esdf_rollout.py` | `--phase-boundaries` 기본값이 `None`(기록에서 파생). 옛 동작을 원하면 `--phase-boundaries 24 56 72` 를 **명시** |
| `studies/step7_state_lag.py` | `--out` 신규. ms 축의 뜻이 바뀌었다 |
| `studies/n1_self_collision_clearance.py` | 신규 |

**재생산이 필요한 산출물**: `/tmp/rollout_frames.npz` · `/tmp/rollout_fields.npz` ·
`/tmp/rby1_frame.npz` 는 도장이 없어 **이제 거부된다.** `esdf_rollout --dump-frames` →
`build_rollout_fields` → (필요하면) `export_frame` 순서로 다시 굽고, `/tmp` 밖에 둘 것.

**옛 기록과 호환이 깨지는가**: 아니다. `run_0004` 로 돌려도 파생 경계가 `(24,56,72)` 로
나오므로 단계 정의가 그대로다. 단위만 옛 출력과 다르다(아래).

**R4·R5 는 여전히 못 잰다.** 코디네이터가 확인한 대로 미세 계층이 0/15 다. 내가 고친 것은
*측정을 여는 것이 아니라* (a) 옛 그림을 안 덮어쓰게, (b) 혼합 실행을 막게, (c) 못 재는
이유를 스크립트가 **직접 말하게** 한 것이다.

**R7 의 단위 주의** (P7 경고 그대로): 14D 값 *"16 ms 에서 851 점"* 은 **옛 단위**로 읽힌
것이고, 새 코드가 내는 ms 는 **새 단위**다. 두 표를 나란히 놓을 때 ms 축을 직접 비교하면
안 된다. **비교할 수 있는 것은 `--lags` 의 칸 수(기록 스텝 단위)이지 ms 가 아니다.**
A3 가 표에 적을 때 이 문장이 같이 가야 한다.

**P5 는 이식했다** — 아래 절 참고.

---

# P5 — R3 을 16D 기록의 파지 구간으로 (이식 완료)

`benchmark/ag3s/experiments/live/sweep_attached_threshold.py` 를 다시 썼다.

| 무엇이 | 전 | 후 |
|---|---|---|
| 씬 | `TransportScene(settle_steps=400)` + seed 로 **새로 만듦** | **기록의 실제 파지 구간** (`--records` 필수) |
| 운반 | target 점을 합성으로 50·100·200 mm 들어올림 | 로봇이 실제로 들고 움직인 프레임들 |
| `source` | `fresh_mujoco_scene` / `used_saved_run: false` | `recorded_grasp` / `used_saved_run: true` |
| 후보 | `(0, .5, 1, 1.5, 2, 3)` — 3.0 이 상한 | `(0, .5, 1, 1.5, 2, 3, 4, 5, 6, 8)` |
| 상한 경고 | 없음 | `chosen_is_top_of_list` 를 산출물에 박고 stdout 에 경고 |
| 쓸 프레임 | — | **세어서 정한다** (`--min-points`, 기본 8). 범위를 박지 않는다 |

산출물에 `n_grasp_frames` · `frames`(프레임별 카메라별 사과 픽셀 수) · `frames_skipped` ·
`sample_note` 가 실린다 — **표본이 작다는 것이 숫자 옆에 같이 간다.**

## 돌려 본 결과 (기능 확인. 측정은 A2 의 것)

```
문턱  0.0 복셀  잔여음수     0  공유      0  교정    226  최악    +1.20 mm
문턱  1.5 복셀  잔여음수    34  공유     71  교정    155  최악   -16.83 mm
문턱  3.0 복셀  잔여음수    34  공유    119  교정    107  최악   -16.83 mm
문턱  5.0 복셀  잔여음수    34  공유    135  교정     91  최악   -16.83 mm
문턱  8.0 복셀  잔여음수    34  공유    145  교정     81  최악   -16.83 mm
고른 값: 0.0 복셀 (통과 후보 [0.0], 표본 10 프레임)
```

고른 값이 후보 목록의 **아래쪽 끝**이라 상한 잘림은 아니다. 다만 0.0 은 *"부호를 항상 고친다"*
쪽 끝이라 공유 판정이 0 건이 되는 값이다 — 머리말의 *"0 이면 판정이 무력해진 것 — 숨김 위험"*
경고에 정확히 걸린다. **이 기록·이 표본에서 통과하는 문턱이 0.0 하나뿐**이라는 것이 결과다.

## 표본 수가 코디네이터 수치와 다르다 — **릴리스 감지가 없다**

지시받은 것은 6 프레임(32~35, 44~45)인데 내 실행은 **10** 을 썼다: 32~35 와 **44~49**.
원인은 내 단계 파생이 **파지 시작만 잡고 끝을 안 잡기 때문**이다 —
`phase_boundaries_from_record` 는 그리퍼가 열린 값에서 벗어나는 지점을 찾을 뿐이고,
`phase_for` 는 그 뒤 **기록 끝까지 전부 `grasp`** 으로 라벨한다. 46~49 는 파지가 유지되지
않는 구간인데 들어왔다.

산출물의 프레임별 카메라 픽셀 수는 이렇다 (내 실행에서 그대로 찍힌다):

```
[46] t=368 점 139   [47] t=376 점 113   [48] t=384 점 298   [49] t=392 점 307
     — 세 대 중 wrist_cam_l 만 0 이 아니다 (zed_left·wrist_cam_r 는 전부 0)
[36]~[43] 점 0 — 건너뜀 (그리퍼가 완전히 가림)
```

**이것을 내가 임의로 잘라 6 으로 맞추지 않았다.** 파지 종료를 무엇으로 정의할지가 판정이기
때문이다(그리퍼가 다시 열리는 지점인가, 물체가 목적지에 놓이는 지점인가). A0 이 정해 주면
`phase_boundaries_from_record` 에 릴리스 감지를 넣고 `--min-points` 와 함께 표본을 확정한다.
**지금 산출물의 표본은 10 이고, 그 사실이 JSON 에 그대로 적혀 있다.**

## A0 에게 — 남은 판정

1. **P5** (R3 이 기록을 먹게) — 지시대로 멈춰 있다. `--thresholds` 상한도 함께.
2. **P8 의 고정 구 개수가 14D 와 다르다** — 14D 기록은 27 개, 여기서는 42 개다. 움직일 수
   있는 구의 최소 여유는 거의 같은데(185.6 → 중앙 185.1) 가른 집합의 크기가 다르다.
   16D 에서 `arm_6` 이 자유 관절이 되었으니 고정 구는 **줄어야** 맞는데 늘었다.
   쥔 물체 점을 grounding 이 아니라 MuJoCo 분할에서 가져온 차이일 수도 있고, 프레임이
   다른 탓일 수도 있다. **내가 판정할 것이 아니라 A2 가 재야 한다** — 내 숫자는 스크립트가
   도는지 본 부수물이다.
3. **파지 후반에 쥔 물체가 안 보인다** — 호출 36 부터 세 카메라 모두 사과 픽셀이 0 이라
   건너뛴다(그리퍼가 완전히 가린다). 쓸 수 있는 파지 프레임은 32~35 뿐이다. 더 필요하면
   분할 대신 MuJoCo body 자세에서 기하를 직접 만드는 쪽으로 바꿔야 한다.

---

# 부록 — 1 차 예측과 A2 실측의 대조

1 차에서 낸 "깨질 곳 목록" 과 `R.smoke.verify.json` 을 맞춘 결과. **예측이 어긋난 곳이
가장 흥미롭다**는 배분표의 주문에 대한 답이다.

| 예측 | 실측 | 맞았나 |
|---|---|---|
| `ACTION_WIDTH` 14→16 은 7 개 중 어느 것도 안 깨뜨린다 | 폭 때문에 깨진 스크립트 0 건 | ✅ |
| 16D 기록에 depth 없음 (`depth_cameras: []`) | `no_depth_stored: true` | ✅ |
| venv 셋, `.venv-openpi-live` 가 유일한 cuRobo+mujoco | 코디네이터가 실측 확인, CLAUDE.md 수정됨 | ✅ |
| FRESH-1: R1·R3 은 기록을 안 읽는다 | 둘 다 `targets_16d_record: false`, `used_saved_run: false` | ✅ |
| SCRIPT-0: N1·F20 을 재는 스크립트가 없다 | `not_measured` 에 그대로 | ✅ |
| ATT-1: `attention_16d_long.npz` 가 맞는 파일 | A2 가 못 찾아 합성으로 돌렸고 그 수치는 무효 | ✅ (전달이 늦었다) |
| R7 단위가 33 배 어긋남 | 코디네이터 확인, 이번에 고침 | ✅ |
| **ENV-1: R1 은 venv 때문에 깨진다** | **틀렸다.** `.venv-openpi-live` 에서도 깨졌고, 원인은 **T0 불변식**(`pipeline.py:906`)이 한 프로세스에 두 backend 를 금지하는 것이었다 | ❌ |

**틀린 하나가 이 STEP 의 본체가 됐다.** 내가 venv 표를 만들고 "cuRobo+mujoco 가 같이 있는
venv 가 있다" 까지 확인한 뒤 *"그러면 돈다"* 로 넘어갔는데, **스크립트가 한 프로세스에서
backend 를 두 번 바꿔 부른다는 것**(옛 `:128` `for backend in ("legacy", "curobo")`)을
읽고도 불변식과 연결하지 못했다. 환경을 맞추는 것과 설계가 허용하는 것은 다른 질문이고,
나는 앞의 것만 봤다. P1 의 프로세스 분리가 그 답이다.

부차적으로 어긋난 것 둘:

* **PHASE-1·DENOM-1·CELL-1 을 A2 가 `runs` 로 분류할 것**이라고 했고 그대로 됐다 — 죽지
  않기 때문이다. 다만 내가 "가장 조심할 곳" 이라고 적은 것에 비해 1 차에서는 그것이
  분류에 반영되지 않았고, 대신 **`--frames-npz` 기본값이 14D 라서 혼합 실행이 됐다**는
  더 나쁜 형태로 드러났다(A2 가 `targets_16d_record_note` 에 정확히 적었다).
* **`needs-outpath` 라는 분류가 새로 생겼다.** 내 목록에는 "출력 경로가 박혀 있다" 가
  `FIG-1`(R7) 하나뿐이었는데, R4 에서 실제로 **archive 증거 파일을 덮어쓰는** 사고가 났다.
  출력 경로를 "고칠 곳" 이 아니라 "불편한 점" 으로 본 것이 내 판단 착오다.
