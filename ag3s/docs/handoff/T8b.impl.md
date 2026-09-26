# T8b — 구현

> writer: ag3s-implementer (A1) · 2026-09-26 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

**아직 쥐지 않은 target(사과)을 거리장에서 빼는 길을 만들었다.** 필드를 파내는 것이 아니라
**사과 seed 를 지운 미세 계층 한 겹을 더 만들고**, 권한 있는 link 의 질의점만 그 계층에 거리를
묻게 했다. 본 계층(coarse+fine)은 한 복셀도 바뀌지 않으므로 팔뚝·몸통·반대팔은 예전과 똑같은
답을 받는다 — 익명으로 파내는 것과 이름으로 되묻는 것의 차이가 정확히 E1 이다.

**행을 끄지 않는다.** ESDF 의 한 행은 최근접 표면까지의 거리 하나뿐이므로 끄면 그 질의점은
table 에 대해서도 보호를 잃는다. target 없는 계층은 table·crate 를 그대로 담고 있고, 그것이
계층을 한 겹 더 두는 구조를 고른 이유 전부다.

**기본값은 지금 동작 그대로다** (`target_field_policy="relax"`). 정책을 켜지 않으면 계층은
0 겹, 마스크는 `None`, 프레임 비용은 0, 시작 로그는 한 줄도 다르지 않다. 켜는 것은
`--target-field-policy exclude-authorized|exclude-all` 하나이고, 켜면 시작 로그가 크게 말한다.

**쥔 뒤에는 이 정책이 돌지 않는다** (규칙 3). 그때 target 은 crate 이고 crate 는 그 안에
넣어야 하므로 필드에서 빠지면 안 된다. 조건은 코드에 한 줄이다 — `attached is None`.

---

## 마진이 아니라 `d` 를 바꾼 이유 (다시 시험하지 않았다)

제약은 `d − r ≥ m` 이고 `m ≥ 0` 이다. 성공한 shadow 궤적의 손끝 구는 사과 표면을
**−17.96 mm 관통**하므로 `margin = 0`(GRASP 의 `margin_scale = 0.0`)에서도 그 자세는
**feasible set 안에 없다** — 위반이 아니라 해가 없는 것이다 (T7a 실측: `max_violation_m`
0.0 이 75/75, feasible 75/75 인데도 사과에 +89.86 mm 보다 가까워진 적이 없다). 그래서
`m` 을 내리는 길(`manipulated_link_margin`)은 그대로 두고 **`d` 를 다른 계층에서 받는 길**을
새로 냈다. 두 길은 같은 권한 집합(`contact.contact_links`)을 쓰며, 테스트가 그것을 박는다.

---

## 세 값의 합성 — 여기가 이 구현의 핵심이고, 설계를 한 번 고쳤다

`target_free_distance(p)` 는 창 안에서 `min(f, max(d, b))` 다.

| 기호 | 뜻 |
|---|---|
| `d` | 본 계층(target 이 든) 합성 답 — 지금 동작 |
| `f` | target 없는 계층의 답 (창 안에서만 정의) |
| `b` | 그 창의 **경계까지의 거리** |

창 밖은 `d` 그대로다 (= 지금 동작, 보수적).

**왜 `b` 가 필요한가.** `f` 는 창 안 표면만 안다. 창 **밖** 표면은 정의상 `b` 이상 멀고
동시에 `d` 이상 멀다(`d` 는 target 까지 포함한 최소값이므로). 그래서 창 밖 표면까지의 거리는
`max(d, b)` 이상이고, 위 식은 참값의 **하한** — 안전한 쪽으로 틀린다.

**그리고 이 식은 지금 동작보다 더 보수적이 되지 않는다.** `f ≥ d` 이므로
`min(f, max(d, b)) ≥ d` 다. 즉 **이 경로가 켜져서 없던 위반이 생기는 일은 없다** — 창 경계에
유령 장애물이 서지 않는다. (`b` 를 그냥 쓰면 그 일이 생긴다. 그래서 `max(d, b)` 다.)

### 고친 것 — 라벨로 가르는 첫 구현은 파지하는 자리에서 틀렸다

처음에는 라벨 층으로 갈랐다 ("가장 가까운 표면이 target 인 점만 되묻는다"). **단위 시험에서
바로 틀렸다**: 사과 밑면 안쪽(테이블 위 4 mm — 정확히 파지하는 자리)에서 20 mm 거친 계층의
라벨이 "테이블" 이라고 답하는데 값은 사과에서 왔다. 값은 삼선형 보간이고 라벨은 최근접 격자점
조회라, 두 표면이 만나는 자리에서 둘이 어긋난다. 그 결과가 "완화가 조용히 꺼진다" 였다.

그래서 **판정을 라벨에서 기하로 옮겼다.** 지금 합성은 라벨을 한 번도 읽지 않고, 테스트가
"라벨 없이 만든 같은 씬이 같은 값을 낸다" 로 그것을 고정한다.

**`TARGET_LABEL` 은 그래도 만들어 싣는다** (지시서 (b)). 쓰임이 판정에서 진단으로 바뀌었다 —
"이 질의점의 최근접 표면이 사과인가" 를 기록에서 되짚는 데 쓴다. **T8a 가 묻는 질문이 바로
그것**이므로 버리지 않았다.

---

## 세 정책

| 값 | 누가 target 없는 계층에 묻나 | 비고 |
|---|---|---|
| `"relax"` | 아무도 | **기본값 · 지금 동작** |
| `"exclude_authorized"` | `contact.contact_links` 의 link 의 구만 | 사용자 판정 |
| `"exclude_all"` | 제약 구 전부 | 사용자 fallback — **E1 이 되살아난다** |

`"exclude_all"` 에서는 몸통·전완·반대팔에게도 사과가 사라진다 (사과 위로 팔꿈치가 지나가도
아무도 막지 않는다). 그 사실을 `ConstraintConfig` docstring · `ConstraintBuilder` 생성자 경고 ·
서버 시작 로그 세 곳에 **E1 을 이름으로 부르며** 적었다. `"exclude_authorized"` 는 익명이 아니라
이름으로 빼므로 E1 이 재발하지 않는다.

**권한 집합은 `contact.contact_links` 하나다.** 코드에 두 번째 목록을 만들지 않았다 —
T8a 의 답(어느 link 이 실제로 필요한가)은 그 dict 한 줄로 들어온다. 테스트가 그 경로를 본다
(`link_1` 로 바꿔 주면 마스크가 그 link 의 구를 고른다).

---

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/types.py:101` | `TARGET_LABEL = SourceType.TARGET.value` | `DESTINATION_LABEL` 옆, 같은 규약. 라벨 이름의 정본이 하나여야 정책 축과 필드 라벨이 갈라지지 않는다 |
| `benchmark/ag3s/types.py:934` | `CollisionConstraintSet.target_field_exclude` `(S,) bool` | 어느 구가 target 없는 계층에 묻는가. `relax` 면 **언제나 `None`** |
| `benchmark/ag3s/types.py:937` | `.target_field_policy: str = "relax"` | 기록에 남는다. 어느 정책으로 돈 프레임인지 모르는 기록은 비교할 수 없다 |
| `benchmark/ag3s/config.py:724` | `TARGET_FIELD_POLICIES` (세 이름의 정본) | parser 도 이 목록을 본다 — 두 곳에 박으면 CLI 가 config 에 없는 이름을 받는 날이 온다 |
| `benchmark/ag3s/config.py:786` | `ConstraintConfig.target_field_policy` + `excludes_target_from_field` + `validate` | 스위치의 정본. 모르는 이름은 config 단계에서 죽는다 |
| `benchmark/ag3s/fields/curobo_field.py:150` | `CuroboEsdfField.target_free_layers` · `target_label` | `layers` 에 **넣지 않는다** — `distance()` 는 `min` 합성이라 넣으면 모든 질의점에서 사과가 사라지고 그것이 E1 이다 |
| `benchmark/ag3s/fields/curobo_field.py:267` | `target_free_distance()` | 위의 `min(f, max(d, b))`. 없으면 `distance()` 와 같은 값 |
| `benchmark/ag3s/fields/curobo_field.py:317` | `_boundary_distance()` | 경계를 `_covers` 와 **같은 정의**로 잰다. 두 곳이 다르면 하한 논증이 깨진다 |
| `benchmark/ag3s/fields/curobo_builder.py:220` | `update(target_free_points=, target_free_label=)` | `exclude_target`(E1 로 폐기, 여전히 예외)과 **다른 인자**다. 파내지 않고 한 겹 더 만든다 |
| `benchmark/ag3s/fields/curobo_builder.py:308` | `_build_tiers(..., only="fine", tier_name="fine_no_target")` | 쥔 물체를 뺀 tier 를 만드는 **같은 길**로 target 을 뺀다. 부호 교정(`_force_sign_on_pure_voxels`)도 같은 집합에 걸려야 하므로 한 함수인 것이 맞다 |
| 〃 | 본 계층을 **다 만든 뒤에** 만든다 | `_site_index`·`feature_tensor` 가 재사용 버퍼다 (함정 4). 사이에 끼면 본 계층 값이 바뀐다 |
| 〃 | **coarse 는 만들지 않는다** | 권한 있는 link 은 손 근처(미세 창 안)에 있다. 두 겹이면 프레임 비용이 두 배 |
| `benchmark/ag3s/fields/curobo_builder.py:348` | `stats["target_free"]` (`n_layers`·`build_ms`·`n_seeds_excluded`·`label`) | **정책이 꺼져 있어도 키를 싣는다** — 계층이 있었는지 모르는 기록은 켜진 실행과 구별되지 않는다 |
| `benchmark/ag3s/fields/curobo_builder.py:384` | `_check_target_free_request()` — **`import torch` 앞** | 미세 계층이 없거나 target 이 없으면 정책이 아무 일도 못 하는데, 그 상태로 도는 것이 최악이다. torch 앞에 둔 이유는 torch 없는 `.venv-ag3s` 테스트가 이 계약을 실제로 받아 보게 하기 위함 (T7b 의 `announce_diagnostic_scope` 와 같은 이유) |
| `benchmark/ag3s/fields/curobo_builder.py:545` | `_announce_target_free_cost()` | **첫 프레임에 ms 를 크게 찍는다.** 안전 계층을 한 겹 비껴가는 길이므로 조용히 켜져 있으면 안 된다 |
| `benchmark/ag3s/runtime/pipeline.py:972` | `labelled[TARGET_LABEL] = target.points` — **`self._attached is None` 일 때만** | 쥔 뒤의 target 은 crate 이고 crate 는 빠지면 안 된다 (규칙 3) |
| `benchmark/ag3s/runtime/pipeline.py:975` | 정책이 요구할 때만 `target_free_points` 를 넘긴다 + legacy backend·단일 계층이면 **거절** | 기본에서 인자가 아예 없으므로 호출이 예전과 글자 그대로 같다. legacy 로는 계층을 못 만드는데 조용히 지나가면 정책을 켰다고 믿게 된다 |
| `benchmark/ag3s/constraints/to_adapter.py:210` | 마스크를 만드는 네 조건 (정책·쥔 것 없음·target grounded·필드가 계층을 들고 있음) | 하나라도 어긋나면 `None` = 지금 동작. 권한은 `clearance_policy.authorized_links(ctx)` — 마진 완화와 **같은** 집합 |
| `benchmark/ag3s/constraints/constraint_builder.py:179` | 생성자에서 정책을 **크게 경고** (권한 link 목록까지) | `serve_safe` 를 거치지 않는 호출자(실험 스크립트·테스트)도 반드시 듣는다 — 자기 충돌 경고와 같은 자리, 같은 이유 |
| `benchmark/trajopt/linearize.py:125` | `SceneSnapshot.target_free_mask` | `None` 이면 이 축이 없는 것과 같다 |
| `benchmark/trajopt/linearize.py:532` | `_esdf_clearance` 가 마스크 행만 `target_free_distance` 로 받는다 · 짧은 마스크는 `False` 로 채운다 · 능력 없는 필드면 **예외** | 쥔 물체의 점은 권한 대상이 아니다(그 점들이 곧 물체다) → fail closed. 조용한 no-op 금지 |
| `benchmark/trajopt/linearize.py:1004` | `scene_from_constraint_set` 이 마스크를 싣고 길이를 검사 | 길이가 다르면 AG3S 와 optimizer 가 다른 제약 모델을 보고 있다는 뜻 |
| `benchmark/trajopt/serve_safe.py:84,95` | `target_field_policy_choices()` · `resolve_target_field_policy()` | CLI 는 하이픈, config 는 밑줄. 목록의 정본은 config 쪽 하나 |
| `benchmark/trajopt/serve_safe.py:197` | `announce_diagnostic_scope` 가 정책을 크게 찍는다 (권한 link 목록 포함) | 조용히 안전 계층을 느슨하게 한 채 떠 있는 것이 최악이다 (T7b 와 같은 이유) |
| `benchmark/trajopt/serve_safe.py:348` | `constraint_section` 하나로 모아 `AG3SConfig` 에 넣는다 (기본이면 키 없음) | `self_collision` 과 같은 section 이다. 기본값을 여기 다시 적으면 스위치가 둘이 된다 |
| `benchmark/trajopt/serve_safe.py:554` | `--target-field-policy relax\|exclude-authorized\|exclude-all` | 기본 `relax` |
| `benchmark/trajopt/serve_safe.py:630,634` | `--no-safe` · `--esdf-backend legacy` · `--fine-voxel 0` 과의 조합을 **시작 전에 거절** | 첫 프레임에서 죽으면 그때는 체크포인트 두 벌이 이미 GPU 에 올라간 뒤다 |
| `benchmark/trajopt/serve_safe.py:721` | recorder meta 에 `target_field_policy` (기본이 아닐 때만) | target 이 빠진 기록을 안 빠진 것과 나란히 읽는 것이 이 flag 의 가장 나쁜 실패다 |
| `tests/trajopt/test_target_field_policy.py` | 새 파일, 47 test | 아래 |

**T7b 의 세 flag 는 손대지 않았다.** `--links gripper` · `--no-self-collision` ·
`--capsule-radius-scale-link` 는 그대로이고, 이 정책과 서로 독립이다 (같이 줘도 되고 따로 줘도
된다). 자기 필터 모델(218 구 전신)도 그대로다.

---

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**926 passed** = T7b 의 **879** + 새 파일 **47**. 실패 0 (17:24 측정). **기존 테스트 파일은
한 줄도 안 고쳤다** — 기본값을 안 건드렸다는 가장 직접적인 증거다.

> **17:25 이후 이 작업 트리는 내 것이 아닌 이유로 깨져 있다.** 다른 writer 가
> `benchmark/ag3s/robot_models/urdf_sphere_chain.py` 를 고치는 중이고, 그 판이
> `:566` 에서 `self.capsule_extent_by_link` 를 **`:577` 의 대입보다 먼저 읽는다**
> (`AttributeError: 'UrdfSphereChain' object has no attribute 'capsule_extent_by_link'`).
> 로봇 모델을 만드는 모든 테스트가 그 한 줄로 죽는다 — 내 새 파일에서 11 개, **T7b 의
> `test_gripper_only_constraints.py` 에서도 18 개**. 내 파일의 나머지 36 개는 그 상태에서도
> 통과하고, 실패 11 개의 사유는 **전부 그 AttributeError 하나**다 (`grep -E "^E "` 로 확인).
> **그 파일은 지금 다른 세션이 쓰는 중이므로 손대지 않았다** (규칙 D — 두 writer 가 같은
> 파일을 고치면 어느 쪽이 최신인지 모른다). 그쪽 편집이 끝나면 926 이 그대로 돌아온다.
> verifier 는 측정 전에 `pytest tests/trajopt/test_gripper_only_constraints.py -q` 가
> 통과하는지로 트리가 성한지 먼저 확인하기 바란다.

새 테스트가 지키는 것:

- **기본값** — `ConstraintConfig().target_field_policy == "relax"` · `excludes_target_from_field
  is False` · `CollisionConstraintSet.target_field_exclude is None` ·
  `SceneSnapshot.target_free_mask is None` · parser 기본값 · 계층 없는 필드의
  `target_free_distance` 가 `distance` 와 **같은 배열** · `None` 마스크와 전부-False 마스크가
  같은 답 · 기본 실행의 시작 로그가 **빈 문자열**.
- **계층** — 사과 안쪽에서 양수(통과 가능) · 사과 밑면에서 **테이블까지 4 mm 를 그대로 답한다**
  (행을 끈 것이 아니라는 수치적 내용) · 창 밖 표면(벽)이 `max(d, b)` 로 지켜진다 ·
  4,000 점 무작위로 `free ≥ plain` (없던 위반이 생기지 않는다) · 창 밖은 `distance` 와 동일 ·
  본 계층의 값·라벨이 **한 값도** 안 바뀐다 · 라벨 없이 만든 씬과 같은 답.
- **마스크** — 표시한 구만 움직이고 나머지 열은 `array_equal` · 그 구의 값이 '바닥까지의
  거리 − 반지름 − margin' 과 같다 (바닥이 그대로 막는다) · `exclude_all` 은 모든 행이
  단조 증가 · 쥔 물체 점 열은 안 움직인다(fail closed) · `full_violation` 이 이 축을 본다 ·
  능력 없는 필드 + 마스크 = **예외**.
- **배선** — primitive backend 에는 안 실린다 · esdf/both 에는 실린다 · 길이 불일치는 예외 ·
  이 축이 없는 옛 객체도 통과.
- **AG3S 결정** — `relax` 는 계층이 있어도 마스크를 안 만든다 · `exclude_authorized` 가
  **마진 완화와 같은 구**를 고른다 · 권한 집합을 config 로 바꾸면 마스크도 따라간다 ·
  권한이 비면 `None` · 쥔 뒤에는 `None` · target 없으면 `None` · 계층 없으면 `None` 이지만
  정책 이름은 기록에 남는다 · `ConstraintBuilder` 생성자가 E1 을 이름으로 외친다.
- **서버** — CLI 이름과 config 이름의 사전이 맞는다 · 밑줄 표기는 parser 가 거절 ·
  `--no-safe` · legacy backend · `--fine-voxel 0` 조합은 시작 전에 죽는다 · 권장 조합
  (`exclude-authorized --esdf-backend curobo`)은 통과 · 미세 계층 없는 builder 는 정책을
  받으면 죽는다.

---

## verifier 가 알아야 할 것

- **새 flag**: `--target-field-policy relax|exclude-authorized|exclude-all` 하나.
  **기본값 변경은 없다.** 안 주면 T7b 실행과 모든 수치가 같아야 한다.
- **권장 실행 인자** (사용자 판정을 켜는 최소 조합):
  `--esdf-backend curobo --target-field-policy exclude-authorized`.
  T7b 의 세 flag 와 독립이므로 원하면 함께 줄 수 있다.
- **기본 실행에서 딱 하나 달라지는 것 — 라벨이 하나 늘었다.** 쥔 것이 없는 프레임의 필드
  `stats` 에 `label_names` 로 `"target"` 이 들어가고 `n_labels` 가 1 → 2 가 된다
  (목적지가 없는 프레임이면 0 → 1). **거리값·마진·제약 행·위반 수치는 하나도 바뀌지 않는다** —
  라벨 층은 "가장 가까운 표면이 무엇인가" 만 답하고 `is_label(destination)` 의 답도 그대로다.
  회귀 기준선에서 이 두 키가 움직이는 것은 **예상된 것**이다.
  그리고 `stats["target_free"]` 키가 새로 생긴다 (정책이 꺼져 있으면 `n_layers: 0`,
  `build_ms: 0.0`).
- **재생산이 필요한 산출물**: 없음. TSDF/ESDF **생산** 경로(`.venv-curobo`)는 건드리지
  않았다. 새 계층은 같은 프로세스에서 기존 integrator 로 만들어진다.
- **옛 기록과 호환이 깨지는가**: 아니다. recorder meta 는 정책이 기본이 아닐 때만
  `target_field_policy` 키가 늘고, `CollisionConstraintSet` 의 새 두 필드는 기본값이 있다.
  옛 기록을 `scene_from_constraint_set` 에 다시 먹여도 `getattr` 기본값으로 지나간다
  (테스트가 그것을 본다).
- **내가 측정하지 않은 것** (전부 verifier 몫이다):
  - **cuRobo 로 계층을 실제로 만들어 보지 않았다.** `.venv-ag3s` 에는 torch 가 없어
    `CuroboFieldBuilder.update()` 를 돌릴 수 없다. `_tier_specs` 의 분기와 인자 검증은 시험이
    덮지만, **`fine_no_target` tier 의 실제 복셀값·seed 제외 수·`build_ms`** 는 GPU 에서
    처음 나온다. 시작 로그의 `!!! TARGET-FREE ESDF LAYER IS ON !!!` 한 줄과
    `stats["target_free"]` 가 그 첫 수치다.
  - **rollout 을 하나도 돌리지 않았다.** 사과가 움직이는지, 위반이 어떻게 되는지, 프레임
    예산이 어떻게 되는지 나는 재지 않았다.
  - `safe_policy.py:468` 의 진단용 clearance 는 **본 필드**(target 이 든)를 그대로 쓴다.
    일부러 안 고쳤다 — 기록에서 사과가 사라지면 무엇이 일어났는지 되짚을 수 없다. 그래서
    recorder 의 그 수치와 optimizer 가 본 수치가 권한 있는 구에서 **다를 수 있다**. 그것은
    버그가 아니고, 두 수치의 차이가 곧 정책이 준 여유다.
- **경고 하나 — 이 파일을 쓰는 동안 `tests/trajopt/test_target_field_policy.py` 가 17:16 에
  다른 writer 에게 한 번 덮였다.** 덮인 내용은 검증 안 된 초기 설계(라벨 판정)였고 실제로
  13 test 가 실패했다 (`TwoLinkArm` 오타, `Primitive(radius=)` 등). 내 검증된 판을 다시
  써 넣었고 (17:20:26) 그 뒤로는 안 움직였다. 지금 파일이 정본이다 — 47 passed.
  **source 파일 12 개에는 중복·충돌이 없다** (각 정의가 정확히 하나임을 grep 으로 확인).
  lead 는 T8b 에 나 말고 다른 writer 가 붙어 있는지 확인해 주기 바란다.

---

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.**

`exclude_authorized` 에서 손끝 구는 사과를 **관통할 수 있게** 된다. 그래서 두 갈림길이다.

1. 사과가 움직인다 → apple 0.0 mm 를 붙들고 있던 것은 **feasible set 에 grasp 가 없던 것**이
   맞았고, 남은 일은 권한 집합을 T8a 의 답으로 좁히는 것이다.
2. 여전히 0.0 mm → 원인은 제약 모델 밖이다 (refiner 의 실행 창 · grasp latch · shadow 와
   closed-loop 의 차이). T7b 의 `--links gripper` 로도 안 움직였다면 이쪽이 유력하고, 그때는
   "제약을 더 푸는" 방향을 멈춰야 한다 — 이미 손끝은 사과를 통과할 수 있는데도 안 간다는 뜻이다.

수치로 기대하는 것 하나: 권한 있는 구의 ESDF 여유거리는 `esdf_margin`(10 mm) + 구 반지름에
**테이블까지의 높이**로 결정된다. T7b 표대로 손가락 구는 배율 1.0 에서 최저 28.7 mm,
0.35 에서 18.8 mm 다. 사과 중심 높이 40 mm 보다 낮은 값이 처음으로 가능해지므로,
`--capsule-radius-scale-link gripper=0.35` 와 함께 주면 손끝이 사과 **아래 절반**까지 내려갈 수
있어야 한다. 그보다 훨씬 좋은 수가 나오면 무언가 다른 것이 함께 바뀐 것이므로 의심해야 한다.

비용은 미세 계층 한 겹이므로 **coarse+fine 을 만드는 비용의 절반 미만**이어야 한다 (실측 ESDF
0.44 ms 기준 1 ms 아래). `build_ms` 가 그보다 크게 나오면 `only="fine"` 이 안 듣고 두 겹이
만들어진 것이므로 `stats["target_free"]["n_layers"]` 를 먼저 봐야 한다.

---
---

# T8b (2부) — capsule 을 link 별로 **짧게 자르고**, 손바닥을 제약 모델에 넣는다

> writer: ag3s-implementer (A1) · 2026-09-26 18:0x · 1부 뒤에 사용자가 새로 내린 판정 둘.
> **1부의 판정(`target_field_policy`)은 한 줄도 바뀌지 않았다.** 여기 있는 것은 그 위에 얹는
> 별개의 축이고, 서로 독립이다.

## 무엇을 했나 (평이한 요약 먼저)

사용자 판정: **"팔꿈치에서 손가락까지 한 덩어리로 덮지 마라. 그리퍼의 2-finger 부분에 작은 구를
만들어라."** 그리고 **"그리퍼와 나머지 링크들의 capsule 반지름도 대폭 줄여라."**

그래서 **굵기와 다른 축을 하나 새로 냈다 — 길이다.** `capsule_extent_by_link` /
`--capsule-extent-link LINK=Z` 는 capsule 을 **link frame 의 z** 로 자르고, 잘린 구간에는 구를
하나도 놓지 않는다. T7b 의 굵기 손잡이(`capsule_radius_scale_by_link`)는 그대로 쓰되 대상이
바뀌었다 — **`arm_5` 는 길이로, 손가락은 굵기로** 다룬다.

그리고 자르는 순간 드러나는 구멍을 막았다: **손바닥 `ee_left`/`ee_right` 가 제약 모델에 아예
없었다.** `arm_5` 덩어리가 유일하게 덮고 있어서 티가 안 났다. `GRIPPER_LINKS` 에 넣었다.

## 왜 `arm_5` 는 굵기가 아니라 길이인가 (실측, 테스트가 박아 둔다)

| capsule | r | L | 축의 link-frame z | 표면이 닿는 z |
|---|---|---|---|---|
| `link_*_arm_5` | **75.0 mm** | **250 mm** | [−225.0, +25.0] | **−300.0** |
| `link_*_arm_3` | 40.0 mm | 115 mm | [−165.0, −50.0] | |
| `link_*_arm_4` | 35.0 mm | 95 mm | [−30.0, +65.0] | |

같은 frame 에서 손목(`link_*_arm_6`) 원점 **0.0** · `FT_sensor_*` **−108.7** · 손바닥(`ee_*`)
**−154.8** · 2 지 그리퍼 뿌리 **−227.8** mm 다. **capsule 하나가 넷을 전부 삼킨다.** 이웃한
`arm_3`·`arm_4` 는 40 / 35 mm 로 얇으므로 **튀는 것은 이 하나**이고, 튀는 이유는 굵기가 아니라
**이 팔에서 가장 긴 250 mm 가 손목 아래로 내려가 있다**는 것이다.

손가락은 반대다. 손가락 capsule 3 개의 축은 **link z 에 수직**이다 (실측 `R[2,2] = 0.000`,
축이 x/y 방향). z 로 자를 것이 없으므로 손가락은 **굵기**로 다룬다. 두 테스트가 이 두 사실을
고정한다 — 숫자가 바뀌면 판정의 전제가 바뀐 것이다.

## 기계 — 자르는 규칙 하나, 근사 없음

축의 z 는 `z(t) = p_z + R[2,2]·t`, `t ∈ [−L/2, +L/2]` 다.

| 경우 | 하는 일 |
|---|---|
| `R[2,2] ≠ 0` | z 구간을 t 구간으로 **정확히** 옮긴다 (`t = (z − p_z)/R[2,2]`), `[−L/2, L/2]` 와 교집합 |
| `R[2,2] = 0` (축이 z 평면에 누움) | z 값이 하나뿐이다 — 유지 구간 안이면 **통째로 남기고** 밖이면 **통째로 뺀다** |
| 교집합이 빈다 | 그 capsule 은 **구 0 개** |

남은 구간에는 **같은 간격 규칙**으로 구를 다시 놓는다 (`n = ceil(span/(spacing·r)) + 1`).
구를 골라 버리는 것이 아니라 다시 놓는 것이므로, 자른 뒤 `arm_5` 는 구 5 → **6 개**가 되고
유효 반지름은 81.25 → **19.53 mm** 가 된다 (배율 0.2 · spacing 0.4 · cap 32 에서).
**A2 의 T8c 표에 적힌 20.44 mm 와 0.9 mm 다르다** — 그쪽은 전체 길이로 놓은 사슬에서 구를
**빼는** 방식이었고, 여기는 남은 구간에 **다시 놓는다**. 둘 다 A2 의 `r_crit` 46.64 mm 아래이며,
다시 놓는 쪽이 더 촘촘하고 더 얇다.

## 자르는 것만으로는 표면이 안 물러난다 (읽는 사람이 꼭 알아야 할 것)

`z_cut = −100 mm` 하나만 주면 (배율 1.0):

| | 자르기 전 | 자른 뒤 |
|---|---|---|
| `arm_5` 구 수 | 5 | 3 |
| 표면이 닿는 z | **−306.2 mm** | **−181.3 mm** |

−181.3 mm 는 여전히 손바닥(−154.8)보다 **아래**다. 구 중심은 −100 에서 멈추지만 그 구가
자기 반지름(81.25 mm)만큼 부풀기 때문이다. 그래서 **길이와 굵기는 같이 걸어야 하고, 코드는 그
조합을 막지 않는다** (테스트가 한 link 에 둘을 동시에 거는 판을 본다). 배율 0.2 를 같이 주면
−119.5 mm 로 올라와 손바닥 위로 물러난다.

## 무엇이 풀려났는지 반드시 말한다

자르면 시작 로그가 **잘린 구간에 원점이 있는 하위 link 을 이름으로** 적고, 그 link 이 자기
구를 갖고 있는지까지 적는다 (관절값 0 기준 z — 구조적 질문이므로 자세에 무관하다).

`z_cut = −100` 에서 실제로 나가는 줄:

```
!!! CAPSULES ARE CUT SHORT (capsule_extent_by_link) !!!
    - link_left_arm_5: 축 z [-225.0, +25.0] → 유지 [-100.0, +inf] mm, 구 10 → 6 개,
      표면 도달 z -245.4 → -119.5 mm (+125.9 mm 물러남)
        풀려난 하위 link (자기 구 있음): ee_left, ee_finger_l1, ee_finger_l2
        **풀려났고 자기 구가 없다**: FT_sensor_L — 이 구간은 어떤 구에도 안 덮인다
```

**`FT_sensor_L`/`FT_sensor_R` 이 벌거벗는다.** 손목과 손바닥 사이의 force/torque sensor 로,
`UNCOVERED_LINKS` 에도 없고 MJCF alias 도 없어 gap-filling capsule 이 없다. 나는 그것을 **고치지
않았다** (지시 범위 밖이고, MJCF body 이름을 찾아 alias 를 더하는 일이다). 로그가 매 실행 크게
말하므로 lead 가 결정할 수 있다 — 두께 몇 cm 짜리 원통이고 손바닥 구가 바로 아래에 있으므로
실질 위험은 작지만, **"빼는 것이 위험을 지우지 않는다"** 는 이 프로젝트의 원칙 그대로다.

## 손바닥 — 넣는 것과 권한, 둘 다 필요하다

`ee_left`/`ee_right` 는 URDF 에 collision capsule 이 **없다.** geometry 는 MJCF mesh 실측에서
온다 (`gap_filling_capsules` 의 `EE_BODY_L`/`EE_BODY_R`, r 30.4 / 35.6 / 35.8 mm). self-filter 는
2026-09-25 부터 이미 그것을 쓰고 있었고 (손목 카메라가 자기 손바닥을 매 프레임 보는데
90,690 / 90,688 px 가 샜던 건), **제약 모델에만 없었다.**

- `GRIPPER_LINKS = PALM_LINKS + FINGER_LINKS` 로 넣었다. `--links gripper` 가 `--links arms` 의
  부분집합이라는 성질은 그대로다 (테스트가 `<` 로 본다).
- 구 수: `arms` **120 → 144**, `gripper` **44 → 68** (기본 굵기·간격에서).
- **권한(A2 T8e 의 지적)은 이미 되어 있었다.** `DEFAULT_CONTACT_LINKS` 가 예전부터
  `{"left": ("ee_left", "ee_finger_l1", "ee_finger_l2"), "right": (...)}` 이므로 손바닥은
  접촉 권한 집합에 있다. 지금까지 그 권한이 **아무 일도 안 하고 있었을 뿐**이다 — 제약 모델에
  손바닥 구가 없어서 걸릴 행이 없었다. 손바닥을 넣은 지금 1 부의 `exclude_authorized` 가
  손바닥 행에 그대로 걸린다. 테스트가 그 사실을 박는다 (그래서 새로 넣을 것이 없었다).

## `max_spheres_per_capsule` 이 `sphere_spacing` 을 무력화한다 (A2 T8e)

`r_eff = sqrt((r·scale)² + (간격/2)²)` 이므로 **배율만 내리면 `간격/2` 바닥에서 멈춘다.**
그런데 `sphere_spacing < 0.5` 에서는 기본 상한 8 이 구 수를 먼저 잘라 간격이 안 줄어든다.

그래서 상한에 잘린 capsule 을 기억하고 **시작 로그에 크게 경고**한다:

```
!!! sphere_spacing IS NOT TAKING EFFECT — max_spheres_per_capsule IS BINDING !!!
    N capsule(s) wanted more spheres than the cap (8); worst <link> wanted W, got 8.
    ... **얇게 만들었다고 믿는데 안 얇아진 상태다** — max_spheres_per_capsule 을 올리십시오.
```

테스트가 수치로 본다: `spacing 0.4 · scale 0.2` 에서 상한 8 일 때와 32 일 때 `arm_5` 반지름이
다르고, 32 쪽이 더 얇다. 두 flag 는 **T6f 부터 이미 노출돼 있었다** (`--sphere-spacing`,
`--max-spheres-per-capsule`) — 새로 만들 것은 경고뿐이었다.

## 구 개수와 행 개수를 갈라 찍는다

`announce_row_budget` 이 시작 로그에 한 줄 더한다.

```
row budget: 제약 구 266 개 → ESDF 질의 행 2128/청크 (8 스텝 × 266 구) ·
            QP 행 192/청크 (8 × rows_per_step 24, **구 개수와 무관**) · 자기 필터 구 218 개
```

A2 가 실측으로 확인한 것을 로그가 말하게 한 것이다 — 구를 120 → 218 로 늘려도 QP 행은 192 로
그대로이고 (활성 띠가 `horizon × rows_per_step`), 늘어나는 것은 질의점과 FK 비용
(0.509 → 0.690 ms) 뿐이다. 그 구분이 로그에 없으면 "구를 늘렸다" 가 "실시간을 잃었다" 로
읽힌다.

## launch 후보 판이 flag 만으로 만들어진다 (값은 박지 않았다)

```bash
--sphere-spacing 0.4 --capsule-radius-scale 0.2 --max-spheres-per-capsule 32 \
--capsule-extent-link link_left_arm_5=-0.10 link_right_arm_5=-0.10 \
--target-field-policy exclude-authorized --esdf-backend curobo --fine-voxel 0.005
```

내가 그 조합으로 `build_ag3s` 를 실제로 띄워 읽은 값 (rollout 아님, 체크포인트 없음):

| | 값 |
|---|---|
| 제약 구 | **266** (기본 120) |
| 자기 필터 구 | **218 그대로** |
| `link_*_arm_5` | 구 6 개 × **19.53 mm** (A2 `r_crit` 46.64 아래) |
| `ee_finger_*` | 구 19 개 × **4.72 mm** (A2 `r_crit` 7.75 아래, A2 값과 **일치**) |
| `ee_left`/`ee_right` | 구 23 개 × **8.92 mm** (A2 값과 **일치**) |
| `link_*_arm_6` | 구 29 개 × 7.5 mm |
| 절단 | 양팔 `arm_5`, 표면 −245.4 → **−119.5 mm** |

**값은 코드에 박지 않았다.** 기본값은 전부 지금 그대로이고 (`capsule_extent_by_link=None`,
`capsule_radius_scale=1.0`, `max_spheres_per_capsule=8`, `sphere_spacing=1.0`), 위 숫자는 전부
CLI 에서 온다.

## 바뀐 파일 (2부)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:41` | `CapsuleTrim` 기록 (유지 구간·구 수·**표면 도달 z**·풀려난 하위 link) | 굵기(`CoverageShortfall`)와 **다른 축**이다. 한 숫자에 섞으면 "얼마나 얇은가" 와 "어디까지 덮는가" 를 구분할 수 없다 |
| 〃`:145` | `_parse_extent()` — 숫자 하나(원위 절단면) 또는 `(z_min, z_max)` | 팔의 원위 방향이 −z 이므로 숫자 하나가 곧 절단면이다. 빈 구간·양쪽 무한은 거절 |
| 〃`:474` | `__init__(capsule_extent_by_link=None)` + 없는 이름·capsule 없는 link **거절** | 굵기 손잡이와 **같은 규율**. 조용히 넘기면 아무것도 안 잘린 채 잘랐다고 믿는다 |
| 〃`:711` | `_capsule_sphere_centres` 가 유지 구간에만 구를 놓는다 (`span` 으로 간격·덮개 계산) | 구를 빼는 것이 아니라 다시 놓는다 — 간격 규칙이 한 곳이어야 로그와 모델이 안 갈라진다 |
| 〃`:731` | `_spacing_capped` — 상한에 잘린 capsule 기억 | `sphere_spacing` 이 안 듣는 것을 조용히 지나가면 얇게 만들었다고 믿는데 안 얇아진다 |
| 〃`:766` | `_kept_axis_span()` — `R[2,2]=0` 은 통째로 남기거나 뺀다, 근사 없음 | 근사하면 어느 구가 남는지 아무도 예측할 수 없다 |
| 〃`:793` | `_record_trims()` · `_sphere_zs()` · `_descendant_origins_z()` | 무엇이 풀려났는지 **이름으로** 말한다. 자기 구가 없는 link 을 따로 표시한다 |
| 〃`:900` | `coverage_report()` 가 세 절(굵기 부족·절단·상한 포화)과 **link 별 유효 반지름**을 함께 낸다 | 결과를 한 자리에서 읽어야 `r_eff` 를 손으로 계산하지 않는다 |
| 〃`:597` | 생성자 경고가 절단·상한 포화에도 나간다 | 호출자가 여럿이다 (서버·실험·테스트) |
| `benchmark/ag3s/experiments/reports/grounding_report.py:88` | `PALM_LINKS` 신설 · `FINGER_LINKS` · `GRIPPER_LINKS = PALM_LINKS + FINGER_LINKS` | 이름 정본을 하나로. `arm_5` 를 자르면 손바닥이 벌거벗으므로 절단 손잡이와 같이 와야 한다 |
| `benchmark/trajopt/serve_safe.py:196` | `parse_link_extents()` — `LINK=Z` · `LINK=Zmin:Zmax`, **mm 로 적으면 거절** | `-100` (mm 의도)은 유지 구간 `[−100 m, inf]` 가 되어 **아무것도 안 잘린다**. 이 flag 에서 가장 있을 법한 실수다 |
| `benchmark/trajopt/serve_safe.py:245` | `announce_row_budget()` | 구 개수와 QP 행을 갈라 말한다 |
| `benchmark/trajopt/serve_safe.py:283` | `announce_diagnostic_scope` 가 배율·절단·**link 별 유효 반지름**을 한 warning 에 | 둘은 다른 축이지만 같은 link 에 함께 걸리는 것이 정상이다 |
| `benchmark/trajopt/serve_safe.py:361` | `sphere_options` 가 `capsule_extent_by_link` 를 싣는다 (빈 dict 면 키 없음) | 안 준 flag 는 키가 아예 없다 = 호출이 예전과 글자 그대로 같다. `--no-safe` 거절도 이 통로를 그대로 탄다 |
| `benchmark/trajopt/serve_safe.py:625` | `--capsule-extent-link LINK=Z` | 기본 `()` |
| `benchmark/trajopt/serve_safe.py:452` | `build_ag3s(plan_horizon_steps=, rows_per_step=)` → 행 예산 로그 | `main` 이 `to_config` 에서 읽어 넘긴다. 안 주면 예전처럼 구 수만 찍는다 |
| `tests/trajopt/test_capsule_extent_trim.py` | 새 파일, **37 test** | 아래 |

**self-filter 모델은 손대지 않았다.** `build_robot_model` 에 `sphere_options` 통로가 없다는 것을
테스트가 확인한다. 그 비대칭은 의도된 것이고 (거기서 가늘어지면 그 link 의 점이 필터를 통과해
장애물로 샌다), 편의를 위해 통로를 뚫지 않았다.

## 단위 검증 (2부 포함, 트리 전체)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**963 passed**, 실패 0 (18:0x 측정). 1 부가 적은 "17:25 이후 트리가 깨져 있다" 는
**해소됐다** — 그 `AttributeError` 는 이 2 부 작업이 반쯤 들어간 상태였고, 지금은
`test_gripper_only_constraints.py` 를 포함해 전부 통과한다.

### 기존 테스트 두 줄을 고쳤다 — 그 이유를 밝혀 둔다

1 부는 "기존 테스트 파일을 한 줄도 안 고쳤다" 고 적었고 그것은 1 부에서 사실이다. **2 부는
두 줄을 고쳤다**, 손바닥이 들어오면서 박아 둔 link 목록이 바뀌었기 때문이다.

| 파일 | 고친 것 |
|---|---|
| `tests/trajopt/test_exclude_constraint_links.py:61` | `ARM_LINKS` 글자 고정에 `ee_left, ee_right` 추가 + 이유 주석 |
| `tests/trajopt/test_gripper_only_constraints.py:106` | `GRIPPER_LINKS` 글자 고정에 손바닥 추가 + 이유 주석 |

**둘 다 "기본 집합이 흔들리면 걸리게" 만든 핀이고, 흔들린 것이 맞다** (사용자 판정으로).
핀의 목적은 변경을 막는 것이 아니라 **의식하지 못한 변경을 막는 것**이므로, 값을 갱신하고
갱신 이유를 그 자리에 적었다. 다른 테스트는 `GRIPPER_LINKS` 를 이름으로 참조하므로 자동으로
따라갔다.

새 테스트가 지키는 것:

- **전제(실측)** — `arm_5` capsule 이 r 75 · L 250 · 축 z [−225, +25] 이고 표면이 −300 까지
  가서 손목·FT·손바닥·그리퍼 뿌리를 전부 삼킨다 · 이웃 `arm_3`/`arm_4` 는 40/35 mm ·
  이 팔에서 가장 긴 capsule 이다 · 손가락 capsule 축은 link z 에 **수직**이다.
- **기본값** — 생성자 기본 `None` · 빈 dict 도 같은 모델 (`radii` `array_equal`) ·
  `capsule_trims == ()` · `covers every capsule` · parser 기본 `()` · `sphere_options == {}` ·
  기본 시작 로그는 **빈 문자열**.
- **자르면** — 구 수가 줄고 **다른 link 은 반지름·구 수가 그대로** · 남은 구 중심이 전부 유지
  구간 안 · 명시 구간은 그 폭만 남는다 · 빗나간 구간은 **구 0 개** · 표면이 물러나지만
  절단면까지는 못 간다(그 차이가 곧 유효 반지름) · 굵기와 길이를 한 link 에 같이 걸면 둘 다
  듣는다.
- **말한다** — 풀려난 하위 link 을 이름과 z 로 (실측 −154.8 mm 와 일치) · 손바닥을 빼면
  "자기 구가 없다" 로 · 생성자에서 WARNING.
- **거절** — 없는 이름 · 이 모델에 capsule 없는 이름 · `z_min ≥ z_max` · 숫자 아님 ·
  집합 이름 · **mm** · `--no-safe` 와의 조합.
- **손바닥** — `GRIPPER_LINKS == PALM_LINKS + FINGER_LINKS` · `gripper ⊂ arms` ·
  두 link 집합에서 손바닥 구가 **같다** · URDF 에는 없고 MJCF 에서 온다 ·
  `DEFAULT_CONTACT_LINKS` 에 손바닥과 손가락이 다 있다 · self-filter 에 `sphere_options`
  통로가 **없다**.
- **배율·간격** — 전역 배율이 손바닥 포함 **모든** 제약 link 에 걸린다(구 수 불변) ·
  두 간격 flag 가 서버에 노출돼 있다 · `spacing 0.4` 는 상한 8 에 걸려 경고가 나가고 32 로
  올리면 안 나간다 · 상한을 올리면 `arm_5` 반지름이 실제로 더 내려간다 ·
  행 예산 로그가 2128 과 192 를 갈라 찍는다.
- **launch 후보** — flag 만으로 그 판이 만들어지고, `arm_5` < 46.64 mm · 손가락 < 7.75 mm ·
  손바닥 구 존재 · 양팔 `arm_5` 가 잘렸다.

## verifier 가 알아야 할 것 (2부)

- **새 flag**: `--capsule-extent-link LINK=Z` 하나. 나머지(`--sphere-spacing` ·
  `--max-spheres-per-capsule` · `--capsule-radius-scale`)는 T6f/T7b 부터 있던 것이다.
- **기본값 변경 하나 — 제약 모델의 link 집합이 넓어졌다.** 손바닥이 들어와
  `arms` 120 → **144** 구, `gripper` 44 → **68** 구다. **이것은 flag 없이도 바뀐다.**
  회귀 기준선의 구 수·제약 행 수·최악 위반 link 이 이것 때문에 움직일 수 있다 — 특히
  **`ee_left` × apple 행이 새로 생긴다** (A2 T8e: 배율 1.0 에서 exec8 279 행 · −31.55 mm).
  1 부의 `exclude_authorized` 를 켜면 손바닥은 권한이 있으므로 그 행이 target 없는 계층으로
  간다. **정책을 끈 채 기준선을 다시 재면 손바닥 행이 새 위반으로 나타나는 것이 정상이다.**
- **자기 필터는 218 구 그대로다.** 어느 flag 로도 안 바뀐다.
- **재생산이 필요한 산출물**: 없음.
- **옛 기록과 호환**: recorder meta 는 준 flag 만 늘어난다 (`sphere_options` 안에
  `capsule_extent_by_link`). 기본 실행의 키 집합은 그대로다.
- **내가 측정하지 않은 것**: rollout 0 개. 위반·성공률·프레임 예산은 재지 않았다.
  `build_ag3s` 를 체크포인트 없이 띄워 시작 로그를 읽은 것이 전부다 (위 표의 반지름들).
- **`FT_sensor_L`/`FT_sensor_R` 이 절단 뒤 어떤 구에도 안 덮인다.** 고치지 않았고 로그가
  매 실행 말한다. lead 의 결정이 필요한 자리다.

## 내가 기대하는 결과 (2부)

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.**

`arm_5` 를 자르는 것이 굵기로 푸는 것보다 **덮개를 덜 잃는다**는 것이 A2 의 T8c 가 이미 보인
것이고, 나는 그 기계를 만들었을 뿐이다. 그래서 기대는 하나뿐이다 — **`link_*_arm_5` 가 최악
위반 link 목록에서 사라지고, 남는 것이 `ee_finger_*` 대 table 과 (배율을 안 내리면)
`ee_left` 대 apple 이어야 한다.** 다른 link 이 새로 올라오면 그것은 이 절단이 만든 것이 아니라
**손바닥이 새로 제약에 들어온 결과**이므로, `--links` 를 손바닥 없는 집합으로 한 번 더
돌려 가려야 한다.

전역 배율 0.2 · spacing 0.4 판에서는 `coverage_shortfall` 이 36/36 capsule 에 차고 최악
56.9 mm 다. **그 수치는 "로봇을 그만큼 못 덮는다" 는 뜻이고, 이 판은 진단용이지 배포용이 아니다.**
사과를 잡은 다음에는 반드시 되돌려야 한다 — 되돌리는 방법은 flag 를 빼는 것뿐이다.
