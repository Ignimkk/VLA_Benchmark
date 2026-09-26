# T8b (A1 · 구현) — 잡을 대상은 필드에서 빼고, 넣을 대상은 남긴다

**사용자 판정이다. 설계를 다시 묻지 말고 이대로 만들어라.** 단, 기본값은 지금 동작 그대로
두고 flag 로 켠다 — T8a 의 측정이 아직 안 끝났다.

너의 T7b 작업(커밋 안 된 6 파일)은 그대로 살려라. 그 위에 얹는다. 커밋하지 않는다
(push 는 사용자가 한다).

## 규칙 (사용자 말 그대로)

1. target 은 attention 이 정한다. 초기 step 에서 사과가 target 이다.
2. **사과를 집을 때까지 target 인 사과는 필드에서 제외한다.** 필드에 있으면 충돌 때문에
   못 잡는다.
3. **잡은 뒤에는** 사과가 gripper 에 포함되어 로봇 link 로 작용한다. 이때 target 은 crate 인데
   **crate 는 필드에서 빼지 않는다** — 그 안에 넣어야 하므로 충돌을 회피하며 내부 공간에
   들어가야 한다.
4. 3 은 이미 구현되어 있다 — `curobo_builder._build_tiers(itg, attached, ...)` 가 쥔 물체를
   필드에서 빼고, `to_adapter` 가 `attached.points` 를 로봇 쪽 질의점으로 넣는다
   (E3 — 쥔 물체가 optimizer 에 도달하지 않던 문제의 수정). **건드리지 마라.**
   crate 도 이미 맞다 — `manipulated_object` 가 쥔 뒤엔 attached 를 돌려주므로 crate 는
   `manipulated` 가 아니고, `destination_label`/`destination_margin` 으로만 다뤄진다.
5. **그래서 새로 만들 것은 "쥐기 전 target 의 제외" 하나다.**

## 왜 margin 완화로는 안 되는가 (T7a 실측, 다시 시험하지 마라)

성공한 shadow 궤적은 손끝 구가 사과 표면을 **−17.96 mm 관통**한다. `margin = 0`
(GRASP phase 의 `margin_scale = 0.0`) 조차 `d − r ≥ 0` 을 요구하므로 **음수 여유가
필요하고 margin 은 음수가 될 수 없다.** closed-loop 은 `max_violation_m` 0.0 이 75/75,
feasible 75/75 인데도 사과에 +89.86 mm 보다 가까워진 적이 없다 — 위반이 아니라
**feasible set 안에 grasp 가 없는 것**이다.

## 함정 — 이것 때문에 "행을 끈다" 로 끝내면 안 된다

ESDF 의 한 행은 `(step, sphere)` 당 하나이고 **최근접 표면까지의 거리 하나**뿐이다.
그 행을 끄면 그 질의점은 **table 에 대해서도** 보호를 잃는다. cuRobo 의
`enable_obstacle(name, False)` 는 거리 질의가 **다음 장애물**을 답하게 만들므로 그렇지 않다.
ESDF 에서 같은 의미를 얻는 방법은 **target 없이 만든 layer 를 한 겹 더 두고, 권한 있는
link 만 그 layer 에 묻는 것**이다.

## 만들 것

### (a) `ConstraintConfig.target_field_policy`

| 값 | 뜻 | 비고 |
|---|---|---|
| `"relax"` | 지금 동작 — `manipulated_link_margin` 으로 margin 만 완화 | **기본값** |
| `"exclude_authorized"` | 권한 있는 link 에 대해 target 을 제외 | 사용자 판정 |
| `"exclude_all"` | 모든 link 에 대해 target 을 제외 | 사용자 fallback (아래 5 번) |

`"exclude_all"` 은 사용자가 *"만약에 이렇게 해서 안되면 그냥 target은 필드에서 빼세요"*
라고 한 것이다. **flag 하나로 가야 한다** — 다시 구현하는 일이 되면 안 된다.
E1(조작 대상을 필드에서 통째로 파내면 손끝뿐 아니라 몸통·전완·반대팔에게도 사라진다)이
`"exclude_all"` 에서 되살아난다는 것을 docstring 과 시작 로그에 **크게** 적어라.
`"exclude_authorized"` 는 익명이 아니라 이름으로 빼므로 E1 이 재발하지 않는다.

### (b) target 을 label 로 싣는다

`pipeline.py:959` 는 지금 `{DESTINATION_LABEL: destination_points}` **하나만** 싣는다.
`TARGET_LABEL` 을 `types.py` 에 만들고(`DESTINATION_LABEL` 옆, 같은 규약) grounding 의
target 점을 함께 실어라. **쥔 것이 없을 때만** 싣는다 — 쥔 뒤에는 target 이 crate 이고
crate 는 빠지면 안 되기 때문이다(규칙 3). 이 조건을 코드에 한 줄로 적어라.

### (c) target 없는 fine layer

`curobo_builder` 는 이미 쥔 물체를 뺀 tier 를 만드는 길을 갖고 있다
(`_build_tiers(itg, attached, ...)`). **같은 길로 target 을 뺀 fine tier 를 한 겹 더** 만들고
필드가 그것을 함께 들고 나오게 하라. **coarse 는 만들지 마라** — 권한 있는 link 은 손 근처에
있고, 두 겹을 다 만들면 frame 당 비용이 두 배가 된다.
비용을 시작 로그에 ms 로 찍어라.

### (d) `linearize._esdf_clearance` 가 그 layer 를 쓴다

`"exclude_authorized"` 일 때, 권한 있는 link 의 질의점은 **target 없는 layer** 에 거리를
묻는다. 권한 없는 link 은 지금 그대로 target 이 든 layer 에 묻는다. 그러면 손가락은
사과를 통과할 수 있으면서 **table 과 crate 에 대한 보호는 유지한다.**
`"relax"` 는 지금 경로를 한 줄도 바꾸지 않아야 한다.

권한 집합은 `ClearancePolicy.authorized_links` 다. **T8a(A2)가 어느 link 이 실제로 필요한지
재고 있다** — 그 결과가 오기 전에는 지금 집합을 그대로 두고, 집합을 **config 에서 바꿀 수
있게만** 해 두어라. T7b 가 손목 `link_*_arm_6` 을 지목했으니 거기에 들어갈 가능성이 높다.

### (e) 서버 flag

`serve_safe.py` 에 `--target-field-policy relax|exclude-authorized|exclude-all` 을 더하고,
`relax` 가 아니면 시작 로그에 **어느 정책으로 떠 있는지 크게** 찍어라 — 조용히 안전 계층을
느슨하게 한 채 떠 있는 것이 최악이다(네가 T7b 에서 쓴 것과 같은 이유).

## 지키는 것

- **자기 필터(218 구 전신)는 건드리지 않는다.**
- 기본값은 셋 다 지금 그대로. 테스트로 고정하라.
- 기존 테스트 파일을 고치지 않는다. 새 파일에 쓴다.
- 포트 8000 서버, `outputs/verify/**`, `figures/**`, `docs/*.md` 는 건드리지 않는다.
- 끝나면 `T8b.impl.md` 로 넘겨라. 무엇을 **측정하지 않았는지** 반드시 적어라.
