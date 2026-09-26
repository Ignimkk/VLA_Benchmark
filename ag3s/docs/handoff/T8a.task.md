# T8a (A2 · 검증) — target 을 빼면 성공 궤적이 feasible 해지는가

**MuJoCo 를 돌리지 않는다. 기록만 쓴다.** T7a 의 오프라인 재구성을 그대로 재사용하라
(`outputs/verify/T7a/s1_active_rows.py`, `s2_joint_axis.py`). 코드를 고치지 않는다.

## 왜 이걸 재는가

사용자 판정: **잡을 대상(사과)은 필드에서 빼고, 넣을 대상(crate)은 남긴다.** 그 판정이
실제로 성공 궤적을 feasible 하게 만드는지를 **rollout 없이** 확정하는 것이 이 task 다.
성공 궤적은 이미 있다 — shadow 실행의 `actions_reference` 다.

T7a 가 낸 것: shadow 는 손끝이 사과 표면을 **−17.96 mm** 지나고 `link_left_arm_5` 가
62/75 chunk 에서 `margin_slack ≤ 0` 이다. closed-loop 은 사과에 **+89.86 mm** 보다
가까워진 적이 없다. 이제 물을 것은 **"사과를 빼면 shadow 가 통과하는가"** 다.

## 측정 (셋)

**M1 — 사과에 대해 음수가 되는 link 은 누구인가.**
shadow 의 `actions_reference` chunk 전량에 대해, link 별로 **사과 geom 하나만**을 상대로
`clearance = surface_distance − r` 을 재라. margin **10 mm**(지금 서버 설정 `--esdf-margin 0.01`)
기준으로 `≤ 0` 인 chunk 수와 최소값(mm)을 link 별로 내라. margin 0 과 50 mm 도 같은 표에
같이 넣어라 — 세 값이 나란히 있어야 사용자가 고를 수 있다.
**이 결과가 authorized link 집합을 정한다.** T7b 가 손목 `link_*_arm_6`(구 25.9~29.2 mm,
margin 10 mm 를 더하면 35.9~39.2 mm 이고 사과 반지름이 38 mm)을 지목했는데, 그것이
측정으로 확인되는지 여기서 갈린다.

**M2 — 그 link 들에서 사과를 빼면 나머지가 통과하는가. 이것이 판정이다.**
M1 이 고른 link 집합에 대해 사과를 제외하고, shadow 궤적의 **남은 모든 행**을 margin 10 mm 로
검사하라. 통과하면 사용자의 설계가 충분하다는 뜻이다. 통과하지 않으면 **어느 (link, obstacle)
쌍이 얼마나 남는지** 전부 내라 — 특히 table 과 crate. 정수 개수와 mm 를 같이.

**M3 — 손가락이 사과를 빼앗기면 table 에 대해 얼마나 위험해지는가.**
ESDF 의 한 행은 최근접 표면 하나만 답하므로, 사과를 그 행에서 지우면 그 질의점은 table 에
대해서도 보호를 잃는다. 그래서 묻는다: M1 의 link 들이 사과를 상대로 음수인 그 순간,
**같은 질의점에서 table 표면까지의 거리**는 얼마인가 (chunk 별 최소, mm). 이 숫자가
"fine layer 를 target 없이 한 겹 더 만드는 것"이 필요한지 없는지를 정한다 —
table 여유가 넉넉하면 행을 끄는 싼 방법으로 충분하다.

## 내는 것

- `benchmark/ag3s/docs/handoff/T8a.verify.json` — 모든 수치는 `numbers` 안에만.
- figure: 실제 씬 + 그래프 + 표, `benchmark/ag3s/docs/figures/t8a/`.
  그래프는 **chunk 축 × link 별**, 사과만 / 사과 제외 / table 만 세 계열을 겹쳐라.
- raw 와 script 는 `outputs/verify/T8a/`.

## 규칙

- **해석을 쓰지 않는다.** 수치와 figure 만. 판정은 lead 와 사용자가 한다.
- 못 잰 것은 `not_measured` 에 이유와 함께 적는다. 지어내지 않는다.
- 단독으로 돌려라. 포트 8000 서버와 A1 의 작업 트리는 건드리지 않는다.
- regression baseline 을 다시 돌릴 필요는 없다 — T7a 가 이미 닫았다
  (HEAD 는 `planned_horizon` 8 이라 13/15 · +0.185978 mm 이고, `5c5291e^` 는 32 라
  14/15 · 0.15718632962849477 mm 로 기준값과 소수점까지 같다).
