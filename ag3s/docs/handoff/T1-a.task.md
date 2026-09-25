# T1-a — 16D 모델의 attention 셀 선택을 정한다

> writer: lead (A0) · 2026-09-25 · 사용자 판정으로 R 앞에 끼워 넣음

## 왜 이것이 먼저인가

R-port 중에 **의존이 반대였다는 것**이 드러났다. R4(C5 — 거친 20 mm 계층이 판정 지점에서
clearance 를 넓게 답한다)와 R5(C3 — 미세 계층의 창 경계에서 거리장이 낙관적으로 불연속이다)는
**미세 계층이 있어야 잴 수 있는데, 미세 계층이 하나도 안 붙는다.**

```
target grounding 실패 ──▶ 미세 계층 0 개 ──▶ R4·R5 잴 대상 없음
```

실측: `outputs/verify/R/rollout_fields_16d.npz` 의 키가 `coarse 45 / fine 0`.
그리고 R2 를 7 번 돌린 **모든** 실행에서 `target` 이 **0/15** 다.

**실측 attention 을 붙여도 아무것도 안 바뀐다** — legacy 는 합성이든 실측이든 `feasible 12`,
cuRobo 는 둘 다 `7`. 셀 선택이 잘못되어 attention 이 grounding 까지 **도달하지 못하기** 때문이다.

순환이 아니다. **T1 의 첫 과제인 셀 선택은 R 의 수치를 하나도 쓰지 않는다.** 거기서 끊는다.

## 무엇을 재나

14D 모델에서 정해진 셀은 **`L8H2`** 다. 16D 에서도 성립하는지가 열린 물음이고,
**추측으로 닫지 않는다** (로그 `AG3S_T0T6_LOG.md:490-497`).

탐색 공간은 저장된 npz 가 그대로 담고 있다:

```
attention (프레임, 카메라 3, denoise 3, layer 18, head 8, agg 3, 16, 16)
                                └──────── 3 × 18 × 8 × 3 = 1296 조합 ────────┘
```

| 축 | 크기 | 뜻 |
|---|---|---|
| layer × head | 18 × 8 | 트랜스포머의 어느 층 · 어느 head 의 attention 인가 |
| denoise step | 3 | π0.5 는 동작을 여러 번 걸쳐 다듬는다. **그 몇 번째 단계**인가 |
| aggregation | 3 | 여러 suffix token 의 attention 을 **어떻게 합치나** |

자산 둘:
- `benchmark/ag3s/asset/data/attention_16d_ep1800.npz` — 15 프레임, 회귀 기준선 기록용
- `benchmark/ag3s/asset/data/attention_16d_long.npz` — 50 프레임, 파지 포함 긴 기록용

## A2 가 할 것

### 1 차 — 전수 sweep (MuJoCo 실행 없음, 순수 numpy 로 되는 범위)

1296 조합 × 프레임에 대해 **grounding 성공률**을 낸다. `layer`/`head` 는 `ag3s/config.py:37` 이
*ablation axes* 라고 적어 둔 대로 config 로 들어간다 — 거기서 갈아 끼운다.

낼 것:
- 조합별 `has_target` 성공 프레임 수 (전수 1296 행).
- **14D 의 `L8H2` 가 16D 에서 몇 프레임 성공하나.**
- 16D 최선 조합과 그 성공률. 동률이면 전부.
- **live 서버가 실제로 고르는 셀** (`trajopt/serve_safe.py:143` 의 `attention_extractor`)과
  sweep 최선이 같은가. T0 에서 live 는 성공했으므로 **여기 답이 있을 가능성이 높다.**

두 npz 모두에 대해 돈다. 긴 기록 쪽이 파지를 포함하므로 단계별로 갈리는지도 본다.

### 2 차 — 재현성 원인 추적 (사용자 판정)

같은 명령 5 회에서 legacy `feasible` 이 **12·12·7·12·12** 로 갈렸다. SQP 반복 **중앙값은
다섯 다 2** 라서 "반복 수가 줄어서" 만으로는 설명이 안 된다. **새로 돌리기 전에 이미 있는
자료부터 본다:**

`outputs/verify/R/R2_repeat_{1,2,3}.json` 과 `R2_legacy_attn.json`·`R2_legacy_control.json` 의
`frames` 를 프레임 단위로 대조해라.

- **어느 프레임이 뒤집히나.** 5 회 내내 같은 프레임인가, 매번 다른가.
- 뒤집힌 프레임의 `clearance_after_mm` 가 0 에서 얼마나 가까운가. **`feasible`/`violated` 는
  0 을 기준으로 한 이진 딱지일 뿐**이라 +0.5 mm 와 −0.5 mm 가 갈린다 — 실제로 흔들린 것이
  판정인지 mm 인지를 가려라.
- 그 프레임들의 `iterations`·`to_ms`·`ag3s_ms` 가 실행마다 어떻게 다른가.

**결론을 내지 말고 수치만 내라.** 원인 후보가 좁혀지면 그때 A1 이 코드를 본다.

## 산출물

- `benchmark/ag3s/docs/handoff/T1-a.verify.json`
- figure 는 **`benchmark/ag3s/docs/figures/t1-a/`** 아래 (규칙 A 3 종):
  실제 씬(최선 셀의 attention 을 점군에 얹은 것) · 그래프(1296 조합 성공률 분포) ·
  표(14D `L8H2` 대 16D 최선 대조).
- raw 는 `outputs/verify/T1-a/`.

## 이 STEP 이 답하는 질문

**14D 에서 정한 `L8H2` 가 16D 에서도 grounding 을 세우는가. 아니면 어느 셀이 세우는가.**
그리고 **오프라인이 실패하고 live 가 성공하는 차이가 정말 셀 선택인가.**
