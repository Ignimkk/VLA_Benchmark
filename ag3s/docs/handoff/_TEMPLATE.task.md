# <STEP> — <한 줄 제목>

> writer: lead (A0) · 읽는 쪽: implementer, verifier

## 왜 지금 이것인가
<전체 파이프라인에서 어디인가. 앞 STEP 에서 무엇이 닫혔고 이것이 무엇으로 이어지나 (규칙 B)>

```
카메라 depth → 로봇 마스크 → [attention lifting → target grounding] → TSDF/ESDF
            → 거리장 어댑터 → SQP 선형화 → QP 해
```

## 구현에 요청하는 것 (→ implementer)
- <무엇을 고치나. 관련 파일:줄>
- 손대지 않을 것: <범위를 좁혀 둔다>

## 측정에 요청하는 것 (→ verifier)
| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | | | |

- 회귀 기준선을 먼저 돌린다 (skill: `regression-baseline`).
- figure 3종 중 최소 <n>종.

## 이 STEP 이 답하는 질문
<한 문장. 이것에 답하지 못하면 STEP 이 안 끝난 것이다>

## 되돌아올 지점
<고르지 않은 선택지와, 그리로 돌아갈 전환 신호>
