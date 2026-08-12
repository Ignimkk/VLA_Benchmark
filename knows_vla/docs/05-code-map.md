# Phase 7 — Paper ↔ Code Mapping

## 0. 매핑 대상의 대체

**KNOWS의 공식 구현은 존재하지 않는다.** arXiv abs·프로젝트 페이지·웹 검색 모두 링크 없음
([00-evidence.md](00-evidence.md) §1). 따라서 통상적인 "논문 ↔ 저자 코드" 대조가 불가능하며,
다음 세 축으로 대체한다.

| 축 | 저장소 / 문헌 | 고정 리비전 | 이 논문과의 관계 |
|---|---|---|---|
| A. 벤치마크 · 베이스라인 | [THU-RCSCT/vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis) (MIT) | `main` @ **57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b** (2026-06-30) | SafeLIBERO 정의, Naive 베이스라인의 원 설계 [9] |
| B. 정책 스택 | 로컬 [src/openpi](../../../src/openpi) | 워크스페이스 사본 | π0.5, attention 슬라이스 지점, 액션 규약 |
| C. CBF 수식 원출처 | Wu & Liu, arXiv:**2505.20847** (IROS 2025) | v2 | Eq. (5)–(6), (10)의 원형. **공개 코드 없음** |

> **신뢰도 경고**: A의 고정 커밋은 2026-06-30로 **KNOWS 제출일(2026-06-08) 이후**다. 저자가
> 실제로 참조한 상태가 아닐 수 있다. `workflow.md` Phase 0 원칙에 따라 약한 증거로 취급하고,
> 필요하면 2026-06-08 이전 커밋으로 되짚어야 한다.

---

## 1. 매핑 표

Found?: FOUND / PARTIAL / MISSING — 해당 요소를 어딘가에서 확인했는가.
Discrepancy: MISSING이 아닐 때만 의미 있음.

| Paper Element | 의미 | 대응 위치 | Found? | Discrepancy | 비고 |
|---|---|---|---|---|---|
| §3.1 frozen $\pi_\theta$ = π0.5 | 블랙박스 정책 | [pi0.py](../../../src/openpi/src/openpi/models/pi0.py), `pi05_libero` | FOUND | Different hyperparameter | 논문 $H{=}8$ vs 로컬 `action_horizon=10` |
| §3.1 액션 = EEF delta $(\Delta x,\Delta\theta)\in\mathbb{R}^6$ + gripper | 액션 의미 | [libero_policy.py:99](../../../src/openpi/src/openpi/policies/libero_policy.py#L99) `[...,:7]` | FOUND | Unclear | 차원은 일치. **프레임·회전표현 논문 미기재** → OPEN-Q 2 |
| §3.3 attention 그리드 $A_t\in\mathbb{R}^{g\times g}$ | 타깃 신호 | [gemma.py:228](../../../src/openpi/src/openpi/models/gemma.py#L228) `probs` | FOUND | — | $g{=}16$ 확인 (So400m/14, 224²) |
| §3.4 layer 12, head 3 | 헤드 선택 | `gemma_2b`: depth 18, heads 8 | FOUND | — | 인덱스 범위 정확히 일치 |
| §3.4 agent view | 뷰 선택 | `base_0_rgb` = prefix 토큰 [0:256] | FOUND | — | [02-architecture.md](02-architecture.md) §3.4 |
| 부록 §7.2 hook + 수동 재계산 | fused kernel 우회 | 해당 없음 | **MISSING (의도적)** | Different implementation | openpi는 unfused → 불필요. **Layer B 어댑터** |
| Eq. (2) attention mass | 마스크 커버리지 가중 | 없음 | MISSING | — | 신규 구현 |
| Eq. (3) density, $\beta{=}-1$ | 면적 정규화 | 없음 | MISSING | — | 신규 구현. $K$ 미상 |
| Eq. (4) gap 판정 $\delta$ | 타깃 확정 | 없음 | MISSING | — | 신규 구현. $\delta$ 미상 |
| Eq. (5) $h_R, h_O$ | 분리 초평면 CBF | [36] Eq. 22a/22b (형태 상이) | PARTIAL | Different implementation | [36]은 $\lambda,\mu$ 매개변수화, KNOWS는 $\sqrt{n^\top Qn}$ 지지함수 형태. 동치 |
| Eq. (6) $h = h_R + h_O$ | $\gamma$ 소거 결합 | [36]에 **없음** | **PARTIAL** | **Paper-only** | [36]은 두 조건을 개별 유지(Eq. 27b–c)하고 합치지 않음. **KNOWS 고유 변경** — [03-math.md](03-math.md) 참조 |
| Eq. (7) 이산시간 CBF | $\Delta h \ge -\gamma_h h$ | [15] Agrawal & Sreenath | FOUND | — | 표준 형태. $\gamma_h$ 미상 |
| Eq. (10) 회전 gradient | $n\times Q_Rn / \sqrt{\cdot}$ | [36] Eq. 26a/26b | FOUND | Different implementation | [36]의 $(\!R^\top n)^\wedge{}^\top Q\,\partial h/\partial\mu$ — skew 연산자가 외적에 대응. 구조 일치 |
| Eq. (12) QP | OSQP 볼록 QP | [39] OSQP | FOUND | — | 솔버 옵션(tol, max_iter, warm start 설정) **전부 미기재** |
| 가상 normal $n$ | 스텝 간 유지 상태 | [36] Eq. 19 $\dot n = (I-nn^\top)\eta$ | PARTIAL | Different implementation | [36]은 연속시간 투영 동역학, KNOWS는 박스 제약 $\|\delta n\|_\infty\le\epsilon$ + 사후 재정규화 |
| §3.2 MVEE | 타원체 적합 | [35] Khachiyan–Todd | FOUND | — | 표준. 수렴 허용오차 미기재 |
| §3.2 HSV Bhattacharyya | identity swap 복구 | 없음 | MISSING | — | 신규 구현. 임계값 미기재 |
| §4.1 YOLOe 파인튜닝 | 세그멘테이션 | [34] YOLOe 공개, **파인튜닝 가중치 미공개** | PARTIAL | — | 학습 레시피 전무 → OPEN-Q 9 |
| §4.1 SafeLIBERO Level I/II | 벤치마크 | vlsa-aegis @ 57b1aef | FOUND | — | 공개. 태스크·장애물 배치 대조 필요 |
| §4.1 SafeLIBERO **Level III** | 동적 장애물 | **없음** | **MISSING** | — | 저자 확장, 미공개. 재구현 필요 |
| §4.1 Naive 베이스라인 | init-only 필터 | [9] 설계의 **대리 구현** | PARTIAL | Different implementation | 논문이 "stand-in"이라 명시. GT 세그멘테이션 + 특권 상태 사용 |
| §4.1 스텝 예산 300/550 | 롤아웃 길이 | openpi 예제는 220–520 + wait 10 | PARTIAL | Different hyperparameter | 저자는 SafeLIBERO 하네스 사용 추정 [DERIVED] |
| §4.1 `replan_steps` | 실행 horizon | [main.py:29](../../../src/openpi/examples/libero/main.py#L29) `=5` | PARTIAL | Unclear | **논문은 이 값을 밝히지 않음** → OPEN-Q 12 |
| Table 2 지연 | 20 Hz 예산 | 해당 없음 | MISSING | — | 하드웨어·추출 경로 상이 → UNVERIFIABLE |

---

## 2. 불일치 분류 요약

### Paper-only (논문에만 있고 어디에도 구현 근거 없음)
- Eq. (6)의 $\gamma$ 소거 결합 barrier — [36]과의 실질적 차이. **KNOWS의 실제 기여 중 하나**임에도
  논문 본문은 이를 §3.5 한 문장으로만 처리하고 부록에 밀어 두었다.
- Eq. (2)–(4) 타깃 식별 전체.
- SafeLIBERO Level III.

### Different implementation (같은 목적, 다른 방식)
- 부록 §7.2 attention 추출: 논문은 fused kernel 우회, 우리는 직접 슬라이스. **의도적 Layer B 변경**.
- 가상 normal 갱신: [36] 연속시간 투영 동역학 vs KNOWS 박스 제약 + 재정규화.

### Different hyperparameter
- $H$: 논문 8 vs 로컬 체크포인트 10.
- 에피소드 스텝 예산.

### Unclear (재현을 막는 것)
- 실행 horizon $K_{\mathrm{exec}}$ — 논문에 없음.
- 액션 프레임·회전 표현.
- $K, \delta, \gamma_h, W, \epsilon$ 다섯 개 — [00-evidence.md](00-evidence.md) §2.
- OSQP 솔버 옵션.

### Likely engineering detail (재현 결과를 좌우하지 않을 가능성이 높음)
- MVEE 수렴 허용오차.
- HSV 히스토그램 bin 수.

---

## 3. 대조가 필요한 후속 작업

1. **vlsa-aegis에서 실제로 확인할 것** (아직 클론하지 않음):
   - SafeLIBERO Level I/II의 태스크 목록·장애물 스폰 규칙이 논문 §4.1 서술(4 suite, level당
     50 에피소드)과 일치하는가
   - CR이 에피소드 단위 이진 플래그로 계산되는가 (논문 정의와 일치 확인 — `reproduction.md` §6)
   - [9]의 원 CBF 레이어가 어떤 기하 표현을 쓰는가 (타원체? 구? SDF?) — Naive 대리 구현의
     충실도를 판정하려면 필요
   - 2026-06-08 이전 커밋 존재 여부

2. **[36] 원문 PDF 정독** — Eq. 26의 $\partial h/\partial\mu$를 KNOWS Eq. (10)–(11)로
   환원하는 과정을 손으로 검산. 회전 gradient의 부호 규약이 우리 프레임 선택과 맞는지 확인.

3. **openpi 실측** — [02-architecture.md](02-architecture.md) §3.4의 슬라이스 인덱스
   (prefix 968, agent view [0:256], `probs` 축 `[B,1,8,T,S]`)를 실제 텐서로 검증.
   Phase 8 프로브에 포함.
