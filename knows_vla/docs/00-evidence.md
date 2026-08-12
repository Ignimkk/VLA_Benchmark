# Phase 0 — Evidence Inventory

**논문**: *Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action
Models* — Seongbin Park, Fan Zhang, Baharan Mirzasoleiman, Shahriar Talebi, Nader Sehatbakhsh
(University of California Los Angeles).

**버전**: arXiv:2606.09749**v1**, 제출 2026-06-08, comments "Under review". [PAPER]
분석 시점(2026-08-11) 기준 v2 없음.

**약어**: KNOWS = **K**nowledge-driven, **N**o-retraining, **O**nline **W**rapper for **S**afety
(각주 1, p.2). [PAPER]

---

## 1. 증거 표

| 증거 | 가용 | 위치 / 버전 | 신뢰도 |
|---|---|---|---|
| 본문 | O | `papers/KNOWS.pdf` pp.1–8 | 높음 |
| 부록 | O | 동 pp.13–14 (§7.1 CBF-QP, §7.2 attention 추출) | 높음 — §2 결함 참조 |
| 참고문헌 | O | 동 pp.9–12 (41건) | 높음 |
| 프로젝트 페이지 | **X** | 없음 | — |
| **공식 코드** | **X** | arXiv abs·웹 검색 모두 링크 없음 | — |
| 저자 체크포인트 | **X** | 배포 없음 | — |
| π0.5 체크포인트 | O (대체) | 로컬 `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero` | 중간 — §4 참조 |
| SafeLIBERO Level I/II | O | [THU-RCSCT/vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis), MIT. HF 데이터셋 [THURCSCT/SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO) | 높음 — 커밋 해시 미고정(TODO) |
| **SafeLIBERO Level III** | **X** | 저자가 추가한 동적 장애물 변형. 미공개 | — |
| 원 벤치마크 LIBERO | O | 논문 [16], 공개 | 높음 |
| 저장된 관측 | O | `data/policy_records/` 50MB, 스텝별 `.npy` | 중간 — 출처 확인 필요 |
| LIBERO / robosuite 런타임 | **X** | 이 워크스테이션의 어느 venv에도 미설치 | — |

### 논문이 의존하는 외부 저작물

| 항목 | 논문 참조 | 상태 |
|---|---|---|
| SafeLIBERO 벤치마크 + Naive 베이스라인의 원 설계 | [9] VLSA/AEGIS, Hu et al., arXiv:2512.11891 | 코드 공개 |
| 분리 초평면 타원체 CBF (Eq. 5–6의 원 출처) | [36] Wu & Liu, IROS 2025, "eq. 26의 d=3 형태" | 원문 대조 필요 |
| 이산시간 CBF 조건 (Eq. 7) | [15] Agrawal & Sreenath, RSS 2017 | 공개 |
| 세그멘테이션 모델 | [34] YOLOe (ICCV 2025), 저자가 파인튜닝 | 파인튜닝 가중치 미공개 |
| MVEE | [35] Khachiyan & Todd 1990 | 표준 알고리즘 |
| QP 솔버 | [39] OSQP | 공개 |
| 헤드 선택 선행연구 | [32] Kang et al. CVPR 2025 (VLM localization heads), [33] Jeong et al. (VLA navigation heads) | 공개 |

---

## 2. 결함 1 — 부록 참조가 끊겨 있음

본문 §3.3 마지막 문장 [PAPER]:

> "The values of $K, \beta,$ and $\delta$ used for evaluation in section 4 were determined
> empirically; details are in the Appendix."

그러나 부록은 §7.1(CBF-QP 유도)과 §7.2(attention 추출 방법)뿐이며 **어떤 하이퍼파라미터 값도
없다**. 논문 전체에서 수치로 복구되는 것은 $\beta$ 하나뿐이다.

| 기호 | 역할 | 출처 | 값 |
|---|---|---|---|
| $\beta$ | Eq. (3) density의 면적 정규화 지수 | §3.3 본문 "Normalizing by area ($\beta{=}-1$)" | **−1** [PAPER] |
| layer, head | 타깃 신호 추출 위치 | §3.4 "We select the top-scoring unit (layer 12, head 3)" | **layer 12, head 3** [PAPER] |
| 카메라 | 신호 추출 뷰 | §3.4 + Fig. 3 "agent camera, layer 12, head 3" | **agent view** [PAPER] |
| $K$ | Eq. (3) attention 누적 sliding window 프레임 수 | 부록 약속, 부재 | **미상** [ASSUMPTION] |
| $\delta$ | Eq. (4) top-1/top-2 density gap 임계 | 부록 약속, 부재 | **미상** [ASSUMPTION] |
| $\gamma_h$ | Eq. (7) 이산시간 CBF 감쇠율 | $\gamma_h \in (0,1]$ 범위만 명시 | **미상** [ASSUMPTION] |
| $W$ | Eq. (12) 회전 대 병진 추종 가중 | 정의만 명시 | **미상** [ASSUMPTION] |
| $\epsilon$ | Eq. (12) 가상 normal 스텝당 변화 상한 | 정의만 명시 | **미상** [ASSUMPTION] |
| $d_{\text{safe}}$ | 안전 여유 | 명시적 여유항 없음 — Eq. (6)의 $h \ge 0$ 자체가 접촉 경계 | 해당 없음 [DERIVED] |

공식 코드가 없으므로 이 5개는 **재현자가 스윕으로 정해야 한다**. $\delta$와 $\gamma_h$는 각각
"타깃 오분류 시 전 객체를 장애물로 취급하는 빈도"와 "필터 보수성"을 직접 지배하므로 Table 1의
SR/CR 트레이드오프를 사실상 결정한다.

## 3. 결함 2 — 헤드라인 "43%"의 단위

초록·§1·Fig. 1(b)의 "reduces collision rate by up to / more than 43%"는 **상대 감소가 아니라
절대 퍼센트포인트**다. Table 1 Level III, Naive → KNOWS의 CR: [DERIVED]

| Suite | Naive CR | KNOWS CR | 절대차(pt) | 상대 감소 |
|---|---:|---:|---:|---:|
| SPATIAL | 62.0 | 29.0 | 33.0 | 53.2% |
| OBJECT | 50.0 | 14.0 | 36.0 | 72.0% |
| GOAL | 90.5 | 30.5 | 60.0 | 66.3% |
| LONG | 80.5 | 34.0 | 46.5 | 57.8% |
| **평균** | 70.75 | 26.875 | **43.875** | 62.0% |

즉 "43%"는 43.9 **퍼센트포인트**다. 재현 시 PASS 판정 기준을 어느 쪽으로 두느냐로 결론이
갈리므로 `reproduction.md` §6 metric fidelity 항목으로 관리한다.

## 4. 결함 3 — 체크포인트 프로파일 불일치

| 양 | 논문 | 로컬 `pi05_libero` | 근거 |
|---|---|---|---|
| $H$ (action horizon) | **8** | **10** | 논문 Table 2 캡션 "action chunking ($H{=}8$) amortizes the policy forward to $\sim$30 ms/step" (243/8 = 30.4) [PAPER] vs [config.py:745](../../../src/openpi/src/openpi/training/config.py#L745) `action_horizon=10` [CODE] |
| $K$ (실행 스텝) | 미상 | **5** | [main.py:29](../../../src/openpi/examples/libero/main.py#L29) `replan_steps=5` [CODE] |
| 에피소드 스텝 예산 | 300 / 550 (LONG) | 220–520 + wait 10 | 논문 §4.1 [PAPER] vs openpi 예제 [CODE] |

스텝 예산이 다르다는 것은 저자가 **openpi의 LIBERO 예제 하네스가 아니라 SafeLIBERO(vlsa-aegis)
자체 하네스를 썼음**을 시사한다 [DERIVED]. 저자가 어떤 π0.5 체크포인트를 썼는지는 불명이며,
$H{=}8$은 공개된 `pi05_libero`로 재현 불가하다.

SEAM에서 이미 동종의 제약을 겪었다 — [reproduction_notes.md](../../seam_vla/docs/reproduction_notes.md)
참조. 대응 원칙도 동일하게 간다: **체크포인트를 논문에 맞추려 조용히 바꾸지 않고, 같은
체크포인트 위에서 baseline 대 KNOWS를 비교**한다.

---

## 5. 우리 스택에 유리한 사실 (코드로 확인)

### 5.1 openpi의 attention은 fused가 아니다 — 부록 §7.2가 불필요

논문 부록 §7.2는 FlashAttention이 $T \times T$ 행렬을 materialize하지 않기 때문에 hook으로
hidden state를 캐싱하고 RoPE를 재적용하고 GQA key head를 확장해 수동으로
$\mathrm{softmax}(Q_\text{act}K_\text{vis}^\top/\sqrt{d})$를 재계산한다 [PAPER].

openpi의 JAX Gemma는 확률을 **명시적으로 만든다** [CODE]:

```python
# src/openpi/src/openpi/models/gemma.py:217-230
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
```

따라서 §7.2의 **재계산**(RoPE 재적용, GQA key 확장, 수동 softmax)은 불필요하다. 단, 논문의
"0.8 ms attention 추출" 수치는 이 경로에 대한 것이 아니므로 그대로 비교할 수 없다.

> **정정 (P0a로 확인)**: 최초 분석에서 "부록 §7.2가 통째로 불필요"라고 적었으나 이는 부정확했다.
> `probs`가 materialize되는 것과 **접근 가능한 것은 별개**다. `Attention.__call__`은 `probs`를
> 반환하지 않고, 18개 레이어는 `nn.scan` + `nn.remat`으로 스캔되어 레이어를 모듈 경로로 지정할
> 수도 없다. 따라서 **꺼내는 경로를 직접 내야 한다**. 다만 그 작업이 논문의 재계산보다 훨씬
> 작다는 결론은 유지된다 — 실제로 openpi 3개 파일에 36줄 추가로 끝났다.
> [07-p0a-results.md](07-p0a-results.md) 참조.

이는 `reproduction.md` §11의 **Layer B(임베디먼트 어댑터)** 변경이다 — 수학적 결과는 동일하고
논문의 방법을 바꾸지 않는다.

### 5.2 layer 12 / head 3이 그대로 주소 지정 가능

[gemma.py:79–96](../../../src/openpi/src/openpi/models/gemma.py#L79) `gemma_2b` [CODE]:
`depth=18` (layer 0–17), `num_heads=8` (head 0–7), `num_kv_heads=1` (GQA).

논문 Fig. 2의 x축은 0–17, 헤드 컬러바는 0–7 [PAPER] — **정확히 일치**한다. `probs`의 축은
`BKGTS`이고 $K{=}1$이므로 head 3 = G축 인덱스 3이다 [DERIVED].

### 5.3 vision token 그리드 $g$ 확정

SigLIP 변형 `So400m/**14**` ([pi0.py:85](../../../src/openpi/src/openpi/models/pi0.py#L85)),
입력 해상도 $224\times224$ ([model.py:47](../../../src/openpi/src/openpi/models/model.py#L47)) [CODE]
→ $224/14 = 16$ → **$g = 16$, $g^2 = 256$ 토큰/카메라** [DERIVED].

논문의 $A_t \in \mathbb{R}^{g \times g}$와 정합한다.

### 5.4 agent view 키 블록의 위치

[libero_policy.py:57–70](../../../src/openpi/src/openpi/policies/libero_policy.py#L57) [CODE]:
이미지 dict 순서는 `base_0_rgb`(= agent view) → `left_wrist_0_rgb` → `right_wrist_0_rgb`.
`right_wrist_0_rgb`는 zero 배열이고 π0/π0.5에서 `image_mask=False`지만 **토큰 자체는 시퀀스에
존재**하고 attention mask로만 배제된다.

→ agent view 키 블록 = prefix 토큰 인덱스 **[0:256]** [DERIVED]. Phase 8 프로브에서 실측 검증 대상.

### 5.5 주입 지점의 전례

[pi0.py:218–282](../../../src/openpi/src/openpi/models/pi0.py#L218) `sample_actions`는 SEAM이 추가한
`guidance_fn` post-Euler 훅을 이미 갖고 있다 [CODE]. attention 반환도 동일 패턴으로 jitted
sampler 시그니처를 확장하면 된다 — 새 메커니즘이 필요 없다.

---

## 6. 실행 환경

| 항목 | 로컬 | 평가용 |
|---|---|---|
| GPU | RTX 4060 Ti **8 GB** — π0.5 추론 불가([reproduction_notes.md](../../seam_vla/docs/reproduction_notes.md)) | **H200** 서버 (사용자 확인) |
| RAM | 62 GB | 미확인 |
| 논문 하드웨어 | — | RTX PRO 6000 96GB ×2, Xeon w5-3425, ~755 GB RAM [PAPER] |

로컬은 문서 작성과 CPU 프로브 전용, 폐루프 평가는 H200에서 수행한다. H200 서버의 저장소·
체크포인트·LIBERO 배치는 미확인 → `OPEN-QUESTIONS.md` 7번.

---

## 7. 재현 가능성 결론

`reproduction.md` §1 기준 달성 가능 상한:

| 대상 | 상한 | 이유 |
|---|---|---|
| 구조 재구성 (Phase 1–7) | **L0** 달성 가능 | 논문 본문+부록으로 충분 |
| 핵심 알고리즘 (attention 타깃 식별, CBF-QP) | **L1** 달성 가능 | 로컬 체크포인트 + 저장 관측 + 합성 타원체로 검증 가능 |
| SafeLIBERO Level I/II | **L2–L3** 달성 가능 | 벤치마크 공개. 단 $H$ 불일치와 미상 하이퍼파라미터로 절대 수치 일치는 기대 불가 |
| SafeLIBERO Level III | **L1–L2** | 벤치마크 자체를 우리가 재구현해야 함 |
| Table 2 (지연) | **UNVERIFIABLE** | 하드웨어 상이 + 추출 경로 상이(§5.1) |
| Fig. 2 (헤드 선택 스윕) | **L1** 달성 가능 | 절차가 §3.4에 기술됨. 단 "ground-truth ellipsoid safety filter"로 프로파일링했다는 전제를 우리는 갖추기 어려움 |

**공식 코드가 없다는 것이 이 논문의 지배적 제약이다.** 미상 하이퍼파라미터 5개를 코드에서
확인할 방법이 없으므로, 모든 정량 재현은 스윕 위에서만 의미를 갖는다.
