# SEAM-VLA 구현 및 결과 리뷰

> 대상: `benchmark/seam_vla`  
> 목적: 교수님께 **왜 필요한 방법인지 → 어떤 수식으로 동작하는지 → 코드에 어떻게
> 연결했는지 → 현재 어떤 결과까지 확인했는지** 순서로 설명하기 위한 리뷰 자료  
> 현재 결과 재검증일: 2026-07-27

---

## 0. 먼저 전달할 핵심 결론

이 구현의 목적은 VLA가 action chunk를 새로 생성할 때 발생하는 **chunk 경계의 급격한 동작
변화**를 줄이는 것이다.

기존 π0.5는 한 번에 길이 `H`의 action chunk를 생성하지만, 실제로는 앞의 `K`개만 실행한 뒤
새 chunk를 다시 생성한다. 이때 이전 chunk의 실행되지 않은 tail과 새 chunk의 head가 서로
다르면 경계에서 속도 변화와 jerk가 커질 수 있다.

SEAM의 핵심인 VLS(Velocity-guided Loss Steering)는 모델을 다시 학습하지 않는다. 대신 π0.5의
flow-matching denoising 과정에서 매 Euler step 직후의 후보를 이전 chunk의 tail 방향으로 조금씩
보정한다.

현재 확인된 결과는 다음과 같이 세 단계로 구분해야 한다.

1. **구현 정확성:** baseline 경로와 SEAM-off 경로가 실제 π0.5 모델에서 완전히 동일하다.
2. **보정 작동 여부:** SEAM-on은 유한한 크기의 non-zero 보정을 만들며 NaN/Inf가 없다.
3. **오프라인 효과:** RB-Y1의 기존 114개 chunk 기록을 prompt별 두 run으로 분리하고
   decoded-stage 보조 방법을 사후 적용했을 때 BJ 2.4%, AVb 8.8%, overlap residual 1.7% 감소를
   확인했다. 반면 CD는 0.1% 증가하여 사실상 개선되지 않았다.

다만 LIBERO 전체 task success와 실제 로봇에서의 denoising-stage VLS A/B 결과는 아직 없다.
따라서 현재 결론은 **“핵심 방법이 구현·통합·검증되었고, 기록 데이터에서 smoothing 가능성을
확인했다”**까지이며, **“실제 task success를 유지하면서 로봇 움직임이 개선되었다”**고
결론 내리면 안 된다.

---

## 1. 문제 정의: 왜 action chunk 경계가 불연속적인가?

### 1.1 기존 action-chunk 실행

π0.5는 한 번의 inference로 다음과 같은 chunk를 생성한다.

\[
C_n=[a_{n,0},a_{n,1},\ldots,a_{n,H-1}]
\]

- `H`: 한 번에 예측하는 action 수
- `K`: 실제로 실행한 뒤 다시 계획하는 action 수
- `L=H-K`: 이전 예측에 남아 있는 미실행 tail 길이

실행기는 `C_n`의 앞 `K`개만 사용한다.

\[
a_{n,0},\ldots,a_{n,K-1}
\]

그 후 새로운 관측으로 다음 chunk `C_{n+1}`을 독립적으로 생성한다. 이전 chunk에는 이미
미래에 대한 예측 `C_n[K:H]`가 있었지만 baseline은 이것을 다음 추론에 사용하지 않는다.

### 1.2 경계 불연속의 발생

실제 실행 action stream은 다음과 같이 이어진다.

\[
\ldots,a_{n,K-2},a_{n,K-1},a_{n+1,0},a_{n+1,1},\ldots
\]

이때 `a_{n+1,0}`이 이전 계획의 다음 값 `a_{n,K}`와 크게 다르면, chunk가 바뀌는 시점에
action의 1차·2차 차분이 커진다.

직관적으로는 다음과 같다.

```text
이전 예측:  실행 구간 [0 ... K-1] | 미실행 tail [K ... H-1]
실제 실행: -----------------------> |
새 예측:                            | 새 head [0 ...]부터 다시 실행
                                     ↑
                            두 계획이 만나지 않으면 경계 jerk 발생
```

핵심 문제는 action chunk 자체가 아니라 **매번 새로 생성되는 chunk 사이에 연속성을 강제하는
정보가 없다는 것**이다.

---

## 2. 방법론: SEAM의 VLS는 무엇을 하는가?

## 2.1 핵심 아이디어

새 chunk를 생성할 때 이전 chunk의 미실행 tail을 “정답 action”으로 강제하지 않고,
**soft prior**로 사용한다.

이 방식은 다음 두 목표 사이의 균형을 잡는다.

- 새 관측과 언어 명령에 맞게 정책이 새 계획을 만들 수 있어야 한다.
- 새 계획의 시작 부분은 직전 계획의 tail과 지나치게 달라지지 않아야 한다.

따라서 chunk를 생성한 뒤 단순 평균하는 것이 아니라, flow-matching ODE가 action을 생성하는
중간 과정에서 작은 크기의 방향 보정을 반복한다.

## 2.2 Step 1 — 이전 chunk의 tail 추출

이전 model-space chunk를 `C_n ∈ R^(H×D)`라 하면, 앞의 `K`개는 이미 실행되었으므로 tail은
다음과 같다.

\[
T_n=C_n[K:H]\in\mathbb{R}^{L\times D},\qquad L=H-K
\]

중요한 점은 이 값이 물리 단위 action이 아니라 **정규화된 model space의 raw sampler
output**이라는 것이다. VLS가 작동하는 denoising state와 반드시 같은 좌표계를 써야 한다.

## 2.3 Step 2 — aligned prior 생성

sampler tensor의 shape `[B,H,D]`에 맞추기 위해 tail의 마지막 값을 반복해 길이 `H`로 확장한다.

\[
A_n=\operatorname{concat}
\left(T_n,\operatorname{repeat}(T_n[-1],K)\right)
\in\mathbb{R}^{H\times D}
\]

그러나 실제 guidance는 첫 `M`행에만 적용된다.

\[
M_{\mathrm{eff}}=\min(M,L)
\]

따라서 뒤에 반복한 `K`행은 shape을 맞추기 위한 padding이며 실제 보정 대상은 아니다.

## 2.4 Step 3 — π0.5의 Euler candidate 계산

π0.5 sampler는 noise에서 action으로 이동하는 flow-matching ODE를 Euler 방법으로 적분한다.
이 저장소의 시간 방향은 다음과 같다.

- `t=1`: Gaussian noise
- `t=0`: 완성된 action
- `N=10`: Euler step 수
- `dt=-1/N=-0.1`

기존 Euler update는 다음과 같다.

\[
x_{\mathrm{cand}}=x_t+\Delta t\,v_\theta(x_t,t,o)
\]

여기서 `o`는 image, language 등의 observation이고 `v_θ`는 π0.5가 예측한 velocity field다.

## 2.5 Step 4 — 시간에 맞춘 consistency target 계산

완성 action의 prior `A_n`을 denoising 중간 상태와 직접 비교하면 시간 좌표가 맞지 않는다.
따라서 현재 `t_next`에 맞추어 target을 다음처럼 만든다.

\[
r(t_{\mathrm{next}})=(1-t_{\mathrm{next}})A_n
\]

초기 noise 구간에서는 `1-t`가 작아 prior의 영향이 약하고, action이 완성되는 `t→0` 구간에서는
prior의 영향이 커진다.

## 2.6 Step 5 — consistency loss의 negative gradient 계산

후보가 target에서 얼마나 떨어져 있는지를 다음 loss로 생각한다.

\[
\mathcal{L}_{cons}
=\left\|x_{\mathrm{cand}}-r(t_{\mathrm{next}})\right\|_2^2
\]

후보를 loss가 감소하는 방향으로 옮기기 위한 closed-form negative gradient는 다음과 같다.

\[
g=-\nabla_{x_{\mathrm{cand}}}\mathcal{L}_{cons}
=-2\left(x_{\mathrm{cand}}-r(t_{\mathrm{next}})\right)
\]

이 식은 수동으로 계산할 수 있으므로 backpropagation이나 policy parameter gradient가 필요 없다.

## 2.7 Step 6 — VLS 보정 적용

최종 post-Euler 보정식은 다음과 같다.

\[
x_{\mathrm{next}}
=x_{\mathrm{cand}}
+\lambda(1-t_{\mathrm{next}})g
\]

이를 전개하면 다음과 같다.

\[
x_{\mathrm{next}}
=x_{\mathrm{cand}}
-2\lambda(1-t_{\mathrm{next}})
\left[x_{\mathrm{cand}}-(1-t_{\mathrm{next}})A_n\right]
\]

- `λ`: guidance strength
- `λ=0`: baseline과 정확히 동일
- 현재 기본값: `λ=0.1`

보정은 다음 mask의 교집합에만 적용한다.

- position mask: chunk의 앞 `M_eff`개 action
- dimension mask: 실제 제어에 사용하는 action dimension

나머지 position과 padding dimension은 Euler candidate와 bitwise 동일하게 유지된다.

---

## 3. baseline과 SEAM의 실행 흐름 비교

| 단계 | Baseline | SEAM |
|---|---|---|
| 1. 관측 입력 | image/state/language 입력 | 동일 |
| 2. 입력 변환 | repack → quantile normalize → pad | 동일 |
| 3. 초기 noise | `[B,H,D]` Gaussian noise | 동일 |
| 4. velocity 예측 | π0.5 flow velocity | 동일 |
| 5. Euler update | `x + dt·v` | 동일 |
| 6. post-Euler 처리 | 없음 | 이전 tail을 향한 VLS 보정 |
| 7. 반복 | 총 `N`회 | 총 `N`회 |
| 8. 출력 변환 | unnormalize → physical action | 동일 |
| 9. 실행 | 첫 `K`개만 실행 | 동일 |
| 10. 다음 chunk 상태 | 이전 chunk를 사용하지 않음 | 이전 model-space chunk를 보관 |

즉 SEAM은 모델, checkpoint, 관측 전처리, output transform, 실행 cadence를 바꾸지 않는다.
**유일한 알고리즘 차이는 각 Euler update 직후의 masked correction**이다.

---

## 4. 코드 구현: 실제로 어디에 어떻게 연결했는가?

## 4.1 OpenPI의 최소 변경

수정된 OpenPI 파일은 `src/openpi/src/openpi/models/pi0.py` 하나다.

`sample_actions`에 optional `guidance_fn(candidate, t_next)` 인자를 추가하고, 각 Euler step에서
다음 순서로 호출한다.

```python
x_next = x_t + dt * v_t
time_next = time + dt
if guidance_fn is not None:
    x_next = guidance_fn(x_next, time_next)
```

`guidance_fn=None`이 default이므로 기존 호출자는 영향을 받지 않는다. 또한 OpenPI 내부에서
SEAM 모듈을 import하지 않아, 이 hook은 특정 알고리즘에 종속되지 않는 일반적인 inference
extension point다.

근거 코드:

- `src/openpi/src/openpi/models/pi0.py:218-290`
- `benchmark/seam_vla/serving/pi0_guidance_hook.patch`

## 4.2 VLS 순수 연산

`guidance/vls.py`가 다음 계산을 그대로 구현한다.

```python
target = (1.0 - t_next) * aligned_prior
g = -2.0 * (candidate - target)
correction = lambda_ * (1.0 - t_next) * g
result = candidate + masked(correction)
```

이 코드는 `jax.lax.while_loop` 안에서 실행되므로 host I/O나 Python-side mutable state 없이
pure JAX 연산으로 작성되어 있다.

근거 코드:

- `benchmark/seam_vla/guidance/vls.py:31-62`
- `benchmark/seam_vla/integration/openpi_jax.py`

## 4.3 aligned prior

`priors/aligned_tail.py`는 이전 model-space chunk의 `K:H` 구간을 잘라 aligned prior를 만든다.

```python
tail = previous_chunk[..., K:H, :]
aligned = concatenate([tail, repeat(tail[-1], K)])
```

근거 코드:

- `benchmark/seam_vla/priors/aligned_tail.py:17-55`

## 4.4 JIT sampler

`SeamSampler`는 다음 두 JIT 함수를 별도로 가진다.

- `_call_baseline`: `guidance_fn=None`
- `_call_seam`: `VLSGuidance.as_fn()` 전달

aligned prior, `λ`, enable flag는 traced array로 넘기고, position/dimension mask와 `N`은 static
constant로 유지한다. 따라서 chunk마다 값이 달라져도 같은 compiled function을 재사용할 수 있다.

## 4.5 episode별 상태

`SeamState`가 다음 정보를 보관한다.

- `previous_chunk_model_space`: 다음 VLS prior용 `[H,D]`
- `previous_chunk_physical_space`: 기록·검증·decoded refiner용
- `previous_proprio_state`: RB-Y1 base 보정용
- `chunk_index`

첫 chunk 또는 reset 직후에는 이전 chunk가 없으므로 baseline sampler를 사용한다. 두 번째
chunk부터 VLS를 사용할 수 있다. 상태는 process-global이 아니라 session/episode별로 분리된다.

## 4.6 전체 `predict_chunk` 호출 순서

1. raw observation을 OpenPI input transform에 통과시킨다.
2. 현재 episode가 첫 chunk인지 검사한다.
3. 첫 chunk이거나 SEAM-off이면 baseline sampler를 호출한다.
4. 그 외에는 이전 model-space chunk에서 aligned prior를 만든다.
5. 동일 π0.5 sampler를 VLS guidance와 함께 호출한다.
6. model-space `[1,H,D]` 출력을 unnormalize한다.
7. task-specific output transform으로 physical chunk를 만든다.
8. 필요하면 decoded-stage refiner를 호출한다.
9. 현재 model/physical chunk를 새로운 `SeamState`에 저장한다.
10. 실행기는 앞 `K`개 action만 queue에 넣어 순서대로 실행한다.

근거 코드:

- `benchmark/seam_vla/policy/seam_policy.py:132-199`
- `benchmark/seam_vla/state.py`
- `benchmark/seam_vla/rollout/chunk_executor.py`

---

## 5. 좌표계와 tensor shape

SEAM 구현에서 가장 쉽게 발생할 수 있는 오류는 **model-space prior와 physical-space action을
혼합하는 것**이다.

### 5.1 LIBERO π0.5

| 값 | Shape | 좌표계 |
|---|---:|---|
| sampler noise/state/velocity | `[B,10,32]` | normalized model space |
| 이전 model chunk | `[10,32]` | normalized model space |
| aligned prior | `[B,10,32]` | normalized model space |
| 출력 physical chunk | `[10,7]` | LIBERO action space |
| 한 번에 실행하는 action | 첫 5개 | LIBERO action space |

LIBERO action은 유효 dimension 7개이고, model action dimension은 32이므로 7~31번은 padding이다.
VLS는 0~6번 dimension만 보정한다.

π0.5의 quantile normalization은 다음과 같다.

\[
x_{norm}
=\frac{x-q_{01}}{q_{99}-q_{01}+10^{-6}}\times2-1
\]

\[
x
=\frac{x_{norm}+1}{2}(q_{99}-q_{01}+10^{-6})+q_{01}
\]

VLS prior는 반드시 첫 번째 식을 거친 model space에 있어야 한다.

### 5.2 RB-Y1

RB-Y1의 client-facing action은 14차원 absolute joint target이다.

```text
[왼팔 6 joint, 왼 gripper, 오른팔 6 joint, 오른 gripper]
```

하지만 denoising model space는 현재 proprio state를 기준으로 한 delta다. 따라서 이전 chunk의
delta tail은 이전 base state 기준이고, 새 denoising은 현재 base state 기준이라는 차이가 있다.

이를 보정하기 위해 다음 식을 사용한다.

\[
\Delta s=s_{curr}-s_{prev}
\]

\[
A^{newbase}_{norm}
=A^{oldbase}_{norm}
-\Delta s\odot\frac{2}{q_{99}-q_{01}}
\]

arm joint dimension `0..5, 7..12`만 보정하고 near-binary gripper `6,13`은 제외한다.

---

## 6. 실제 설정: 논문 profile과 현재 checkpoint의 차이

### 6.1 LIBERO

| 항목 | 논문 π0.5 | 로컬 `pi05_libero` | 의미 |
|---|---:|---:|---|
| `H` | 50 | 10 | checkpoint에 고정 |
| `K` | 10 | 5 | LIBERO replan cadence |
| `L=H-K` | 40 | 5 | 사용 가능한 overlap |
| 요청 `M` | 20 | 20 | config 값 |
| 실제 `M_eff` | 20 | 5 | `min(M,L)` |
| `N` | 10 | 10 | Euler step |
| model/valid `D` | - / 7 | 32 / 7 | padding 제외 |

따라서 현재 LIBERO checkpoint로는 논문의 `L=40, M=20` 조건을 재현할 수 없다. 방법은 동일하게
적용했지만 더 짧은 overlap에 적용한 것이다. 논문 수치와 절대값을 직접 비교하지 말고,
동일한 `H=10,K=5` baseline과 SEAM을 비교해야 한다.

### 6.2 RB-Y1

| 항목 | 값 |
|---|---:|
| `H` | 50 |
| `K` | 8 |
| `L` | 42 |
| `M` | 20 |
| `N` | 10으로 설정, 실제 server에서 확인 필요 |
| physical/model `D` | 14 / 32 |
| guided dims | arm 12개, gripper 제외 |

RB-Y1은 `L=42`라 논문의 `M=20` window를 그대로 사용할 수 있지만, 실제 checkpoint/model
통합은 GPU server에서 아직 확인해야 한다.

---

## 7. 평가 지표

평가는 실제로 실행된 physical action stream `a_0,...,a_(T-1)`에서 계산한다.

### 7.1 per-step jerk

\[
j_t=\left\|a_{t+1}-2a_t+a_{t-1}\right\|_2
\]

이는 discrete action의 2차 차분 크기다.

### 7.2 boundary와 interior

chunk마다 `K`개를 실행하므로 경계 index 집합은 다음과 같다.

\[
B=\{t\mid t\bmod K=0,\;t>0\}
\]

유효한 jerk 중심 중 `B`가 아닌 index는 interior 집합 `I`다.

### 7.3 보고 지표

| 지표 | 정의 | 좋은 방향 |
|---|---|---|
| BJ | boundary에서의 평균 jerk | 낮을수록 좋음 |
| IJ | chunk 내부의 평균 jerk | 낮을수록 좋음 |
| CD | boundary에서 `‖a_t-a_(t-1)‖` 평균 | 낮을수록 좋음 |
| AVb | boundary jerk의 분산 | 낮을수록 좋음 |
| Success | task 성공률 | 높을수록 좋음 |
| overlap residual | 이전 tail과 새 head의 평균 거리 | 낮을수록 좋음 |

주의할 점은 smoothing metric만 낮추면 정지하거나 반응이 느린 정책도 좋은 점수를 받을 수 있다는
것이다. 따라서 최종 평가는 반드시 **Success를 유지하면서 BJ/CD/AVb가 감소하는지** 확인해야 한다.

---

## 8. 결과 1 — 구현 및 baseline 보존 검증

### 8.1 현재 fast test

2026-07-27 현재 다음 명령으로 재검증했다.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu \
src/openpi/.venv/bin/python -m pytest benchmark/seam_vla/tests/ -q
```

결과:

```text
79 passed, 7 skipped
```

skip은 flag가 필요한 heavy real-model test와 GPU 전용 test다.

### 8.2 실제 π0.5 model parity

동일한 RNG와 동일한 Gaussian noise를 사용한 real-model CPU 검증에서:

| 비교 | 최대 element-wise 차이 |
|---|---:|
| baseline `guidance_fn=None` vs SEAM `enabled=0` | `0.000e+00` |
| baseline vs SEAM `λ=0` | `0.000e+00` |
| baseline vs SEAM `enabled=1, λ=0.1` | 약 `1.459e-01` |

해석:

- SEAM을 끄면 기존 sampler와 정확히 동일하다.
- SEAM을 켜면 실제로 출력이 바뀐다.
- 보정 결과는 finite하며 NaN/Inf가 없다.

이 결과는 “성능이 좋아졌다”는 증거가 아니라, **baseline을 훼손하지 않고 guidance가 의도대로
작동한다**는 통합 검증이다.

---

## 9. 결과 2 — RB-Y1 기록에 대한 decoded-stage 오프라인 적용

### 9.1 실험 설정

- 데이터: `data/policy_records/step_*.npy`
- 기록 수: 114 chunks
- chunk shape: `[50,14]`
- `K=8`, `L=42`, `M=20`
- `λ=0.2`
- arm 12 dimensions만 평가
- 두 run: blue-block 63 chunks, red-block 51 chunks
- prompt가 바뀌는 reset 지점은 chunk 경계 metric에서 제외
- 기존에 기록된 baseline chunk의 새 head를 이전 tail 방향으로 decaying steering

재현 명령:

```bash
src/openpi/.venv/bin/python \
  -m benchmark.seam_vla.experiments.rby1.offline_boundary_eval \
  --records data/policy_records --K 8 --M 20 --lam 0.2
```

### 9.2 결과

| 지표 | Baseline | SEAM(decoded) | 변화 |
|---|---:|---:|---:|
| BJ | 0.0226 | 0.0221 | −2.4% |
| IJ | 0.0129 | 0.0115 | −10.9% |
| CD | 0.0180 | 0.0181 | +0.1% |
| AVb | 약 0.0001 | 약 0.0001 | −8.8% |
| overlap residual | 0.1275 | 0.1254 | −1.7% |
| BJ/IJ | 1.75 | 1.91 | 증가 |

추가 관찰:

- 같은 run 안의 chunk 사이 proprio base drift arm L2 평균: `0.0796 rad`
- 최대 drift: `0.3193 rad`

### 9.3 결과 해석

긍정적인 부분:

- boundary jerk(BJ)와 boundary jerk 분산(AVb)이 소폭 감소했다.
- 이전 tail과 새 head 사이의 overlap residual도 소폭 감소했다.

신중하게 해석할 부분:

- CD는 0.1% 증가하여 측정 정밀도를 고려하면 개선되지 않았다.
- IJ가 BJ보다 더 크게 감소하여 `BJ/IJ` 비율은 1.75에서 1.91로 오히려 증가했다.
- 이 실험은 기록된 chunk에 물리 공간 refiner를 사후 적용한 counterfactual 분석이다.
- 새 action을 실제 robot/simulator에 실행해 다음 observation이 바뀌는 closed-loop 효과는
  반영하지 않는다.
- 이 결과는 primary denoising-stage VLS 자체의 실기 성능 증거가 아니다.

따라서 가장 정확한 표현은 다음과 같다.

> “기존 RB-Y1 기록에서 chunk-boundary artifact가 interior보다 1.75배 크게 관찰되었고,
> decoded-stage steering을 사후 적용하면 BJ와 AVb, overlap residual은 소폭 감소했지만 CD는
> 개선되지 않았다. 이는 overlap 정보를 이용한
> smoothing의 가능성을 보여 주지만, primary VLS의 실제 closed-loop 효과는 별도 A/B 실험이
> 필요하다.”

---

## 10. 아직 결과로 주장할 수 없는 항목

### 10.1 LIBERO full benchmark

다음 결과는 아직 측정되지 않았다.

- LIBERO-10 10 tasks × 130 episodes task success
- 동일 episode 조건의 baseline vs SEAM BJ/IJ/CD/AVb
- GPU steady-state latency overhead

이유:

- 로컬 환경에 LIBERO/robosuite/MuJoCo simulator가 완전하게 준비되어 있지 않다.
- 로컬 8 GB GPU에 약 2B 규모 π0.5 전체 모델이 올라가지 않는다.

### 10.2 RB-Y1 primary VLS

다음도 아직 측정되지 않았다.

- 실제 `pi05_rby1_lora` checkpoint에서 VLS correction
- 실제 robot 또는 MuJoCo closed-loop baseline/SEAM 비교
- task completion 또는 안전성 유지 여부
- server GPU latency

그러므로 오프라인 decoded 결과를 primary VLS의 실기 결과로 표현해서는 안 된다.

---

## 11. 교수님께 설명할 때의 추천 순서

### 1단계 — 문제를 한 문장으로 제시

> “π0.5가 50개 action을 예측해도 일부만 실행하고 다시 계획하기 때문에, 이전 계획의 tail과
> 새 계획의 head가 달라 chunk 경계에서 jerk가 커질 수 있습니다.”

### 2단계 — baseline 그림 설명

`H`, `K`, `L=H-K`를 먼저 정의하고, 이전 tail이 버려진다는 점을 보여 준다.

### 3단계 — 핵심 아이디어 설명

> “SEAM은 이전 tail을 새 action에 강제로 복사하지 않고, denoising 중 soft prior로 사용합니다.”

### 4단계 — 수식 세 줄 설명

\[
r=(1-t)A,\qquad
g=-2(x-r),\qquad
x\leftarrow x+\lambda(1-t)g
\]

세 식의 의미를 각각 “시간 정렬 target”, “target 방향”, “작은 크기의 steering”으로 설명한다.

### 5단계 — 코드 연결 설명

Euler update 직후에 optional hook 하나를 넣었고, off일 때는 기존 코드가 그대로 실행된다는
점을 강조한다.

### 6단계 — 좌표계 설명

VLS는 normalized model space에서, metric은 executed physical space에서 계산된다고 구분한다.
RB-Y1은 model space가 base-relative delta이므로 base compensation이 추가된다고 설명한다.

### 7단계 — 결과를 근거 수준별로 제시

1. test 통과
2. real-model baseline parity와 non-zero correction
3. RB-Y1 offline decoded-stage metric 개선
4. 아직 없는 LIBERO full/robot closed-loop 결과

### 8단계 — 다음 실험 제안

> “다음 단계는 동일 초기조건과 동일 task를 사용해 baseline/SEAM을 실제 closed loop로 비교하고,
> success가 유지되는 범위에서 BJ·CD·AVb와 latency를 함께 보는 것입니다.”

---

## 12. 예상 질문과 답변

### Q1. 이전 tail을 새 chunk 앞부분과 그냥 평균하면 안 되는가?

단순 평균은 이미 생성된 physical action을 사후 변형하므로 새 관측에 대한 policy 의도와
동역학적 일관성을 깨뜨릴 수 있다. VLS는 생성 과정 안에서 작은 gradient 방향을 반복 적용해,
모델의 velocity field와 overlap prior를 함께 반영한다. 실제 기록에서도 uniform steering은
경계 jerk를 악화시킬 수 있어, 언제·어디서 smoothing하는지가 중요하다.

### Q2. 추가 학습이 필요한가?

필요 없다. checkpoint와 parameter는 바꾸지 않는 training-free inference-time 방법이다.

### Q3. backpropagation 때문에 느려지지 않는가?

policy backward pass를 하지 않는다. consistency loss의 gradient가
`-2(x-r)`로 닫힌 형태이므로 element-wise 연산만 추가된다. 다만 실제 GPU latency overhead는
아직 측정해야 한다.

### Q4. 첫 chunk는 어떻게 처리하는가?

이전 tail이 없으므로 baseline으로 생성한다. episode reset 뒤 첫 chunk도 동일하다.

### Q5. 왜 tail 전체가 아니라 앞 `M`개만 guide하는가?

새 chunk가 시작되는 경계 근처만 부드럽게 연결하고, 더 먼 미래는 새 관측에 맞게 자유롭게
재계획하도록 하기 위해서다.

### Q6. `λ`가 크면 더 부드러워지는가?

보정은 강해지지만 task에 필요한 새 계획을 억제할 수 있다. 따라서 success와 smoothing 사이의
trade-off가 있으며, 현재 기본값은 `0.1`이다.

### Q7. padding 32차원도 보정하는가?

아니다. LIBERO는 유효한 7차원만, RB-Y1은 gripper를 제외한 arm 12차원만 guide한다.

### Q8. 현재 가장 강한 결과는 무엇인가?

구현 측면에서는 real-model baseline exact parity다. 효과 측면에서는 RB-Y1 기록에 대한
decoded-stage 오프라인 결과지만, 이는 primary VLS의 closed-loop 실기 결과보다 증거 수준이 낮다.

### Q9. 논문과 같은 조건인가?

LIBERO local checkpoint는 `H=10,K=5`라 논문의 `H=50,K=10`과 다르다. RB-Y1은 `H=50,K=8`이고
`M=20`을 사용할 수 있지만 checkpoint가 있는 server에서 통합 확인이 남아 있다.

---

## 13. 다음 실험 계획

### Phase 1 — server preflight

1. 실제 `pi05_rby1_lora`의 `H`, `D`, `N`을 확인한다.
2. action normalization `q01/q99`와 dimension layout을 확인한다.
3. 첫 guided chunk의 correction norm과 NaN/Inf를 확인한다.
4. baseline-off parity를 server checkpoint에서 확인한다.

### Phase 2 — 짧은 안전 A/B

1. 동일 prompt와 동일 초기 자세를 사용한다.
2. baseline과 `λ=0.05,0.1`을 짧게 실행한다.
3. executed target과 measured qpos를 모두 기록한다.
4. BJ, IJ, CD, AVb, overlap residual, inference latency를 계산한다.
5. gripper와 joint limit, emergency stop 이벤트를 별도로 확인한다.

### Phase 3 — task-level 평가

1. 여러 seed와 task에서 success를 측정한다.
2. success가 유지되는 `λ,M` 영역을 찾는다.
3. 평균뿐 아니라 episode별 분산과 실패 사례를 분석한다.
4. smoothing 개선과 latency 증가를 함께 보고한다.

최종 의사결정 기준은 다음과 같이 잡는 것이 적절하다.

\[
\text{Success 유지}
\quad\land\quad
\text{BJ/CD/AVb 감소}
\quad\land\quad
\text{latency 허용 범위}
\]

---

## 14. 발표 마무리 문장

> “이번 구현은 π0.5의 학습이나 checkpoint를 변경하지 않고, denoising Euler step에 이전
> action tail 기반의 closed-form guidance를 추가했습니다. Off일 때 baseline과 정확히 동일하고,
> on일 때 유한한 보정이 발생함을 확인했습니다. 또한 RB-Y1 기록을 task reset 기준으로 나누어
> 계산했을 때 chunk 경계 jerk가 interior의 1.75배였고, decoded-stage 보조 실험에서 BJ 2.4%,
> AVb 8.8%, overlap residual 1.7% 감소를 얻었지만 CD는 개선되지 않았습니다. 따라서 primary
> VLS의 실제 closed-loop 효과와 task success
> 유지는 아직 검증 전이므로, 다음 단계는 server checkpoint 및 실제 robot A/B 평가입니다.”
