# P0a 결과 — Attention 추출 배관 검증

실행: `JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.knows_vla.probe_p0a`
코드: [benchmark/knows_vla/probe_p0a.py](../probe_p0a.py)
모델: `pi05_libero` (로컬 체크포인트), CPU, `make_libero_example()` 합성 관측(seed 0)

**결과: PASS**

> **범위**: 이것은 논문의 과학적 주장(layer 12 / head 3이 타깃을 국소화한다)을 검증하지
> **않는다**. 올바른 텐서에서 올바른 숫자를 꺼낼 수 있다는 것만 확인한다. 신호 검증은 P0b이며
> 실제 장면과 객체 마스크가 필요하다.

---

## 1. 검증된 shape 주장

[02-architecture.md](02-architecture.md) §3.4의 예측이 **전부 실측과 일치**했다.

| 항목 | 예측 | 실측 | |
|---|---:|---:|---|
| prefix 카메라 슬롯 | 3 | 3 | OK |
| 카메라당 vision token | 256 | 256 | OK |
| prefix 길이 | 968 | 968 | OK |
| suffix 길이 ($H$) | 10 | 10 | OK |
| `probs` 레이어 $L$ | 18 | 18 | OK |
| `probs` KV head $K$ | 1 | 1 | OK |
| `probs` query head $G$ | 8 | 8 | OK |
| `probs` query 길이 $T$ | 10 | 10 | OK |
| `probs` key 길이 $S$ | 978 | 978 | OK |
| agent view 그리드 $g$ | 16 | 16 | OK |

확정된 추출 슬라이스:

```python
A_raw = probs[12, 0, 0, 3, :, 0:256]   # [H, 256] = [10, 256]
A_t   = A_raw.mean(axis=0).reshape(16, 16)
```

softmax 행 합 = 1.000067 (float64 누적 기준). **주의**: `probs`는 bfloat16으로 저장되므로
978개를 bf16으로 누적하면 약 2% 손실이 난다(첫 실행에서 0.9766으로 관측). 다운스트림 계산은
반드시 float32 이상으로 캐스팅할 것 — Eq. (2)의 mass 합산에 직접 영향.

## 2. openpi 변경 사항

3개 파일, 36줄 추가.

| 파일 | 변경 |
|---|---|
| [gemma.py](../../../src/openpi/src/openpi/models/gemma.py) | `Attention.__call__`이 `probs` 반환(플래그 off면 `None`), `Block`/`Module`에 정적 `return_probs` 속성, `Module.__call__`이 플래그 on일 때만 3-tuple 반환 |
| [pi0_config.py](../../../src/openpi/src/openpi/models/pi0_config.py) | `Pi0Config.return_attn_probs: bool = False` |
| [pi0.py](../../../src/openpi/src/openpi/models/pi0.py) | 설정값을 `gemma.Module`로 전달 |

### 설계상 걸림돌 — 플래그는 반드시 정적이어야 한다

처음에는 `return_probs`를 호출 인자로 넣었으나 실패했다. 18개 레이어는
`nn.scan(..., in_axes=(..., nn.broadcast))`으로 스캔되는데, **스캔에 넘긴 값은 tracer가 되고
tracer로는 Python 분기를 할 수 없다** (`TracerBoolConversionError`). 즉
`probs if return_probs else None`이 성립하지 않는다.

해결: `Block`과 `Module`의 **dataclass 속성**으로 옮겨 구성 시점에 고정. 파라미터를 추가하지
않으므로 동일 체크포인트가 양쪽 모두에 로드된다.

부수 효과: `return_attn_probs=True`인 모델에서는 `sample_actions`가 동작하지 않는다
(2-tuple 언팩 가정). 프로브는 prefix/suffix 패스를 직접 구동한다.

## 3. 회귀 검증

| 검사 | 결과 |
|---|---|
| 플래그 off → `probs is None` | OK |
| **플래그 on vs off, 동일 코드 경로** | **max\|diff\| = 0.000e+00 (byte-identical)** |
| SEAM 단위 테스트 | 85 passed (1건 실패는 `rby1_bringup` 미설치로 사전 존재, 무관) |
| **SEAM heavy 모델 테스트** (`SEAM_RUN_MODEL_TESTS=1`) | **8 passed** — baseline parity 포함 |

수동 loop vs `sample_actions`: max\|diff\| = 1.6e-03 (액션 스케일 ~1.0). 이는 jitted
`while_loop`와 eager 실행의 bfloat16 누적 순서 차이이며 플래그와 무관하다 — 위의 on/off 비교가
정확히 0인 것이 그 근거다.

### 프로브 자체에서 나온 교훈

`make_libero_example()`은 **난수 이미지**를 생성한다. 두 모델을 각각 로드하면 서로 다른 관측을
먹게 되어 9.5e-02의 가짜 불일치가 나온다. 비교 실험에서는 반드시 seed를 고정할 것.

## 4. 해소된 미해결 항목

| 항목 | 상태 |
|---|---|
| [02-architecture.md](02-architecture.md) §3.4 슬라이스 인덱스 | **확정** |
| OPEN-Q 16 (`data/policy_records/` 출처) | **해소 — 부적합 판정**. LIBERO가 아니라 RB-Y1 기록이다: state 14차원, actions (50,14), 카메라 `cam_high`/`cam_left_wrist`/`cam_right_wrist`, prompt "put the blue block in the brown box". 이를 생성한 `pi05_rby1_lora` 체크포인트는 로컬에 없다 |
| OPEN-Q 10 ($\bar A_t$ 집계) | 미해소 — 배관은 준비됨. 평균/합/마지막 쿼리 비교는 P0b에서 |
| OPEN-Q 11 (디노이징 스텝) | 미해소 — 프로브에 `denoise_step` 인자로 스윕 가능하게 만들어 둠 |

## 5. P0b로 넘어가기 위해 필요한 것

신호 검증에는 **실제 장면 + 타깃 라벨 + 객체 마스크**가 필요하다. 현재 워크스페이스에는 셋 다 없다.

| 후보 | 장점 | 문제 |
|---|---|---|
| LIBERO LeRobot 데이터셋 (`physical-intelligence/libero`) | `pi05_libero`와 정확히 일치, 태스크 문자열 포함 | 다운로드 필요. **객체 마스크 없음** — 수동 주석 또는 색 분할 필요 |
| `data/policy_records/` (RB-Y1) | 로컬 보유, 2객체 장면("blue block", "brown box")이라 색 분할이 쉬움 | 생성 체크포인트(`pi05_rby1_lora`)가 로컬에 없음. `pi05_libero`로 돌리면 임베디먼트 불일치 |
| LIBERO 시뮬레이터에서 직접 렌더 | GT 세그멘테이션 확보 가능 | LIBERO/robosuite 미설치 |
