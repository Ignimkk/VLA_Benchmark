# P2a 결과 — 지각 (깊이 역투영 + MVEE) 및 액션 규약 확정

논문 §3.2의 지각 단계를 구현하고, 프레임·단위 규약을 **추론이 아니라 실측으로** 확정했다.

| 항목 | 내용 |
|---|---|
| 구현 | [perception/ellipsoid_fit.py](../perception/ellipsoid_fit.py) |
| 검증 | [validate_backprojection.py](../validate_backprojection.py) |
| 카메라 파라미터 | [dump_camera_params.py](../dump_camera_params.py) → `benchmark/knows_vla/camera_params.json` (20 태스크) |

---

## 1. 카메라 파라미터는 태스크마다 다르다

P0b 수집 시 intrinsic/extrinsic을 저장하지 않아 별도로 덤프했다. 환경 생성만 하면 되므로
재수집보다 훨씬 싸다.

깊이가 **정규화 버퍼**라 미터로 바꾸려면 $z = z_{\text{near}}/(1 - d(1 - z_{\text{near}}/z_{\text{far}}))$가
필요한데, `znear`/`zfar`가 `sim.model.stat.extent`에 비례한다 — 즉 **장면 속성**이다. 실제로
`libero_10_task2/3`만 `znear=0.0118, zfar=591.55`로 나머지(`0.0106 / 530.49`)와 다르다.
전역 상수로 두면 그 두 태스크에서 조용히 틀린다.

## 2. 역투영 규약을 실증으로 확정했다

프레임 오류는 조용하고 치명적이다 — 좌우가 뒤집혀도 타원체는 그럴듯해 보이지만 엉뚱한 위치에
놓이고, CBF 필터가 **반대 방향으로 민다**. 그래서 추론 대신 측정했다.

**검증 방법**: 그리퍼 자신의 세그멘테이션 마스크를 역투영한 중심이 에피소드에 기록된
`robot0_eef_pos`와 맞아야 한다.

| flip_row | flip_col | 평균 오차 (m) | median | p90 |
|---|---|---:|---:|---:|
| False | False | 0.2656 | 0.3364 | 0.3778 |
| False | **True** | **0.0671** | **0.0693** | **0.0711** |
| True | False | 0.3573 | 0.3733 | 0.4283 |
| True | True | 0.1829 | 0.1216 | 0.4122 |

`libero_object`에서도 동일하게 `(False, True)`가 0.0665 m로 최소. **결론: 저장 배열은 원본
렌더 대비 수평 반전만 적용된 상태다.** robosuite가 `IMAGE_CONVENTION`으로 이미 수직 반전해
저장하고, `collect_p0b.py`가 학습 전처리에 맞추려 180° 회전하므로 순효과가 수평 미러다.

잔차 6.7 cm는 오차가 아니라 **그리퍼 가시 표면의 중심과 eef 사이트(손가락 사이) 사이의 물리적
오프셋**이다. median 0.069 / p90 0.071로 매우 좁아 잡음이 아닌 일관된 편향임을 보여준다.

## 3. MVEE는 이상치 제거 없이는 쓸 수 없다 — 논문 미기재

세그멘테이션 마스크는 경계에서 전경/배경 픽셀을 몇 개씩 포함한다. MVEE는 **최소부피 외접**
타원체이므로 모든 점을 담아야 하고, 이상치 하나가 시선 방향으로 타원체를 늘린다.

`alphabet_soup_1` 실측: p5–p95 깊이 폭은 4.8 cm인데 min–max는 **38 cm**.

| 객체 | 트리밍 전 반축 (cm) | 트리밍 후 (cm) |
|---|---|---|
| alphabet_soup_1 | 4.9 × 5.0 × **23.8** | 3.4 × 4.1 × 6.2 |
| salad_dressing_1 | 3.6 × 8.6 × 16.2 | 1.8 × 3.7 × 9.9 |
| milk_1 | 4.2 × 9.3 × 18.1 | 3.0 × 4.4 × 9.5 |
| glazed_rim_porcelain_ramekin_1 | 3.2 × 5.3 × 8.1 | 3.2 × 4.8 × 5.1 |

트리밍 후 형상이 물리적으로 맞는다 — 수프 캔은 뭉툭하고, 드레싱 병은 길쭉하다.

**대응**: 마스크 내 깊이 2–98 백분위로 자른 뒤 적합한다. 논문은 MVEE([35])만 지정하고 이상치
처리를 전혀 언급하지 않으므로 **이것은 방법의 일부가 아니라 우리의 공학적 추가**다.
`depth_trim=None`으로 끄면 논문 문자 그대로가 된다.

### 남는 한계 — 단일 시점

논문 §3.2는 "fuse it across the available camera views"라고 하지만, LIBERO의 손목 카메라는
움직이므로 정적 장면 기하에는 쓸 수 없다. agentview 한 시점만 쓴다(Layer B 이탈).

그 결과 큰 객체는 **보이는 면만** 적합된다: `wooden_cabinet_1` 12.6 × 15.0 × 22.1 cm,
`flat_stove_1` 3.3 × 9.6 × 22.6 cm. 실제보다 얇다. 안전 필터 입장에서는 **과소 추정**이라
위험한 방향이다 — P3에서 충돌이 남는다면 여기가 후보다.

## 4. OPEN-Q 2 완전 해소 — 액션 프레임과 회전 표현

P1에서 Eq. (10)이 world 프레임 축각 증분과 $Q_R \to RQ_RR^\top$을 전제한다는 것을 유도했다.
LIBERO/robosuite OSC_POSE가 실제로 그 규약인지 확인했다.

**설정** (`robosuite/controllers/config/osc_pose.json`):
- `control_delta: true`, `input_max: 1`
- `output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`

→ 물리 단위 변환: **위치 = action[0:3] × 0.05 m**, **회전 = action[3:6] × 0.5 rad**

**프레임** (`osc.py`, `control_utils.set_goal_orientation`):
- 위치: `goal_pos = ee_pos + scaled_delta[:3]` → **world 프레임**
- 회전: `goal_ori = R(delta) @ R_current` — **좌측 곱셈 = world 프레임 회전**, delta는 축각

**P1의 유도와 정확히 일치한다.** $\delta\theta$는 world 프레임 축각 증분이고, 그리퍼에 붙은
$Q_R$은 같은 회전으로 $RQ_RR^\top$ 변환된다. Eq. (10)을 그대로 쓸 수 있다.

이로써 OPEN-Q 2의 남은 절반이 닫혔다. 이 규약을 틀렸다면 필터가 잘못된 축으로 밀었을 것이다.

## 5. 다음 단계

지각·기하·QP·액션 규약이 모두 갖춰졌다. 남은 것은 **P2b 오프라인 통합**:

수집된 에피소드에 대해 매 스텝 (a) attention으로 $\tau_t$ 식별 → (b) 나머지를 장애물로 →
(c) 명목 액션을 물리 단위로 변환 → (d) CBF-QP 통과, 그리고 다음을 계측한다.

- $\|\delta c - \delta c^{\mathrm{nom}}\|$ 분포 — 필터가 실제로 얼마나 개입하는가
- emergency stop 발생률
- 최소 $h$ 궤적 — 필터가 없었다면 침범했을 스텝 수
- $\tau_t$ 오분류가 장애물 집합에 미치는 영향 (특히 [OPEN-Q 18](OPEN-QUESTIONS.md)의 배치 단계)

폐루프가 아니므로 실행 결과는 바뀌지 않는다. 필터의 **개입 정도와 타당성**만 본다.
