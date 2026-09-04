# 전체 시나리오 — attention + ESDF 로 충돌 제약을 만든다

**요약.** RB-Y1 사과 집기 롤아웃 44청크를 처음부터 끝까지, primitive 도형을 한 번도 만들지 않고
돌렸다. 충돌 제약을 만든 것은 **attention 과 거리장뿐**이다 — `n_candidates` 는 44청크 전부 0 이다.

두 모드를 나란히 돌렸다. **전신**(`--constraint-links all`)은 194구 전부에 제약을 걸고,
**양팔**(`arms`, 기본값)은 양팔 링크와 손끝 120구에만 건다. 결론부터: 양팔 모드가 44청크 전부에서
복셀 이산화 바이어스 이내로 들어오고, TO 가 116 → 70 ms 로 빨라지며, 참조 궤적 편차가
1.69 → 0.99 rad 로 줄어든다.

## 무엇을 돌렸나

```bash
MUJOCO_GL=osmesa XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python -m benchmark.trajopt.experiments.esdf_rollout \
  --records outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0002 \
  --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz \
  --voxel 0.020 --esdf-margin 0.05 --support-surfaces field \
  --constraint-links arms \
  --out-json benchmark/trajopt/asset/esdf_rollout_arms.json
```

전신 모드는 `--constraint-links all --out-json .../esdf_rollout_field.json` 로 같은 명령을
한 번 더 돌린 것이다.

* **attention 은 실측이다.** π0.5 파인튜닝 체크포인트에서 뽑은 `attention_step1_run0002.npz`,
  1단계에서 채점한 그 텐서다. 합성 가우시안이 아니다.
* **depth 도 실측 롤아웃의 재생이다.** `run_0002` 의 관절 궤적을 MuJoCo 에 되먹여 3-카메라
  depth 를 다시 만든다.
* 복셀 20 mm, `esdf_margin` 50 mm, 계획 지평 32 스텝.

## 파이프라인 — 어디에도 도형 근사가 없다

```
π0.5 attention ─→ 2D→3D 역투영 ─→ target grounding ─→ target 점군
                                                          │
3-카메라 depth ─→ 점군 융합 ─→ TSDF ─→ 점유 ─→ ESDF ←──────┘ (target 복셀 파냄)
                                                  │
                          d_esdf(p(q)) − r − margin ≥ 0  ─→ SQP/QP ─→ 안전한 청크
```

`n_candidates` 는 44청크 전부에서 **0** 이다. 클러스터링도 primitive fitting 도 돌지 않았다.
target 은 44청크 중 **42청크에서 잡혔다**. 못 잡은 2청크는 3단계에서 이미 확인한 가림 프레임이고,
그 청크에서는 target 복셀을 파내지 않으므로 거리장이 **더 보수적**으로 동작한다 — 안전한 방향의
실패다.

## 결과 — 두 모드

| | 전신 (`all`) | 양팔 (`arms`, 기본) |
|---|---|---|
| 제약 구 | 194 | **120** (양팔 링크 76 + 손끝 44) |
| 잔여 ≥ 0 mm | 40/44 | 34/44 |
| 잔여 ≥ −10 mm (복셀 바이어스 이내) | 43/44 | **44/44** |
| 최악 잔여 | −11.1 mm | **−5.1 mm** |
| 개선 중앙값 | +85.1 mm | +77.4 mm |
| 상태 | violated 44 | **feasible 27** / violated 17 |
| TO | 116 ms | **70 ms** |
| 참조 편차 (중앙) | 1.690 rad | **0.991 rad** |

**"완전 해소 40 → 34" 는 같은 것을 재고 있지 않다.** 전신 모델에서 "팔 구만" 은 `base`/`wheel`
만 뺀 마스크라 토르소가 포함돼 있고, 양팔 모드의 제약 집합에는 토르소가 없다. 물리적으로 의미
있는 기준으로 보면 방향이 반대다 — 20 mm 복셀의 이산화 바이어스가 정확히 −voxel/2 = **−10 mm**
(보수적 방향)인데, 양팔 모드는 44청크 전부가 그 안에 들어오고 최악 잔여가 −11.1 → −5.1 mm 로
줄었다. −0.8 mm 같은 잔여는 거리장 해상도의 노이즈이지 실제 관통이 아니다.

진짜 개선은 세 가지다. 상태 보고가 고칠 수 없는 상수에 더는 지배되지 않고(feasible 27청크),
TO 가 40% 빨라지고, **참조 궤적 편차가 절반 아래로 떨어진다**. 앞 실험에서 편차가 1.69 rad 로
컸던 이유의 상당 부분이 바퀴 행이 해를 밀어내고 있었기 때문이다.

### 전신 모드의 세부 — 왜 바퀴를 빼게 됐나

전신 모드 44청크 전부에서 "전체 구" 기준 완전 해소는 **0** 이고, 최악 잔여가 정확히 −33.7 mm
로 **상수**다. 해에서 가장 나쁜 구를 이름으로 뽑으면:

| 구 링크 | 여유 |
|---|---|
| `wheel_l` | −33.7 mm |
| `wheel_r` | −33.7 mm |
| `link_left_arm_5` | −32.7 mm |
| `link_right_arm_5` | −31.4 mm |
| `base` | −25.3 mm |

위반 구 25/194 중 11개가 `base`/`wheel` 이다. 이들은 결정 변수(양팔 12관절)에 의존하지 않으므로
어떤 해에서도 같은 값을 낸다. "해소 0" 은 최적화기의 실패가 아니라 그 상수를 세고 있다는 뜻이다.

그래서 **제약 모델을 자기 필터 모델과 분리**했다. `UrdfSphereChain` 에 이미 `link_filter` 가
있었고 docstring 이 "arms only" 를 예시로 들고 있어서, 새 기계 없이
`build_constraint_robot_model()` 을 추가해 배선만 했다.

| | 자기 필터 모델 | 제약 모델 |
|---|---|---|
| 구 수 | 194 (전신, 두 모드 동일) | 120 (`arms`) / 194 (`all`) |
| 왜 | 머리 카메라가 자기 몸을 내려다본다. 바퀴를 빼면 그 점이 클라우드에 남아 **로봇에 용접된 유령 장애물**로 뭉친다 | 최적화기가 움직일 수 있는 구에만 제약이 의미가 있다 |

AG3S 가 `robot_model` 과 `constraint_robot_model` 을 처음부터 분리해 받은 이유가 정확히 이것이다.

**빼는 것이 위험을 지우지는 않는다.** 바퀴는 여전히 바닥에 닿아 있고, 바꾼 것은 그것을 최적화
문제로 취급하지 않기로 한 것뿐이다. 베이스가 움직이는 순간 이 가정은 깨지므로
`--constraint-links all` 로 되돌리는 경로를 남겼다.

## 거리장이 실제로 무엇을 담았나

| 항목 | 값 |
|---|---|
| 점유 복셀 | 중앙값 6,228 |
| 미관측 비율 | 70% |
| target 으로 파낸 복셀 | 중앙값 10 |

미관측 70% 는 `unknown_policy: free` 로 자유 처리된다. primitive 도 관측 안 된 기하는 후보를
만들지 않으므로 성질은 같지만, ESDF 는 처음으로 그 비율을 **세어서 보고한다**.

## 지연

| 단계 | 전신 | 양팔 |
|---|---|---|
| AG3S (지각 + 거리장) | 391 ms | 393 ms |
| TO (선형화 + QP) | 116 ms | **70 ms** |
| 합계 | 508 ms | **463 ms** / 예산 66.7 ms |

**예산의 6.9배다.** 실시간이 아니다. 이 실험은 표현이 옳은 제약을 만드는지를 보는 것이지 지연을
맞추는 것이 아니다. AG3S 391 ms 의 대부분은 target grounding 이고, TO 116 ms 는 SQP 1회
(RTI) 다. 앞선 초안에 적힌 AG3S 1259 ms 는 지지면 추출이 꺼진 잘못된 설정의 값이었다 —
[ESDF-BACKEND.md](ESDF-BACKEND.md) 의 "지지면: 추출과 소비는 별개다" 참고.

## 정직하게 적어 둘 두 가지

**참조 궤적에서 여전히 벗어난다.** 양팔 모드에서도 SEAM 참조 대비 편차가 중앙값 **0.99 rad**
다. 시작 궤적이 −151 mm 까지 파고든 상태였으니 크게 밀어낼 수밖에 없지만, 이 크기는 "충돌만
피하면 된다" 를 넘어선다. 비용 가중치(`w_track` 대 `w_slack`)를 이 씬에서 재조정하지 않았다.

**양팔 모드에서도 17청크가 `violated` 다.** 그 잔여는 −0.8 ~ −5.1 mm 로 전부 복셀 바이어스
안쪽이다. 상태 판정 임계값이 0 mm 에 걸려 있어 해상도 노이즈를 위반으로 부른다. 임계값을 표현
오차와 묶을 것인지가 미결이다.

## 재현

두 모드의 청크별 수치가 각각 `benchmark/trajopt/asset/esdf_rollout_field.json`(전신)과
`esdf_rollout_arms.json`(양팔)에 있다
(`arm_before_mm`, `arm_after_mm`, `clearance_*`, `esdf_unknown`, `target_voxels_carved`,
`n_candidates`, `iterations`, `reference_deviation`).
