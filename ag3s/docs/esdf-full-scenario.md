# 전체 시나리오 — attention + ESDF 로 충돌 제약을 만든다

**요약.** RB-Y1 사과 집기 롤아웃 44청크를 처음부터 끝까지, primitive 도형을 한 번도 만들지 않고
돌렸다. 충돌 제약을 만든 것은 **attention 과 거리장뿐**이다. 팔 구 기준 44청크 중 **40청크에서
위반이 완전히 해소**되고 44청크 전부에서 개선됐다. 남은 것은 팔이 아니라 바퀴다 —
`wheel_l`/`wheel_r` 이 관측된 바닥면을 33.7 mm 파고드는데, 팔 관절로는 움직일 수 없다.

## 무엇을 돌렸나

```bash
MUJOCO_GL=osmesa XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python -m benchmark.trajopt.experiments.esdf_rollout \
  --records outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0002 \
  --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz \
  --voxel 0.020 --esdf-margin 0.05 --support-surfaces field \
  --out-json benchmark/trajopt/asset/esdf_rollout_field.json
```

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

## 결과

| | 위반으로 시작 | 완전 해소 | 개선 | 최악(전) | 최악(후) | 개선 중앙값 |
|---|---|---|---|---|---|---|
| 팔 구만 | 44/44 | **40** | 44 | −151.5 mm | −11.1 mm | **+85.1 mm** |
| 전체 구 | 44/44 | 0 | 42 | −151.5 mm | −33.7 mm | +43.1 mm |

두 줄을 나눠 세는 이유는 **이동 베이스가 바닥에 닿아 있는 것이 정상**이기 때문이다. 해에서 가장
나쁜 구를 이름으로 뽑으면:

| 구 링크 | 여유 |
|---|---|
| `wheel_l` | −33.7 mm |
| `wheel_r` | −33.7 mm |
| `link_left_arm_5` | −32.7 mm |
| `link_right_arm_5` | −31.4 mm |
| `base` | −25.3 mm |

위반 구 25/194 중 11개가 `base`/`wheel` 이다. 이들은 결정 변수(팔 12관절)에 의존하지 않으므로
44청크 내내 정확히 −33.7 mm 로 **상수**다. "전체 구 해소 0" 은 최적화기의 실패가 아니라 그
상수를 세고 있다는 뜻이다. 제약에서 빼지 않은 것은 의도다 — 무엇을 충돌로 셀지는 로봇 모델의
결정이지 실험 스크립트가 조용히 정할 일이 아니다.

## 거리장이 실제로 무엇을 담았나

| 항목 | 값 |
|---|---|
| 점유 복셀 | 중앙값 6,228 |
| 미관측 비율 | 70% |
| target 으로 파낸 복셀 | 중앙값 10 |

미관측 70% 는 `unknown_policy: free` 로 자유 처리된다. primitive 도 관측 안 된 기하는 후보를
만들지 않으므로 성질은 같지만, ESDF 는 처음으로 그 비율을 **세어서 보고한다**.

## 지연

| 단계 | 평균 |
|---|---|
| AG3S (지각 + 거리장) | 391 ms |
| TO (선형화 + QP) | 116 ms |
| 합계 | **508 ms** / 예산 66.7 ms |

**예산의 7.6배다.** 실시간이 아니다. 이 실험은 표현이 옳은 제약을 만드는지를 보는 것이지 지연을
맞추는 것이 아니다. AG3S 391 ms 의 대부분은 target grounding 이고, TO 116 ms 는 SQP 1회
(RTI) 다. 앞선 초안에 적힌 AG3S 1259 ms 는 지지면 추출이 꺼진 잘못된 설정의 값이었다 —
[ESDF-BACKEND.md](ESDF-BACKEND.md) 의 "지지면: 추출과 소비는 별개다" 참고.

## 정직하게 적어 둘 두 가지

**참조 궤적에서 많이 벗어난다.** SEAM 참조 대비 편차가 중앙값 **1.69 rad** 다. 시작 궤적이
−151 mm 까지 파고든 상태였으니 크게 밀어낼 수밖에 없지만, 이 크기는 "충돌만 피하면 된다" 를
넘어선다. 비용 가중치(`w_track` 대 `w_slack`)를 이 씬에서 재조정하지 않았다.

**모든 청크 상태가 `violated` 다.** 바퀴 때문이다. 상태를 전체 구 기준으로 판정하므로 팔이
완전히 안전해져도 `violated` 로 나온다. 상태 판정을 어떤 구 집합에 걸 것인지가 아직 미결이다.

## 재현

`benchmark/trajopt/asset/esdf_rollout_field.json` 에 44청크 전부의 청크별 수치가 있다
(`arm_before_mm`, `arm_after_mm`, `clearance_*`, `esdf_unknown`, `target_voxels_carved`,
`n_candidates`, `iterations`, `reference_deviation`).
