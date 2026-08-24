# fine-tuning된 VLA가 나오면 할 일

**이 문서가 존재하는 이유.** AG3S와 trajopt의 정확도·성능 수치는 전부 합성 fixture와 MuJoCo에서
나왔고, **attention은 대역품이다.** π0.5는 LIBERO의 224×224 agent view로 학습됐으므로 RB-Y1 프레임에
들이대면 의미 없는 숫자가 나온다. 그래서 실험은 지정한 물체 위에 가우시안 블롭을 놓는
`gaussian_attention`을 쓴다 — **지금까지 검증된 것은 attention *이후*의 전 단계이며 attention 자체가
아니다.**

RB-Y1으로 fine-tuning된 모델이 나오는 순간 그 대역품을 걷어낼 수 있다. 아래는 그때 순서대로 할 일이고,
각 항목은 "무엇을 확인해야 하는가"와 "어떻게 하면 틀렸다는 것을 알 수 있는가"를 함께 적었다.

---

## 0. 전제 — 먼저 확인할 것

체크포인트를 받자마자 확인해야 실패를 며칠 늦게 발견하지 않는다.

- [ ] **action_horizon이 H=50인가?** `seam_rby1.yaml`의 `seam_horizon`과 반드시 일치해야 한다.
      다르면 SEAM의 prior 정렬(`aligned_tail`)이 조용히 어긋난다.
- [ ] **action 포맷이 `[L 6 abs joint, L grip, R 6 abs joint, R grip]` 그대로인가?**
      바뀌었다면 `ChunkLayout.rby1()`을 새 매핑으로 다시 만들어야 한다.
      `benchmark/trajopt/types.py`의 `ChunkLayout`이 그 매핑을 데이터로 들고 있으므로 코드 변경은 없다.
- [ ] **attention을 꺼낼 수 있는가?** 서빙 경로가 attention을 노출하지 않으면 아래의 대부분이
      불가능하다. `benchmark/knows_vla/probe_p0b.py`의 `AttnSampler`가 π0.5에서
      `[L,B,K,G,T,S]` → 패치 그리드를 뽑는 경로를 이미 갖고 있으니 그것부터 확인한다.
- [ ] **이미지 해상도와 패치 격자 크기.** LIBERO는 224×224 → 16×16이었다. RB-Y1 fine-tune이
      다른 해상도면 `AttentionConfig.interpolation`이 다루는 재샘플 배율이 달라진다.

> **무거운 작업은 GPU 서버에서.** 체크포인트 로딩과 추론은 로컬에서 돌리지 않는다.
> 아래 항목 중 π0.5 추론이 필요한 것은 전부 서버용 런북으로 넘긴다.

---

## 1. Attention 추출 경로 확정 — 가장 먼저

**목표.** `AttentionAdapter` Protocol을 만족하는 실제 어댑터 하나.

AG3S는 이미 `AttentionAdapter` 뒤에 VLA를 격리해 두었다. 할 일은 그 Protocol을 구현하는 클래스를
하나 쓰는 것뿐이고, **AG3S 코어는 한 줄도 바뀌지 않는다.**

```python
class Pi05RBY1AttentionAdapter:
    def to_pixel_map(self, raw, out_hw: tuple[int, int]) -> np.ndarray:
        """(layer, head) 셀을 골라 패치 그리드를 뽑고 (H, W) float32로."""
```

- [ ] `benchmark/ag3s/experiments/` 아래에 어댑터 작성
- [ ] `attention.layer` / `attention.head`가 **ablation 축**임을 잊지 말 것.
      KNOWS 프로브가 논문의 (12, 3) 셀이 이 체크포인트에서 최선이 아님을 이미 측정했다
      (`benchmark/knows_vla/docs/08-p0b-results.md`). **어느 셀을 쓸지 스윕으로 정한다.**
- [ ] 스윕 지표: 아래 §3의 grounding IoU. 시각적 인상이 아니라 숫자로 고른다.

**틀렸다는 신호.** attention map이 이미지 전체에 균일하거나, 물체가 아니라 프레임 가장자리·로봇
자신에 몰린다. 그러면 셀 선택이 틀렸거나 전처리(정규화·리사이즈)가 학습 때와 다르다.

---

## 2. 대역품과 실물의 정면 비교 — 이 작업의 핵심 측정

**지금 있는 것이 이 비교를 위한 준비다.** `gaussian_attention`은 "GT 물체 위의 이상적인 블롭"이므로,
실제 attention과 나란히 돌리면 **파이프라인의 성능 상한과 실제치를 동시에** 얻는다.

- [ ] `experiments/rby1_transport.py`에 `--attention {gaussian,vla}` 플래그 추가
- [ ] 같은 씬·같은 프레임에서 두 번 돌려 아래 표를 채운다

| 지표 | gaussian (상한) | VLA (실제) | 어디서 |
|---|---|---|---|
| target IoU | | | `target_report()` |
| target purity | | | `target_report()` |
| grounding confidence | | | 〃 |
| centroid 오차 | | | 〃 |
| `GroundingStatus` 분포 | | | 프레임별 집계 |
| seed 개수 (p95) | | | `debug["seed_indices"]` |

**이 표가 이 문서 전체의 이유다.** 격차가 작으면 attention 이후 단계가 실물에서도 작동한다는 뜻이고,
크면 어느 단계에서 벌어지는지 위 지표가 지목해 준다.

---

## 3. Grounding 파라미터 재조정

합성 fixture에서 IoU 1.0000을 준 값들은 **실제 attention의 분포를 본 적이 없다.**

- [ ] **`attention.seed_percentile`** — 합성에서는 p95가 933점(2.3%)을 뽑았다. 실제 attention은
      훨씬 퍼져 있을 가능성이 높다. seed가 수만 점이 되면 연결성 성장이 씬 전체로 범람한다.
- [ ] **`attention.normalization`** — 4종 전부 순위를 보존하므로 seed 임계값과 독립적으로 고를 수 있다.
      percentile 범위 `(5, 99)`가 실제 분포에 맞는지 히스토그램으로 확인한다.
- [ ] **`clustering.target_score_threshold`** — 저신뢰 target을 지어내면 phase 규칙이 그 물체의
      여유를 줄여 **제약을 능동적으로 비활성화**한다. 실제 attention에서 confidence 분포를 보고
      **보수적으로** 정한다. 애매하면 `target=None`이 정답이다.
- [ ] **`clustering.w_attention` / `w_geometry`** (기본 0.7 / 0.3) — 실제 attention이 노이지하면
      기하 쪽 비중을 올리는 것이 맞을 수 있다.

**운용 범위는 이미 측정돼 있다**: `voxel_size < eps < 최소 물체 간격`. eps 3 cm는 2 cm 간격 물체를
병합한다(Euclidean 연결성의 본질적 한계이며 버그가 아님).

---

## 4. 다중 카메라 attention

- [ ] **세 카메라 전부에 attention이 있는가?** 서빙이 head 뷰만 attention을 노출한다면
      wrist 카메라는 `attention_map=None`으로 들어간다. AG3S는 그것을 정상 처리한다 —
      0을 기여하되 **지오메트리는 그대로 유지**하고 노트를 남긴다.
- [ ] **동일 정규화 강제 확인.** `process_observation`은 `config.attention` 하나만 받으므로
      구조적으로 보장되지만, `metrics["cameras_with_attention"]`로 실제 확인한다.
- [ ] **`max` 융합이 맞는 선택인지 재확인.** 합성 실험에서 세 카메라에 서로 다른 target을 지시했더니
      max가 가장 강한 신호를 골랐다(`seed_camera=right_wrist`). 실제로는 세 뷰가 같은 물체를 볼
      것이므로 동작이 다를 수 있다. `target.seed_camera` / `supporting_cameras`로 추적한다.

---

## 5. 실기 캘리브레이션 — MuJoCo가 숨기고 있는 것

**시뮬레이션에서 정합 오차가 0인 이유는 URDF FK와 시뮬레이터가 정확히 같기 때문이다**
(측정 0.0 mm). 실기에서는 그렇지 않다.

- [ ] **hand-eye calibration.** `T_link_cam`을 실측한다. 현재는 MuJoCo에서 역산한 값이다.
- [ ] **그 오차를 `geometry.perception_uncertainty`에 반영.** 현재 0이다.
      이 값은 **primitive 반지름에만** 더하고 `d_safe`에는 더하지 않는다 — 같은 1 cm를 두 번 세면
      모든 우회가 소리 없이 두 배가 된다.
- [ ] **depth noise 측정.** RealSense D435i는 0.8 m에서 2–3 mm로 알려져 있고, 합성 실험에서
      3 mm 노이즈에서 IoU 0.9983이었다. 실제 값을 재서 그 곡선의 어디에 있는지 확인한다.
- [ ] **timestamp skew 실측.** 현재 30 ms는 씬에 주입한 숫자다. 실제 파이프라인의 지터·드롭·클럭
      오프셋을 재고 `timing.max_camera_skew_sec`를 그에 맞춘다.
- [ ] **융합 보셀 12 mm 재검토.** "같은 표면이 세 장의 평행한 시트로 갈라지지 않을 만큼" 크게 잡은
      값이지 카메라 간 정합 오차를 측정해서 유도한 값이 아니다. 위의 캘리브레이션 오차를 먼저 재고
      되짚는다.

---

## 6. 로봇 충돌 모델 — 실기 전 반드시

- [ ] **손끝 collision geometry 확보.** RB-Y1 URDF는 torso + arm 0–5의 캡슐만 준다.
      접촉 허용 링크(`ee_right` · `ee_finger_r1/r2` · `ee_left` · `ee_finger_l1/l2`)에 sphere가
      없으면 **clearance 정책이 완화할 대상이 없어 grasp를 표현할 수 없다** — 그런데 아무 오류도
      나지 않고 target이 조용히 full margin을 유지한다.
      `tests/ag3s/test_robot_models.py::test_the_arm_only_urdf_model_carries_no_gripper_spheres`가
      이 사실을 고정하고 있다.
- [ ] 현재는 MuJoCo 시각 메시에서 유도한 캡슐을 `extra_capsules`로 주입해 우회한다.
      **실기에서는 제조사 충돌 모델을 확보하거나 손끝 기하를 직접 측정한다.**
- [ ] **self-filter 모델과 constraint 모델을 계속 분리해 둔다.** 필터는 카메라에 보이는 전신이
      필요하고(194구) 제약은 움직이는 체인만 있으면 된다(61구). 같이 쓰면 행 수가
      41,480 → 132,000으로 뛴다.

---

## 7. Phase와 active manipulator의 주입원 결정

AG3S도 trajopt도 **둘 다 추론하지 않는다.** 누가 어떻게 공급할지는 아직 미정이고, 실기 전에 정해야 한다.

- [ ] **Phase** — `TRANSIT / APPROACH / PRE_GRASP / GRASP` 전이를 누가 판단하는가?
      태스크 플래너? 사람? 거리 기반 휴리스틱? **AG3S 안에 FSM을 만들지 않는다**는 것이 제약이다.
- [ ] **Active manipulator** — `Phase.GRASP`는 잡고 있다는 사실만 말할 뿐 어느 손인지는 말하지 않는다.
      아무도 지명되지 않으면 아무것도 완화되지 않는다(fail-closed).
- [ ] **Grasp 확정 신호** — `ag3s.attach()`를 부를 주체. AG3S는 grasp 성공을 추론하지 않는다.
      힘 센서? 그리퍼 폭? 태스크 상태?
- [ ] **`ConstraintValidity` / `TrajOptStatus`에 대한 정책** —
      `GEOMETRY_INCOMPLETE`나 `VIOLATED` 프레임에서 정지할지, 유지할지, 직전 인증 씬을 재사용할지.
      두 계층 다 보고만 하고 결정하지 않는다. **이 정책이 없으면 안전 계층이 무의미하다.**

---

## 8. trajopt 실측 재확인

trajopt의 28.4 ms는 **합성 시나리오**에서 나왔고 실제 π0.5 청크로는 미검증이다.

- [ ] **실제 청크가 로봇의 운동학 한계를 위반하는가?** 데모에서 학습한 정책은 충분히 그럴 수 있다.
      속도·가속 행이 slack으로 부드럽게 돼 있으므로 QP는 항상 풀리지만, `limit_report`가 잔여
      위반을 보고한다. **그 값을 프레임별로 로깅한다.**
- [ ] **latency 재측정.** 실제 씬은 후보가 더 많고 활성 행이 더 많다.
      한 제어 주기(66.7 ms)를 넘기면 루프가 멈춘다.
- [ ] **`plan_horizon` 재조정.** 32는 합성 측정에서 나온 값이다. 이음매 불연속
      (`metrics["plan_join_discontinuity_rad"]`, 합성에서 0.02 rad)을 실제 데이터로 확인한다.
- [ ] **solver 재벤치마크.** `python -m benchmark.trajopt.qp --benchmark`를 **실기 장비에서** 돌린다.
      osqp가 이겼지만 그것은 이 개발 머신의 측정이다.

---

## 9. SEAM ↔ TO 되먹임 — 미해결로 남겨둔 것

**`seam_policy.py:194`가 정제 *전* 모델 공간 청크를 다음 prior로 저장한다.** 정제된 physical chunk는
로깅용으로만 남는다. 즉 **TO의 수정은 SEAM의 prior에 전혀 반영되지 않는다.**

방치하면 SEAM이 매 청크 같은 충돌 궤적을 다시 제안하고 TO가 매번 밀어내면서, SEAM이 없애려던
경계 jerk가 TO 계층에서 되살아난다.

- [ ] **현재의 완화책이 실제로 충분한지 측정.** 연속성 항(`w_continuity`)이
      `context["previous_physical_chunk"]`를 쓴다. 실기 데이터에서 청크 경계 jerk를 재서
      SEAM 단독 대비 얼마나 나쁜지 확인한다.
- [ ] 충분하지 않으면 **physical → model 역변환**이 필요하고 이는 SEAM 수정을 수반한다.
      그때는 별도 작업으로 분리하고, **SEAM을 건드린다는 사실을 명시적으로 결정한다.**

---

## 10. 실기 롤아웃 전 최종 체크

- [ ] `src/openpi/.venv/bin/python -m pytest tests/ag3s tests/trajopt -q` (현재 501 passed)
- [ ] 3-카메라 융합 실행에서 같은 물체가 뷰 전반에 걸쳐 단일 id를 갖는지 확인
- [ ] 실제 attention으로 `GroundingStatus` 분포 확인 — `OK`가 아닌 프레임의 비율
- [ ] `ConstraintValidity` 분포 확인 — `VALID`가 아닌 프레임의 비율
- [ ] TO의 `TrajOptStatus` 분포와 프레임별 latency 히스토그램
- [ ] **정지 정책이 실제로 배선돼 있는가** (§7의 마지막 항목)

---

## 참고 — 지금 무엇이 이미 준비돼 있는가

| 필요 | 이미 있는 것 |
|---|---|
| VLA 격리 | `AttentionAdapter` Protocol — AG3S 코어 수정 불필요 |
| attention 추출 경로 | `benchmark/knows_vla/probe_p0b.py`의 `AttnSampler` |
| 다중 카메라 | `AG3S.process_multi()` — capture-time FK, provenance CSR |
| 접촉 정책 | `clearance.ClearancePolicy` — phase × manipulator × source × link |
| 잡은 물체 | `ag3s.attach()` / `detach()` — 명시적 외부 이벤트만 |
| TO | `benchmark/trajopt/` — SQP + OSQP, RTI, 하드 캡 |
| SEAM 훅 | `TrajOptChunkRefiner` — SEAM 코드 수정 없음 |
| ablation | 전부 config 한 줄. 코드 수정도 브랜치도 없음 |

**아직 없는 것**: MuJoCo 재생 실험(`benchmark/trajopt/experiments/rby1_replay.py`),
`pi05_infer.py`의 `--trajopt` 플래그 배선.
