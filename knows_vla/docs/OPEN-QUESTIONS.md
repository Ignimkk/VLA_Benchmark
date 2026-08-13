# OPEN QUESTIONS — KNOWS

논문에서 복구되지 않아 구현자가 결정해야 하는 항목. 모든 `[ASSUMPTION]`은 여기에 등재된다.

**영향도**: ●●● 재현 결과를 좌우 / ●● 동작에 영향 / ● 세부사항

---

## 1. ●●● 하이퍼파라미터 5개가 논문에 없음

본문 §3.3은 "$K, \beta, \delta$ 값은 부록에 있다"고 하나 **부록에 없다**. 추가로 CBF 쪽
$\gamma_h, W, \epsilon$도 정의만 있고 값이 없다. 복구되는 것은 $\beta = -1$ 뿐.

| 기호 | 역할 | 영향 |
|---|---|---|
| $K$ | Eq. (3) attention 누적 윈도 길이 | 신호 평활도 ↔ 반응 지연 |
| $\delta$ | Eq. (4) top-1/top-2 gap 임계 | **SR/CR 트레이드오프를 직접 지배** |
| $\gamma_h$ | Eq. (7) CBF 감쇠율 | 필터 보수성 |
| $W$ | Eq. (12) 회전 대 병진 가중 | 자세 추종 품질 |
| $\epsilon$ | Eq. (12) normal 변화 상한 | **실효 안전 제약을 $\epsilon\|\nabla_nh\|_1$만큼 완화** ([03-math.md](03-math.md)) |

**해소 방법**: 저자 문의. 불가 시 스윕 — 단, 두 가지 결합을 주의할 것.
(a) $\delta$는 $\bar A_t$ 정규화 규약(#10)에 스케일 의존적이므로 #10을 먼저 고정한다.
(b) **$\epsilon$과 $\gamma_h$는 독립 축이 아니다.** 실효 제약이 $\epsilon\|\nabla_nh\|_1$만큼
완화되는데, 실제 장면에서 $\|\nabla_nh\|_1 \sim 1$이라 $\epsilon=0.05$면 완화량이 요구량
$-\gamma_h h \approx 0.04$를 넘는다 — 필터가 사실상 비활성이 된다(P2b에서 $h=-3.3$ cm인데
개입 0.00 cm 관측). 반드시 공동 스윕할 것. [11-p2b-results.md](11-p2b-results.md) §4
**참조**: [00-evidence.md](00-evidence.md) §2, [03-math.md](03-math.md)

## 2. ~~EEF delta의 프레임과 회전 표현~~ — **완전 해소 (P1 + P2a)**

부록 §7.1은 $\delta c_R \in \mathbb{R}^3$, $\delta\theta \in \mathbb{R}^3$라고만 한다.
- $\delta c_R$이 world/base 프레임인가 EEF 프레임인가?
- $\delta\theta$가 축각인가, 오일러각인가, body-frame 각속도 증분인가?

**왜 중요한가**: Eq. (10)의 $\nabla_{R_R}h = (n \times Q_Rn)/\sqrt{n^\top Q_Rn}$는 특정
회전 매개변수화를 전제한다. 틀리면 필터가 잘못된 축으로 회전시켜 **충돌 방향으로 밀 수 있다.**

**해소된 부분 (P1)**: Eq. (10)이 전제하는 규약을 확정했다. $Q_R \to RQ_RR^\top$이고
$R \approx I + [w]_\times$일 때 $d(-\sqrt{n^\top Qn}) = w\cdot(n\times Qn)/\sqrt{n^\top Qn}$로
Eq. (10)과 정확히 일치한다. 즉 $\delta\theta$는 **world 프레임 축각 증분**이고 EEF 타원체는
그리퍼와 강체로 회전한다. 논문은 둘 다 명시하지 않는다.
[09-p1-results.md](09-p1-results.md) §2.1, `test_grad_rotation_matches_finite_difference_under_R_Q_Rt`

**확인 (P2a)**: LIBERO/robosuite OSC_POSE가 정확히 그 규약이다.
- `goal_pos = ee_pos + scaled_delta[:3]` → world 프레임
- `goal_ori = R(delta) @ R_current` → **좌측 곱셈 = world 프레임 회전**, delta는 축각
- 스케일: 위치 × 0.05 m, 회전 × 0.5 rad (`osc_pose.json` `output_max`)

Eq. (10)을 그대로 쓸 수 있다. [10-p2-perception.md](10-p2-perception.md) §4

## 3. ●● EEF 타원체 $Q_R$의 오프라인 캘리브레이션 절차

§3.1은 "semi-axes are calibrated offline"이라고만 한다. 방법·데이터·안전 여유 포함 여부 미기재.
과대 추정 → 과보수, 과소 추정 → 충돌.
**잠정 가정** [ASSUMPTION]: 그리퍼 CAD/충돌 메시의 MVEE.

## 4. ●●● 저자가 사용한 π0.5 체크포인트

논문 Table 2는 $H{=}8$(243 ms ÷ 8 ≈ 30 ms/step)인데 공개 `pi05_libero`는
`action_horizon=10`이다. 어떤 체크포인트인지 밝히지 않았고, $H{=}8$은 공개 가중치로 재현 불가.
**대응**: SEAM과 동일 원칙 — 체크포인트를 논문에 맞추려 바꾸지 않고, 같은 체크포인트 위에서
baseline 대 KNOWS를 비교. 절대 수치 비교는 포기.
**참조**: [00-evidence.md](00-evidence.md) §4

## 5. ●●● SafeLIBERO Level III 재구현 사양

논문이 밝힌 것: 두 waypoint 사이 직선, 30 제어 스텝(20 Hz에서 1.5 s), 이후 정지.
**밝히지 않은 것**: waypoint 선정 규칙, 장애물 개수, 이동 시작 타이밍, 태스크별 배치,
"adversarially"의 구체적 의미(로봇 경로를 향해 오는가?).
**대응**: 우리 선택을 전부 `[ASSUMPTION]`으로 명시 문서화. Level III 결과는 논문과 절대
비교하지 않고 **조건 간 상대 비교**로만 사용.

## 6. ●● Naive 베이스라인이 [9]의 실제 구현인가

논문은 "a strong stand-in for prior init-only filters"라고 표현한다 — 즉 [9]의 코드가 아니라
저자의 대리 구현이다. 게다가 **ground-truth 세그멘테이션 + 특권 시뮬레이터 상태**를 쓴다.
**함의**: "KNOWS ≈ Naive"(Level I/II)는 *지각 오차를 감수하고도 oracle과 대등하다*는 더 강한
주장이다. 반대로 Naive가 [9]보다 약하게 구현됐다면 KNOWS의 우위가 과대평가된다.
**해소 방법**: vlsa-aegis @57b1aef에서 [9]의 실제 안전 레이어 기하 표현을 확인.

## 7. ●● H200 서버 배치

저장소·`pi05_libero` 체크포인트·LIBERO/robosuite/vlsa-aegis가 서버에 어떻게 놓이는지 미확인.
로컬 RTX 4060 Ti 8GB로는 π0.5 추론이 불가하므로 P3/P4는 전부 서버 의존.
**해소 방법**: 실행 착수 시 확인.

## 8. ●●● Naive vs KNOWS는 변수가 두 개 다르다

Naive는 (a) init-only이고 (b) **단일 장애물**이다. KNOWS는 (a) 매 스텝 갱신이고
(b) **추적된 모든 비타깃 객체**가 장애물이다. 두 변수가 동시에 바뀌므로
`reproduction.md` §8("통제 비교당 한 변수만")에 어긋난다.
**대응** [Proposed improvement, 논문에 없음]: "다중 장애물 + init-only" 조건을 추가해
attention 기반 매 스텝 갱신의 순수 기여를 분리. 논문 재현과 분리 표기.
**참조**: [01-problem.md](01-problem.md) §5, [06-repro-plan.md](06-repro-plan.md) §8.3

## 9. ●●● YOLOe 파인튜닝 레시피 전무

§4.1 "We finetune YOLOe to segment manipulable objects"가 전부다. 데이터·에폭·LR·증강·
프롬프트 설정 미기재이고 가중치도 미공개. 지각 품질이 타원체 기하 → 안전성에 직결된다(§5).
**함의**: **"training-free"는 VLA 정책에 한정된 주장**이며 시스템 전체는 학습된 모듈에 의존한다.
**대응**: P0/P1에서는 GT/수동 마스크를 써서 지각 오차와 방법론을 분리. P3에서만 실제 검출기 도입.

## 10. ~~$\bar A_t$의 정의~~ — **부분 해소 (P0b)**

부록 §7.2는 추출 블록이 $H \times g^2$라 하고 본문 Eq. (2)는 $\bar A_t \in \mathbb{R}^{g\times g}$를
쓴다. 바(bar)가 평균인지 합인지 마지막 쿼리인지 정의가 없다. 정규화 여부도 불명.
**왜 중요한가**: $\delta$(#1)가 $d_i$의 절대 스케일에 의존하므로, 집계 규약이 바뀌면
$\delta$도 함께 바뀐다. **#1을 스윕하기 전에 이것부터 고정해야 한다.**
**해소**: mean과 sum은 타깃 선택에 대해 **수학적으로 동일**하다(sum = $H\times$mean이고 Eq. (3)
분자만 균일 배율 → 순서 불변). `last`는 일관되게 나쁘다(0.788 vs 0.846). **mean 권장.**
**남은 부분**: $\delta$는 절대 임계라 집계 규약에 따라 $H$배 달라진다 — #1 스윕 전에 고정할 것.
[08-p0b-results.md](08-p0b-results.md) §3

## 11. ~~어느 디노이징 스텝의 attention인가~~ — **해소 (P0b), 영향 ●**

π0.5는 청크 하나에 $N$번(openpi 기본 10) Euler 스텝을 돌고 매 스텝 suffix forward pass가 있다
→ 청크당 attention 그리드가 $N$개. 논문은 "During each policy query we obtain an attention
grid"라고 단수로만 말한다. 노이즈가 큰 초기 스텝과 마지막 스텝의 attention은 다를 수 있다.
**해소**: Euler 스텝 0/5/9의 적중률이 0.846/0.829/0.833으로 **거의 무관**하다. 편한 스텝을
쓰면 된다. [08-p0b-results.md](08-p0b-results.md) §3

## 17. ● layer 12 안에서 head 3이 최적이 아니다

1,479스텝 스윕에서 (12,4)=0.715 > (12,3)=0.682이다. 차이가 0.03이고 표본이 40 에피소드라
유의하다고 말할 수 없다. **논문대로 head 3을 쓰되**, 표본을 늘렸을 때 h4가 일관되게 앞서면
재검토한다. 레이어 선택(12)은 논문과 일치하므로 유지.
[08-p0b-results.md](08-p0b-results.md) §2

## 18. ●●● 배치 단계에서 타깃 식별이 열화된다

접근 0.853 vs 배치 0.568 (우연 0.199). 원인이 둘 중 무엇인지 미확정:
(a) 첫 그리퍼 폐쇄를 경계로 삼는 우리 단계 판정이 거칠어 재시도 구간이 섞인 것,
(b) 정책이 배치 중에도 쥔 물체를 계속 보는 것.

**(b)라면 KNOWS에 직접적 결함이다** — $\tau_t$가 쥔 물체로 남으면 목적지가 장애물 집합에
들어가 필터가 놓아야 할 곳을 회피한다. 논문 §5의 한계 목록에 없는 항목이다.
**해소 방법**: `success=True` 에피소드만 골라 재채점.
[08-p0b-results.md](08-p0b-results.md) §3

## 12. ●● 기호 $K$의 중복 사용 / 실행 horizon 미기재

논문은 Eq. (3)의 슬라이딩 윈도를 $K$로 쓰는데, **실행 horizon(청크에서 몇 개를 실행하고
재계획하는지)은 아예 밝히지 않는다.** 이 문서에서는 후자를 $K_{\mathrm{exec}}$로 구분해 쓴다.
연쇄 효과: attention 그리드는 $K_{\mathrm{exec}}$ 스텝 동안 고정이고 마스크만 갱신되므로
$m_{i,t}$는 매 스텝 변한다 [DERIVED]. Eq. (3)의 "last $K$ frames"가 제어 스텝인지 정책
질의인지도 불명.
**참조**: openpi LIBERO 예제는 `replan_steps=5`.

## 13. ●●● $\delta c_R, \delta\theta$의 크기 제한이 없음 — **실측 확인 (P2b)**

Eq. (12)는 $\|\delta n\|_\infty \le \epsilon$만 제한하고 $\delta c_R$, $\delta\theta$에는 상한이 없다.

**측정 (5,879 스텝)**: 최대 보정이 **181 cm**(`target=gt`), **510 cm**(`none`). OSC 액션 한계가
5 cm이므로 각각 36배, 102배다. 실제 로봇이라면 부서진다.

**더 중요한 귀결**: 상한이 없으면 QP가 비가능해지지 않으므로 **논문의 emergency-stop 폴백이
단 한 번도 실행되지 않는다(0.0%)**. 즉 부록 §7.1의 그 분기는 Eq. (12) 그대로면 도달 불가능한
경로다. $\|\delta c\|_\infty \le 5$ cm를 걸면 비상정지가 **23.8%**로 뛴다.
[11-p2b-results.md](11-p2b-results.md) §3

## 19. ●●● Eq. (4)에 "쥐고 있는 물체" 범주가 없다 — **P2b 신규**

Eq. (4)는 객체를 타깃 하나 vs 나머지 전부 장애물로만 나눈다. 배치 단계에 타깃이 목적지로
옮겨가면 **쥔 물체가 장애물로 강등**되는데, 그리퍼가 물리적으로 잡고 있으므로 $h<0$이 구조적으로
불가피하다. 필터가 자기가 든 물건에서 영원히 도망치려 한다.

**측정**: BDDL 정답으로 타깃을 완벽히 제외해도 $h<0$이 **82.1%의 스텝**에서 발생한다.
즉 attention 오차와 무관한 구조적 문제다. 논문 §5의 한계 목록에 없다.

**대응**: Eq. (4)에 세 번째 범주(`held`)가 필요하다. 그리퍼 상태로 파지 감지는 쉽지만 **논문에
없는 추가이며, "attention만으로 타깃/장애물을 나눈다"는 주장의 범위를 좁힌다.**
[11-p2b-results.md](11-p2b-results.md) §2

## 14. ●● Table 2 지연 합산이 예산을 초과한다

wrapper 49.3 ms + 정책 상각 30 ms = 79 ms > 50 ms(20 Hz). §4.3은 "holds control rate"라고
결론짓는다. §4.1이 정책 서버(GPU)와 클라이언트(렌더링+안전필터)를 분리했다고 밝히므로
**두 항이 파이프라인으로 겹친다**는 전제라야 성립하나, 논문은 이를 명시하지 않는다.
**대응**: 재현 시 동기 지연과 파이프라인 지연을 구분 측정. SEAM의 synchronized latency 개념 재사용.
**참조**: [04-pipelines.md](04-pipelines.md) §2.2

## 15. ●● 청크 경계에서의 타깃 전환 처리

$\tau_t$가 바뀔 때 히스테리시스가 있는가? 새 타깃/장애물의 가상 normal $n^{(j)}$는 어떻게
초기화되는가? Eq. (3)의 $K$ 누적이 attention 노이즈는 완화하지만 청크 경계 불연속은 다른 문제다.
**잠정 가정** [ASSUMPTION]: 히스테리시스 없음, 새 장애물의 $n$은 중심 간 방향으로 초기화.

## 16. ~~`data/policy_records/`의 출처와 형식~~ — **해소됨 (P0a)**

**LIBERO 데이터가 아니다.** state 14차원, actions (50,14), 카메라
`cam_high`/`cam_left_wrist`/`cam_right_wrist`, prompt "put the blue block in the brown box" —
RB-Y1 파이프라인 기록이며, 이를 생성한 `pi05_rby1_lora` 체크포인트는 로컬에 없다
(로컬 보유: `pi05_base`, `pi05_droid`, `pi05_libero`). 객체 GT 마스크도 없다.
→ P0b 입력원 결정이 새 과제. [07-p0a-results.md](07-p0a-results.md) §5

---

## 그 밖의 소소한 미기재 (영향 ●)

- Eq. (3)에서 $\sum_K \alpha_{i,t} = 0$(윈도 내내 완전 가림)일 때의 처리
- $Q$ 고유값 하한(floor). 납작한 객체로 $Q$가 특이에 가까워지면 Eq. (10)(11)의 분모
  $\sqrt{n^\top Qn} \to 0$. 논문에 규정 없음 [ASSUMPTION]
- 객체가 장면에 새로 등장하거나 사라질 때의 트랙 생성/소멸 규칙
- 멀티뷰 깊이 융합에서 LIBERO의 가용 뷰 수와 외부 파라미터 획득 방법
- **MVEE 이상치 처리 — 논문 전무, 그러나 필수 (P2a)**. 마스크 경계 픽셀 몇 개가 시선 방향으로
  타원체를 늘린다(`alphabet_soup_1` 반축 23.8 cm → 트리밍 후 6.2 cm). 깊이 2–98 백분위 트리밍을
  **공학적 추가**로 넣었다(방법의 일부 아님). [10-p2-perception.md](10-p2-perception.md) §3
- MVEE 수렴 허용오차, HSV 히스토그램 bin 수 및 Bhattacharyya 임계
- ~~OSQP 솔버 옵션~~ — **결정 (P1)**: 기본 허용오차 1e-3에서 Eq. (8) primal residual이 4e-6
  남는다. `eps_abs = eps_rel = 1e-9`, `max_iter = 20000`으로 조였고 비용은 무시할 만하다
  (10개 장애물에서 ~1 ms). 논문 미기재이므로 [ASSUMPTION].
  [09-p1-results.md](09-p1-results.md) §4
- 타원체 볼록껍질을 이미지 평면에 래스터화하는 구체적 방법(§3.3)

---

## 논문 서술과 다른 결론에 도달한 항목 (참고 — 방법은 논문대로 구현)

| 항목 | 논문 서술 | 우리 분석 | 근거 |
|---|---|---|---|
| Eq. (6)의 성격 | "practical safety margin rather than a formal collision-free certificate" | 고정 단위 $n$에 대해 $h(n)\ge0$은 분리의 **충분조건**. $\gamma$ 소거는 최적 $\gamma$ 선택과 동치이므로 인증을 약화시키지 않음. 실제 인증 손실은 선형화·$\epsilon$ 박스·OSC 추종오차에서 옴 | [03-math.md](03-math.md), [36] 원문 대조 |
| 헤드라인 "43%" | 상대 감소로 읽히는 서술 | **절대 43.9 퍼센트포인트** (상대로는 53–72%) | [00-evidence.md](00-evidence.md) §3 |
| 지연 예산 | "holds control rate" | 순차 합산 시 79 ms > 50 ms. 파이프라이닝 전제 필요 | #14 |
