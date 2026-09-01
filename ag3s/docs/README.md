# AG3S — 실모델 검증 (fine-tuned π0.5 `rby1_transport_14d`)

`asset/doc/`는 **설계 문서**(파이프라인 설명, 발표 자료)입니다. 이 폴더는 **실측 검증 기록**입니다.
합성 fixture가 아니라 실제로 추론 중인 파인튜닝 모델의 출력으로 각 단계를 하나씩 확인하고,
단계마다 표와 그림을 남깁니다.

## 왜 단계별인가

AG3S는 8단계 파이프라인이고, 각 단계는 앞 단계의 출력을 신뢰합니다. 마지막 단계(제약 집합)만
보고 "된다/안 된다"를 판정하면, 실패했을 때 어느 단계가 원인인지 알 수 없고 성공했을 때도
운이 좋았던 것인지 알 수 없습니다. 그래서 **각 단계의 출력을 그 단계의 ground truth와 직접
대조**합니다. MuJoCo 씬이므로 모든 단계에 정확한 ground truth가 존재한다는 것이 이 검증의
전제이자 최대 강점입니다.

## 단계

| # | 문서 | 검증 대상 | Ground truth | 상태 |
|---|---|---|---|---|
| 0 | [record_check](../experiments/record_check.py) | 관측 기록 — 추론 스텝마다 qpos + 정책이 실제로 본 3장의 224×224 이미지 | — | **run_0002 검사 통과** — 28/44 채점 가능 |
| 1 | [step-01-attention.md](step-01-attention.md) · [서버 프롬프트](step-01-server-prompt.md) | π0.5가 프롬프트가 지목한 물체를 보는가 | MuJoCo 세그멘테이션 | **실행 대기** |
| 2 | step-02-backprojection.md | 깊이 → 3D 점, 카메라 규약 | 물체 중심의 실제 좌표 | 미착수 |
| 3 | step-03-attention-lifting.md | 2D attention → 3D 점 매핑 | 점별 body id | 미착수 |
| 4 | step-04-target-grounding.md | 클러스터링 + 채점이 target을 고르는가 | 물체별 point mask | 미착수 |
| 5 | step-05-target-obstacle.md | target/obstacle 분리 후에도 기하가 남는가 | 후보 집합 vs 실제 물체 | 미착수 |
| 6 | step-06-geometry.md | primitive 근사가 점을 포함하는가 | 포함 검사 + 실제 크기 | 미착수 |
| 7 | step-07-constraints.md | 제약 집합 + TO가 충돌을 없애는가 | MuJoCo 접촉 | 미착수 |

## 실행 명령

### ① 서버 (GPU 서버에서)

```bash
cd <워크스페이스 루트>/src/openpi
.venv/bin/python scripts/serve_policy.py \
    --port 8123 \
    policy:checkpoint \
    --policy.config=pi05_rby1_lora \
    --policy.dir=<체크포인트 절대경로>
```

로컬 `--remote localhost:8123`은 VS Code SSH 포워딩을 타므로, 서버 쪽 포트는 8123이어야
합니다. 이 커맨드라인 전체를 어딘가에 적어 두세요 — 1단계 probe가 서버를 잠시 내리고
같은 인자로 다시 띄워야 합니다.

### ② 로컬 (기록을 남기는 롤아웃)

기존에 쓰시던 명령에 `--record-ag3s` 하나만 추가된 것입니다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
RUN=outputs/rby1_atomic_infer/ag3s_step1

src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d --remote localhost:8123 \
  --prompt "put the apple in the basket" \
  --fruit-layout-index 0 --fruit-slot-order apple banana orange pear \
  --obstacle-profile clear --max-steps 350 --start-delay 2 --speed 1.0 --view front \
  --record        $RUN/third_person.mp4 \
  --trajectory-out $RUN/trajectory.npz \
  --record-ag3s   $RUN/ag3s_records
```

→ `$RUN/ag3s_records/run_0002/` (스텝당 `.npz`, 44스텝에 3.7 MB). **이 하나의 기록을 1–7단계가
전부 재사용합니다.** 단계마다 롤아웃을 다시 돌리면 매번 다른 씬을 검증하게 되어 단계 간
비교가 불가능해집니다.

### ③ 현재 기준선 — 합성 attention으로 도는 AG3S 전체

실모델 검증에 들어가기 전에, 파이프라인이 지금 어떤 답을 내는지 찍어 둡니다. 여기 쓰이는
attention은 `gaussian_attention`(정답 위치에 놓은 가우시안)이고, 1단계가 대체하려는 대상이
바로 이것입니다.

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.rby1_transport \
    --json benchmark/ag3s/docs/baseline_synthetic_index.json \
    --images benchmark/ag3s/asset/image/baseline_synthetic
```

### ④ 1단계 — attention map

```bash
# (서버) 서버를 내리고 → probe → 같은 인자로 재기동. 상세: step-01-server-prompt.md
src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.pi05_attention \
    --records run_0002 --config pi05_rby1_lora \
    --checkpoint <서버가 로드했던 절대경로> \
    --out attention_step1_run0002.npz

# (로컬) 채점 — GPU 불필요
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.attention_report \
    --records $RUN/ag3s_records/run_0002 \
    --attention benchmark/ag3s/asset/attention/attention_step1_run0002.npz
```

→ `docs/step-01-attention.md` + 그림 4장 + 터미널 한 줄 판정.

### ⑤ 2–7단계 — 아직 도구가 없습니다

각 단계는 그 단계 고유의 ground truth 대조가 필요하고, 그 대조 코드가 아직 없습니다.
1단계 결과를 보고 하나씩 만듭니다 — 순서가 뒤집히면 안 되는 이유가 있습니다: 예를 들어
1단계가 "쓸 만한 head가 없다"로 끝나면 3단계(attention lifting)는 검증할 대상 자체가
달라지고, 4단계의 채점 기준도 다시 정해야 합니다.

각 단계가 필요로 할 대조는 다음과 같습니다.

| 단계 | 비교 대상 | 필요한 것 |
|---|---|---|
| 2 back-projection | 역투영한 점 vs 물체 중심 실좌표 | `run_0002` + `TransportScene`. 새 스크립트 하나 |
| 3 attention lifting | 점별 attention vs 점별 body id | 1단계의 `.npz` + `pixel_labels` |
| 4 target grounding | 고른 클러스터 vs 정답 물체의 point mask | 3단계 출력 + 클러스터 채점 |
| 5 target/obstacle | 후보 집합 vs 실제 존재하는 물체 | 4단계 출력. **개수 보존이 핵심** |
| 6 geometry | primitive가 점을 포함하는가, 크기가 맞는가 | `containment_report` + 실제 치수 |
| 7 constraints + TO | 최적화된 궤적의 MuJoCo 접촉 | `benchmark/trajopt` + 재생 하네스 |

## 도구

| 스크립트 | 실행 위치 | 하는 일 |
|---|---|---|
| `pi05_infer.py --record-ag3s DIR` | 로컬 | 추론 스텝마다 `.npz` 하나 — qpos, state(14), 정책 이미지 3장, action chunk |
| `benchmark.ag3s.experiments.record_check` | 로컬 | 기록이 forward pass를 쓸 값어치가 있는지 — 대상 가시성·이미지 정상성·씬 재생 |
| `benchmark.ag3s.experiments.pi05_attention` | **GPU 서버** | 체크포인트를 `return_attn_probs=True`로 로드해 3개 카메라 grid의 attention 추출 |
| `benchmark.ag3s.experiments.attention_report` | 로컬 (GPU 불필요) | 세그멘테이션 대조 채점 → 표 4개 + 그림 4장 + 이 폴더의 문서 |

### 왜 attention 추출만 GPU 서버인가

`localhost:8123`은 VS Code SSH 포워딩이고, 실제 정책 서버는 GPU 서버에 있습니다. websocket
프로토콜은 `actions`만 돌려주므로 실행 중인 서버에서 attention을 꺼낼 방법이 없습니다.
attention은 `Pi0Config.return_attn_probs=True`로 prefix/suffix를 직접 구동해야 나오고,
그러려면 체크포인트가 있는 곳에서 forward를 돌려야 합니다. **서버를 재시작할 필요는 없습니다** —
기록된 관측에 대해 오프라인으로 한 번 돌리는 별도 스크립트입니다.

## 기록 형식이 qpos만 저장하는 이유

깊이와 세그멘테이션은 저장하지 않습니다. MuJoCo는 결정론적이라 `qpos`가 로봇·크레이트·과일을
모두 고정하므로, `TransportScene`이 나중에 정확히 같은 깊이/세그멘테이션/내부·외부 파라미터를
재생성합니다 — AG3S가 이미 테스트를 통과해 온 바로 그 코드 경로로. 저장하면 파일이 약 50배가
되고, 더 나쁘게는 기록 시점의 렌더링과 분석 시점의 렌더링이 아무도 모르게 어긋날 수 있습니다.

정책 이미지 3장만은 그대로 저장합니다. 이것은 재생성하면 안 되는 유일한 것입니다 —
네트워크에 실제로 들어간 텐서이고, attention 검증은 "일치할 것으로 기대되는 재렌더링"이 아니라
모델이 본 픽셀 위에서 이뤄져야 합니다.

## 좌표 규약 (모든 단계가 여기에 의존)

`render_cam(..., match_rby1_dataset=True)`는 299×224(4:3)로 렌더한 뒤 224×224로 **찌그러뜨립니다**
— crop도 pad도 아닌 순수 가로 압축. 따라서 **정규화 좌표는 보존됩니다**: 정책 열 `u_p`는 정규화
열 `u_p/224`이고, AG3S가 재구성하는 640×480(역시 4:3) 프레임에서도 같은 정규화 열입니다.
attention은 offset 없이 단순 정규화 스케일링으로 점군에 올라갑니다.

토큰 배치는 추측이 아닙니다. `AlohaInputs`가 base → left wrist → right wrist 고정 순서로 이미지
dict를 만들고 `Pi0.embed_prefix`가 그 순서로 이어붙이므로:

| 정책 키 | MuJoCo 카메라 | AG3S CameraID | prefix 토큰 |
|---|---|---|---|
| `cam_high` | `zed_left` | head | 0–255 |
| `cam_left_wrist` | `wrist_cam_l` | left_wrist | 256–511 |
| `cam_right_wrist` | `wrist_cam_r` | right_wrist | 512–767 |
| (언어) | — | — | 768– |
