# 새 컨테이너 환경 구성 — trajopt 어댑터 작업 인계

> 이 문서 하나로 새 컨테이너에서 작업을 이어받을 수 있게 하는 것이 목적이다.
> 같이 읽어야 하는 문서 둘: [`AG3S_REVIEW_PLAN.md`](AG3S_REVIEW_PLAN.md) (무엇을 왜 하는가),
> [`AG3S_REVIEW_LOG.md`](AG3S_REVIEW_LOG.md) (지금까지 무엇을 찾고 고쳤는가).
>
> **다음 작업**: cuRobo `VoxelGrid` → trajopt `SceneSnapshot` 어댑터. 배경은 이 문서 §6.

---

## 새 세션을 시작할 때 — 이 순서로 읽으면 된다

새 컨테이너의 Claude 에게 줄 지시문은 이것으로 충분하다.

> `benchmark/ag3s/docs/` 의 문서 세 개를 이 순서로 읽고 작업을 이어받아라.
>
> 1. **`CONTAINER_SETUP.md`** (이 문서) — 환경 구성, 동작 확인, 다음 작업의 배경.
>    §3 의 확인 3단계를 **먼저 돌려서** 환경이 맞는지 증명하고 시작할 것. 특히 (2) 의
>    회귀 기준선 숫자가 다르면 코드를 고치기 전에 그 원인부터 찾아라.
> 2. **`AG3S_REVIEW_PLAN.md`** — 무엇을 왜 하는가. "방향 결정" 절이 핵심이다
>    (cuRoboV2 를 인프라로 쓰고, attention 은 해상도를 배분한다).
> 3. **`AG3S_REVIEW_LOG.md`** — 지금까지 무엇을 찾고 고쳤는가. 길다(991줄). 맨 위의
>    **진행 현황 · 누적 발견 표 · 모듈 처분 표** 세 개만 먼저 읽으면 전체 위치를 알 수 있고,
>    상세는 필요할 때 해당 Step 절을 찾아 읽으면 된다.
>
> 그리고 **`VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md`** 가 같은 디렉터리에 있다 —
> 미세 ESDF 의 중심을 action chunk 의 swept volume 으로 잡는다는 결정이 거기 있고,
> 이 문서 §7 이 그것과 검토 결과의 정합을 정리해 두었다.
>
> 첫 작업은 **cuRobo `VoxelGrid` → trajopt `SceneSnapshot` 어댑터**다. 배경·이미 확인된 것·
> 밟았던 함정은 이 문서 §6 에 있다.

### 작업 방식 — 이어서 지킬 것

`AG3S_REVIEW_PLAN.md` "진행 방식" 에 있는 규칙이 계속 유효하다. 요약하면:

- 각 스텝은 `설명 → 사용자 질문(여기서 멈춤) → 공동 판정 → 확정된 것은 그 자리에서 수정+검증
  → 기록` 순서.
- **추측을 기록에 남기지 않는다.** 가능하면 그 자리에서 실행해 수치로 확정한다.
- **설명은 평이한 요약을 먼저, 코드 근거(`파일:줄`)는 그 다음.**
- 테스트가 필요하면 **먼저 사용자에게 요청**한다 — 체크포인트/MuJoCo 실행은 임의로 돌리지 않는다.
- 기록은 `AG3S_REVIEW_LOG.md` 에 **이어 쓴다.** 새 기록 파일을 만들지 않는다.

---

## 0. 지금 상태 요약 (2026-09-11)

- AG3S/trajopt 검토 Step 0~4 완료, Step 5~11 남음. 발견 17건 중 8건 수정, 1건 기각, 8건 미판정.
- ESDF 인프라를 **cuRoboV2 로 교체**하기로 결정했고, 그 API 가정을 이 장비에서 검증 완료.
- **아직 안 한 것이 trajopt 어댑터**다 — cuRobo 가 주는 `VoxelGrid` 를 우리 최적화기가
  읽는 `SceneSnapshot` 에 물리는 일. 가장 불확실한 부분이라 다음 차례로 잡았다.

---

## 1. 옮겨야 하는 것

| 대상 | 크기 | 방법 | 필수 |
|---|---|---|---|
| `benchmark/` (코드) | 45 MB | **git clone** — `https://github.com/Ignimkk/VLA_Benchmark.git` | 필수 |
| `src/rby1_description/` | 350 MB | 복사 (MuJoCo 모델·메시). `pi05_TO_hybrid/rby1_description` 의 심볼릭 경로 | 필수 |
| `run_0004/` | 9.4 MB | 복사 — 검증에 쓰는 기록된 롤아웃 | 필수 |
| `attention_step1_run0004.npz` | 63 MB | 복사 — 그 롤아웃의 실측 attention | 필수 |
| `papers/` | 8.2 MB | 복사 (cuRoboV2, cuRobo v1 PDF) | 권장 |
| `attention_step1_run0002.npz` + `run_0002` | 63 MB | 복사 | 선택 (이전 비교용) |
| **체크포인트 `29999`** | **8.9 GB** | rsync | **필수** — 아래 |
| `pi05_TO_hybrid/assets/pi05_rby1_atomic_lora/` | 16 KB | 복사 | 필수 (체크포인트와 짝) |
| `pi05_TO_hybrid/openpi/` | — | git clone | 체크포인트를 띄울 때만 |

### 체크포인트

```
pi05_TO_hybrid/checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999/
├── params/         6.0 GB   <- 추론에 필요한 것
├── train_state/    3.0 GB   <- 옵티마이저 상태. 추론에는 안 쓴다
├── assets/          16 KB
└── _CHECKPOINT_METADATA
```

**추론만 할 거면 `train_state/` 는 빼도 된다** — `policy_config.py:57` 이
`restore_params(checkpoint_dir / "params")` 만 부른다. 그러면 8.9 GB 가 **6.0 GB** 로 준다.
학습을 재개할 계획이면 통째로 옮긴다.

크기가 크므로 tar 로 묶지 말고 **rsync 로 이어받기 가능하게** 보내는 편이 낫다:

```bash
CKPT=pi05_TO_hybrid/checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999
rsync -avP --exclude 'train_state' "$CKPT" <새컨테이너>:<경로>/   # 추론만 할 경우
rsync -avP "$CKPT" <새컨테이너>:<경로>/                            # 학습 재개까지 할 경우
```

**config 와 체크포인트는 반드시 짝이 맞아야 한다** (`pi05_rby1_atomic_lora` ↔ `29999`).
둘은 norm stats 가 다르고 pi0.5 는 state 를 이산 언어 토큰으로 prefix 에 넣으므로, 틀리면
**prefix 토큰 자체가 달라져 attention 이 달라지는데 shape 도 범위도 정상으로 보인다.**
(`RUNBOOK.md` 의 경고)

체크포인트로 정책 서버를 띄우는 명령은 `RUNBOOK.md` §1 에 있다. 다만 **trajopt 어댑터
작업 자체에는 체크포인트가 필요 없다** — 기록된 롤아웃(`run_0004` + 추출해 둔 attention)으로
충분하다. 체크포인트가 필요해지는 것은 새 롤아웃을 따거나 closed-loop 로 검증할 때다.

### 옮기기 전에 — 미커밋 변경을 반드시 push

이 서버의 `benchmark` 작업 트리에 **커밋되지 않은 변경 18건**이 있다 (Step 1~4 수정 전부 +
문서 + 새로 만든 `experiments/curobo/`). clone 만 하면 그게 전부 사라진다.

```bash
cd /mnt/dev/work/benchmark
git status --short          # 무엇이 바뀌었는지 먼저 확인
git add -A && git commit -m "ag3s/trajopt 검토 Step 0~4 + cuRobo 검증"
git push
```

### 복사 (예시 — 경로는 새 컨테이너에 맞게)

```bash
# GPU 서버 컨테이너에서 tar 로 묶어 보내는 쪽이 빠르다
cd /mnt/dev/work
tar czf /tmp/ag3s_migrate.tgz \
    run_0004 attention_step1_run0004.npz papers \
    src/rby1_description
# 새 컨테이너에서
tar xzf ag3s_migrate.tgz -C <새 작업 루트>
```

**디렉터리 구조는 유지해야 한다.** `mujoco_source.py` 가
`src/rby1_description/models/rby1a/mujoco/model_transport.xml` 를 **작업 루트 기준 상대경로**로
찾는다. `run_0004/meta.json` 안의 `model_xml` 은 로컬 PC 의 절대경로라 무시되고, 이 상대경로가
실제로 쓰인다.

---

## 2. 파이썬 환경 — **두 개를 따로 만든다**

`numpy`/`scipy` 버전이 충돌해서 하나로 합칠 수 없다. curobo 는 최신 numpy 를 원하고,
AG3S 쪽은 검증된 조합에 고정되어 있다.

### 2-A. AG3S/trajopt 환경 (CPU 파이프라인)

```bash
python3.11 -m venv .venv-ag3s
.venv-ag3s/bin/pip install -r benchmark/requirements-ag3s.txt
```

확인:

```bash
MUJOCO_GL=osmesa PYTHONPATH=<작업루트> .venv-ag3s/bin/python -c "
import mujoco, casadi, osqp, scipy, numpy
print('ok', mujoco.__version__, numpy.__version__)
from benchmark.ag3s.pipeline import AG3S
from benchmark.trajopt.linearize import CollisionLinearizer
print('benchmark import ok')"
```

### 2-B. cuRobo 환경 (GPU)

torch 는 **새로 받지 말고** 컨테이너에 이미 있는 CUDA 빌드를 빌려 쓴다.

```bash
# 컨테이너의 torch 확인 — CUDA 가 True 여야 한다
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 그 python 으로 --system-site-packages venv 를 만든다
python3 -m venv --system-site-packages .venv-curobo
.venv-curobo/bin/python -c "import torch; print('torch 보임:', torch.__version__)"

# curobo 소스 설치. **nvcc 불필요** — setup.py 의 USE_PYBIND 기본값이 0 이라
# CUDA 확장 컴파일은 선택이고 기본은 JIT 백엔드다
git clone https://github.com/NVlabs/curobo.git ~/curobo_src    # 검증 시점 commit 78fd485
cd ~/curobo_src && ../.venv-curobo/bin/pip install -e . --no-build-isolation

# JIT 커널 백엔드 — 이것을 빠뜨리면 ESDF 단계에서
#   RuntimeError: No curobo kernel backend available!
.venv-curobo/bin/pip install -r benchmark/requirements-curobo.txt
```

확인:

```bash
.venv-curobo/bin/python -c "
import torch; from curobo.perception import Mapper, MapperCfg
from curobo._src.geom.types import SceneCfg, VoxelGrid
print('curobo ok, cuda:', torch.cuda.is_available())"
```

**검증된 조합 (2026-09-11, H200 NVL, driver 580.173):**
torch 2.4.1+cu124 / warp-lang 1.17.0 / cuda-core 1.2.0 / curobo `78fd485`.
CUDA toolkit(nvcc)은 이 서버에 **없었고, 없이 동작했다.**

---

## 3. 동작 확인 — 3단계

### (1) AG3S 파이프라인이 도는가

```bash
MUJOCO_GL=osmesa PYTHONPATH=<작업루트> .venv-ag3s/bin/python -m \
  benchmark.ag3s.experiments.curobo.export_frame
# -> /tmp/rby1_frame.npz, target centroid [0.554 0.302 0.854], 제약 구 120, 카메라 3
```

이게 되면 MuJoCo 모델·기록·AG3S 가 전부 제대로 연결된 것이다.

### (2) trajopt 까지 end-to-end

```bash
MUJOCO_GL=osmesa PYTHONPATH=<작업루트> .venv-ag3s/bin/python -m \
  benchmark.trajopt.experiments.esdf_rollout \
  --records run_0004 --attention attention_step1_run0004.npz --frames 15 \
  --voxel 0.020 --esdf-margin 0.05 --support-surfaces field --constraint-links arms \
  --out-json /tmp/check.json --out-doc /tmp/check.md
```

**기대값 (이 서버에서 Step 4 직후 측정, 회귀 기준선으로 쓸 것):**

```
청크  0 transit   전체  -28.7 →  -0.6 mm  violated
청크  5 approach  전체  -68.0 →  +1.7 mm  violated
청크 10 grasp     전체 -140.3 →  +3.7 mm  feasible
청크 14 grasp     전체  -73.9 →  +3.2 mm  feasible
15청크  해소 14  개선 15   상태: feasible 8, violated 7
```

숫자가 다르면 이관 과정에서 뭔가 달라진 것이다 — 코드를 고치기 전에 그것부터 찾아야 한다.

### (3) cuRobo 가 도는가

```bash
cd benchmark/ag3s/experiments/curobo
../../../../.venv-curobo/bin/python check_occupancy_frame.py   # 좌표계
../../../../.venv-curobo/bin/python verify_two_tier.py         # 2계층 정확성
../../../../.venv-curobo/bin/python bench_two_tier.py          # 시간
```

기대값은 `experiments/curobo/README.md` 와 `AG3S_REVIEW_LOG.md` 의 "cuRoboV2 API 검증" 절에 있다.
요지: 테이블 상판(참값 z=0.823)을 거친 20 mm 계층은 0.790(−33 mm), 미세 5 mm 계층은
0.821(−2 mm)로 본다.

---

## 4. 없는 것 — 알고 있어야 할 두 가지

**`tests/ag3s`, `tests/trajopt` 가 이 저장소에 없다.** `AG3S_REVIEW_LOG.md` Step 0 에 적힌
"560 passed" 기준선은 이 서버에서 재현할 수 없었고, 그동안의 회귀 확인은 전부 위 (2) 의
`esdf_rollout` 실측 대조로 대신했다. 그 스위트는 로컬 PC(`/home/mk/dev_ws/vla/pi0_TO_ws`)에만
있는 것으로 보인다. Step 11 이 그 스위트를 읽는 스텝이므로 그때 가져와야 한다.

**한글 폰트.** 그림을 만들려면 필요하다. 없으면 라벨이 전부 네모로 깨진다.

```bash
apt-get install -y fonts-noto-cjk
```

Noto CJK 는 `.ttc` 라 matplotlib 이 자동으로 못 잡는다. 코드에서 직접 등록해야 한다
(`experiments/curobo/` 밖의 그림 스크립트들이 쓰는 방식):

```python
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
from benchmark.ag3s.experiments import figstyle
assert figstyle.use_korean()      # JP 이름으로 등록되지만 한글이 들어 있다
```

---

## 5. 두 머신의 분업

```
로컬 PC (/home/mk/dev_ws/vla/pi0_TO_ws)        새 컨테이너 (GPU)
 ├─ pi05_infer.py  (VLA 추론, MuJoCo 뷰어)      ├─ AG3S 지각 파이프라인
 └─ tests/ag3s, tests/trajopt                  ├─ cuRobo TSDF/ESDF
                                               └─ trajopt (SQP/QP)
```

접속: `ssh blunex@ai.amrc.kr -p 21151` → `ssh root@172.21.121.112 -p 32542`
(현재 컨테이너 기준. 새 컨테이너 주소는 다를 수 있다.)

체크포인트 `29999` 로 서버를 띄우는 절차와 롤아웃 기록 절차는 `RUNBOOK.md` 에 있다.

---

## 6. 다음 작업 — trajopt 어댑터

**목표**: cuRobo `Mapper.compute_esdf()` 가 주는 `VoxelGrid` 를 우리 `SceneSnapshot` 에 물려서,
`benchmark/trajopt` 가 지금과 똑같이 돌게 한다.

### 알고 있는 것

우리 소비부가 필드에 요구하는 것은 **두 메서드뿐**이다 (`trajopt/linearize.py`):

```python
scene.esdf.distance(points)   # (N,3) -> (N,)   _esdf_clearance 에서
scene.esdf.gradient(points)   # (N,3) -> (N,3)  linearize 의 방향 벡터로
scene.esdf.grid.voxel_size    # E1 의 target 판정 허용오차로
```

그래서 어댑터 한 겹이면 될 것으로 **보이지만 확인 전이다.** cuRobo `VoxelGrid` 는
`feature_tensor`(128³ 거리장) + `pose`(중심) + `dims` + `voxel_size` 를 들고 있고,
월드 좌표는 `create_xyzr_tensor(transform_to_origin=True)` 가 준다.

### 밟았던 함정 (반복하지 말 것)

1. **`esdf_origin` 은 격자 코너가 아니라 중심이다.** 코너로 넣으면 관심 물체가 창 모서리에
   걸려 두 계층이 130~257 mm 어긋난다. 근거: `integrator_esdf.py` 의 `"Pose at center"` 주석,
   그리고 `vg.pose[:3]` 가 넘긴 값과 같게 나온다.
2. **인덱싱을 손으로 쓰지 말 것.** `create_xyzr_tensor(transform_to_origin=True)` 가 복셀
   월드좌표를 `feature_tensor.reshape(-1)` 과 같은 순서로 준다. 직접 계산해서 두 번 틀렸고,
   그때마다 "좌표계 불일치" 처럼 보였지만 **필드는 옳았다.**
3. **첫 `compute_esdf` 호출 시간을 쓰지 말 것** — JIT 컴파일 + CUDA graph capture 로 1302 ms 다.
   정상 상태는 0.5~0.7 ms. 반드시 워밍업 후 측정한다.

### 아직 정하지 않은 것

- **두 계층의 합성 규칙.** `SceneCfg(voxel=[coarse, fine])` 가 받아들여지는 것은 확인했지만,
  충돌 질의가 둘을 `min()` 으로 합치는지 각각 별도 행이 되는지는 커널을 더 읽어야 한다.
  우리 쪽에서 직접 `min()` 을 해도 되지만, 그러면 cuRobo 의 질의 커널을 안 쓰게 된다.
- **미세 계층의 중심.** `VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md` 가 이 질문에
  답을 냈다 — **action chunk 의 swept volume**. 실측도 그쪽을 지지한다 (같은 반경에서 장애물
  유지율 75.3% 대 target 중심 46.5%). §7 참고.
- **증분 갱신.** 지금 측정은 한 프레임이다. 프레임 간 재사용의 정상 상태 비용은 따로 재야 한다.
- **gradient.** cuRobo `VoxelGrid` 에서 기울기를 어떻게 얻는지 아직 안 봤다. 없으면 우리가
  중심차분해야 하고, 그러면 `distance` 와 같은 보간을 써야 한다 (안 그러면 SQP 가 수렴 안 함 —
  우리 `esdf.py` 의 `gradient()` 주석이 그 이유를 적어 두었다).

---

## 7. `VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md` 와의 정합

같은 디렉터리에 그 문서가 있다. 검토 결과 **방향이 일치하고, 우리가 열어 둔 질문 하나에
답을 준다.**

| 항목 | 그 문서 | 이 검토 | 정합 |
|---|---|---|---|
| cuRoboV2 를 참고/사용 | ✓ | ✓ | 일치 |
| TSDF/ESDF 해상도 분리 | ✓ | ✓ | 일치 |
| 미세 ESDF 의 중심 | **action chunk 의 swept volume** | 미정이었음 | **그 문서가 답을 줌** |
| ESDF 범위 | swept volume ROI 로 **국소화** | 전체 유지 + 국소 미세 계층 | **차이 있음 — 아래** |

**차이 하나와 그 판정.** 그 문서는 full-workspace dense ESDF 를 local ROI 로 **대체**한다고
적고 있다. 검토 중에 "target 주변만 남기면 장애물의 84% 를 잃는다" 는 실측이 있었지만,
**그 우려는 ROI 가 swept volume 일 때는 적용되지 않는다** — 충돌은 로봇이 실제로 지나가는
곳에서만 일어나므로, `swept volume ⊕ (로봇 반경 + margin + activation band + trust region)`
밖의 기하는 그 청크 동안 닿을 수 없다. 그 문서 9항이 TO 가 ROI 를 벗어나는 경우를 이미 짚고 있다.

**다만 구현 시 반드시 지킬 것 둘:**

1. **ROI 패딩은 최소 `max_distance` 이상.** 잘린 ESDF 는 경계 근처 복셀의 거리가 과대평가된다
   (진짜 최근접 표면이 ROI 밖에 있을 수 있으므로). 우리 `esdf.py` 의 `_dirty_blocks` 가 같은
   이유로 `max_distance` 만큼 부풀린다.
2. **ROI 안에서는 해상도를 올릴 것.** 그 문서는 ESDF 를 10~20 mm 로 적었는데, ROI 가 작아진
   덕에 5 mm 를 감당할 수 있고 **이득의 대부분이 거기서 나온다** — 20 mm 의 이산화 편향
   −10 mm 는 `esdf_margin` 50 mm 의 20% 를 먹지만 5 mm 는 4% 다. 실측: 같은 테이블 상판을
   20 mm 계층은 33 mm, 5 mm 계층은 2 mm 틀리게 본다.

그 문서 7항의 "진행 상황 기록용 md 를 만들어 누적 기록" 은 `AG3S_REVIEW_LOG.md` 가 이미
그 역할을 하고 있으므로 **새로 만들지 말고 거기 이어 쓰는 것을 권한다** — 기록이 갈라지면
어느 쪽이 최신인지 알 수 없게 된다.
