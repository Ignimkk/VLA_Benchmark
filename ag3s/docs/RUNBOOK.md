# AG3S 실모델 검증 — 처음부터 끝까지 다시 돌리는 법

이 문서 하나로 전체 검증을 재실행할 수 있다. 지금까지의 결과는 `--record-depth` 없이 얻은
것이므로, **실제 depth로 다시 도는 것이 다음 실행의 목적**이다.

## 0. 전체 흐름

```
로컬                                        GPU 서버 (172.21.121.112:32542)
────                                        ──────────────────────────────
① 서버 시작 요청 ──────────────────────────▶  serve_policy.py  (포트 8123)
② 롤아웃 + 기록 (--record-depth)
   → run_XXXX/  (qpos · 정책이미지 · depth)
③ 기록 검사 (record_check)
④ run 디렉터리 scp ────────────────────────▶  ⑤ 서버 중단 → attention 추출 → 재기동
   attention.npz ◀───────────────────────────    (pi05_attention)
⑥ 1~5단계 채점 (전부 로컬, GPU 불필요)
⑦ 갤러리 렌더
```

**서버가 필요한 곳은 ①과 ⑤ 둘뿐이다.** ⑤는 서버를 잠시 내렸다가 같은 인자로 다시 띄운다.

## 1. 서버 시작 (GPU 서버)

```bash
cd /mnt/dev/work/pi05_TO_hybrid/openpi
XLA_FLAGS='--xla_gpu_enable_command_buffer=' XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python scripts/serve_policy.py \
    --port 8123 \
    policy:checkpoint \
    --policy.config=pi05_rby1_atomic_lora \
    --policy.dir=/mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999
```

지난 실행에서 확인된 경로다. **커맨드라인을 어딘가에 적어 두라** — ⑤에서 같은 인자로 재기동해야
한다. config와 체크포인트는 반드시 짝이 맞아야 한다: 둘은 norm stats가 다르고 pi0.5는 state를
이산 언어 토큰으로 prefix에 넣으므로, 틀리면 **prefix 토큰 자체가 달라지고 따라서 attention이
달라지는데 shape도 범위도 정상으로 보인다.**

## 2. 롤아웃 + 기록 (로컬)

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
  --record-ag3s   $RUN/ag3s_records \
  --record-depth                            # ← 이번에 추가되는 것
```

`--record-depth`는 세 카메라의 depth와 내부·외부 파라미터를 **uint16 밀리미터**로 함께 저장한다
(스텝당 +0.13 MB). 실제 depth 카메라가 내보내는 형식이고, 그래서 1 mm 양자화가 그대로 남는다.

**출력 디렉터리 번호가 올라간다.** 기존에 `run_0000`~`run_0002`가 있으므로 이번은 `run_0003`이
된다. 아래 모든 명령의 `run_0002`를 새 번호로 바꿔야 한다. 아래 변수로 한 번만 정해 두면 편하다:

```bash
REC=$RUN/ag3s_records/run_0003        # ← 실제로 생성된 번호로
TAG=run0003
```

## 3. 기록 검사 (로컬) — forward pass를 쓸 값어치가 있는가

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.record_check \
    --records $REC
```

`READY`가 나와야 다음으로 간다. 확인하는 것: target 가시 프레임 수, 이미지가 정상인지, 씬이
재생되는지. `NOT READY`면 롤아웃을 다시 돈다 — **서버에 보내고 나서 알게 되면 왕복 하나가
낭비된다.**

## 4. 서버로 전송

```bash
tar czf /tmp/ag3s_records.tgz -C $RUN/ag3s_records $(basename $REC)
scp -J blunex@ai.amrc.kr:21151 -P 32542 /tmp/ag3s_records.tgz root@172.21.121.112:/mnt/dev/work/
```

ProxyJump가 막히면 2단으로:

```bash
scp -P 21151 /tmp/ag3s_records.tgz blunex@ai.amrc.kr:~/
ssh -p 21151 blunex@ai.amrc.kr 'scp -P 32542 ~/ag3s_records.tgz root@172.21.121.112:/mnt/dev/work/'
```

코드는 git으로 간다 (`benchmark` 저장소를 push → 서버에서 pull). **`outputs/`는 어느 저장소에도
속하지 않으므로** 기록만 직접 보낸다.

## 5. attention 추출 (GPU 서버)

전문은 [step-01-server-prompt.md](step-01-server-prompt.md)에 있다. 요지만:

```bash
# (a) 실행 중인 서버 커맨드라인을 파일로 남긴다 — 재기동에 필요
ps aux | grep serve_policy | grep -v grep
PID=<위 PID>
tr '\0' '\n' < /proc/$PID/cmdline | tee /tmp/serve_policy_cmdline.txt

# (b) 코드 동기화 + 기록 풀기
cd /mnt/dev/work/benchmark && git pull
cd /mnt/dev/work && tar xzf ag3s_records.tgz && cat run_0003/meta.json | head -20

# (c) 서버 중단 → GPU 를 probe 에게 전부 준다
kill $PID && sleep 10 && nvidia-smi

# (d) 먼저 2프레임 스모크 (layers=18 heads=8 action_tokens=50 확인)
PY=/mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python
cd /mnt/dev/work
$PY -m benchmark.ag3s.experiments.pi05_attention --records run_0003 --limit 2 \
    --checkpoint <(a)의 경로> --config <(a)의 config> --out /tmp/attn_smoke.npz

# (e) 전체
XLA_FLAGS='--xla_gpu_enable_command_buffer=' XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -m benchmark.ag3s.experiments.pi05_attention     --records run_0004     --checkpoint /mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999     --config pi05_rby1_atomic_lora     --out /mnt/dev/work/attention_step1_run0004.npz

# (f) 서버 재기동 — 이것까지가 작업의 일부다
cat /tmp/serve_policy_cmdline.txt   # 그대로 재실행
ps aux | grep serve_policy | grep -v grep; ss -tlnp | grep 8123
```

**서버를 내리기 전에 (a)를 반드시 먼저** 한다. 커맨드라인을 잃으면 재기동 시 인자를 추측하게
되고, 그러면 이후 붙는 정책이 원래 것과 같다는 보장이 사라진다.

산출물 `attention_step1_run0003.npz`는 float16으로 약 81 MB (44프레임 기준). 로컬로 가져와
`benchmark/ag3s/asset/data/`에 둔다.

## 6. 채점 — 1~5단계 (전부 로컬, GPU 불필요)

**순서가 중요하다.** 3·4·5단계가 1단계의 결과 파일(`step-01-attention.json`)에서 (층, 헤드,
Euler step, pooling)을 읽는다. 1단계를 먼저 돌리지 않으면 나머지가 실패한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
REC=outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0003
ATT=benchmark/ag3s/asset/data/attention_step1_run0003.npz
PY="MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m"

# 1단계 — attention map  (~1분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.attention_report \
    --records $REC --attention $ATT

# 2단계 — 3D back-projection  (~2분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.backprojection_report \
    --records $REC

# 3단계 — attention lifting  (~4분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.lifting_report \
    --records $REC --attention $ATT

# 4단계 — target grounding  (~8분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.grounding_report \
    --records $REC --attention $ATT

# 5단계 — target/obstacle 분리  (~3분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.separation_report \
    --records $REC --attention $ATT

# 6단계 — geometry 변환 (primitive)  (~5분)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.geometry_report \
    --records $REC --attention $ATT
```

### `--tag` — 이전 결과를 덮어쓰지 않기

기본값으로 돌리면 `docs/step-0N-*.md` 와 `asset/image/<단계>/` 를 덮어쓴다. **depth 유무를
비교하는 것이 이번 실행의 목적이므로 덮어쓰면 비교 대상이 사라진다.** 모든 채점 스크립트가
`--tag` 를 받는다:

```bash
TAG=004     # run_0004 에 맞춘 꼬리표
for step in attention_report backprojection_report lifting_report \
            grounding_report separation_report geometry_report; do
  MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.$step \
      --records $REC --attention $ATT --tag $TAG
done
```

(2단계 `backprojection_report` 는 `--attention` 을 받지 않으므로 그 인자만 빼고 돌린다.)

| | 기본 | `--tag 004` |
|---|---|---|
| 그림 | `asset/image/<단계>/` | `asset/image_004/<단계>/` |
| 문서 | `docs/step-05-separation.md` | `docs/step-05-separation_004.md` |
| 문서 안 링크 | `../asset/image/<단계>/` | `../asset/image_004/<단계>/` |

링크는 하드코딩이 아니라 **문서에서 그림 폴더까지의 상대 경로로 계산**하므로
(`experiments/outputs.py`), `--out-figs` 로 아무 데나 지정해도 문서의 이미지가 깨지지 않는다.
`--tag` 없이 돌리면 지금까지와 **한 글자도 다르지 않은** 결과가 나온다.

## 7. 갤러리 (로컬)

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.cloud_gallery \
    --records $REC --attention $ATT --frame 0
```

기록에 depth가 있으면 **자동으로 그것을 쓰고**, 그림 제목에 "기록된 depth (uint16 mm)"라고
표시된다. 없으면 재생 렌더를 쓰고 그렇게 표시된다. 재생 렌더와 대조하려면 `--replay-depth`.

## 7-b. ESDF backend (선택)

primitive 대신 관측 표면을 그대로 쓰는 충돌 표현. 자세한 것은
[ESDF-BACKEND.md](ESDF-BACKEND.md).

```yaml
# AG3S 설정에 추가
collision_backend: both        # primitive | esdf | both
esdf:
  voxel_size: 0.010            # 5 / 10 / 20 mm 비교 축
```

기본값은 `primitive` 이므로 위 1~7 단계 결과는 이 backend 와 무관하게 재현된다. `both` 로 돌리면
두 표현이 한 제약 집합에 함께 들어가 행 단위로 비교할 수 있다.

## 8. 회귀

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m pytest tests/ag3s tests/trajopt -q   # 538 passed
```

---

## 재실행 시 주의할 점

| 항목 | 내용 |
|---|---|
| **run 번호** | `--record-ag3s`는 매번 새 `run_XXXX`를 만든다. 모든 명령의 경로를 새 번호로 바꾼다 |
| **단계 순서** | 3·4·5단계가 `step-01-attention.json`을 읽는다. 1단계를 먼저 |
| **체크포인트·config 짝** | 서버가 실제 로드한 것을 확인해서 쓴다. 틀려도 결과가 정상으로 보인다 |
| **`range_max`** | 4·5·6단계 기본값 `2.0 m`. 없으면 voxel이 15.9 mm로 커져 군집이 불가능해진다 (아래 참조) |
| **결과 덮어쓰기** | 채점 스크립트가 같은 파일에 쓴다. 비교하려면 미리 복사 |
| **서버 재기동** | probe 후 반드시. 커맨드라인은 내리기 *전에* 저장 |

### `range_max`를 반드시 켜야 하는 이유

AG3S 기본 설정으로 돌리면 이 씬에서 **모든 프레임이 `NO_CLUSTER`다.** 방 전체가 카메라에
들어와 `max_points=60000`을 넘고, `coverage_preserving_cap`이 voxel을 5 mm → 15.9 mm로 키우며,
그 간격에서 3 cm 반경 이웃은 9개뿐이라 `clustering.min_points=20`을 구조적으로 만족할 수 없다.
`range_max=2.0`의 근거는 측정이다 — 롤아웃에서 팔이 base로부터 최대 1.389 m까지 닿고, **팔이
닿을 수 없는 기하는 팔과 충돌할 수 없다.**

4·5·6단계 스크립트는 이 값을 기본값으로 갖는다. 1·2·3단계는 영향받지 않는다.

### depth 유무 비교에서 볼 것

이번 실행의 목적이다. 기대되는 차이:

* **2단계** — 표면 오차가 1 mm 양자화만큼 늘어나야 한다. 크게 늘면 다른 곳에 문제가 있다.
* **1·3단계** — attention은 정책 이미지에서 오므로 **변하지 않아야 한다.** 변하면 무언가 잘못됐다.
* **4·5단계** — 점군이 조금 굵어지므로 클러스터 점 수가 미세하게 달라질 수 있다. 선택 결과가
  바뀌면 그것 자체가 결과다.
* **6단계** — 여기가 가장 민감하다. depth 양자화는 도형을 **더 작게** 만드는 방향으로도 작용할
  수 있고, 6단계가 재는 것이 정확히 "도형이 실제보다 얼마나 작은가"다. 최대 침투가 늘어나면
  `perception_uncertainty`에 필요한 예산도 그만큼 늘어난다.
