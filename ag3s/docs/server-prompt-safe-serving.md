# 서버 프롬프트 — 안전 서빙 통합 검증

아래를 GPU 서버의 claude 에게 그대로 전달한다. 로컬과 서버는 git 으로 같은 워크스페이스를
공유하므로 `git pull` 로 받으면 된다.

---

## 배경

로컬에서 **π0.5 + SEAM + AG3S + TO 를 서버 한 프로세스로 묶는** 계층을 구현했다. 로컬 MuJoCo 는
3카메라 관측을 보내고, 서버가 청크와 안전 판정을 돌려주고, 로컬은 그 판정이 유효할 때만
앞 8개 action 을 실행한다. 아니면 현재 관절을 hold 한다.

**openpi 는 한 줄도 고치지 않았다.** `PolicyRecorder` 가 이미 만들어 둔 자리(정책을 감싸는
정책)를 쓴다. `serve_policy.py` 대신 `benchmark/trajopt/serve_safe.py` 로 띄운다.

새로 들어온 것:

| 파일 | 역할 |
|---|---|
| `benchmark/trajopt/wire.py` | 요청/응답 계약. pack/unpack 이 한 파일에 |
| `benchmark/trajopt/safe_policy.py` | `SafePolicy` — 정책을 감싸 AG3S+TO 를 돌린다 |
| `benchmark/trajopt/serve_safe.py` | 서버 진입점 |
| `benchmark/trajopt/client.py` | 로컬 안전 게이트 (서버에서는 안 쓴다) |
| `benchmark/trajopt/experiments/safe_replay.py` | 서버 단독 replay 검증 |
| `benchmark/ag3s/trace.py`, `benchmark/ag3s/experiments/constraint_record.py` | 타임스탬프·제약 기록 |

로컬에서 이미 확인된 것: **테스트 560 passed**, 서버 단독 replay 6프레임 통과.

---

## 0. 접속 경로

컨테이너는 점프 호스트 내부망(172.21.121.112)에 있어 로컬에서 **직접 닿지 않는다.** 로컬은
SSH 터널로 붙는다.

```
로컬 :8123  ──터널──▶  blunex@ai.amrc.kr:21151  ──▶  root@172.21.121.112:32542  ──▶  컨테이너 :8123
```

서버 쪽(당신)이 할 일은 컨테이너 안에서 `0.0.0.0:8123` 에 띄우는 것뿐이다. 터널은 사용자가
로컬에서 건다.

```bash
# 서버 접속 (사용자가 이미 이렇게 들어와 있다)
ssh blunex@ai.amrc.kr -p 21151
ssh root@172.21.121.112 -p 32542
```

## 1. pull 과 회귀 테스트

```bash
cd /mnt/dev/work          # benchmark/ 가 있는 곳
git pull

# 회귀. 로컬에서 560 passed 다. 서버에서 같은 수가 나와야 한다.
MUJOCO_GL=osmesa /mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python \
    -m pytest tests/ag3s tests/trajopt -q
```

**MuJoCo 렌더 백엔드를 확인해 달라.** 로컬은 `MUJOCO_GL=osmesa`(CPU 소프트웨어 래스터라이저)라
3카메라 depth 렌더에 167 ms 가 걸렸다. 서버에 GPU 렌더가 있으면 `MUJOCO_GL=egl` 이 훨씬 빠를
것이고, 이 계층이 실시간에 들어가느냐가 거기 달려 있다. **둘 다 재서 알려 달라.**

---

## 2. 서버 단독 replay

기록된 MuJoCo 관측으로 서버 경로 전체를 돌린다. 네트워크도 없고 정책도 스텁이라, 문제가 나면
그것은 AG3S+TO 와 배선뿐이다.

```bash
cd /mnt/dev/work

# 기본 — 인증 요구 켜짐
MUJOCO_GL=osmesa /mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python \
    -m benchmark.trajopt.experiments.safe_replay \
    --records run_0002 --frames 10

# 인증 요구 끔
MUJOCO_GL=osmesa /mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python \
    -m benchmark.trajopt.experiments.safe_replay \
    --records run_0002 --frames 10 --allow-uncertified \
    --out-json /tmp/safe_replay_uncertified.json

# 카메라 누락
MUJOCO_GL=osmesa /mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python \
    -m benchmark.trajopt.experiments.safe_replay \
    --records run_0002 --frames 5 --drop-camera zed_left \
    --out-json /tmp/safe_replay_dropcam.json
```

`run_0002` 는 이전에 scp 로 올린 기록 디렉터리다. 이름이 다르면
`ls /mnt/dev/work | grep run_` 로 확인해서 맞춰 달라.

**로컬에서 나온 결과 (이것과 같아야 한다):**

| 시나리오 | 결과 |
|---|---|
| 기본 | **전부 hold** — `geometry_certified=False`, AG3S `degraded`/`no_target` |
| `--allow-uncertified` | 전부 실행, `feasible`, 위반 0.0 mm |
| `--drop-camera zed_left` | 전부 hold, `no_target` |
| 그리퍼·형태 | 전 프레임 불변 |
| 지연 | 전체 중앙 610 ms (osmesa 로컬) |

**기본에서 아무것도 실행되지 않는 것은 버그가 아니다.** 이 씬의 미관측 비율이 72% 라 AG3S 가
`degraded` 를 내고, `require_certified_geometry: true` 가 그것을 `VIOLATED` 로 내린다. 위반이
0.0 mm 라도 hold 다 — "제약을 다 지켰다" 는 모델에 대한 진술이지 세계에 대한 진술이 아니라는
계약이다. 서버에서도 같은 결과가 나오는지 확인만 해 달라.

---

## 3. attention 을 응답에 싣기 ← **이 작업이 핵심이다**

지금 AG3S 가 `no_target` 을 내는 이유가 이것이다. attention 이 없으면 target 을 못 잡고, 거리장이
target 복셀을 파내지 않아 제약이 필요 이상으로 보수적이 된다.

**요청: 추론 응답에 attention 맵을 함께 실어 달라. 전체 텐서가 아니라 선택 셀 하나만.**

1단계 채점(`benchmark/ag3s/docs/step-01-attention.json` 의 `best`)에서 확정한
(층, 헤드, aggregation, denoise step) 조합이 있다. 카메라마다 그 셀 하나면 카메라당 맵 한 장이라
**수십 KB** 다. 전체 텐서는 프레임당 184 MB 라 불가능하지만 이건 부담이 없다.

### 함정 두 가지

**(1) `return_attn_probs=True` 를 켜면 서빙이 깨진다.** `gemma.Module.__call__` 이 3-tuple 을
돌려주는데 `Pi0.sample_actions` 는 `pi0.py:245` 에서 2개만 언팩한다. 거기를 고쳐야 한다.
**`return_attn_probs` 를 재구현하지 말 것** — 이미 있는 기능이다. sync/checkout 은 괜찮다.

**(2) 서버 재시작은 괜찮다.** 사용자가 확인했다. JAX 가 GPU 메모리 75% 를 미리 잡으므로 두
프로세스를 동시에 띄우려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`, 그리고 이 워크스페이스에서는
`XLA_FLAGS='--xla_gpu_enable_command_buffer='` 없이 돌리면 CUDA graph replay 오류가 난다
(`src/docs/RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md:773`).

### 형식

`benchmark/trajopt/serve_safe.py` 의 `attention_extractor()` 가 이렇게 읽는다:

```python
raw = result.get("attention")      # 정책 응답의 딕셔너리
ALIAS = {"cam_high": "zed_left",
         "cam_left_wrist": "wrist_cam_l",
         "cam_right_wrist": "wrist_cam_r"}
```

즉 `result["attention"]` 이 **딕셔너리**이고 키는 정책 입력 카메라 이름(`cam_high` 등),
값은 그 카메라의 정책 이미지 격자에 대한 2D 맵(float16/32)이다. **정규화하지 말고 원본 값
그대로** — AG3S 가 정규화와 역투영을 알아서 한다. 1단계에서 임계값은 정규화본을, 피크는 원본을
요구한다는 것을 이미 확인했다.

일부 카메라만 보내도 된다. 없는 카메라는 attention 없이 기하만 기여한다.

### 확인해 줄 것

1. `result["attention"]` 이 실제로 실리고 크기가 수십 KB 인가.
2. attention 을 켠 상태의 `result["actions"]` 가 **끄고 돌렸을 때와 같은가.** 달라지면 샘플링
   경로를 건드린 것이고, 그러면 지금까지의 모든 롤아웃과 비교가 불가능해진다. **이게 가장
   중요한 확인이다.**
3. 추론 지연이 얼마나 늘었나 (켜기 전/후 ms).
4. 어느 (층, 헤드, agg, denoise) 셀을 골랐고, `step-01-attention.json` 의 `best` 와 일치하는가.

---

## 4. 안전 서버 띄우기

```bash
cd /mnt/dev/work
XLA_FLAGS='--xla_gpu_enable_command_buffer=' XLA_PYTHON_CLIENT_PREALLOCATE=false \
/mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python -m benchmark.trajopt.serve_safe \
    --config pi05_rby1_atomic_lora \
    --checkpoint /mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999 \
    --model-xml /mnt/dev/work/src/rby1_description/models/rby1a/mujoco/model_transport.xml \
    --port 8123
```

**XML 은 반드시 `model_transport.xml` 이어야 한다.** 로컬이 `--model rby1_transport_14d` 로
돌고 그 모델의 씬이 이것이다. 다른 XML 을 쓰면 서버가 만든 로봇 모델이 로컬이 움직이는 로봇과
달라지고, 그 제약은 로봇을 설명하지 않는다. 그 경로가 서버에 없으면 **추측해서 다른 XML 을
쓰지 말고 물어봐 달라.**

`--model-xml` 은 로봇 모델(자기 필터·제약 구)을 만드는 데만 쓴다. 서버는 시뮬레이션을 돌리지
않는다 — 로컬이 보낸 관측만 본다. 띄울 때 로그에 이렇게 나와야 한다:

```
AG3S: self-filter 194 spheres, constraints 120 spheres (arms)
```

**숫자가 다르면 멈추고 알려 달라.** 194/120 이 아니면 로봇 모델이 다른 것이고, 그 상태의 제약은
로봇을 설명하지 않는다.

`--no-safe` 로 띄우면 감싸지 않는다 — 기존 서빙과 같으므로, 문제가 안전 계층에 있는지 아닌지를
플래그 하나로 가를 수 있다.

---

## 5. 로컬 연결 (사용자가 로컬에서 실행)

서버가 뜨면 사용자가 로컬에서 이걸 돌린다. **서버 쪽에서 할 일은 로그를 보는 것**이다.

```bash
# ① 터널 (별도 터미널에서 띄워 둔다)
ssh -N -L 8123:localhost:8123 -J blunex@ai.amrc.kr:21151 root@172.21.121.112 -p 32542

# ② 로컬 루프
cd /home/mk/dev_ws/vla/pi0_TO_ws
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \
    --model rby1_transport_14d --remote localhost:8123 \
    --prompt "put the apple in the basket" \
    --fruit-layout-index 0 --fruit-slot-order apple banana orange pear \
    --obstacle-profile clear --max-steps 350 --start-delay 2 --speed 1.0 --view front \
    --safe-remote --safe-timeout 2.0 \
    --trace outputs/live/trace \
    --record outputs/live/third_person.mp4
```

**여기에는 `MUJOCO_GL=osmesa` 를 붙이지 않는다.** 이 루프는 `--headless` 가 없으면
`mujoco.viewer.launch_passive` 로 대화형 뷰어를 띄우는데, osmesa 는 화면 없는 소프트웨어
렌더러라 기본 프레임버퍼가 없다. 붙이면 시작하자마자 죽는다:

```
ERROR: Default framebuffer is not complete, error 0x0
```

뷰어 없이 돌리려면 `--headless` 를 함께 준다. 그때는 osmesa 를 붙여도 되고, `--record` 가
있으므로 영상은 그대로 나온다:

```bash
MUJOCO_GL=osmesa XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \
    --model rby1_transport_14d --remote localhost:8123 \
    --prompt "put the apple in the basket" \
    --fruit-layout-index 0 --fruit-slot-order apple banana orange pear \
    --obstacle-profile clear --max-steps 350 --start-delay 2 --speed 1.0 --view front \
    --headless --safe-remote --safe-timeout 2.0 \
    --trace outputs/live/trace \
    --record outputs/live/third_person.mp4
```

오프라인 분석 스크립트(`safe_replay`, `*_report`)는 뷰어를 띄우지 않으므로 osmesa 가 맞다.
**렌더 백엔드는 뷰어를 띄우느냐로 갈린다** — 스크립트에서 복사해 오면 이 오류를 만난다.

서버 로그에서 확인할 것:
1. 요청에 `ag3s/depth/*`, `ag3s/K/*`, `ag3s/T_base_cam/*`, `ag3s/robot_state/*` 가 카메라마다
   들어오는가.
2. 첫 요청의 `ag3s/reset` 이 True 이고 리셋이 실제로 도는가.
3. `timing_ms` 의 `infer` / `ag3s` / `trajopt` / `total` 분해. **로컬 osmesa 에서 total 610 ms
   였다. 서버에서 몇 ms 인가?** 청크 예산은 533 ms 다.
4. 요청 payload 크기 (3카메라 depth uint16 640×480 ≈ 1.8 MB/프레임).

---

## 하지 말 것

- **openpi 파일을 고치지 말 것.** 3절의 `pi0.py:245` 만 예외이고, 그것도 2-tuple 언팩을 고치는
  최소 변경이어야 한다.
- **로컬 파일을 고치지 말 것** — `src/rby1_bringup/pi05_infer.py`, `benchmark/trajopt/client.py`.
  그쪽은 준비돼 있고 서버는 응답 형식만 맞추면 된다.
- **전체 attention 텐서를 응답에 싣지 말 것** (프레임당 184 MB).
- **합성 attention 으로 채우지 말 것.** 없으면 없는 대로 두면 AG3S 가 `no_target` 을 보고하고
  제약이 더 보수적이 된다 — 안전한 방향의 실패다. 가짜로 채우면 이 파이프라인이 실측인 척하게
  되고, 결과를 읽는 사람이 그것을 알 방법이 없다.
- `--allow-uncertified` 를 기본으로 켜지 말 것. 그건 위험을 인수하는 결정이고 아직 아무도
  내리지 않았다.

## 우선순위

3절(attention)이 나머지를 막고 있다. 1·2절은 pull 직후 바로 되고, 4·5절은 3절 이후다.
1·2절 결과를 먼저 알려주면 로컬에서 다음을 준비할 수 있다.
