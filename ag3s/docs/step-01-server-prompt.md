# Step 1 — GPU 서버 Claude에게 보낼 프롬프트

아래 "프롬프트 본문"을 그대로 복사해서 GPU 서버에서 돌고 있는 Claude에게 보내면 됩니다.
코드는 git으로 동기화되고, 롤아웃 기록만 따로 전송하면 됩니다 (아래 "사전 준비").

## 사전 준비

로컬과 GPU 서버의 워크스페이스는 git으로 함께 관리되므로 **코드는 push/pull로 갑니다.**
`benchmark/` 저장소(`github.com/Ignimkk/VLA_Benchmark`)에 probe 코드가 들어 있습니다.

```bash
# 로컬
cd /home/mk/dev_ws/vla/pi0_TO_ws/benchmark
git add ag3s/docs && git commit -m "docs(ag3s): step-1 runbook and server prompt"
git push
```

**롤아웃 기록은 git으로 갈 수 없습니다.** `outputs/`는 어느 저장소에도 속하지 않는
워크스페이스 루트 아래에 있고 (루트의 `.git`은 빈 디렉터리입니다), `.npz` 44개(3.7 MB)는
어차피 저장소에 넣을 것이 아닙니다. 이것만 직접 전송하세요:

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
tar czf /tmp/ag3s_records.tgz -C outputs/rby1_atomic_infer/ag3s_step1/ag3s_records run_0002
scp /tmp/ag3s_records.tgz <서버>:<워크스페이스 루트>/
```

---

## 프롬프트 본문 (여기부터 복사)

RB-Y1 조작 과제의 충돌 회피 파이프라인(AG3S)을 검증 중입니다. 1단계는 "파인튜닝된 π0.5가
프롬프트가 지목한 물체를 실제로 보는가"이고, 그걸 재려면 **attention map**이 필요합니다.

지금 이 서버에서 `scripts/serve_policy.py`가 `rby1_transport_14d`(config `pi05_rby1_lora`)를
서빙 중이고 로컬 MuJoCo가 websocket으로 붙어 추론하고 있습니다. websocket 프로토콜은
`actions`만 돌려주므로 실행 중인 서버에서는 attention을 꺼낼 수 없습니다. 그래서 **이미
기록해 둔 관측에 대해 오프라인으로 forward를 한 번 더** 돌리는 별도 스크립트를 만들어
보냈습니다.

### 실행 순서 — 서버를 세우고, probe를 돌리고, 다시 띄웁니다

정책 서버를 **중단해도 됩니다** (사용자 확인 완료). 그 편이 더 간단합니다: 이 저장소는
`serve_policy`에 JAX 메모리 설정을 하지 않으므로 서버가 기본값으로 GPU 메모리의 75%를
선점하고 있고, 두 번째 JAX 프로세스인 probe를 그 옆에 띄우면 남은 것의 75%를 잡으려다
OOM 납니다. 총량이 아니라 선점이 문제라 H200이어도 마찬가지입니다.

그러니 GPU를 probe에게 통째로 주세요:

1. 실행 중인 `serve_policy` 프로세스의 **정확한 커맨드라인을 먼저 기록** (아래 1-(b))
2. 서버 중단
3. probe 실행
4. 기록해 둔 커맨드라인 그대로 서버 재기동, 뜨는지 확인

**2단계 전에 1단계를 건너뛰지 마세요.** 커맨드라인을 잃으면 재기동할 때 인자를 추측하게 되고,
그러면 사용자가 이후 붙는 정책이 원래 것과 같다는 보장이 사라집니다.

### 그래도 하지 말 것 (그리고 왜)

- **서버가 attention도 반환하도록 고치지 마세요.** 재시작이 허용됐다고 이 길이 열린 건
  아닙니다. `return_attn_probs=True`를 켜면 `gemma.Module.__call__`이 2-tuple 대신 3-tuple을
  반환하는데, `Pi0.sample_actions`는 `_, kv_cache = self.PaliGemma.llm(...)`로 2개를 언패킹합니다
  (`pi0.py:245`). 즉 **그 플래그를 켠 채로는 서빙 경로 자체가 동작하지 않고**, 살리려면
  샘플링 루프를 다시 써야 합니다 — 지금 우리가 측정하려는 대상, 즉 action이 만들어지는 방식을
  건드리는 것입니다. 게다가 attention 블록은 청크당 약 22 MB라 websocket으로 흘리면
  latency 측정까지 오염됩니다. 오프라인 probe는 우회로가 아니라 이 경우의 정공법입니다.

- **`return_attn_probs`를 직접 구현하지 마세요.** 없으면 그 기능이 있는 커밋을 sync 하거나
  checkout 하는 것이 올바른 조치입니다 (그건 해도 됩니다). 재구현이 위험한 이유는 구체적입니다:
  `pi05_attention.py`가 `probs[:, 0, 0, :, :, :768]`로 슬라이스하는데, 이건 `gemma.py`의
  `einsum("BKGTS,BSKH->BTKGH")`가 확정하는 `[L, B, K, G, T, S]` 축 순서와 gemma_2b가
  multi-query라 `K = 1`이라는 사실에 하드코딩돼 있습니다. 축 순서나 마스킹이 조금 다른 두 번째
  구현은 여전히 `[0, 1]` 범위의, shape가 맞는, 그림이 그려지는 숫자를 냅니다 — 다른 것을 재면서.

- **체크포인트 경로를 추측하지 마세요.** 아래 방법으로 *중단하기 전 서버가 실제로 로드했던*
  경로를 알아내야 합니다. 다른 체크포인트를 쓰면 사용자가 돌리던 정책이 아니라 다른 모델의
  attention을 재게 되고, 그 사실이 결과 어디에도 드러나지 않습니다.

### 1. 사전 점검 (여기서 하나라도 실패하면 멈추고 보고)

```bash
cd <워크스페이스 루트>

# (a) openpi에 attention-probs 기능이 있는가
grep -n "return_attn_probs" src/openpi/src/openpi/models/pi0_config.py
grep -n "return_probs" src/openpi/src/openpi/models/gemma.py
```
`pi0_config.py`에 `return_attn_probs: bool = False` 필드가, `gemma.py`에 `return_probs`
인자와 `probs = jax.nn.softmax(...)` 가 있어야 합니다. 없으면 이 서버의 openpi가 커밋
`a8902aa "feat: optional attention-probability output for KNOWS probes"`
(브랜치 `knows-attn-probs`)를 포함하지 않는 것입니다. `git log --oneline -5`와 현재 브랜치를
보고해 주세요 — 그 커밋을 가져오는 것은 괜찮고, 기능을 새로 작성하는 것은 안 됩니다.

```bash
# (b) 실행 중인 서버가 로드한 체크포인트 — 재기동에 필요하므로 커맨드라인을 파일로 남깁니다
ps aux | grep -i "serve_policy\|serve_seam" | grep -v grep
PID=<위에서 찾은 PID>
tr '\0' '\n' < /proc/$PID/cmdline | tee /tmp/serve_policy_cmdline.txt
ls -l /proc/$PID/cwd
```
`/tmp/serve_policy_cmdline.txt`는 3-b에서 서버를 되살릴 때 씁니다 — **이 파일을 만들기 전에
서버를 내리지 마세요.** 거기서 `--policy.dir` / `--policy.config` 혹은 그에 해당하는 인자를
읽어 **체크포인트 절대경로**와 **config 이름**을 확보하세요. 인자가 커맨드라인에 없으면
(환경변수나 설정 파일 경유 등) 추측하지 말고 보고해 주세요.

### 2. 코드 동기화 + 기록 풀기

probe 코드는 `benchmark/` 저장소에 커밋되어 있습니다. 워크스페이스가 git으로 함께 관리되므로
pull 하시면 됩니다.

```bash
cd <워크스페이스 루트>/benchmark
git pull
ls -l ag3s/experiments/pi05_attention.py ag3s/experiments/policy_record.py
```

두 파일이 없으면 pull이 안 된 것입니다 — 현재 브랜치와 `git log --oneline -3`을 보고해 주세요.

롤아웃 기록은 git 밖입니다 (`outputs/`는 어느 저장소에도 속하지 않습니다). 별도로 받은
tarball을 푸세요:

```bash
cd <워크스페이스 루트>
tar xzf ag3s_records.tgz
ls run_0002 | head; cat run_0002/meta.json
```

`meta.json`이 아래와 일치하는지 확인해 주세요. 다르면 다른 롤아웃입니다 — 멈추고 보고해 주세요.

| 키 | 기대값 |
|---|---|
| `prompt` | `put the apple in the basket` |
| `n_steps` | `44` |
| `nq` | `66` |
| `policy_model` | `rby1_transport_14d` |

이 기록은 로컬에서 이미 검사했습니다: head 카메라에서 **apple이 44프레임 중 28프레임에서
200 px 이상 보입니다** (t_step 72–192 구간은 파지 중 팔에 가려짐 — 정상이고 채점에서
제외됩니다).

### 3. 서버 중단 후 attention 추출

```bash
kill $PID           # SIGTERM. 몇 초 뒤 nvidia-smi 로 메모리가 반환됐는지 확인
nvidia-smi
```

```bash
src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.pi05_attention \
    --records run_0002 \
    --checkpoint <1-(b)에서 확인한 절대경로> \
    --config pi05_rby1_lora \
    --out attention_step1_run0002.npz
```

먼저 `--limit 2`를 붙여 2 프레임만 돌려서 shape 로그(`layers=18 heads=8 action_tokens=50`)를
확인한 뒤 전체를 돌리는 편이 안전합니다.

호스트 RAM이 빠듯하면 `--denoise-steps 9` 하나만, `--agg mean` 하나만으로 줄일 수 있습니다.
다만 그러면 로컬 분석에서 "어떤 Euler step / query pooling이 더 나은가"를 비교할 수 없으니,
가능하면 기본값(`--denoise-steps 0 4 9 --agg mean first last`)으로 돌려주세요.

시간이 남으면 `--noise-seeds 0 1 2`로 한 번 더 돌려 별도 파일(`attention_step1_run0002_seeds012.npz`)로 저장해
주시면 좋습니다. flow-matching 초기 노이즈는 서버의 요청별 RNG 상태를 복원할 수 없어 고정
시드를 쓰는데, 시드를 바꿔도 순위가 그대로인지가 그 선택이 무해했는지에 대한 유일한 증거입니다.
시간이 부족하면 생략해도 됩니다 — 기본 1개 시드로도 1단계 판정은 가능합니다.

### 3-b. 서버 재기동

```bash
cat /tmp/serve_policy_cmdline.txt      # 1-(b)에서 남긴 것
# 그대로 재실행 (원래 cwd에서, 원래 방식대로 백그라운드/tmux 등)
```
재기동 후 프로세스가 살아 있고 포트가 LISTEN 상태인지 확인해 주세요. 여기까지가 이 작업의
일부입니다 — probe만 돌리고 서버를 내려둔 채 끝내지 마세요.

### 4. 결과 회수

```bash
ls -lh attention_step1_run0002.npz
src/openpi/.venv/bin/python -c "
import numpy as np; d=np.load('attention_step1_run0002.npz')
print({k: getattr(d[k],'shape',d[k]) for k in d.files})
print('finite:', np.isfinite(d['attention']).all(), 'range:', float(d['attention'].min()), float(d['attention'].max()))
"
```

`attention`은 `[frame, denoise, agg, layer, head, camera, 16, 16]` **float16**이어야 하고,
값은 softmax 확률의 부분합이므로 `[0, 1]` 안에 있어야 합니다.

**크기: 약 81 MB** (44프레임 × Euler 3 × pooling 3 × 18층 × 8헤드 × 3카메라 × 256패치).
float32였다면 184 MB인데, attention 값이 1/768 근처라 float16으로 담아도 16×16 패치 합산
후 상대오차 0.2% 수준입니다 — 두 헤드의 순위를 뒤집을 수 있는 크기가 아닙니다. 채점은
어차피 로컬에서 float64로 합니다.

전송이 부담이면 `--denoise-steps 9 --agg mean`으로 약 10 MB까지 줄일 수 있지만, 그러면
"어떤 Euler step / pooling이 나은가"라는 질문을 포기하는 것입니다.

### 보고해 줄 것

1. 사전 점검 (a)(b) 결과 — 특히 **체크포인트 절대경로와 config 이름**
2. shape 로그 한 줄 (`layers=... heads=... action_tokens=...`)
3. 위 검증 스크립트 출력 (shape / finite / range)
4. 소요 시간과 `.npz` 파일 크기·경로
5. **서버가 원래 커맨드라인 그대로 재기동되어 살아 있는지** — 이게 확인 안 되면 나머지는 무의미합니다
6. 도중에 뭔가 예상과 달랐다면 그게 무엇이었는지

`.npz`는 제가 로컬로 가져가서 MuJoCo 세그멘테이션과 대조 채점합니다. 서버에서는 채점하지
않아도 됩니다 (`attention_report.py`는 로컬 전용이고 MuJoCo 렌더가 필요합니다).
