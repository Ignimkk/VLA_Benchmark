# T5a — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-09-25

## 무엇을 했나 (평이한 요약 먼저)

shadow 모드를 배선했다. 서버(`--shadow`)는 **하던 일을 하나도 줄이지 않고** — AG3S 지각·cuRobo
ESDF·SQP·안전 판정이 전부 돈다 — 응답에 정책의 **원본** 청크를 선택 키 `actions_reference` 로
하나 더 싣는다. `actions` 는 shadow 에서도 여전히 refined 다 (서버가 거짓말하지 않는다).
로컬(`--safe-shadow`)이 그 둘 중 **원본**을 실행하고, 판정이 `unsafe` 여도 멈추지 않는다.

**두 플래그 중 하나만 켜면 즉시 죽는다.** 접속 시점(서버 metadata)과 첫 왕복(응답에 키가 있나)
두 곳에서 본다. 조용히 refined 를 실행하면 *"shadow 라고 적힌 T6 실행"* 이 남고, 그것이 지시서가
말한 가장 비싼 버그다.

**플래그가 없으면 한 바이트도 안 바뀐다.** 응답 키 집합·클라이언트가 돌려주는 객체(**같은 객체**,
복사조차 안 한다)·`stats` 의 키 집합·frame record 의 키 집합·`SafePolicy.metadata` 전부 T0 때와
같고, 그것을 테스트로 고정했다.

## lead 의 설계에서 바꾼 것 — 셋

지시서의 모양(`actions_reference` / 서버 `--shadow` / 클라이언트 `--safe-shadow`)을 코드로 확인해
그대로 갔다. 세 곳만 더했고, 이유는 전부 *"기록이 거짓이 되는 길을 막는다"* 다.

1. **`last_safe` 를 건드리지 않았다.** shadow 에서 hold 를 건너뛰는 가장 짧은 길은
   `last_safe = True` 로 올리는 것인데, 그러면 프레임 기록에 *"safe 였다"* 가 남아 **unsafe 인데
   움직인 프레임이 사라진다**. 대신 `should_execute`(실행하나) 와 `last_executed_chunk`(무엇을)
   를 새로 뒀다. **shadow 가 아니면 `should_execute is last_safe` 가 정확히 성립**하고
   (`client.py:216-223`), 그 동치를 옛 여섯 시나리오 전부에 대해 테스트로 고정했다.
2. **`--safe-shadow` 없이 서버만 `--shadow` 인 경우도 즉시 실패로 만들었다.** 지시서는 반대
   방향만 명시했지만, 이쪽도 똑같이 기록을 거짓으로 만든다 — 로봇은 refined 를 실행해 **닫힌
   고리로 도는데** 서버 쪽 기록(`--record-constraints` 의 meta)은 shadow 라고 말한다.
3. **shadow 에서 연속성 기준을 reference 로 바꿨다** (`safe_policy.py:234-241`).
   `_continuity_reference` 는 *"앞 `execution_length` 스텝은 이미 실행됐다"* 를 전제로 지난
   청크의 꼬리를 쓴다(`refiner.py:179-195`). shadow 에서 실행된 것은 reference 이므로 refined 를
   물려주면 SQP 가 **날아간 적 없는 궤적**에서 이어지는 것으로 계획한다. **A2 가 알아야 할
   유일한 "shadow 내부 동작 차이"** 이고, shadow=False 에서는 예전과 같다.

## 짝이 안 맞을 때 무슨 일이 나나 (지시서가 명시를 요구한 항목)

| 서버 | 로컬 | 무슨 일 |
|---|---|---|
| `--shadow` | `--safe-shadow` | 정상. 로봇이 reference 를 실행, 판정은 프레임마다 기록 |
| 없음 | 없음 | T0 과 **완전히 같다**. 응답에 `actions_reference` 가 없고 로컬은 키를 찾지도 않는다 |
| 없음 | `--safe-shadow` | **즉시 `RuntimeError`.** metadata 가 있으면 `SafeRemoteClient` 생성자에서 (카메라 한 번 안 찍고, 로봇이 아무것도 실행하기 전에), 없으면 첫 왕복 직후. 메시지: `Restart the server with --shadow` |
| `--shadow` | 없음 | **즉시 `RuntimeError`.** 같은 두 지점. 메시지: `--safe-shadow is off` |

hold 로 수렴시키지 **않은** 이유: hold 는 "이 청크를 실행하지 않는다" 이고 실행은 계속된다.
설정이 어긋난 채 끝까지 도는 rollout 은 결과가 무슨 뜻인지 아무도 모른다.

**shadow 에서도 hold 인 것 넷**: timeout · 오래된 `seq` · 서버 오류 · 청크 차원 불일치.
그 넷은 *"이 청크가 위험하다"* 가 아니라 *"이 청크를 신뢰할 근거가 없다"* 이고, 원본이든
수정본이든 똑같이 근거가 없다. `reference` 와 `actions` 의 **모양이 다른 경우도 hold** 다 —
어느 쪽이 어느 스텝인지 모르는 청크는 실행하지 않는다.

## 바뀐 파일

`benchmark/` 와 `pi05_TO_hybrid/` 는 **서로 다른 git repo** 다 (커밋할 때 둘 다 봐야 한다).

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/wire.py:93` | `ACTIONS_REFERENCE = "actions_reference"` 상수 | 세 곳이 같은 문자열을 쓴다. 박아 두면 갈라진다 |
| `benchmark/trajopt/wire.py:228,261-268` | `pack_response(..., actions_reference=None)`. **`None` 이면 키를 아예 안 싣는다**; 모양이 `actions` 와 다르면 `ValueError` | 키 없음이 곧 "shadow 아님" 신호다. `field` 와 규칙이 반대인 것은 뜻이 반대이기 때문 (docstring 에 적었다) |
| `benchmark/trajopt/wire.py:274-284` | `unpack_actions_reference()` | `pack_*`/`unpack_*` 짝을 이 파일에 두는 규약 |
| `benchmark/trajopt/wire.py:38,44-57` | 응답 표에 행 하나 + 절 하나 | 와이어의 정본은 이 파일 머리말이다 |
| `benchmark/trajopt/safe_policy.py:88,121` | `shadow: bool = False` | |
| `benchmark/trajopt/safe_policy.py:251-257` | shadow 면 `actions_reference=chunk` 를 함께 넘긴다. `actions` 는 계속 refined | 서버는 판정만 하고 무엇을 실행할지는 로컬이 고른다 |
| `benchmark/trajopt/safe_policy.py:234-241` | shadow 면 `_previous_chunk = chunk` | 위 3번 |
| `benchmark/trajopt/safe_policy.py:174-178` | shadow 일 때만 `metadata["shadow"] = True` | 로컬이 접속 즉시 짝을 보는 근거 |
| `benchmark/trajopt/client.py:62,89` | `shadow` · `last_executed_chunk`(`refined`/`reference`/`none`, 기본 `none`) | 기본이 `none` 인 것은 `last_safe` 가 False 로 시작하는 것과 같은 이유 |
| `benchmark/trajopt/client.py:157-212` | reference 를 읽고, 짝을 검사하고, shadow 면 `unsafe` 에서 hold 하지 않고 reference 를 실행. 응답에 `actions_refined` · `executed_chunk` 를 더해 돌려준다 | refined 를 조용히 덮지 않는다 — 호출부가 무엇을 받았는지 알아야 한다 |
| `benchmark/trajopt/client.py:216-223` | `should_execute` property | hold 판단이 보는 **유일한** 값 |
| `benchmark/trajopt/client.py:225-270` | `_check_server_metadata`(생성자) · `_check_shadow_pairing`(프레임) · `_pairing_message` | 두 지점 검사. 메시지는 두 곳에서 모두 참인 문장만 쓴다 |
| `benchmark/trajopt/client.py:192` | `stats["shadow_override"]` — **일어났을 때만** 키를 만든다 | 비-shadow 실행의 `stats` 키 집합을 T0 과 같게 둔다 (`--trajectory-out` 의 `ipc_stats`) |
| `benchmark/trajopt/serve_safe.py:198,242,307,312-321` | `--shadow`; `--no-safe` 와 같이 주면 `ap.error`; 어느 모드로 떴는지 시작할 때 `logging.info` | legacy backend 경고와 같은 이유 — 조용히 떠 있는 것이 가장 나쁘다 |
| `pi05_TO_hybrid/rby1_bringup/pi05_infer.py:713,921,1175` | `--safe-shadow`; `--safe-remote` 없으면 `ap.error`; 클라이언트에 전달 | |
| 같은 파일 `:1582,1594-1598` | hold 게이트가 `last_safe` → **`should_execute`**; shadow + unsafe 인 프레임은 `[shadow] t=… verdict UNSAFE, executing the reference chunk anyway — <사유>` 를 찍는다 | 조용히 넘기지 않는다. shadow 의 목적이 unsafe 프레임을 **보는** 것이다 |
| 같은 파일 `:1610-1616`, `:1527-1532`, `:1226` | control/planning frame 에 `executed_chunk`·`shadow`, manifest 에 `safe_shadow` — **셋 다 shadow 일 때만** | T0 기록의 키 집합을 그대로 두면서, 키가 있으면 그 자체로 "이 실행은 shadow" 표시가 된다 |

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**692 passed** (직전 기준 667 + 새로 25). 새 테스트가 고정하는 것:

* **wire 왕복** (`tests/trajopt/test_safe_policy.py`) — reference 가 **있을 때**(키가 생기고,
  `actions` 는 refined 그대로, float32 ndarray 라 msgpack_numpy 가 그대로 넘긴다) / **없을
  때**(`unpack_actions_reference() is None`) / **없을 때의 키 집합이 예전과 같다**
  (`BASE_RESPONSE_KEYS` 상수로 고정) / 모양이 다르면 `pack_response` 가 `ValueError`.
* **서버** — shadow 면 `actions_reference == 정책 청크`이고 `actions` 는 refined,
  `metadata["shadow"]` 는 켤 때만 존재, `_previous_chunk` 가 shadow 에서 reference 다.
  (`refiner.refine` 을 `+0.5` 로 바꿔 refined ≠ reference 로 만들었다 — 합성 씬에서는 refiner 가
  청크를 그대로 통과시켜 둘이 같아지고, 그러면 "무엇이 실렸나" 를 구별할 수 없다.)
* **클라이언트** — reference 를 실행한다 / unsafe 여도 실행하지만 `last_safe`·`last_reason`·
  `stats["unsafe"]` 는 그대로 남는다 / 짝 불일치는 프레임과 **생성자** 양쪽에서 `RuntimeError` /
  timeout·stale·error 는 shadow 에서도 hold / 모양 불일치도 hold.
* **기본 동작 불변** — 옛 일곱 시나리오에 대해 `should_execute == last_safe`, 그리고 비-shadow
  에서 클라이언트가 **응답 객체를 그대로**(`out is result`) 돌려주고 `stats` 에
  `shadow_override` 키가 생기지 않는다.

CLI 도 확인했다: `serve_safe --shadow --no-safe` → `ap.error`,
`pi05_infer --safe-shadow`(`--safe-remote` 없이) → `ap.error`.

`SafePolicy(shadow) ↔ SafeRemoteClient(shadow)` 를 **한 프로세스에서 실제로 맞물려** 한 왕복
돌려 보았다(합성 씬, 임시 스크립트): shadow=True 는 `executed=reference`, shadow=False 는 키가
없고 hold, 짝 불일치 두 방향 모두 생성자에서 거절. 전송 계층(msgpack)은 `.venv-ag3s` 에
`msgpack` 이 없어 여기서 못 태웠다 — 실물 왕복은 A2 의 live 실행이 처음이다.

## verifier 가 알아야 할 것

* **새 플래그** — 서버 `benchmark.trajopt.serve_safe --shadow`, 로컬
  `pi05_infer.py --safe-shadow`. **반드시 둘을 같이** 켠다 (한쪽만이면 즉시 죽는다).
  기본값은 둘 다 off 이고 그때 동작은 T0 과 같다.
* **포트 8000 의 서버는 내 변경 이전 코드다.** 그대로 두면 비-shadow 경로가 예전처럼 돈다
  (metadata 에 `shadow` 키가 없고 클라이언트도 안 찾으므로 아무 문제 없다). shadow 측정에는
  `--shadow` 로 **재기동이 필요하고 그것은 lead 가 한다.**
* **재생산이 필요한 산출물** — 없음. 옛 npz·frame record 를 다시 만들 필요가 없다.
* **옛 기록과 호환이 깨지는가** — 깨지지 않는다. shadow 기록에 키가 **더해질** 뿐이다
  (frame record 의 `executed_chunk`·`shadow`, manifest 의 `safe_shadow`, `stats` 의
  `shadow_override`, 서버 record meta 의 `shadow`). 비-shadow 실행에는 그 키들이 **아예 없다** —
  옛 파서가 그대로 읽는다.
* **shadow 기록을 읽을 때**: `safe=False` 인데 `executed=true` 인 control frame 이 **정상**이다.
  그 조합은 `shadow: true` 가 같은 줄에 있을 때만 나온다.
* 내가 측정하지 않은 것: clearance 비교·지연·회귀 기준선. 전부 A2 다. 서버 쪽
  `--record-constraints` 가 이미 `reference_chunk` 와 `refined_chunk` 를 둘 다 저장하므로
  (`safe_policy.py:360-392`) 두 궤적의 MuJoCo 참값 여유거리를 같은 npz 하나에서 뽑을 수 있다.

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.** 기대가 보이면 판정이 뒤집힌다 (C2·D2).

`--safe-shadow` 없는 실행의 세 기준선(`위반으로 시작 14/15` · `has_target 9/15` ·
`frame0 clearance_before +0.157 mm`)이 그대로 나올 것으로 본다 — 그 경로의 코드가 바뀌지
않았고 응답 키도 그대로다. shadow 실행에서는 판정이 unsafe 인 프레임이 hold 되지 않으므로
**에피소드가 끝까지 간다**(T0 에서는 섰다). refined 대 reference 의 여유거리 우열은 **예상하지
않는다** — T5 가 존재하는 이유가 그것을 모르기 때문이다. 지연은 shadow 로 줄지 않는다
(서버가 하는 일이 그대로다): 왕복 약 2725 ms 가 청크 예산 533 ms 를 넘는 상태가 유지될 것이다.
