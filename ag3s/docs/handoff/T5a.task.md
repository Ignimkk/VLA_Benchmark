# T5a — shadow 모드를 배선한다 (판정은 하되 수정을 실행하지 않는다)

> writer: lead (A0) · 2026-09-25 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T5a.impl.md`

## 확정된 아키텍처 (사용자 판정 2026-09-25) — 이것을 깨지 말아라

**AG3S · cuRobo · TO 는 서버에 둔다.** attention 과 cuRobo 가 GPU 를 요구하기 때문이다.
로컬은 `pi05_infer.py --safe-remote` 로 관측과 프롬프트를 보내고 **action 만 받는다.**

물려받은 결정 *"T5·T6 에서 AG3S 를 클라이언트 in-process 로 옮긴다"* 는 **철회됐다.**
근거: IPC 는 왕복 2725 ms 중 **약 206 ms(7.6 %)** 라 옮겨도 청크 예산 533 ms 를 못 맞춘다
(지배 항은 AG3S 지각 1943 ms = 서버 시간의 77 %). **in-process 로 옮기는 코드를 쓰지 말아라.**

## 지금 없는 것 — shadow 모드

T5 의 정의(로그 용어 절): *"AG3S · cuRobo · SQP 를 전부 돌리되 **수정된 청크를 로봇에
보내지 않는** 실행."* 실제로 보내는 T6 앞에 두는 이유는 **수정이 여유거리(clearance)를 나쁘게
만드는지를 로봇을 움직이기 전에 보기 위해서**다.

**플래그가 없다** (2026-09-25 확인). `pi05_infer.py` · `serve_safe.py` · `client.py` 어디에도
`shadow`/`dry-run` 이 없다.

## 이미 있는 것 — 절반은 되어 있다

| 어디 | 무엇 |
|---|---|
| `benchmark/trajopt/safe_policy.py:215-229` | `refined = self.refiner.refine(chunk, ...)` → `_preserve_grippers(chunk, refined)` → `_record(seq, chunk, refined)` → `wire.pack_response(refined, verdict, ...)` |
| `benchmark/trajopt/safe_policy.py:360-386` | **`_record` 가 `reference_chunk` 와 `refined_chunk` 를 둘 다 저장한다** (`--record-constraints DIR` 일 때) |
| `benchmark/trajopt/wire.py` (`pack_response`) | 응답에는 `actions`(= refined) 하나만 실린다. reference 는 **로컬이 볼 수 없다** |
| `pi05_TO_hybrid/rby1_bringup/pi05_infer.py:1557` | `not safe_client.last_safe` 면 **HOLD** 한다 |
| 같은 파일 `:1577` | 기록에 `executed=bool(safe_client is None or last_safe)` 를 남긴다 |

**즉 서버는 reference 를 이미 알고 있고 기록도 한다. 없는 것은 그것을 로컬에 주는 길과,
로컬이 그것을 실행하는 길이다.**

## 요구사항 — 무엇이 되어야 하나

1. **서버는 전부 돈다.** AG3S 지각 · cuRobo ESDF · SQP/QP · 안전 판정. 하나도 건너뛰지 않는다.
2. **로컬은 정책의 원본 청크를 실행한다** — TO 가 수정한 것이 아니라.
3. **판정과 refined 청크는 기록된다.** 나중에 *"refined 가 reference 보다 나빠지지 않는가"* 를
   MuJoCo 참값으로 잴 수 있어야 한다 (그 측정은 A2 의 몫이다).
4. **shadow 에서는 unsafe 여도 멈추지 않는다.** 멈추면 에피소드가 서고 볼 것이 없어진다.
   단 **판정과 사유는 프레임마다 기록**된다 — 조용히 넘기는 것이 아니다.
5. **기본 동작은 한 바이트도 바뀌지 않는다.** shadow 플래그가 없으면 T0 때와 같은 응답이어야
   한다 — 회귀 기준선과 T0 기록이 그것에 달려 있다.

## 내가 보는 모양 — **코드를 읽고 확인하거나 더 나은 것을 제시해라**

- `wire.py` 의 응답에 **선택 키 `actions_reference`** 를 더한다 (`field` 가 선택 키인 것과 같은
  방식). **shadow 일 때만 싣는다** — 그래야 기본 응답이 그대로다.
- `safe_policy.py` 에 `shadow: bool = False` 를 두고, True 면 `pack_response` 에
  `actions_reference=chunk` 를 함께 넘긴다. **`actions` 는 여전히 refined 다** — 서버가
  거짓말을 하지 않게. 무엇을 실행할지는 로컬이 고른다.
- `serve_safe.py` 에 `--shadow` 플래그.
- `pi05_infer.py` 에 `--safe-shadow`: `actions_reference` 가 오면 **그것을** 실행하고,
  `last_safe` 가 False 여도 HOLD 하지 않는다. 기록의 `executed` 는 참이지만 **무엇을
  실행했는지**(reference 냐 refined 냐)를 기록에 남겨야 한다 — 새 키를 하나 더해라.
- `actions_reference` 가 안 왔는데 `--safe-shadow` 면 **즉시 실패**해라. 조용히 refined 를
  실행하면 shadow 가 아닌데 shadow 라고 기록된다 — 이 프로젝트에서 가장 비싼 종류의 버그다.

**`--safe-shadow` 와 `--shadow` 가 짝이 안 맞을 때(한쪽만 켠 경우) 무슨 일이 나는지 명시해라.**

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 기준 **667 passed**. `wire` 왕복 테스트를 하나 더한다 — `actions_reference` 가 있을 때와
**없을 때 둘 다**, 그리고 **없을 때의 응답이 예전과 같다**는 것.

## 규칙

- 네 소유: `benchmark/trajopt/*.py` · `tests/**` · `pi05_TO_hybrid/rby1_bringup/pi05_infer.py`
  (이번 라운드에 한해 명시적으로 준다) · `handoff/T5a.impl.md`.
  **`docs/*.md`·figure·`*.verify.json` 은 건드리지 않는다.**
- **측정하지 않는다.** 긴 rollout 을 돌리지 않는다 — 단위 테스트까지다. clearance 비교는 A2 다.
- **서버가 포트 8000 에 떠 있다** (lead 가 `pi05_TO_hybrid/logs/serve_safe_rby1_16d.sh` 로 띄웠다).
  **죽이지 말아라.** 네 변경은 재기동이 필요하고 그것은 lead 가 한다.
- 커밋하지 않는다. `pkill -f` 금지 — PID 를 뽑아 `kill <PID>`.
- 위에 박힌 수치(2725 ms · 206 ms · 1943 ms · 533 ms · 667) 밖의 숫자를 지어내지 마라.
