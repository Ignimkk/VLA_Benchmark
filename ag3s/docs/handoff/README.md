# handoff — 세션 사이의 계약

병렬 세션은 서로의 대화 이력을 물려받지 않는다. CLAUDE.md·rules·skill 은 받지만
"아까 우리가 정한 것" 은 못 받는다. 그래서 넘길 것을 **파일로** 쓴다.

산문 요약으로 넘기면 받는 쪽이 빈칸을 상상으로 채운다. 그래서 형식을 고정한다.

```
A0 lead ──<STEP>.task.md──► A1 구현 ──<STEP>.impl.md──► A2 검증 ──<STEP>.verify.json──► A3 기록
         무엇을 왜                    무엇을 어떻게 고쳤나      명령·raw·숫자·figure       로그 + 논문
         성공 기준                    바뀐 파일:줄            해석 없음
```

| 파일 | writer | 템플릿 |
|---|---|---|
| `<STEP>.task.md` | lead | `_TEMPLATE.task.md` |
| `<STEP>.impl.md` | ag3s-implementer | `_TEMPLATE.impl.md` |
| `<STEP>.verify.json` | ag3s-verifier | `_SCHEMA.verify.json` |

`<STEP>` 은 `AG3S_CUROBO_LIVE_TEST_PLAN.md` 의 게이트 이름을 쓴다 (`T3`, `T3-a` …).

## 철칙

**`verify.json` 의 `numbers` 에 없는 숫자는 로그에 들어가지 않는다.**
이것이 *"추측을 기록에 남기지 않는다"* 를 규칙이 아니라 구조로 만드는 장치다.
숫자가 필요한데 없으면 → 쓰지 말고 verifier 에게 측정을 요청한다.

`verify.json` 은 **해석을 담지 않는다.** 판정은 lead 가 사용자와 함께 한다.
