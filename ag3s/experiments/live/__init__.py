"""T0~T6 전체 통합 테스트의 라이브 하네스.

실행 계획과 판정 기준은 `docs/AG3S_CUROBO_LIVE_TEST_PLAN.md` 의
"T0~T6 전체 통합 테스트 — 구현 및 실험 계획" 절에 있다.

기존 `experiments/curobo/` 와 다른 점은 입력이다 — 저장된 `run_XXXX` 나 attention npz 를
재생하지 않고 실행 시점에 MuJoCo 씬을 새로 만들고 실측 정책 attention 을 쓴다.
"""
