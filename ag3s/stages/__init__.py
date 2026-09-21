"""AG3S 파이프라인 스테이지 — 점에서 도형까지.

깊이에서 나온 점이 `reconstruction` → `robot_filter` → `support_surface` →
`attention_lifting` → `target_grounding` → `collision_candidates` → `geometry` 순서로
충돌 후보 primitive 가 된다. 각 모듈은 앞 단계의 출력만 받고, 서로를 직접 부르지 않는다.
"""
