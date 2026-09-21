"""충돌 제약 조립 — 여유거리에서 CasADi 사양까지.

`clearance` 가 phase 별 요구 여유거리를 정하고, `attached` 가 쥔 물체를 링크에 붙이며,
`constraint_builder` 가 고정 슬롯 제약을 만들고, `to_adapter` 가 그것을
`CollisionConstraintSet` 으로 묶어 최적화기에 넘긴다.
"""
