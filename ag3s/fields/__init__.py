"""거리장(distance field) 백엔드 — TSDF/ESDF 생산과 소비.

`esdf` 는 순수 numpy 거리장, `curobo_field` 는 cuRobo 가 만든 격자를 읽는 어댑터,
`static_scene` 은 씬에 고정된 기하를 필드에 넣는 경로다.
"""
