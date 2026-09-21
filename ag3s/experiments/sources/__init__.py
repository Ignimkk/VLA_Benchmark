"""실험의 입력을 만드는 쪽 — MuJoCo 씬, 정책 롤아웃 기록, 제약 기록.

`mujoco_source` 는 MuJoCo 카메라를 AG3S 입력(depth·K·T_base_cam)으로 바꾸고,
`policy_record`/`constraint_record` 는 롤아웃과 제약 생성 과정을 디스크에 남긴다.
`record_scripted`/`record_check` 는 기록을 뜨고 쓸 만한지 판정한다.
"""
