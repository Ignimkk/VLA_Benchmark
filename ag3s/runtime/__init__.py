"""실행 기반 — 오케스트레이션 · 계측 · 자산 경로 · 시각화.

`pipeline` 은 조율만 하고 알고리즘을 갖지 않는다. `multiview` 는 카메라별 처리를
base frame 에서 융합하고, `profiler`/`trace` 는 계측, `asset_path` 는 MJCF/URDF 자산
탐색, `visualization` 은 단계별 PNG 덤프를 맡는다.
"""
