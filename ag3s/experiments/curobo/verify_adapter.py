"""`CuroboEsdfField` 어댑터 검증. **ag3s venv 에서** 돌린다 (cuRobo 없이).

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python \\
        -m benchmark.ag3s.experiments.curobo.verify_adapter

이 스크립트가 cuRobo 를 임포트하지 않고 도는 것 자체가 검증의 일부다 — 질의 경로에 cuRobo
의존이 남아 있지 않다는 뜻이고, 그래서 trajopt 를 검증된 환경에서 그대로 돌릴 수 있다.

확인하는 것 다섯:
  1. 왕복   — 어댑터가 복셀 중심에서 cuRobo 가 준 값을 그대로 답하는가
  2. 기울기 — `gradient` 가 `distance` 의 유한차분과 맞는가 (어긋나면 SQP 가 수렴 안 함)
  3. 합성   — `min()` 이 어느 계층을 고르는가, 경계에서 튀지 않는가
  4. 마스킹 — 로봇 자기 관측이 실제로 사라졌는가 (masked 대 raw)
  5. 정확도 — 테이블 상판을 두 계층이 각각 얼마나 틀리게 보는가 (33 mm 대 2 mm 재현)
"""

import argparse

import numpy as np

from benchmark.ag3s.curobo_field import CuroboEsdfField, layer_from_arrays

TABLE_Z = 0.823  # 참값 (MuJoCo 모델)


def load(path, *, outside_distance=None):
    d = np.load(path)
    layers = []
    for name in ("coarse", "fine"):
        if f"{name}_values" not in d:
            continue
        layers.append(layer_from_arrays(
            d[f"{name}_values"], d[f"{name}_origin"], float(d[f"{name}_voxel_size"])))
    return CuroboEsdfField(tuple(layers), outside_distance=outside_distance), d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field", default="/tmp/rby1_field.npz")
    ap.add_argument("--raw-field", default="/tmp/rby1_field_raw.npz")
    args = ap.parse_args()

    field, d = load(args.field, outside_distance=0.5)
    tc = np.asarray(d["target_centroid"], np.float64)
    print("계층")
    print(field.layer_report())
    print(f"\ngrid.voxel_size (trajopt 이 읽는 허용오차) = {field.grid.voxel_size*1000:.1f} mm")
    print(f"coverage_grid (G4 가 읽는 범위)            = {field.coverage_grid.voxel_size*1000:.1f} mm")

    # 1. 왕복 --------------------------------------------------------------
    print("\n" + "=" * 68)
    print("1. 왕복 — 복셀 중심에서 어댑터가 cuRobo 값을 그대로 답하는가")
    print("=" * 68)
    rng = np.random.default_rng(0)
    for i, layer in enumerate(field.layers):
        g = layer.grid
        idx = rng.integers(0, np.asarray(g.shape), size=(4000, 3))
        pts = g.origin + idx * g.voxel_size
        expect = layer.distance_grid[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
        got = layer.distance(pts)
        err = np.abs(got - expect)
        print(f"  계층 {i} ({g.voxel_size*1000:.0f} mm)  최대오차 {err.max()*1e6:.3f} µm  "
              f"평균 {err.mean()*1e6:.3f} µm   {'OK' if err.max() < 1e-6 else '불일치!'}")

    # 2. 기울기 ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("2. 기울기 — gradient 와 distance 의 유한차분이 맞는가")
    print("=" * 68)
    # 미세 계층 안, 표면에서 떨어진 자유 공간에서 본다 (표면 위는 거리장이 꺾이는 자리)
    pts = tc + rng.uniform(-0.12, 0.12, size=(600, 3))
    g = field.gradient(pts)
    h = 0.002
    fd = np.zeros_like(g)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        fd[:, k] = (field.distance(pts + e) - field.distance(pts - e)) / (2 * h)
    ok = np.linalg.norm(g, axis=1) > 1e-6
    err = np.linalg.norm(g[ok] - fd[ok], axis=1)
    print(f"  표본 {ok.sum()} 점   |∇d - FD| 중앙값 {np.median(err):.4f}  "
          f"90분위 {np.percentile(err, 90):.4f}")
    norm = np.linalg.norm(g[ok], axis=1)
    print(f"  |∇d| (eikonal, 자유공간에서 ≈1)  중앙값 {np.median(norm):.3f}  "
          f"[{norm.min():.3f}, {norm.max():.3f}]")

    # 3. 합성 --------------------------------------------------------------
    print("\n" + "=" * 68)
    print("3. 합성 — min() 이 어느 계층을 고르는가")
    print("=" * 68)
    fine = field.layers[-1].grid
    lo = fine.origin
    hi = fine.origin + (np.asarray(fine.shape) - 1) * fine.voxel_size
    probe = rng.uniform(lo, hi, size=(20000, 3))
    _, winner = field._evaluate(probe)
    print(f"  미세 창 안에서 표본 {len(probe):,}  ->  미세 계층이 이긴 비율 "
          f"{100.0*(winner == 1).mean():.1f} %")
    out = rng.uniform([-0.5, -0.9, 0.0], [1.2, 0.9, 1.5], size=(20000, 3))
    _, winner_out = field._evaluate(out)
    print(f"  작업공간 전체 표본 {len(out):,}     ->  미세 계층이 이긴 비율 "
          f"{100.0*(winner_out == 1).mean():.1f} %")
    # 경계 연속성: 미세 창 가장자리를 가로지르며 값이 튀는지
    edge = hi[0]
    xs = np.linspace(edge - 0.04, edge + 0.04, 81)
    line = np.stack([xs, np.full_like(xs, tc[1]), np.full_like(xs, tc[2])], axis=1)
    dl = field.distance(line)
    jump = np.abs(np.diff(dl)).max()
    print(f"  미세 창 x 경계 가로지르기: 최대 이웃 간 도약 {jump*1000:.2f} mm "
          f"(표본 간격 {(xs[1]-xs[0])*1000:.1f} mm)")

    # 4. 마스킹 ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. 마스킹 — 로봇 자기 관측이 사라졌는가")
    print("=" * 68)
    sc = np.asarray(d["sphere_centers"], np.float64)
    sr = np.asarray(d["sphere_radii"], np.float64)
    margin = 0.05
    for label, path in (("masked", args.field), ("raw", args.raw_field)):
        try:
            f2, _ = load(path, outside_distance=0.5)
        except FileNotFoundError:
            print(f"  {label:<7} — {path} 없음, 건너뜀")
            continue
        dv = f2.distance(sc)
        clr = dv - sr - margin
        print(f"  {label:<7} d 범위 [{dv.min():+.3f}, {dv.max():+.3f}] m   "
              f"d-r-margin 최악 {clr.min()*1000:+7.1f} mm   위반 {int((clr < 0).sum()):>3}/{len(sc)}")

    # 5. 정확도 ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("5. 정확도 — 테이블 상판을 두 계층이 얼마나 틀리게 보는가")
    print("=" * 68)
    for i, layer in enumerate(field.layers):
        g = layer.grid
        vs = g.voxel_size
        c = g.centres().reshape(-1, 3)
        v = layer.distance_grid.reshape(-1)
        sel = ((c[:, 0] > 0.45) & (c[:, 0] < 0.85) & (np.abs(c[:, 1]) < 0.4)
               & (np.abs(v) < vs))
        if sel.sum():
            z = float(np.median(c[sel, 2]))
            print(f"  계층 {i} ({vs*1000:4.1f} mm)  |d|<{vs*1000:.0f}mm 복셀의 z 중앙값 "
                  f"{z:.3f} m   참값 {TABLE_Z} 대비 {(z-TABLE_Z)*1000:+6.1f} mm   (n={sel.sum():,})")
    print(f"\n  target centroid 에서 합성 필드 d = {float(field.distance(tc[None])[0]):+.4f} m")


if __name__ == "__main__":
    main()
