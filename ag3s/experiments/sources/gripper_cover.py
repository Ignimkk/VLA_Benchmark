"""T43 R — 제약 모델의 그리퍼 질의 구가 **실제 collision mesh 를 덮게** 한다 (개수 · 배치로, 반지름을 키우지 않고).

T40 I 실측: E3b 설정(`--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32`)의
손가락 구는 손가락 collision mesh 정점의 4.3 % 만 담았고, 막대에 가장 가까운 손 점은 가장 가까운 구에서
1.7–8.2 mm 밖이었다. 원인은 굵기가 아니라 **배치**다. 지금 구는 MJCF 정점에 맞춘 slab capsule 3 개
(`mujoco_source.gap_filling_capsules`) 의 **축선 위** 사슬이고, 반지름이 0.05 배라 축선 둘레 3–6 mm
만 덮는다. 손바닥(`EE_BODY_*`) 은 127 × 66 × 74 mm 인데 구 23 개가 축 셋 위에 있다.

이 모듈이 하는 것은 둘이다.

1. **지표** (`coverage_metrics`) — link 마다 collision mesh(= MuJoCo 가 실제로 충돌시키는 볼록 껍질)
   의 정점 · 표면 표본이 구 합집합 안에 드는 비율, 합집합 표면까지의 최대 바깥 거리. 자기 link 구만
   (관절과 무관) · 그리퍼 전체 합집합 (손가락 관절 값마다) 둘 다.
2. **덮개 구** (`outer_cover_capsules`) — 바깥 표면을 **안쪽에서 접하는 구**로 덮는다. 표면점 `p` 와
   바깥 법선 `n` 에서 `c = p − r n` 인 구는 그 점에서 표면에 접하고, `c` 에서 표면까지 거리가 `r`
   이상이면 mesh 밖으로 나가지 않는다 (내접). 그러므로 **덮개 구는 실제 그리퍼보다 크지 않다** —
   사용자 지시("그리퍼 구는 최소로, 과일을 잡을 수 있게") 를 반지름 상한이 아니라 기하로 지킨다.
   반지름 상한은 그 link 의 **지금 구의 최대 반지름** 이다 (손가락 3.25 · 손바닥 6.22 mm, E3b).
   후보 (표면점 × 반지름) 중 아직 안 덮인 표면점을 가장 많이 덮는 것을 고르는 greedy set cover.

**파지 면은 그대로 둔다.** 손가락의 안쪽 면 (법선이 반대 손가락 쪽, `u_in · n > INNER_FACE_COS`) 은
덮을 대상이 아니고, 덮개 구는 `u_in · c + r ≤ (지금 그 손가락 구의 가장 안쪽 끝)` 을 지킨다 — 손가락
사이 공간 (사과 자리) 으로 지금보다 한 점도 더 나오지 않는다. 그래서 T26 의 max opening
(`gripper_openings`) 이 그대로다. 손바닥은 손가락 사이 통로 (`|x| <` 열린 손가락 안쪽 면) 에서 손끝
쪽을 보는 바닥면이 파지 면이다 — 덮지 않고, 덮개 구가 그 통로에서 지금 손바닥 구보다 손끝 쪽으로
더 내려가지 않는다.

**기본은 꺼져 있다.** `build_constraint_robot_model(..., gripper_cover=False)` 가 기본이고 그때 이
모듈은 불리지 않는다 — 모델이 T43 전과 글자 그대로 같다. 켜는 길은 `serve_safe --gripper-cover`.

이름 뒤바뀜 (T11 · T40 I): URDF `ee_finger_l1` ≡ MJCF `ee_finger_l2` (그 반대도). 이 모듈은 URDF
link 이름으로 일하고 mesh 는 `MJCF_BODY_ALIASES` 로 찾는다 — gap-filler 와 **같은 표** 다. 그래서
덮개는 그 link 의 실제 mesh 위에 놓인다 (`test_t43r_gripper_cover.py` 가 기하로 고정).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Mapping, Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

#: `UrdfCapsule.role` 의 덮개 구. 길이 0 capsule = 구 하나. `UrdfSphereChain` 은 이 역할의 구를
#: **반지름 그대로** 쓴다 (`capsule_radius_scale` · 간격 inflation 을 받지 않는다) — 이미 내접 구라
#: 키우면 mesh 밖으로 나가고, 줄이면 덮개가 아니다. 정의는 한 곳 (`urdf_sphere_chain.EXACT_SPHERE_ROLE`).
from benchmark.ag3s.robot_models.urdf_sphere_chain import EXACT_SPHERE_ROLE as GRIPPER_COVER_ROLE  # noqa: E402

#: 덮개 목표: 바깥 표면점에서 구 합집합까지 이 거리(m) 안. 사용자 (T43 task) "최대 틈 약 2–3 mm".
COVER_MAX_GAP = 0.0025
#: 표면 표본 간격 (m). 삼각형 안쪽 최대 오차 ≈ 0.43 × 간격.
COVER_SAMPLE_SPACING = 0.001
#: 후보 접점 간격 (m). 표면 표본을 이 격자로 솎은 것이 후보 중심의 접점이다.
COVER_CANDIDATE_SPACING = 0.002
#: 내접 허용오차 (m). `hull_depth` 가 정확하므로 부동소수 오차만 — 접점 `p` 가 껍질 면 위라 `c = p − r n`
#: 의 깊이는 정확히 `r` 이고, 허용오차 0 이면 반올림으로 절반이 떨어진다. 1 µm 는 기하적으로 0 이다.
COVER_INSCRIBE_TOL = 1e-6
#: 후보 반지름 = 상한 × 이 비율들. 상한은 그 link 의 지금 구 최대 반지름.
COVER_RADIUS_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
#: 법선이 반대 손가락 쪽을 이 코사인보다 더 보면 손가락 안쪽 (파지) 면.
INNER_FACE_COS = 0.5
#: 같은 body 의 다른 볼록 조각 안으로 이만큼(m) 넘게 들어간 표면점은 겉면이 아니다.
_INTERIOR_TOL = 0.0002
#: 이웃 강체 body (손목 · FT 센서) 의 collision 껍질에서 이 거리(m) 안의 표면점은 닿을 수 없는 면이다.
_NEIGHBOUR_TOL = 0.0005


@dataclasses.dataclass(frozen=True)
class LinkSurface:
    """한 URDF link 의 collision 겉면 (link frame).

    `points`·`normals` = 볼록 껍질 삼각형 표본 중 같은 body 의 다른 조각 안에 묻히지 않은 것.
    `grasp_face` = 파지 면 (덮지 않는 곳). `hidden` = 강체로 붙은 이웃 body (손목 · FT 센서) 의 껍질
    안 — 아무것도 닿을 수 없다. 덮을 대상 = `outer` = 둘 다 아닌 것. `vertices` = collision mesh 정점
    (T40 I 지표와 같은 것).
    """

    link: str
    body: str
    points: np.ndarray
    normals: np.ndarray
    grasp_face: np.ndarray
    vertices: np.ndarray
    hidden: np.ndarray
    hulls: tuple  # (A[k,3], b[k]) per convex piece: inside ⇔ A p + b ≤ 0

    @property
    def outer(self) -> np.ndarray:
        """덮을 대상: 파지 면도, 이웃 body 에 묻힌 면도 아닌 겉면."""
        return ~self.grasp_face & ~self.hidden


def _hull_samples(verts: np.ndarray, spacing: float):
    """볼록 껍질 표면 표본과 바깥 법선. MuJoCo 는 mesh geom 을 볼록 껍질로 충돌시킨다.

    정점 + 모서리마다 `spacing` 간격 + 삼각형 안쪽은 넓이에 비례한 개수 (R2 저불일치 수열 — 결정적).
    최대 변 기준 무게중심 격자는 가늘고 긴 껍질 삼각형에서 점이 제곱으로 불어난다 (손바닥 849 k 점).
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(verts)
    eq = hull.equations                                  # (f, 4): n · p + d ≤ 0 inside, n outward
    simp = hull.simplices
    vidx = np.unique(simp)
    vn = np.zeros((verts.shape[0], 3))
    for k in range(3):
        np.add.at(vn, simp[:, k], eq[:, :3])
    vn = vn[vidx]
    pts = [verts[vidx]]
    nrm = [vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)]
    # 모서리 (두 면이 만나는 곳 — 틈이 가장 큰 자리) : 이웃 두 면 법선의 평균
    edges: dict = {}
    for f, tri in enumerate(simp):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(tri[i], tri[j]), max(tri[i], tri[j]))
            edges.setdefault(key, []).append(f)
    for (i, j), faces in edges.items():
        length = float(np.linalg.norm(verts[j] - verts[i]))
        n_seg = int(np.ceil(length / float(spacing)))
        if n_seg < 2:
            continue
        t = (np.arange(1, n_seg) / n_seg)[:, None]
        pts.append(verts[i] + t * (verts[j] - verts[i]))
        en = eq[faces, :3].sum(0)
        nrm.append(np.repeat((en / max(np.linalg.norm(en), 1e-12))[None], n_seg - 1, axis=0))
    a, b, c = verts[simp[:, 0]], verts[simp[:, 1]], verts[simp[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    count = np.floor(area / (0.5 * float(spacing) ** 2)).astype(np.int64)
    g = 1.32471795724474602596                            # plastic number (R2 sequence)
    for f in np.nonzero(count > 0)[0]:
        k = np.arange(1, count[f] + 1)[:, None]
        uv = np.mod(np.hstack([k / g, k / (g * g)]) + 0.5, 1.0)
        flip = uv.sum(1) > 1.0
        uv[flip] = 1.0 - uv[flip]
        pts.append(a[f] + uv[:, :1] * (b[f] - a[f]) + uv[:, 1:] * (c[f] - a[f]))
        nrm.append(np.repeat(eq[f:f + 1, :3], uv.shape[0], axis=0))
    return np.vstack(pts), np.vstack(nrm), (eq[:, :3].copy(), eq[:, 3].copy())


def _geom_pieces(model, body_id: int) -> list:
    """`[(verts[n,3] in the body frame)]` of the body's **collision** mesh geoms."""
    import mujoco

    pieces = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        mesh = int(model.geom_dataid[geom])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[v0:v0 + nv], np.float64).reshape(-1, 3)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[geom])
        pieces.append(local @ R.reshape(3, 3).T + model.geom_pos[geom])
    return pieces


def _rigid_neighbours(model, body_id: int) -> list[tuple[int, np.ndarray]]:
    """`[(other body, T_body_other)]` — bodies **rigidly** attached to `body_id` (no joint between),
    other than itself. RB-Y1 `EE_BODY_L`: `link_left_arm_6`, `FT_SENSOR_L`, bracket, D435i."""
    import mujoco

    root = body_id
    while int(model.body_jntnum[root]) == 0 and int(model.body_parentid[root]) not in (root, 0):
        root = int(model.body_parentid[root])
    group, stack = [], [root]
    while stack:
        b = stack.pop()
        group.append(b)
        stack += [k for k in range(model.nbody)
                  if int(model.body_parentid[k]) == b and k != b and int(model.body_jntnum[k]) == 0]
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    def world(b):
        T = np.eye(4)
        T[:3, :3] = data.xmat[b].reshape(3, 3)
        T[:3, 3] = data.xpos[b]
        return T

    inv = np.linalg.inv(world(body_id))
    return [(b, inv @ world(b)) for b in group if b != body_id]


def body_collision_surface(model, body: str, T: np.ndarray, spacing: float = COVER_SAMPLE_SPACING):
    """`(points, normals, vertices, hulls, buried_by_neighbour)` of `body`'s **collision** mesh
    geoms, in the frame `T` maps the body frame into (`static_urdf_ancestor`).

    Visual geoms are not collision geometry. A sample buried in another convex piece of the same
    body is dropped (it is not surface). A sample inside a **rigidly attached neighbour**'s collision
    hull (the palm's top face under the FT sensor) is kept but flagged — nothing can reach it.
    Samples are thinned to one per `spacing` voxel (skinny hull triangles oversample their edges).
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise KeyError(f"no body {body!r}")
    T = np.asarray(T, np.float64)
    pieces = [v @ T[:3, :3].T + T[:3, 3] for v in _geom_pieces(model, bid)]
    if not pieces:
        raise KeyError(f"body {body!r} has no collision mesh geoms")
    pts, nrm, hulls = [], [], []
    for verts in pieces:
        p, n, h = _hull_samples(verts, spacing)
        pts.append(p)
        nrm.append(n)
        hulls.append(h)
    owner = np.concatenate([np.full(p.shape[0], i) for i, p in enumerate(pts)])
    pts, nrm = np.vstack(pts), np.vstack(nrm)
    buried = np.zeros(pts.shape[0], bool)
    for i, ((A, b), verts) in enumerate(zip(hulls, pieces)):
        # 조각의 AABB 안에 든 점만 판정한다 (전부 하면 손바닥에서 36 s)
        cand = np.nonzero((owner != i) & np.all(pts >= verts.min(0), axis=1)
                          & np.all(pts <= verts.max(0), axis=1))[0]
        if cand.size:
            buried[cand] |= np.max(pts[cand] @ A.T + b, axis=1) < -_INTERIOR_TOL
    pts, nrm = pts[~buried], nrm[~buried]
    cell = np.floor(pts / float(spacing)).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    first = np.sort(first)
    pts, nrm = pts[first], nrm[first]
    neighbour = np.zeros(pts.shape[0], bool)
    for other, T_bo in _rigid_neighbours(model, bid):
        for verts in _geom_pieces(model, other):
            from scipy.spatial import ConvexHull

            V = verts @ (T @ T_bo)[:3, :3].T + (T @ T_bo)[:3, 3]
            cand = np.nonzero(np.all(pts >= V.min(0) - _NEIGHBOUR_TOL, axis=1)
                              & np.all(pts <= V.max(0) + _NEIGHBOUR_TOL, axis=1))[0]
            if cand.size < 1 or V.shape[0] < 4:
                continue
            eq = ConvexHull(V).equations
            neighbour[cand] |= np.max(pts[cand] @ eq[:, :3].T + eq[:, 3], axis=1) <= _NEIGHBOUR_TOL
    return pts, nrm, np.vstack(pieces), tuple(hulls), neighbour


def finger_inward(robot_model, link: str) -> Optional[np.ndarray]:
    """손가락 link frame 에서 **반대 손가락 쪽** 단위 벡터 (닫히는 방향). 손가락이 아니면 None.

    URDF prismatic joint 의 축은 joint frame (= child link frame) 에 있고, 열림 값은
    `gripper_joint_pairs` 가 원점 간격 최대로 구조적으로 정한다 — 부호를 가정하지 않는다.
    """
    for pair in robot_model.gripper_joint_pairs():
        for joint_name, child in zip(pair["joints"], pair["children"]):
            if child != link:
                continue
            joint = robot_model.model.joint_by_name(joint_name)
            opening = float(pair["open"][joint_name]) - float(pair["closed"][joint_name])
            axis = np.asarray(joint.axis, np.float64)
            return -np.sign(opening) * axis / np.linalg.norm(axis)
    return None


def _palm_channel(robot_model, palm: str):
    """손바닥 frame 에서 손가락 사이 통로: `(u, half_width, tip_dir, finger_links)` 또는 None.

    `u` = 열린 두 손가락 원점을 잇는 단위 벡터, `half_width` = 열린 손가락 안쪽 면의 `|u·p|` 최소,
    `tip_dir` = 손바닥 원점에서 손가락 원점 쪽 (손끝 방향) 의 `u` 에 수직인 단위 벡터.
    """
    from benchmark.ag3s.robot_models.urdf_sphere_chain import _joint_transform, _NumpyBackend

    be = _NumpyBackend()
    for pair in robot_model.gripper_joint_pairs():
        if pair["parent_link"] != palm:
            continue
        origins, faces = [], []
        for joint_name, child in zip(pair["joints"], pair["children"]):
            joint = robot_model.model.joint_by_name(joint_name)
            Tj = np.asarray(_joint_transform(joint, float(pair["open"][joint_name]), be), np.float64)
            origins.append(Tj[:3, 3])
            u_in = finger_inward(robot_model, child)
            c, r = robot_model.local_spheres(child)
            inner = float(np.max(c @ u_in + r)) if r.size else 0.0
            faces.append(Tj[:3, 3] + Tj[:3, :3] @ (u_in * inner))
        u = origins[0] - origins[1]
        u = u / np.linalg.norm(u)
        half = float(min(abs(float(u @ f)) for f in faces))
        mid = 0.5 * (origins[0] + origins[1])
        tip = mid - u * float(u @ mid)
        tip = tip / max(np.linalg.norm(tip), 1e-12)
        return u, half, tip, tuple(pair["children"])
    return None


def link_surface(model, robot_model, link: str, *, spacing: float = COVER_SAMPLE_SPACING
                 ) -> LinkSurface:
    """`link` (URDF 이름) 의 collision 겉면 · 파지 면 표시."""
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        MJCF_BODY_ALIASES, static_urdf_ancestor)

    body = MJCF_BODY_ALIASES.get(link, link)
    found, T = static_urdf_ancestor(model, body, robot_model.model.links)
    if found != link:
        raise ValueError(f"{link!r}: MJCF body {body!r} hangs off URDF link {found!r}")
    pts, nrm, verts, hulls, hidden = body_collision_surface(model, body, T, spacing)
    grasp = np.zeros(pts.shape[0], bool)
    inner, channel = grasp_limits(robot_model, link)
    if inner is not None:
        u_in, h = inner
        grasp = (nrm @ u_in > INNER_FACE_COS) | (pts @ u_in > h)
    if channel is not None:
        u, half, tip, z_lim = channel
        grasp |= (np.abs(pts @ u) < half) & ((nrm @ tip > INNER_FACE_COS) | (pts @ tip > z_lim))
    return LinkSurface(link=link, body=body, points=pts, normals=nrm, grasp_face=grasp & ~hidden,
                       vertices=verts, hidden=hidden, hulls=hulls)


def grasp_limits(robot_model, link: str):
    """`(inner, channel)` — 덮개 구가 넘지 않을 파지 쪽 경계 (link frame). 해당 없으면 None.

    * 손가락 `inner = (u_in, h)`: `h` = 지금 그 손가락 구의 가장 안쪽 끝 `max(u_in · c + r)`.
    * 손바닥 `channel = (u, half, tip, z_lim)`: 열린 손가락 사이 통로 `|u · p| < half` 에서 지금
      손바닥 구의 가장 손끝 쪽 끝 `z_lim = max(tip · c + r)` (통로에 걸친 구만).

    지금 구에서 읽으므로 **덮개 구는 이 값을 바꾸지 않는다** (덮개가 이 경계를 지키므로 max 가 그대로)
    — 덮개를 켠 모델에서 다시 물어도 같은 경계가 나온다.
    """
    keep = [i for i, (name, role) in enumerate(zip(robot_model.sphere_link_names,
                                                   robot_model.sphere_roles))
            if name == link and role != GRIPPER_COVER_ROLE]
    local = [robot_model._local_spheres[i] for i in keep]
    if not local:
        return None, None
    c0 = np.asarray([c for _, c, _ in local], np.float64).reshape(-1, 3)
    r0 = np.asarray([r for _, _, r in local], np.float64)
    inner = None
    u_in = finger_inward(robot_model, link)
    if u_in is not None:
        inner = (u_in, float(np.max(c0 @ u_in + r0)))
    channel = None
    ch = _palm_channel(robot_model, link)
    if ch is not None:
        u, half, tip, _ = ch
        in_ch = np.abs(c0 @ u) < half + r0
        z_lim = float(np.max((c0 @ tip + r0)[in_ch])) if in_ch.any() else -np.inf
        channel = (u, half, tip, z_lim)
    return inner, channel


def _gap(points: np.ndarray, centres: np.ndarray, radii: np.ndarray, chunk: int = 4096
         ) -> np.ndarray:
    """`min_s |p − c_s| − r_s` — 합집합 밖이면 합집합 표면까지의 거리, 안이면 ≤ 0."""
    out = np.full(points.shape[0], np.inf)
    if radii.size == 0:
        return out
    for i in range(0, points.shape[0], chunk):
        p = points[i:i + chunk]
        out[i:i + chunk] = np.min(
            np.linalg.norm(p[:, None, :] - centres[None], axis=2) - radii[None], axis=1)
    return out


def _stats(gap: np.ndarray) -> dict:
    if gap.size == 0:
        return {"n": 0}
    return {"n": int(gap.size),
            "frac_inside": float(np.mean(gap <= 0.0)),
            "frac_within_max_gap": float(np.mean(gap <= COVER_MAX_GAP)),
            "max_gap_mm": float(max(gap.max(), 0.0) * 1e3),
            "p95_gap_mm": float(max(np.percentile(gap, 95), 0.0) * 1e3),
            "median_gap_mm": float(np.median(gap) * 1e3)}


def coverage_metrics(model, robot_model, links: Sequence[str], *,
                     finger_values: Sequence[Mapping[str, float]] = (),
                     spacing: float = COVER_SAMPLE_SPACING,
                     surfaces: Optional[Mapping[str, LinkSurface]] = None) -> dict:
    """link 마다 덮임 지표. 모든 거리는 그 link frame 에서, 구는 `robot_model` 의 것.

    * `own` — 그 link 자기 구만 (손가락 관절과 무관 — 구와 mesh 가 같이 움직인다).
      `vertices` = collision mesh 정점 (T40 I 의 4.3 % 와 같은 정의), `outer` = 겉면 표본 중 파지 면이
      아닌 것, `grasp_face` = 파지 면 표본.
    * `gripper_union` — `finger_values` 의 각 손가락 관절 값에서, 같은 그리퍼 (손바닥 + 두 손가락)
      의 **모든** 구 합집합 대 그 link 의 겉면. 다른 link 의 구가 덮어 주는 몫이 여기 들어간다.
    """
    from benchmark.ag3s.robot_models.urdf_sphere_chain import _joint_transform, _NumpyBackend

    be = _NumpyBackend()
    surfaces = dict(surfaces or {})
    out: dict = {}
    pairs = robot_model.gripper_joint_pairs()
    parent_of = {child: p for p in pairs for child in p["children"]}
    joint_of = {child: j for p in pairs for j, child in zip(p["joints"], p["children"])}
    for link in links:
        surf = surfaces.get(link) or link_surface(model, robot_model, link, spacing=spacing)
        surfaces[link] = surf
        c, r = robot_model.local_spheres(link)
        outer = surf.outer
        rec = {"mujoco_body": surf.body, "n_spheres": int(r.size),
               "sphere_radius_mm": ([float(r.min() * 1e3), float(r.max() * 1e3)] if r.size else None),
               "own": {"vertices": _stats(_gap(surf.vertices, c, r)),
                       "outer": _stats(_gap(surf.points[outer], c, r)),
                       "grasp_face": _stats(_gap(surf.points[surf.grasp_face], c, r)),
                       "n_hidden": int(surf.hidden.sum())}}
        # 그리퍼 합집합: 손바닥 frame 으로 모은다
        palm = link if link not in parent_of else parent_of[link]["parent_link"]
        pair = next((p for p in pairs if p["parent_link"] == palm), None)
        if pair is not None and finger_values:
            rows = []
            for values in finger_values:
                def pose(child):
                    joint = robot_model.model.joint_by_name(joint_of[child])
                    q = float(values.get(joint.name, robot_model.joint_parameters.get(joint.name, 0.0)))
                    return np.asarray(_joint_transform(joint, q, be), np.float64)
                frames = {palm: np.eye(4), **{ch: pose(ch) for ch in pair["children"]}}
                cs, rs = [], []
                for name, Tn in frames.items():
                    cc, rr = robot_model.local_spheres(name)
                    if rr.size:
                        cs.append(cc @ Tn[:3, :3].T + Tn[:3, 3])
                        rs.append(rr)
                C, Rr = np.vstack(cs), np.concatenate(rs)
                Tl = frames[link]
                pts = surf.points[outer] @ Tl[:3, :3].T + Tl[:3, 3]
                ver = surf.vertices @ Tl[:3, :3].T + Tl[:3, 3]
                rows.append({"finger_joints": {k: float(v) for k, v in values.items()},
                             "outer": _stats(_gap(pts, C, Rr)),
                             "vertices": _stats(_gap(ver, C, Rr))})
            rec["gripper_union"] = rows
        out[link] = rec
    return out


def hull_depth(points: np.ndarray, hulls) -> np.ndarray:
    """`max_i (−max_j (A_i p + b_i)_j)` — the deepest a point sits in **one** convex piece (m; < 0
    outside every piece). A ball of radius ≤ this around `p` lies inside that piece, hence inside
    the collision geometry — **exactly**, with no sampling error (T43 R: the inscribed test)."""
    depth = np.full(points.shape[0], -np.inf)
    for A, b in hulls:
        depth = np.maximum(depth, -np.max(points @ A.T + b, axis=1))
    return depth


def outer_cover_spheres(surface: LinkSurface, existing: tuple[np.ndarray, np.ndarray], *,
                        r_max: float, max_gap: float = COVER_MAX_GAP,
                        candidate_spacing: float = COVER_CANDIDATE_SPACING,
                        fractions: Sequence[float] = COVER_RADIUS_FRACTIONS,
                        inscribe_tol: float = COVER_INSCRIBE_TOL,
                        inner_limit: Optional[tuple[np.ndarray, float]] = None,
                        channel: Optional[tuple[np.ndarray, float, np.ndarray, float]] = None,
                        max_spheres: Optional[int] = None,
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(centres, radii, report)` — 겉면을 `max_gap` 안으로 덮는 내접 구 (link frame).

    `existing` 구가 이미 덮은 표면점은 대상이 아니다 (빈 곳만 채운다). 후보는 겉면 (파지 면 제외)
    점 `p` · 법선 `n` · 반지름 `r ∈ r_max × fractions` 에서 `c = p − r n`;

    * 내접: `c` 가 어느 볼록 조각 안이고, 겉면 표본까지 최소 거리 ≥ `r − inscribe_tol`;
    * `inner_limit = (u_in, h)` (손가락): `u_in · c + r ≤ h` — 지금 구보다 안쪽 (사과 쪽) 으로 안 나온다;
    * `channel = (u, half, tip, z_lim)` (손바닥): 통로 `|u · c| < half + r` 안이면 `tip · c + r ≤ z_lim`
      — 손가락 사이에서 지금 손바닥 구보다 손끝 쪽으로 안 내려온다.

    greedy: 아직 안 덮인 대상점을 가장 많이 덮는 후보 (동률이면 작은 반지름) 를 고른다.
    `max_spheres` 를 주면 그 개수에서 멈춘다 (덮는 양이 큰 것부터라 예산 안에서 가장 많이 덮는다).
    """
    from scipy.spatial import cKDTree

    outer = surface.outer
    pts, nrm = surface.points[outer], surface.normals[outer]
    c0, r0 = existing
    target = pts[_gap(pts, c0, r0) > max_gap]
    report = {"n_targets": int(target.shape[0])}
    if target.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0), {**report, "n_uncovered": 0}
    # 후보 접점 = 겉면 표본을 candidate_spacing 격자로 솎은 것 (격자 칸마다 첫 점)
    cell = np.floor(pts / float(candidate_spacing)).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    anchors, anchor_n = pts[first], nrm[first]
    cands_c, cands_r = [], []
    for f in fractions:
        r = float(r_max) * float(f)
        c = anchors - r * anchor_n
        # 내접: 한 볼록 조각 안에 통째로 든다 (정확). 표본 거리로 재면 간격만큼 mesh 밖으로 나간다
        # (실측 C1807 손바닥 0.39 mm). 조각 경계를 걸치는 구는 버린다 — 덮개가 조금 늘어난다.
        ok = hull_depth(c, surface.hulls) >= r - inscribe_tol
        if inner_limit is not None:
            u_in, h = inner_limit
            ok &= c @ u_in + r <= h + 1e-12
        if channel is not None:
            u, half, tip, z_lim = channel
            in_channel = np.abs(c @ u) < half + r
            ok &= ~in_channel | (c @ tip + r <= z_lim + 1e-12)
        cands_c.append(c[ok])
        cands_r.append(np.full(int(ok.sum()), r))
    C = np.vstack(cands_c)
    R = np.concatenate(cands_r)
    report["n_candidates"] = int(R.size)
    if R.size == 0:
        return np.zeros((0, 3)), np.zeros(0), {**report, "n_uncovered": int(target.shape[0])}
    t_tree = cKDTree(target)
    covers = [np.asarray(t_tree.query_ball_point(c, float(r) + max_gap), np.int64)
              for c, r in zip(C, R)]
    uncovered = np.ones(target.shape[0], bool)
    chosen, newly = [], []
    gain = np.array([len(cv) for cv in covers])
    while uncovered.any() and (max_spheres is None or len(chosen) < int(max_spheres)):
        # lazy greedy: 위에서부터 실제 이득을 다시 재고, 여전히 최대면 고른다
        order = np.lexsort((R, -gain))
        best, best_gain = -1, 0
        for j in order:
            if gain[j] <= best_gain:
                break
            g = int(uncovered[covers[j]].sum())
            gain[j] = g
            if g > best_gain or (g == best_gain and best >= 0 and R[j] < R[best]):
                best, best_gain = j, g
        if best < 0 or best_gain == 0:
            break
        chosen.append(best)
        newly.append(best_gain)
        uncovered[covers[best]] = False
    report["n_uncovered"] = int(uncovered.sum())
    #: greedy 곡선 — k 번째 구까지 덮은 대상점 누적 수 (구 예산을 정할 때 읽는다)
    report["covered_after"] = np.cumsum(newly).astype(int).tolist()
    idx = np.asarray(chosen, np.int64)
    return C[idx], R[idx], report


def outer_cover_capsules(model, robot_model, links: Sequence[str], *,
                         max_gap: float = COVER_MAX_GAP,
                         spacing: float = COVER_SAMPLE_SPACING,
                         r_max_by_link: Optional[Mapping[str, float]] = None,
                         max_spheres_by_link: Optional[Mapping[str, int]] = None
                         ) -> tuple[list, dict]:
    """`(extra_capsules, report)` — `robot_model` (덮개 없는 제약 모델) 의 그리퍼 link 마다 덮개 구.

    반지름 상한 = 그 link 의 지금 구 최대 반지름 (`r_max_by_link` 로 바꿀 수 있다). 결과는 길이 0
    `UrdfCapsule` (role `GRIPPER_COVER_ROLE`) — `UrdfSphereChain` 이 반지름 그대로 구 하나로 쓴다.
    `max_spheres_by_link` = link 별 구 예산 (없으면 다 덮을 때까지).
    """
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    out: list = []
    report: dict = {}
    for link in links:
        c0, r0 = robot_model.local_spheres(link)
        if r0.size == 0:
            continue
        r_max = float((r_max_by_link or {}).get(link, r0.max()))
        surf = link_surface(model, robot_model, link, spacing=spacing)
        inner, channel = grasp_limits(robot_model, link)
        budget = (max_spheres_by_link or {}).get(link)
        key = _cache_key(model, surf, c0, r0, r_max, max_gap, spacing, inner, channel) + (budget,)
        hit = _CACHE.get(key)
        if hit is None:
            hit = outer_cover_spheres(surf, (c0, r0), r_max=r_max, max_gap=max_gap,
                                      inner_limit=inner, channel=channel, max_spheres=budget)
            _CACHE[key] = hit
        centres, radii, rep = hit
        report[link] = {**rep, "n_spheres": int(radii.size), "r_max_mm": r_max * 1e3,
                        "radius_mm": ([float(radii.min() * 1e3), float(radii.max() * 1e3)]
                                      if radii.size else None)}
        for c, r in zip(centres, radii):
            origin = np.eye(4)
            origin[:3, 3] = c
            out.append(UrdfCapsule(link=link, origin=origin, radius=float(r), length=0.0,
                                   role=GRIPPER_COVER_ROLE))
    return out, report


#: 덮개 구 캐시 — greedy 가 link 당 수 초라 모델을 짓는 곳 (서버 · 테스트) 마다 다시 하지 않는다.
#: 키는 mesh 표본 · 지금 구 · 설정의 digest.
_CACHE: dict = {}


def _cache_key(model, surf, c0, r0, r_max, max_gap, spacing, inner, channel) -> tuple:
    import hashlib

    h = hashlib.sha1()
    for a in (surf.points, surf.normals, surf.grasp_face, c0, r0):
        h.update(np.ascontiguousarray(a).tobytes())
    for x in (inner, channel):
        h.update(repr(None if x is None else [np.asarray(v).tolist() for v in x]).encode())
    # link 이름은 키에 넣지 않는다 — 왼손 · 오른손 (그리고 네 손가락) 은 mesh 와 지금 구가 같아서
    # 같은 답이고 (link frame), 한 번만 푼다.
    return (h.hexdigest(), float(r_max), float(max_gap), float(spacing))


# =============================================================================================
# T43 FC — 손가락 몸통 구 (`serve_safe --finger-cover`, 기본 꺼짐)
# =============================================================================================
#
# 사용자 판정 (2026-10-06 10:00, "B"): 그리퍼마다 **두 손가락을 각각 따로**, 실제 굵기의 구로 감싼다.
# 두 손가락을 한 덩어리로 감싸지 않는다 — 손가락 사이 공간은 비어 있다.
#
# 지금 손가락 구 (E3b: slab capsule 3 개의 축선 위 사슬, 배율 0.05) 는 반지름 2.6–3.3 mm 이고 손가락
# collision mesh 표본의 4.4 % 만 담는다 (가장 먼 표면점 13.6 mm 밖). 배율을 1 로 올리면 URDF 상당
# capsule (r 13.0–17.6 mm) 이 되는데, 손가락 판은 두께 9.5 mm (x −3.3…6.2 mm) 라 그 capsule 은 안쪽
# (사과 쪽) 으로 13 mm, 손끝 아래로 14 mm 넘친다. capsule 축이 판의 폭 방향 (y) 이라 판 길이 (z) 를
# 덮으려면 반지름이 slab 반길이 (~10 mm) 이상이어야 하므로 어떤 배율도 "실제 굵기" 가 되지 않는다.
#
# 그래서 손가락마다 구를 **mesh 에서 다시 만든다**:
#
# * 후보 중심 = mesh 안 격자점 `c` (간격 `grid`). 반지름 = `depth(c) + overflow`.
#   `depth(c)` = max(한 볼록 조각 안의 정확한 깊이 (`hull_depth`), 겉면 표본까지 거리 − `union_eps`).
#   `ball(c, depth) ⊂ mesh` 이므로 `ball(c, depth + overflow)` 의 어느 점도 mesh 에서 `overflow` 보다
#   멀지 않다 — **넘친 양 ≤ overflow** (뒤 항은 표본 간격만큼의 근사; 보고서가 실측을 싣는다).
# * 대상 = 그 손가락의 collision 겉면 표본 **전부** (파지 면 포함 — 실제 굵기). 같은 body 의 다른 조각에
#   맞붙어 묻힌 면 (표본에서 법선 쪽 `face_probe` 가 mesh 안) 은 겉면이 아니므로 뺀다.
# * greedy set cover (아직 안 덮인 대상을 가장 많이 담는 후보) 로 대상 전부를 **구 안에** (틈 0) 담고,
#   다른 구들만으로 이미 담기는 구는 뒤에서부터 지운다.
#
# 결과는 그 손가락 link 의 gap-filling capsule 을 **대신한다** (덮개처럼 뒤에 붙이지 않는다 — 옛 축선
# 사슬은 새 구 안에 들어 있어 행만 늘린다). role = `FINGER_BODY_ROLE` (반지름 그대로).
# 손바닥 · 다른 link 은 그대로다.

from benchmark.ag3s.robot_models.urdf_sphere_chain import FINGER_BODY_ROLE  # noqa: E402

#: 넘친 양 (m): 구 합집합이 손가락 collision mesh 밖으로 나가는 최대 거리. 2.5 mm = T43 R 의 덮개 틈과
#: 같은 크기 (사용자 "최대 틈 약 2–3 mm"). 구 수와 맞바꾼다 (손가락 하나: 1.0 mm 253 · 2.0 mm 91 ·
#: 2.5 mm 56 · 3.0 mm 48 구, `outputs/impl/T43FC/build_check.json`).
FINGER_COVER_OVERFLOW = 0.0025
#: 표면 표본 · 후보 격자 간격 (m).
FINGER_COVER_SPACING = 0.0005
#: 겉면 표본까지 거리로 잰 깊이에서 빼는 여유 (m) — 표본 간격 (0.5 mm 격자에서 표면점과 가장 가까운
#: 표본 사이 거리) 을 흡수한다.
FINGER_COVER_UNION_EPS = 0.0005
#: 맞붙은 면 판정: 표본에서 바깥 법선으로 이만큼 (m) 나간 점이 mesh 안이면 겉면이 아니다.
FINGER_COVER_FACE_PROBE = 0.0003


def finger_body_targets(surface: LinkSurface, face_probe: float = FINGER_COVER_FACE_PROBE
                        ) -> np.ndarray:
    """덮을 대상: 겉면 표본 중 같은 body 의 다른 볼록 조각에 맞붙어 묻힌 면이 아닌 것 (bool mask).

    `body_collision_surface` 는 다른 조각 **안으로** 들어간 표본만 뺀다. 두 조각이 면을 맞대면 그 면의
    표본은 경계 위라 남는다 — 그 면은 겉이 아니고, 그것을 대상으로 두면 구가 쓸데없이 는다.
    파지 면 · 이웃 강체 표시는 쓰지 않는다 (실제 굵기 — 손가락 전부).
    """
    return ~(hull_depth(surface.points + float(face_probe) * surface.normals, surface.hulls) > 0.0)


def _greedy_exact_cover(target: np.ndarray, C: np.ndarray, R: np.ndarray, *, pad: float = 0.0,
                        max_count: Optional[int] = None):
    """`(chosen, n_uncovered)` — 대상점 전부를 구 **안에** 담는 greedy set cover + 중복 제거.

    T43 FR 선택지 (기본값이면 예전과 같은 계산): `pad` > 0 이면 대상이 구 표면에서 `pad` 안에만 있어도
    담긴 것으로 친다 (덮임 허용 틈). `max_count` 를 주면 그만큼 고르고 멈춘다 (남은 대상은 안 덮인다).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    covers = [np.asarray(tree.query_ball_point(c, float(r) + float(pad) if pad else float(r)), np.int64)
              for c, r in zip(C, R)]
    gain = np.array([cv.size for cv in covers])
    uncovered = np.ones(target.shape[0], bool)
    chosen: list[int] = []
    while uncovered.any() and (max_count is None or len(chosen) < int(max_count)):
        order = np.lexsort((R, -gain))
        best, best_gain = -1, 0
        for j in order:
            if gain[j] <= best_gain:
                break
            g = int(uncovered[covers[j]].sum())
            gain[j] = g
            if g > best_gain:
                best, best_gain = j, g
        if best < 0 or best_gain == 0:
            break
        chosen.append(int(best))
        uncovered[covers[best]] = False
    # 중복 제거: 나중에 고른 (덜 담은) 구부터, 그 구의 대상이 모두 다른 구에도 담기면 지운다
    count = np.zeros(target.shape[0], np.int64)
    for j in chosen:
        count[covers[j]] += 1
    kept = list(chosen)
    for j in reversed(chosen):
        if covers[j].size and np.all(count[covers[j]] >= 2):
            count[covers[j]] -= 1
            kept.remove(j)
    return np.asarray(sorted(kept, key=chosen.index), np.int64), int(uncovered.sum())


def _sampled_overflow(C: np.ndarray, R: np.ndarray, hulls, surface_points: np.ndarray,
                      n: int = 600) -> float:
    """구 표면 표본 중 mesh 밖인 점에서 가장 가까운 mesh 표면 표본까지 거리의 최대 (m, 실측 넘친 양)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(surface_points)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = 0.0
    for c, r in zip(C, R):
        P = c + r * v
        outside = P[hull_depth(P, hulls) < 0.0]
        if outside.size:
            out = max(out, float(tree.query(outside)[0].max()))
    return out


#: T43 FR — 손가락 몸통 구를 고르는 방법. `greedy` = FC 그대로 (기본). `ilp` = 같은 후보 · 같은 보장
#: (넘침 ≤ overflow, 대상 전부 덮음) 에서 구 수를 정수계획 (set cover, HiGHS) 으로 최소화한다.
FINGER_COVER_SOLVERS = ("greedy", "ilp")
#: `ilp` 후보 격자 = 0.5 mm 격자에서 축마다 하나 건너 (1 mm) + greedy 가 고른 중심. 0.5 mm 전부 (13.6 만)
#: 는 정수계획이 너무 크다.
FINGER_COVER_ILP_STRIDE = 2
#: `ilp` 를 그 자리에서 풀 때 시간 상한 (s). 미리 푼 답 (`finger_cover_presets`) 이 있으면 풀지 않는다.
FINGER_COVER_ILP_TIME_LIMIT = 300.0


def _finger_candidates(surface: LinkSurface, target: np.ndarray, grid: float, union_eps: float):
    """`(G, depth, ijk)` — mesh 안 격자점 · 그 깊이 · 격자 index (`finger_body_spheres` 머리말의 후보)."""
    from scipy.spatial import cKDTree

    lo, hi = surface.vertices.min(0), surface.vertices.max(0)
    axes = [np.arange(a, b + 1e-9, float(grid)) for a, b in zip(lo, hi)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    piece = hull_depth(G, surface.hulls)
    inside = piece > 0.0
    ijk = np.stack(np.meshgrid(*[np.arange(a.size) for a in axes], indexing="ij"), -1).reshape(-1, 3)
    G, piece, ijk = G[inside], piece[inside], ijk[inside]
    surf = cKDTree(target).query(G)[0] - float(union_eps)
    depth = np.maximum(piece, surf)
    return G, depth, ijk


def _ilp_cover(target: np.ndarray, C: np.ndarray, R: np.ndarray, *, pad: float = 0.0,
               time_limit: float = FINGER_COVER_ILP_TIME_LIMIT):
    """`(chosen, info)` — 대상 전부를 담는 최소 개수 구 (0/1 정수계획, `scipy.optimize.milp`).

    시간 상한에 걸리면 그때까지 찾은 가장 좋은 답 (최적이 아닐 수 있다 — `info["optimal"]`). 답이 없으면
    `chosen = None`."""
    import time

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    cols = [np.asarray(tree.query_ball_point(c, float(r) + float(pad)), np.int64) for c, r in zip(C, R)]
    indptr = np.cumsum([0] + [c.size for c in cols])
    A = csc_matrix((np.ones(int(indptr[-1])), np.concatenate(cols), indptr),
                   shape=(target.shape[0], C.shape[0]))
    t0 = time.perf_counter()
    res = milp(c=np.ones(C.shape[0]), constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(C.shape[0]), bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit), "disp": False, "mip_rel_gap": 0.0})
    info = {"status": int(res.status), "optimal": int(res.status) == 0,
            "dual_bound": None if getattr(res, "mip_dual_bound", None) is None
            else float(res.mip_dual_bound),
            "sec": round(time.perf_counter() - t0, 1), "n_candidates": int(C.shape[0])}
    if res.x is None:
        return None, info
    return np.flatnonzero(res.x > 0.5), info


def _covered(target: np.ndarray, C: np.ndarray, R: np.ndarray, pad: float) -> bool:
    return bool(R.size) and bool(np.all(_gap(target, C, R) <= float(pad) + 1e-12))


def finger_body_spheres(surface: LinkSurface, *, overflow: float = FINGER_COVER_OVERFLOW,
                        grid: float = FINGER_COVER_SPACING,
                        union_eps: float = FINGER_COVER_UNION_EPS,
                        face_probe: float = FINGER_COVER_FACE_PROBE,
                        gap: float = 0.0, max_spheres: Optional[int] = None,
                        solver: str = "greedy",
                        ilp_time_limit: float = FINGER_COVER_ILP_TIME_LIMIT,
                        use_presets: bool = True,
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(centres, radii, report)` — 손가락 collision mesh 전부를 담고 `overflow` 넘게 나가지 않는 구.

    `surface` 는 `link_surface(..., spacing=grid)` (link frame). 위 머리말의 규칙.

    T43 FR — 구 수를 줄이는 선택지 (기본값이면 FC 와 같은 계산 · 같은 답):

    * `gap` (m, 기본 0): 대상이 구 합집합 표면에서 `gap` 안이면 담긴 것으로 친다. 안 덮인 표본이 생기지만
      그 깊이는 `gap` 이하다 (보고서 `max_gap_mm`). 넘침 보장은 그대로다.
    * `max_spheres` (기본 없음, `greedy` 만): 손가락마다 그만큼 고르고 멈춘다. 덮임 보장이 없다.
    * `solver="ilp"`: 같은 후보 (1 mm 격자 + greedy 의 중심) 에서 구 수 최소화. 보장 (넘침 ≤ overflow,
      대상 전부 `gap` 안) 은 greedy 와 같다. 미리 푼 답 (`finger_cover_presets`, 격자 index) 이 있으면
      그 중심의 반지름을 다시 계산하고 덮임을 검사해서 쓴다 (풀지 않는다). 없거나 검사에 떨어지면 그
      자리에서 푼다 (`ilp_time_limit`). greedy 보다 많으면 greedy 답을 쓴다.
    """
    if not overflow > 0.0:
        raise ValueError(f"overflow 는 0 보다 커야 합니다 (m): {overflow}")
    if not (gap >= 0.0 and np.isfinite(gap)):
        raise ValueError(f"gap 은 0 이상이어야 합니다 (m): {gap}")
    if solver not in FINGER_COVER_SOLVERS:
        raise ValueError(f"solver 는 {FINGER_COVER_SOLVERS} 중 하나: {solver!r}")
    if max_spheres is not None:
        if int(max_spheres) < 1 or int(max_spheres) != max_spheres:
            raise ValueError(f"max_spheres 는 1 이상의 정수: {max_spheres}")
        if solver != "greedy":
            raise ValueError("max_spheres 는 solver='greedy' 에서만 씁니다 (ilp 는 전부 덮는 최소 개수)")
    outer = finger_body_targets(surface, face_probe)
    target = surface.points[outer]
    lo, hi = surface.vertices.min(0), surface.vertices.max(0)
    G, depth, ijk = _finger_candidates(surface, target, grid, union_eps)
    R_all = depth + float(overflow)
    idx, n_unc = _greedy_exact_cover(target, G, R_all, pad=gap, max_count=max_spheres)
    extra: dict = {}
    if solver == "ilp":
        extra = {"solver": "ilp", "greedy_n_spheres": int(idx.size)}
        chosen, src = None, None
        preset = None
        if use_presets and float(grid) == FINGER_COVER_SPACING and float(union_eps) == \
                FINGER_COVER_UNION_EPS and float(face_probe) == FINGER_COVER_FACE_PROBE:
            from benchmark.ag3s.experiments.sources.finger_cover_presets import (
                FINGER_BODY_ILP_PRESETS)

            preset = FINGER_BODY_ILP_PRESETS.get((int(round(overflow * 1e6)), int(round(gap * 1e6))))
        if preset is not None:
            key = {tuple(v): i for i, v in enumerate(ijk.tolist())}
            pick = [key.get(tuple(v)) for v in preset["ijk"]]
            if all(p is not None for p in pick) and _covered(
                    target, G[pick], R_all[pick], gap):
                chosen, src = np.asarray(pick, np.int64), "preset"
                extra["ilp"] = {k: v for k, v in preset.items() if k != "ijk"}
            else:
                _LOG.warning("finger cover ilp preset (overflow %.2f mm, gap %.2f mm) 가 이 mesh 에 "
                             "맞지 않습니다 — 그 자리에서 풉니다", overflow * 1e3, gap * 1e3)
        if chosen is None:
            sub = np.all(ijk % FINGER_COVER_ILP_STRIDE == 0, axis=1)
            sub[idx] = True
            cand = np.flatnonzero(sub)
            sel, info = _ilp_cover(target, G[cand], R_all[cand], pad=gap, time_limit=ilp_time_limit)
            extra["ilp"] = info
            if sel is not None and sel.size < idx.size and _covered(
                    target, G[cand[sel]], R_all[cand[sel]], gap):
                chosen, src = cand[sel], "solved"
            else:
                chosen, src = idx, "greedy_fallback"
        extra["source"] = src
        idx, n_unc = chosen, 0
    C, R = G[idx], R_all[idx]
    gaps = _gap(target, C, R)
    if gap or max_spheres is not None:
        n_unc = int((gaps > 1e-12).sum())
        extra.update({"gap_mm": float(gap * 1e3),
                      **({"max_spheres": int(max_spheres)} if max_spheres is not None else {})})
    report = {
        "n_spheres": int(R.size),
        "radius_mm": [float(R.min() * 1e3), float(R.max() * 1e3)] if R.size else None,
        "overflow_mm": float(overflow * 1e3),
        "n_targets": int(target.shape[0]),
        "n_uncovered": int(n_unc),
        "max_gap_mm": float(max(gaps.max(), 0.0) * 1e3) if gaps.size else 0.0,
        "overflow_sampled_mm": _sampled_overflow(C, R, surface.hulls, surface.points) * 1e3,
        "mesh_bbox_mm": [np.round(lo * 1e3, 2).tolist(), np.round(hi * 1e3, 2).tolist()],
        "spheres_bbox_mm": [np.round((C - R[:, None]).min(0) * 1e3, 2).tolist(),
                            np.round((C + R[:, None]).max(0) * 1e3, 2).tolist()],
        **extra,
    }
    return C, R, report


def finger_body_capsules(model, robot_model, links: Sequence[str], *,
                         overflow: float = FINGER_COVER_OVERFLOW,
                         grid: float = FINGER_COVER_SPACING,
                         gap: float = 0.0, max_spheres: Optional[int] = None,
                         solver: str = "greedy") -> tuple[list, dict]:
    """`(capsules, report)` — 손가락 link 마다 몸통 구 (길이 0 `UrdfCapsule`, role `FINGER_BODY_ROLE`).

    `robot_model` 은 mesh 를 찾는 데만 쓴다 (URDF link → MJCF body, `MJCF_BODY_ALIASES`). 네 손가락은
    link frame 에서 mesh 가 같아 한 번만 푼다 (캐시 키 = 표본 digest). `gap` · `max_spheres` · `solver`
    는 T43 FR 의 구 수 선택지 (`finger_body_spheres`; 기본값 = FC 그대로).
    """
    import hashlib

    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    out: list = []
    report: dict = {}
    for link in links:
        surf = link_surface(model, robot_model, link, spacing=grid)
        h = hashlib.sha1()
        for a in (surf.points, surf.normals):
            h.update(np.ascontiguousarray(a).tobytes())
        key = ("finger_body", h.hexdigest(), float(overflow), float(grid), float(gap),
               None if max_spheres is None else int(max_spheres), str(solver))
        hit = _CACHE.get(key)
        if hit is None:
            hit = finger_body_spheres(surf, overflow=overflow, grid=grid, gap=gap,
                                      max_spheres=max_spheres, solver=solver)
            _CACHE[key] = hit
        centres, radii, rep = hit
        report[link] = {**rep, "mujoco_body": surf.body}
        for c, r in zip(centres, radii):
            origin = np.eye(4)
            origin[:3, 3] = c
            out.append(UrdfCapsule(link=link, origin=origin, radius=float(r), length=0.0,
                                   role=FINGER_BODY_ROLE))
    return out, report
