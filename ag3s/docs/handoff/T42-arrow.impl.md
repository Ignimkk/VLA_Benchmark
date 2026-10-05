# T42-arrow — 구현 (회피 화살표 overlay, 사용자 추가 지시 2)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-05 KST · not committed (lead commits)

## 무엇을 했나 (평이한 요약 먼저)

Client recordings (`--record` third-person mp4) and the live viewer can now draw an **avoidance arrow** each control
step. The arrow starts at the active gripper's TCP and shows how far, and in which direction, the TO pushed the
gripper away from what the policy wanted:
**v = TCP(refined row) − TCP(policy reference row)** for the action applied at that step. Both TCPs come from
forward kinematics of the commanded joint targets on the same model. The overlay is **visual only**: physics,
actions, `frames.jsonl`, `traj.npz`, server requests and the RNG are bit-identical with the flag on or off,
which the tests check end to end. A replay tool re-renders recorded runs with the arrow, with no policy server.
The demo clips use T41a E3b C1968 and D1995.

## Vector definition and drawing (scale label)

| item | definition |
|---|---|
| TCP | site `left_ee` / `right_ee` (100 mm below `EE_BODY_*`, between the fingers; the IK EE site) |
| v (execute) | `TCP(q_refined) − TCP(q_reference)`. `q_*` = the 7 arm joint targets of row `chunk_step` of `actions` / `actions_reference` (16-D: `[0:7]`, `[8:15]`; 14-D: `[0:6]`/`[7:13]` + held `arm_6`, as in `apply_action`) |
| v (HOLD) | `TCP(q_hold) − TCP(q_reference)`; `q_hold` = the arm part of the applied HOLD action |
| FK | private scratch `MjData`: `qpos` = current `d.qpos` (torso, base, free bodies), arm joints overwritten, `mj_kinematics`. Both TCPs share the same base pose |
| anchor | `d.site_xpos` of the TCP, read only. This is the TCP as rendered in that frame (after the step's `mj_step` calls) |
| active arm | the episode's `used_arm` (16-D); else a single `--safe-manipulators` arm; else both, and each step shows the arm with the larger \|v\| |
| threshold | arrow only when \|v\| > `--viz-arrow-threshold-mm` (default **5 mm**) |
| **scale** | drawn length = `--viz-arrow-scale` × \|v\| (default **8×**: 10 mm → 80 mm), capped at **0.40 m** (caption says "clipped"). Shaft width 12 mm |
| caption (mp4 only) | top-left: state, \|v\| in mm, `arrow length = 8 x |v|` |

| gate state of the step | drawn |
|---|---|
| execute (`should_execute`) | **green** arrow |
| HOLD, reference row present (unsafe verdict) | **red** arrow `q_hold − reference`; if \|v\| ≤ threshold, a red sphere marker (r = 15 mm) at the TCP |
| HOLD, no reference (comms: timeout / error / no response) | red marker only |
| E0 (`--remote` without `--safe-remote`), in-process `--trajopt` | nothing (no refined/reference pair; the `[viz]` line says so) |
| optional `--viz-obstacle-normal` (off by default) | **blue** arrow (80 mm) along the obstacle normal at the gripper's closest point, within 10 cm. `mj_geomDistance` between the active obstacle collision geoms and the gripper collision geoms (`EE_BODY_*`, fingers), run on a second scratch `MjData` (qpos + mocap copied). Witness pairs inconsistent by > 0.1 mm (the T40 H false zero) and contacts are skipped. This is MuJoCo ground truth, not what the TO saw |

Shadow mode: the arrow is still refined − reference (what the TO would have done), in green, because the step executes.
Gate off: green on every arriving chunk (the step executes refined); `would_hold` is not coloured separately.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `pi05_TO_hybrid/rby1_bringup/avoidance_arrow.py` (new, 397 lines) | `arm_targets` (row → per-arm joint targets, same mapping as `apply_action`) · `TcpKinematics` (scratch-`MjData` FK) · `AvoidanceArrows.step` (vector, threshold, colour, marker, optional normal → `StepOverlay`) · `draw` (`mjGEOM_ARROW` / sphere decor geoms via `mjv_initGeom` + `mjv_connector`, respects `maxgeom`) · `draw_user_scene` (viewer `user_scn`, under `lock()`) · `caption` (PIL text on a copy) | one helper shared by the live path and the replay tool. It only reads `d` |
| `pi05_TO_hybrid/rby1_bringup/pi05_infer.py:887-902` | flags `--viz-avoidance-arrow`, `--viz-arrow-threshold-mm` (5), `--viz-arrow-scale` (8), `--viz-obstacle-normal` | default off |
| `pi05_infer.py:1031-1042` | argument checks: `--headless` without `--record` is refused (nothing to draw on); RB-Y1 action format only; threshold ≥ 0, scale > 0; `--viz-obstacle-normal` needs the arrow flag | same convention as the other "only does something together with" checks |
| `pi05_infer.py:1246-1271` | builds `arrow_viz` only when the flag is set (lazy `import avoidance_arrow` from the script's own dir). With the flag off nothing is imported or built | flag off = old code path |
| `pi05_infer.py:1998-2013` | after the step's `mj_step` loop and before `ctx.sync()`: `arrow_viz.step(d, executed=should_execute, refined_row=last_actions_refined[chunk_step], reference_row=last_actions_reference[chunk_step], applied_row=action)`; viewer `user_scn` update | reads only `safe_client.last_*` and `d` (client.py:356-357, 473; HOLD keeps the response's chunks, client.py:672-674) |
| `pi05_infer.py:2026-2036` | recording: `update_scene` → `draw(renderer_rec.scene, items)` → `render` → caption | pixels only |
| `pi05_TO_hybrid/rby1_bringup/replay_avoidance_arrow.py` (new, 389 lines) | replay renderer: run dir (`manifest.json` + `frames.jsonl`) → mp4 + stills + `vectors.json` + `deflection.png` + `replay_check.json`. Frame t = `control[t+1].qpos` (state after action t, same as the live frame t). Arrow t from planning `chunk_seq` row `step_in_chunk`; HOLD uses `applied_ctrl.arm` (= q_hold). Options: `--inset` (close-up 0.55 m from the TCP), `--rgba-from XML` (copy visual `geom_rgba` by name from the run's own asset tree), `--compare-original` (pixel diff vs the run's mp4 + mp4 compression floor), `--obstacle-normal`, `--start/--stop/--stills` | the demo, without a policy server |
| `tests/sim/test_t42_avoidance_arrow.py` (new, 15 tests) | see below | |

**Not mine, in the same file:** `pi05_infer.py` also has two uncommitted hunks from a parallel session, T41 c
`--safe-hold-mode measured` (diff `@@ -757` and `@@ -1866`). They are not part of this step. Commit them
separately, or know they ride along.
`benchmark/trajopt/client.py` is **not** touched.

## 단위 검증

```bash
cd /mnt/dev/work && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
  PYTHONPATH=/mnt/dev/work .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/sim/test_t42_avoidance_arrow.py
# 15 passed in 64 s
... -m pytest -q -p no:cacheprovider tests/sim tests/trajopt/test_gate_off.py tests/trajopt/test_t39_policy_seed.py \
  tests/trajopt/test_capture_stamps.py tests/ag3s/test_record_cost_guard.py tests/ag3s/test_head_camera_pose.py \
  tests/ag3s/test_finger_joints.py
# 228 passed in 7 m 53 s (before the viewer test was added; the file alone is now 15)
```

| test | what it pins |
|---|---|
| `test_row_layouts_match_apply_action` | 16-D / 14-D (+ arm_6 hold) row → joint targets; bad formats raise |
| `test_vector_equals_independent_fk_and_jacobian[left/right]` | synthetic refined = reference + Δq on one arm: v **bit-identical** to an independent `mj_forward` on a fresh `MjData`; the other arm's vector is exactly 0; the moved arm is chosen; v ≈ `J_site Δq` (rtol 1e-3) at Δq·1e-5 |
| `test_vector_is_taken_at_the_current_base_pose` | torso moved: v follows the current base pose; the anchor = `d.site_xpos` |
| `test_threshold_gates_the_arrow_and_scale_sets_its_length` | 4.9 mm: no arrow; 5.1 mm: one green arrow, `end − start = 8·v`; threshold configurable; the 0.40 m cap keeps the direction; invalid threshold or scale raise |
| `test_hold_is_red_q_hold_minus_reference_or_a_marker` | HOLD: red arrow = `TCP(q_hold) − TCP(ref)` (refined row ignored); `applied_arm` form gives the same result; q_hold = ref gives a red marker at the TCP; no reference gives a marker only and caption "no reference" |
| `test_no_pair_draws_nothing` | execute without refined or reference: state `none`, no items, no caption |
| `test_draw_adds_arrow_and_marker_geoms_with_the_right_pose` | `mjGEOM_ARROW`: pos = start, size[2] = length, mat z = direction, rgba; sphere marker; `maxgeom` respected |
| `test_viewer_user_scene_is_replaced_each_step` | `draw_user_scene` on a handle with `lock()` + `user_scn` replaces the overlay each call |
| `test_only_pixels_change_and_only_where_drawn` | two plain renders are equal; the overlay changes > 30 px and < 5 % of the image; `MjData` unchanged |
| `test_step_reads_data_only_and_uses_no_rng` | 20 `MjData` arrays + time unchanged after execute / HOLD / comms steps; numpy and `random` RNG states unchanged |
| `test_obstacle_normal_matches_the_manager_distance` | T40 F ep1968 hurdle: normal distance = `PickPlaceObstacleManager.pair_distance` minimum (checked GJK path) to 1e-6; unit normal; obstacle witness = `point − dist·normal`; off by default; out of range gives None |
| `test_the_scenario_exercises_execute_hold_and_comms` | `main()` + fake safe server, 32 steps: chunk 1 execute, 2 unsafe HOLD with a reference, 3 comms HOLD (raises), 4 execute → `executed` = 8 T, 16 F, 8 T |
| `test_flag_off_is_bit_identical_to_head` | **pre-arrow client** (`pi05_infer.py` of the newest commit without the flag) vs new code with the flag off: `frames.jsonl`, manifest, `traj.npz`, the 4 server requests, numpy and `random` RNG and all 32 mp4 frames are equal, except on paths where two flag-off runs of the new code also differ |
| `test_flag_on_changes_pixels_only` | flag on + `--viz-obstacle-normal` vs flag off, same comparison: `executed_actions`, `measured_qpos`, `predicted_chunks`, `obstacle_safety_json` equal; requests equal (4 each); argv differs only by the `--viz*` flags; every one of the 32 frames differs (the caption is on every safe-remote step), and the arrow changes pixels below the caption band on the 8 execute frames |

Two flag-off runs differ only in wall clock, and the test pins that: camera and render stamps, `now`,
`field.applied_at`, `timing_ms.policy_infer`, manifest `recorded_at_*`, npz `inference_ms`, request
`ag3s/stamp/*` and `ag3s/render_stamp/*`. No physical path (qpos, applied_ctrl, actions, object_poses,
executed, gate, verdict_reasons, q_hold, gripper) is ever in that set.

Pod: memory.current − inactive_file was 38–45 GiB at the start of each run. Tests ran in one process;
the demo used 3 replay processes with `LP_NUM_THREADS=2`.

## Demo (no policy server) — `/mnt/dev/work/outputs/verify/T42/arrow/`

```bash
RGBA=/mnt/dev/work-t41a/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-openpi-live/bin/python pi05_TO_hybrid/rby1_bringup/replay_avoidance_arrow.py \
  outputs/verify/T41/a/runs/E3b/C/E3b_ep1968_s19681 --out outputs/verify/T42/arrow/C1968 --name T42_arrow_C1968 \
  --inset --compare-original --rgba-from $RGBA
# D1995: .../E3b/D/E3b_ep1995_s19952 --out .../D1995 ; window with the normal: --obstacle-normal --start 180 --stop 260 --out .../D1995_normal
```

| clip | frames | execute / HOLD | arrows drawn | \|v\| max / median | replay vs original mp4 (mean abs px) | mp4 compression floor |
|---|---|---|---|---|---|---|
| `C1968/T42_arrow_C1968.mp4` | 600 | 600 / 0 | 131 | 90.8 / 2.3 mm | 1.81 | 1.87 |
| `D1995/T42_arrow_D1995.mp4` | 415 | 367 / 48 (all with a reference → red arrows/markers) | 222 | 55.5 / 5.7 mm | 1.84 | 1.84 |
| `D1995_normal/T42_arrow_D1995_normal.mp4` (t 180–259, + blue normal) | 80 | 80 / 0 | 59 | 55.5 / 14.2 mm | 1.89 | 1.92 |

- The replay reproduces the original frames to within mp4 compression. Frame rule check on D1995 t 190–219:
  `qpos[t+1]` gives 1.917, `qpos[t]` 2.411 and `qpos[t+2]` 2.425, with a floor of 1.915.
- `--rgba-from` restores the T41a hurdle colours, because T42-2 recoloured obstacle visual geoms black.
  Without it the hurdle renders black and the diff vs the original rises.
- Stills: `C1968/stills/*_t0146, t0178 (44 mm, green), t0214, t0535`; `D1995/stills/*_t0142, t0207, t0247, t0303,
  t0335 (HOLD, red 27 mm)`; `D1995_normal/stills/*_t0183 (blue normal 19 mm), t0207, t0247`.
- `*/deflection.png`: \|v\| per step, coloured execute or HOLD, with the 5 mm threshold. `*/vectors.json`: per-step
  vector, magnitude, arm, state, gate action. `*/replay_check.json`: summary and fidelity.
- Normal in the D1995 window: present on 14 of 80 steps (t 180–193, 18–93 mm). Elsewhere the gripper is more than 10 cm from the hurdle.

## verifier 가 알아야 할 것

- New flags (all default off / unchanged behaviour): `--viz-avoidance-arrow`, `--viz-arrow-threshold-mm 5`,
  `--viz-arrow-scale 8`, `--viz-obstacle-normal`. With the flag on, `manifest.extra.argv` carries it. That
  is the only record difference, by construction.
- Not covered by a run: the live passive viewer (`ctx.user_scn`). It is unit-tested with a stand-in handle
  (`lock()` + `user_scn`), and a headless pod has no display. The live `--record` path is covered by `main()`
  with a fake server; no real GPU-server run was made with the flag.
- The live mp4 frame t shows the state after action t with the arrow of action t. The replay uses the same
  rule.
- Artifacts that need to be produced again: none. Old records are compatible: the replay reads existing
  T41a runs as they are.
- For T42-2 snapshot runs, add `--viz-avoidance-arrow` to the client command (with `--record`). Nothing is sent
  to the server.

## 내가 기대하는 결과

(verifier: read after measuring.) A real `--safe-remote` run with the flag on should give the same
`frames.jsonl` / `traj.npz` as the same (episode, seed) run with the flag off, except the wall-clock fields.
Its mp4 should show green arrows mainly around the hurdle crossing, at about \|v\| 10–50 mm.
