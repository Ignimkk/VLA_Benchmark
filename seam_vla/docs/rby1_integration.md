# Applying SEAM/VLS to the RB-Y1 robot (`pi05_rby1_lora`)

## What was verified (from `data/policy_records/*.npy`, PolicyRecorder dumps)
- Action chunk = **(50, 14)** ⇒ **H=50**, 14-dim `[L 6 joints, L grip, R 6 joints, R grip]`.
- **Client-facing actions are ABSOLUTE joint targets** (`action[0] ≈ state`; arm dims evolve to real
  joint positions). The model's **denoising space is base-relative delta** (`absolute = delta + state`,
  `transforms.AbsoluteActions`). ⇒ VLS in model space needs **base-state compensation**.
- Executed per chunk **K = `OPEN_LOOP_HORIZON` = 8** ⇒ overlap **L = 42**.
- **The chunk-boundary artifact is real:** after splitting the 114 records at the prompt/task reset,
  boundary jerk is **BJ=0.0226 vs interior IJ=0.0129 = 1.75×**. Absolute overlap residual (arm) mean
  is **0.1275 rad**; within-run base drift |Δs| mean **0.0796**, max **0.3193 rad** (⇒ naive
  delta-space VLS can mis-target by up to ~0.3 rad — compensation matters).

## Two SEAM surfaces (both implemented)

### A. Denoising-stage VLS + base compensation — the primary, paper-faithful method
Runs where the model + flow-matching ODE are (the **GPU server**). VLS steers the denoising toward the
previous chunk's tail; the aligned prior is compensated so it targets the previous **absolute** tail:
`prior_delta_norm[j] = prev_tail_delta_norm[K+j] − Δs·(2/(q99−q01))`, `Δs = s_curr − s_prev`.
Guides **arm joints only** (dims 0–5, 7–12); grippers (6, 13) excluded. Config: `configs/seam_rby1.yaml`.

### B. Decoded-stage overlap-steering refiner — offline-verifiable companion
A physical-space `DecodedChunkRefiner` (`refinement/overlap_steer.py`) that nudges the new chunk's first
M absolute steps toward the previous chunk's absolute tail with a decaying weight `λ(1−j/M)`. Correct in
absolute space and testable offline on the recorded chunks. This is also the reserved seam for future
trajectory-optimization / MPC.

**Offline evidence** (`experiments/rby1/offline_boundary_eval.py`, 114 recorded chunks split into two
prompt-contiguous runs, λ=0.2, M=20): BJ **−2.4%**, IJ **−10.9%**, CD **+0.1%**, AVb **−8.8%**,
overlap residual **−1.7%**. BJ/IJ rises from 1.75 to 1.91, so this is modest smoothing evidence rather
than proof that the boundary-specific artifact is solved. Note a *uniform* weight can make BJ worse —
the post-hoc-averaging failure the SEAM paper warns about — which is why the primary method steers
inside the denoising loop.

## Deploy — server-side (the real remote setup)
On the GPU server (which has `pi05_rby1_lora` + checkpoint + openpi + `benchmark/seam_vla`):
```bash
python -m benchmark.seam_vla.serving.serve_seam_policy \
    --config pi05_rby1_lora \
    --dir /mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_lora/full_run_30k/29999 \
    --seam-config benchmark/seam_vla/configs/seam_rby1.yaml \
    --port 8000
```
This wraps the trained policy with `SeamServerPolicy` (per-session VLS state, reset via the
`seam_reset` obs flag) and serves the same websocket protocol as `scripts/serve_policy.py`
(OpenPI itself is untouched). On the local PC:
```bash
python src/rby1_bringup/pi05_infer.py --model rby1 --remote <server-ip>:8000 --seam
```
`--seam` in remote mode sends `seam_reset` on the first request so the server starts a fresh episode.

## Deploy — all-local (running `pi05_infer.py` on the GPU server)
```bash
python src/rby1_bringup/pi05_infer.py --model rby1 --seam \
    --seam-config benchmark/seam_vla/configs/seam_rby1.yaml
```
`--seam` wraps the in-process policy with `SeamPolicy`; the first chunk is baseline, subsequent chunks
use VLS. `K` is taken from `OPEN_LOOP_HORIZON=8`.

## Run the offline evidence (no model / GPU needed)
```bash
python -m benchmark.seam_vla.experiments.rby1.offline_boundary_eval \
    --records data/policy_records --K 8 --M 20 --lam 0.2
```

## Assumptions to confirm on the server (could not be tested here)
The `pi05_rby1_lora` config/checkpoint are server-only, and the 2B model does not fit the local 8 GB
GPU, so RB-Y1 is **wired + unit-tested (synthetic) + offline-validated**, not run against the live model
here. On the first real run, `SeamServerPolicy` logs the first guided chunk's correction; confirm:
1. `action_horizon == 50` and `num_steps == 10` for `pi05_rby1_lora` (else the config assertion fires).
2. 14-dim action/state layout matches `[L6, Lgrip, R6, Rgrip]` (guided arm indices 0–5, 7–12).
3. The rby1 `DeltaActions` mask marks the **arm** dims as delta (compensated) and grippers as absolute.
4. Action norm-stat keys expose `q01/q99` for `inv_scale = 2/(q99−q01)`.
If any differ, adjust `configs/seam_rby1.yaml` (`seam_horizon`, `seam_guided_dimensions`,
`num_valid_physical_dims`) — the config asserts against the live model and fails loudly on mismatch.

## Tuning
- Start with `seam_lambda=0.1` (denoising VLS). The paper shows success degrades past ~0.15; keep it small.
- `seam_guided_window` up to L=42; larger M smooths more of the overlap.
- To A/B on the robot: run baseline (`configs/baseline_rby1.yaml`) vs SEAM (`configs/seam_rby1.yaml`),
  record with `--record-inputs`, and compare boundary jerk with `metrics/motion.py`.
