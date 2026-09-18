#!/usr/bin/env bash
set -eu
cd /mnt/dev/work
export MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT=benchmark/ag3s/docs/figures/reuse-audit-20260915
mkdir -p "$OUT"
# baseline.json is immutable evidence: snapshot.py is intentionally NOT rerun here.
.venv-ag3s/bin/python -m pytest tests/ag3s tests/trajopt -q > "$OUT/pytest.log" 2>&1
.venv-ag3s/bin/python benchmark/ag3s/experiments/reuse_audit_20260915/counterexamples.py > "$OUT/counterexamples.log" 2>&1
.venv-curobo/bin/python benchmark/ag3s/experiments/reuse_audit_20260915/gpu_reprojection.py > "$OUT/gpu_reprojection.log" 2>&1
for run in 4 5; do
 for camera in head all; do
  frames=15
  suffix=""
  if [ "$camera" = all ]; then frames=22; suffix=_all; fi
  .venv-ag3s/bin/python -u -m benchmark.trajopt.experiments.esdf_rollout \
   --records "run_000$run" --attention "attention_step1_run000$run.npz" \
   --frames "$frames" --cameras "$camera" --voxel .020 --esdf-margin .05 \
   --support-surfaces field --constraint-links arms \
   --out-json "$OUT/rollout$run$suffix.json" > "$OUT/rollout$run$suffix.log" 2>&1
 done
done
.venv-ag3s/bin/python benchmark/ag3s/experiments/reuse_audit_20260915/figures.py
