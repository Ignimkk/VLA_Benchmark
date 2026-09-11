# cuRoboV2 검증 스크립트

2계층 ESDF 설계의 전제를 실제 장비에서 확인한 스크립트들. 판정과 수치는
[`../../docs/AG3S_REVIEW_LOG.md`](../../docs/AG3S_REVIEW_LOG.md) 의 "cuRoboV2 API 검증" 절에 있다.

**이 스크립트들은 openpi venv 가 아니라 별도 venv 에서 돈다.** curobo 의존성이 `numpy`/`scipy`
를 올릴 수 있고 그러면 openpi(jax) 가 깨진다.

## 환경 만들기 (재부팅 후 다시 해야 한다 — `/tmp` 는 휘발성이다)

```bash
# 1. 소스
git clone https://github.com/NVlabs/curobo.git /tmp/curobo_src     # 검증 시점 78fd485

# 2. 격리된 venv — conda 의 torch 를 빌려 쓴다 (torch 재설치를 피하려고)
/opt/conda/bin/python -m venv --system-site-packages /tmp/curobo_venv

# 3. 설치. nvcc 불필요 — setup.py 의 USE_PYBIND 기본값이 0 이라 CUDA 확장은 선택이다
cd /tmp/curobo_src && /tmp/curobo_venv/bin/pip install -e . --no-build-isolation
/tmp/curobo_venv/bin/pip install 'cuda-core[cu12]'     # JIT 커널 백엔드. 없으면 ESDF 단계에서 죽는다
```

## 입력 데이터 만들기

스크립트들은 `/tmp/rby1_frame.npz` 를 읽는다. RB-Y1 `run_0004` step 9 의 실제 depth 3 대 +
target centroid + 제약 구를 담은 것으로, **openpi venv** 에서 만든다:

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python - <<'PY'
# (AG3S_REVIEW_LOG.md 의 해당 절 참고 — run_0004 를 replay 해서
#  depth_{head,left_wrist,right_wrist}, K_*, T_*, target_centroid,
#  sphere_centers, sphere_radii 를 npz 로 저장한다)
PY
```

## 실행

```bash
/tmp/curobo_venv/bin/python check_occupancy_frame.py   # 좌표계 일치 — 점유 복셀 vs 실제 테이블 높이
/tmp/curobo_venv/bin/python verify_two_tier.py         # 2계층 정확성 — 거친 33 mm 대 미세 2 mm
/tmp/curobo_venv/bin/python bench_two_tier.py          # 정상 상태 시간 (워밍업 필수)
```

## 함정 두 가지 (둘 다 내가 밟았다)

1. **`esdf_origin` 은 격자 코너가 아니라 중심이다.** `integrator_esdf.py` 의
   `"Pose at center"` 주석이 근거이고, 실행하면 `vg.pose[:3]` 가 넘긴 값과 같게 나온다.
   코너로 넣으면 관심 물체가 창 모서리에 걸려 두 계층이 130~257 mm 어긋난다.
2. **인덱싱을 손으로 쓰지 말 것.** `VoxelGrid.create_xyzr_tensor(transform_to_origin=True)` 가
   복셀 월드 좌표를 `feature_tensor.reshape(-1)` 과 같은 순서로 준다. 직접 계산한 인덱스로
   두 번 틀렸고, 그때마다 "좌표계 불일치" 처럼 보였지만 실제로는 필드가 옳았다.

3. **첫 `compute_esdf` 호출 시간을 쓰지 말 것** — JIT 컴파일 + CUDA graph capture 때문에
   1302 ms 가 나온다. 정상 상태는 0.5~0.7 ms 다. 반드시 워밍업 후 측정한다.
