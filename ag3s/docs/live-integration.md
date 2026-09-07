# 추론 + AG3S + TO 통합 — 배선과 기록

**상태: 로컬 배선 완료, attention 서버 수정 대기.** `--trajopt` 없이는 루프가 예전과 동일하다.

지금까지의 모든 수치는 기록된 롤아웃을 **재생**해서 얻었다. 재생은 값이 실측이어도 지연을
증명하지 못한다 — 청크 예산 안에 들어가는지는 실제로 돌려봐야 안다. 이 문서는 그 실제 루프를
어떻게 돌리고 무엇이 기록되는지를 적는다.

## 실행

```bash
MUJOCO_GL=osmesa XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python -m src.rby1_bringup.pi05_infer \
  --model rby1 --remote <서버:포트> --prompt "..." \
  --trajopt --trajopt-links arms \
  --trace outputs/live/trace \
  --record-constraints outputs/live/constraints
```

| 플래그 | 뜻 |
|---|---|
| `--trajopt` | AG3S+TO 를 켜고 **TO 청크를 실제로 실행**한다 (closed loop) |
| `--trajopt-links {arms,all}` | 제약을 걸 링크. `arms` 기본 |
| `--trajopt-cameras` | AG3S 가 쓸 카메라. 기본 `zed_left wrist_cam_l wrist_cam_r` |
| `--trace DIR` | 벽시계 타임스탬프 |
| `--record-constraints DIR` | 제약 생성 중간 산출물 |
| `--record-constraints-esdf {none,occupancy,full}` | 거리장 저장 범위 |

**`MUJOCO_GL=osmesa` 는 `--headless` 와 함께일 때만 쓴다.** 뷰어를 띄우는 실행에 붙이면
`ERROR: Default framebuffer is not complete, error 0x0` 로 시작하자마자 죽는다 — osmesa 는
화면 없는 소프트웨어 렌더러라 기본 프레임버퍼가 없다. 오프라인 분석 스크립트에서 명령을
복사해 올 때 자주 만나는 오류다.

`zed_right` 를 넣지 않는 이유는 `CameraID` 에 `HEAD` 가 하나뿐이라 `zed_left` 와 겹치기
때문이다. 세 대는 머리 하나 + 손목 둘이다.

## 무엇이 기록되나

### 타임스탬프 (`trace.jsonl`, 한 줄 = 한 스팬)

| 스팬 | 뜻 |
|---|---|
| `depth_capture` | 세 카메라 depth 렌더 |
| `ag3s` | 지각 + 거리장 + 제약 생성 |
| `chunk_total` | 청크 하나의 **전체**. 위 둘을 **포함한다** |

`chunk_total` 을 `to` 라 부르지 않는 이유는 그 안에 지각이 중첩돼 있어서다 — 그렇게 두면 QP 가
800 ms 걸린 것처럼 읽힌다. 순수 최적화 시간은 `chunk_total − depth_capture − ag3s` 이고
`solve_ms` 로도 따로 실린다. 단조시계를 쓰고 `meta.json` 의 `t0_epoch` 이 절대시각을 고정한다.
줄마다 flush 하므로 초과 지연으로 루프를 죽여도 그때까지가 남는다.

### 제약 생성 (`chunk_XXXXX.npz`, 청크당 약 1.2 MB)

| 무엇 | 키 |
|---|---|
| attention | `attention_head`, `attention_left_wrist`, … (카메라별 원본 맵) |
| 점군 | `raw_cloud`, `filtered_cloud`, `attention_cloud` |
| attention 값 | `attention_values`(정규화), `attention_raw`(원본) — **둘 다** |
| target | `target_points`, `target_point_indices`, `target_centroid` |
| 지지면 | `support_mask`, `seed_indices` |
| 거리장 | `esdf_origin`, `esdf_shape`, `esdf_voxel_size`, `esdf_occupancy`, `esdf_distance` |
| 제약 | `sphere_centres`, `sphere_radii`, `clearance` (H×S) |
| 청크 | `reference_chunk`(정책 원본), `refined_chunk`(TO 결과) |
| 요약 | `summary_json` — status, grounding, esdf_stats, TO 상태·반복·**노트** |

두 가지가 의도적으로 중복이다.

* **attention 정규화본과 원본을 둘 다.** 임계값은 정규화본을 보고 피크는 원본을 본다 — 그것이
  `AttentionPointCloud` 가 두 배열을 들고 있는 이유이고, 하나만 저장하면 "왜 저 점이 시드가
  됐나" 를 되짚을 수 없다.
* **정책 원본 청크와 TO 청크를 둘 다.** closed loop 에서는 TO 가 궤적을 바꾸므로 "원본이었다면"
  을 같은 실행에서 볼 수 없다. 원본을 남기는 것이 그 대조군을 부분적으로나마 복원하는 유일한
  방법이다.

여유거리는 **계획 지평 전체** (H, S) 로 남긴다. 첫 스텝만 재면 TO 의 판정(지평 전체를 본다)과
기록이 어긋나 "TO 는 violated 인데 기록은 위반 0" 이 나온다.

## 상태 `violated` 의 두 갈래

이걸 모르면 결과를 잘못 읽는다. `TrajOptStatus.VIOLATED` 는 두 가지에서 나온다.

1. **관통이 남았다** — 실제로 제약을 못 지켰다.
2. **기하를 인증하지 못했다** — 제약은 전부 통과했지만 AG3S 가 `degraded` 를 냈다. 미관측
   비율이 높으면 그렇게 된다(측정된 이 씬의 미관측 72%). "제약을 다 지켰다" 는 모델에 대한
   진술이지 세계에 대한 진술이 아니라는 것이 그 판정의 뜻이다.

스모크 테스트에서 실제로 2번이 나왔다: 여유거리가 전 지평 양수(+4.7 / +3.0 / +1.6 mm)인데
상태는 `violated`. 그래서 기록에 **TO 노트를 남긴다** — 상태값만으로는 두 갈래가 구분되지 않는다.

## 측정된 지연 — 예산 초과다

| | 값 |
|---|---|
| 청크 예산 (`OPEN_LOOP_HORIZON 8 / CTRL_HZ 15`) | **533 ms** |
| depth 렌더 3카메라 (osmesa CPU) | 149~206 ms |
| AG3S | 522~628 ms |
| 청크 전체 | **765~1422 ms** |

**예산의 1.4~2.7배다.** 계획 단계에서 이미 알고 있던 초과이고, 숨기지 않고 기록하는 것이 이
실험의 목적 중 하나다. 병목 후보는 depth 렌더(osmesa 는 CPU 소프트웨어 래스터라이저다)와
AG3S 의 target grounding 이다. 실제 trace 로 확인한 뒤에 고친다.

## 아직 안 된 것

**attention 이 live 로 오지 않는다.** 지금은 롤아웃이 끝난 뒤 `pi05_attention` 으로 따로 뽑는다.
서버 서빙이 1단계에서 확정한 셀 하나를 응답에 실어 보내도록 고쳐야 한다 —
[live-attention-server-prompt.md](live-attention-server-prompt.md) 에 그 프롬프트가 있다.

그때까지 `--trajopt` 는 attention 없이 돈다. AG3S 는 target 을 못 잡고(`no_target`), 거리장이
target 복셀을 파내지 않아 제약이 **더 보수적**이 된다 — 안전한 방향의 실패다. 합성 attention 을
몰래 끼워넣지 않는 이유는, 그러면 이 실행이 실측 파이프라인인 척하게 되기 때문이다. 로그에
경고가 한 번 찍힌다.

## 배선에서 고친 실제 버그 두 개

**`_depth_cameras_from` 이 `resolve_T_base_cam()` 을 인자 없이 불렀다.** 손목 카메라는
`T_base_cam` 대신 `mount_link` + `T_link_cam` 으로 오고 그 조합은 FK 없이 풀리지 않으므로,
ESDF 경로 전체가 예외로 죽었다. `esdf_rollout` 은 재생 프레임이 `T_base_cam` 을 직접 들고 있어
조기 반환돼서 드러나지 않았다 — **ESDF backend 가 mount_link 관측으로 돌아간 적이 없었다.**

**`TransportScene` 은 항상 자기 XML 을 로드해 자기 `MjData` 를 만든다.** live 루프에 그대로
쓰면 로봇이 움직이는 씬과 AG3S 가 보는 씬이 갈라진다. `TransportScene.attach(model, data)` 를
추가해 같은 객체를 가리키게 했다.
