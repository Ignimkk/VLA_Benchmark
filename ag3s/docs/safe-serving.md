# 서버 통합 — π0.5 + SEAM + AG3S + TO 한 프로세스

**상태: 배선 완료, 서버 단독 replay 검증 완료. attention 서버 수정과 end-to-end 대기.**
`--safe-remote` 없이는 로컬 루프가 예전과 동일하다.

## 왜 서버인가

로컬 배선(`trajopt/bringup.py`)을 실측하니 depth 렌더 167 ms + AG3S 393~628 ms 로 청크 예산
533 ms 를 1.4~2.7배 넘었다. 대부분이 CPU 소프트웨어 래스터라이저와 지각이고 둘 다 GPU 서버에서
싸다. 더 결정적인 것은 attention 이다 — 정책 내부에서 나오는 값이라 밖으로 꺼내려면 매 프레임
전송해야 하지만, 같은 프로세스 안이면 그냥 변수다.

## 구조

```
로컬 MuJoCo                                 GPU 서버
─────────────                              ──────────────────────────────────
3카메라 캡처 ──── ag3s/* 요청 ───────────▶  strip_request
  depth uint16 mm                            │
  K, T_base_cam                              ├─ π0.5 (+SEAM) → chunk [H,14]
  촬영시점 robot_state (카메라마다)          ├─ attention (선택 셀)
  phase, manipulators, reset, seq            ├─ CameraObservation × 3
                                             ├─ AG3S.process_multi → 제약
                                             ├─ scene_from_constraint_set
                                             └─ TrajOptChunkRefiner
안전 게이트 ◀──── actions + 판정 ──────────  pack_response
  safe → 앞 8개 실행
  아니면 현재 관절 hold
```

`SafePolicy` 는 `BasePolicy` 데코레이터다 — `PolicyRecorder` 가 이미 만들어 둔 자리를 쓰므로
**openpi 를 한 줄도 고치지 않는다.**

## 실행

```bash
# 서버
python -m benchmark.trajopt.serve_safe \
    --config pi05_rby1_atomic_lora \
    --checkpoint /mnt/dev/work/.../29999 \
    --model-xml <RB-Y1 씬 XML> --port 8000

# 터널 (컨테이너는 점프 호스트 내부망이라 직접 닿지 않는다)
ssh -N -L 8123:localhost:8123 -J blunex@ai.amrc.kr:21151 root@172.21.121.112 -p 32542

# 로컬 루프. `MUJOCO_GL=osmesa` 를 붙이지 않는다 — 아래 참고
cd /home/mk/dev_ws/vla/pi0_TO_ws
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \
    --model rby1_transport_14d --remote localhost:8123 \
    --prompt "put the apple in the basket" \
    --fruit-layout-index 0 --fruit-slot-order apple banana orange pear \
    --obstacle-profile clear --max-steps 350 --start-delay 2 --speed 1.0 --view front \
    --safe-remote --safe-timeout 2.0 \
    --trace outputs/live/trace \
    --record outputs/live/third_person.mp4
```

**렌더 백엔드는 뷰어를 띄우느냐로 갈린다.** `--headless` 가 없으면 이 루프는
`mujoco.viewer.launch_passive` 로 대화형 뷰어를 띄우는데, `MUJOCO_GL=osmesa` 는 화면 없는
소프트웨어 렌더러라 기본 프레임버퍼가 없다. 붙이면 시작하자마자 죽는다:

```
ERROR: Default framebuffer is not complete, error 0x0
```

뷰어 없이 돌리려면 osmesa 와 `--headless` 를 함께 준다. `--record` 가 있으므로 영상은 그대로
나온다. 오프라인 분석 스크립트(`safe_replay`, `*_report`)는 뷰어를 띄우지 않으므로 osmesa 가
맞고, **거기서 명령을 복사해 오면 이 오류를 만난다.**

또한 `--model` 은 `rby1_transport_14d` 여야 한다. 서버의 `--model-xml` 이 그 모델의 씬
(`model_transport.xml`)이고, 둘이 어긋나면 서버가 만든 로봇 모델이 로컬이 움직이는 로봇과
달라진다.

`--no-safe`(서버) 로 감싸기를 끄면 기존 서빙과 같다. 문제가 안전 계층에 있는지 아닌지를 플래그
하나로 가를 수 있다.

## 안전 계약

서버는 **판정만** 한다. 멈출지 실행할지는 로컬이 정한다 — AG3S 가 기하 불확실성을 보고만 하고
결정하지 않는 것과 같은 계약이고, 이유도 같다: 무엇이 허용 가능한 위험인지는 지각도 최적화도
알 수 없다.

응답: `actions`, `safe`, `ag3s_status`, `geometry_certified`, `trajopt_status`,
`max_violation_m`, `timing_ms`, `notes`, `seq`.

로컬은 **확인된 것이 없으면 실행하지 않는다.** `last_safe` 가 기본 False 이고, 네 갈래가 전부
같은 결과로 수렴한다:

| 갈래 | 왜 hold 인가 |
|---|---|
| `unsafe` | 서버가 인증하지 못했다 |
| `timeout` | 응답은 왔지만 늦었다. 그 청크는 **지나간 자세**를 위한 계획이고, 절대 관절 목표라 관절이 튄다 |
| `stale` | `seq` 가 안 맞는다. 지난 청크의 응답이다 |
| `error` | 예외를 밖으로 내면 제어 루프가 죽고, 팔은 마지막 `ctrl` 에 매달린 채 아무도 hold 를 걸어주지 않는다 |

**hold 는 정지가 아니라 유지다.** 현재 관절을 그대로 목표로 준다 — 제어를 끊으면 팔이 중력으로
떨어지고, 그것은 안전 판정이 막으려던 것보다 나쁘다.

에피소드 리셋(`ag3s/reset`)은 정책(SEAM 포함)·AG3S·TO 세 층의 내부 상태와 warm-start 를 모두
버린다. 하나라도 남으면 다음 에피소드 첫 청크가 지난 에피소드의 씬을 warm-start 로 받고, 물체가
옮겨졌다면 **사라진 장애물을 피하려 애쓰는 궤적**이 나온다.

## 서버 단독 replay 결과

```bash
MUJOCO_GL=osmesa python -m benchmark.trajopt.experiments.safe_replay \
    --records outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0002 --frames 6
```

| 시나리오 | 결과 |
|---|---|
| 정상 (기본) | 6/6 **hold** — 전부 `geometry_certified=False` |
| `--allow-uncertified` | 6/6 실행, `feasible`, 위반 0.0 mm |
| `--drop-camera zed_left` | 3/3 hold, `no_target` — 머리 카메라가 없으면 attention 이 갈 곳이 없다 |
| 그리퍼·형태 불변 | **전 프레임 유지** |

지연: 전체 중앙 **610 ms**, 최대 650 ms (osmesa 로컬 기준. GPU 서버에서는 더 낮을 것이고, 그
확인이 end-to-end 단계의 목적 중 하나다).

### 기본 설정에서는 아무것도 실행되지 않는다

이것이 이 검증의 가장 중요한 발견이다. 미관측 비율이 72% 인 이 씬에서 AG3S 는 `degraded` 를
내고, `require_certified_geometry: true` 가 그것을 `VIOLATED` 로 내린다. **위반이 0.0 mm 라도
hold 다.** 계약대로 동작한 것이지 버그가 아니다 — "제약을 다 지켰다" 는 모델에 대한 진술이지
세계에 대한 진술이 아니라는 뜻이다.

실제로 로봇을 움직이려면 셋 중 하나가 필요하다: 카메라 배치를 늘려 미관측을 줄이거나,
`unknown_report_threshold` 를 이 씬에 맞게 다시 잡거나, `--allow-uncertified` 로 위험을 명시적으로
인수하는 것. **지금은 아무것도 고르지 않았다.** 이건 측정이 아니라 정책 결정이다.

## 배선하다 고친 것

**같은 뜻의 스위치가 둘이었다.** `SafePolicy.require_certified` 와
`TrajOptConfig.safety.require_certified_geometry`. SafePolicy 쪽만 끄면 sqp 가 여전히 상태를
`VIOLATED` 로 내려서, 위반 0 mm 인데도 모든 프레임이 hold 로 갔다. 스위치를 하나로 합쳤고
(`to_config.safety` 가 유일한 근거), 테스트가 두 번째 스위치가 생기는 것을 막는다.

**`_depth_cameras_from` 이 `resolve_T_base_cam()` 을 인자 없이 불렀다.** 손목 카메라는
`mount_link` + `T_link_cam` 으로 오고 FK 없이는 안 풀리므로 ESDF 경로가 예외로 죽었다.
`esdf_rollout` 은 재생 프레임이 `T_base_cam` 을 직접 들고 있어 드러나지 않았다 — **ESDF backend
가 mount_link 관측으로 돌아간 적이 없었다.**

**`TransportScene` 은 항상 자기 XML 을 로드해 자기 `MjData` 를 만든다.** live 에 그대로 쓰면
로봇이 움직이는 씬과 AG3S 가 보는 씬이 갈라진다. `TransportScene.attach(model, data)` 를 추가했다.

## 남은 것

1. **attention 이 아직 live 로 오지 않는다.** 서버 서빙이 1단계에서 확정한 셀 하나를 응답에
   실어야 한다 — [live-attention-server-prompt.md](live-attention-server-prompt.md).
   그때까지 `no_target` 으로 돌고, 거리장이 target 을 파내지 않아 제약이 더 보수적이다.
2. **GPU 서버에서 회귀 테스트** (`pytest tests/ag3s tests/trajopt`, 현재 로컬 560 passed).
3. **end-to-end** — 로컬 MuJoCo 연결, 장애물 회피 시나리오, 실제 지연 측정.
