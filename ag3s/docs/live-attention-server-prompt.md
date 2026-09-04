# 서버 프롬프트 — 추론 응답에 attention 선택 셀 싣기

아래 내용을 GPU 서버에서 돌고 있는 claude 에게 그대로 전달한다. 로컬과 서버는 git 으로 같은
워크스페이스를 공유하므로, 로컬 변경은 pull 로 받으면 된다.

---

## 배경

로컬에서 `pi05_infer.py --trajopt` 로 **추론 + AG3S + TO** 를 한 루프에서 돌리려 한다. AG3S 가
충돌 제약을 만드는 방식은 **attention + ESDF** 다: attention 이 target 을 지목하고, 거리장이
그 target 을 파낸 뒤 나머지 기하로 제약을 만든다. attention 이 없으면 target 을 못 잡고
(`no_target`), 거리장이 target 을 파내지 않아 제약이 필요 이상으로 보수적이 된다.

지금은 attention 을 롤아웃이 **끝난 뒤** `benchmark.ag3s.experiments.pi05_attention` 로 따로
뽑는다. 그래서 live 루프에서는 쓸 수가 없다.

## 요청

**추론 응답에 attention 맵을 함께 실어 보내 달라.** 전체 텐서가 아니라 **선택 셀 하나**만.

1단계 채점(`benchmark/ag3s/docs/step-01-attention.json` 의 `best`)에서 이미 확정한
(층, 헤드, aggregation, denoise step) 조합이 있다. 카메라마다 그 셀 하나면 되고, 그러면
카메라당 맵 한 장(정책 이미지 격자 크기)이라 **수십 KB** 다. 전체 텐서는 프레임당 184 MB 라
전송이 불가능하지만, 이건 부담이 거의 없다.

## 알려진 함정 두 가지

**(1) `return_attn_probs=True` 를 켜면 서빙이 깨진다.**
`gemma.Module.__call__` 이 3-tuple 을 돌려주는데 `Pi0.sample_actions` 는 `pi0.py:245` 에서
2개만 언팩한다. 여기를 고쳐야 한다. `return_attn_probs` 를 **다시 구현하지 말 것** — 이미
있는 기능이다. 최신 코드로 sync/checkout 하는 것은 괜찮다.

**(2) 서버를 재시작해도 된다.**
현재 돌고 있는 추론 세션은 종료해도 상관없다고 사용자가 확인했다. 수정 후 재기동하면 된다.
JAX 가 기본으로 GPU 메모리 75% 를 미리 잡으므로, 두 프로세스를 동시에 띄우려면
`XLA_PYTHON_CLIENT_PREALLOCATE=false` 가 필요하다. 그리고 이 워크스페이스에서는
`XLA_FLAGS='--xla_gpu_enable_command_buffer='` 없이 돌리면 CUDA graph replay 오류가 난다
(`src/docs/RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md:773` 에 이미 기록돼 있다).

## 응답 형식

로컬 클라이언트는 이렇게 읽는다 (`src/rby1_bringup/pi05_infer.py`):

```python
result = policy.infer(obs)
chunk = np.asarray(result["actions"])
policy_attention.update(result.get("attention", {}) or {})
```

즉 `result["attention"]` 이 **딕셔너리**이고, 키는 로컬이 쓰는 MuJoCo 카메라 이름이어야 한다:

| 정책 입력 이름 | 로컬 카메라 키 |
|---|---|
| `cam_high` | `zed_left` |
| `cam_left_wrist` | `wrist_cam_l` |
| `cam_right_wrist` | `wrist_cam_r` |

값은 그 카메라의 정책 이미지 격자에 대한 2D attention 맵(float32 또는 float16)이다. AG3S 가
정규화·역투영을 알아서 하므로 **정규화하지 말고 원본 값 그대로** 보내면 된다 — 1단계에서
정규화본과 원본이 서로 다른 것을 요구한다는 것을 이미 확인했다(임계값은 정규화본, 피크는 원본).

세 카메라 중 일부만 보내도 된다. 없는 카메라는 로컬에서 attention 없이 기하만 기여한다.

## 확인해 줄 것

1. `result["attention"]` 이 실제로 실려 오고 크기가 수십 KB 인가 (전체 텐서가 아닌가).
2. attention 을 켠 상태에서 `result["actions"]` 이 **끄고 돌렸을 때와 같은가**. 달라지면
   샘플링 경로를 건드린 것이고, 그러면 지금까지의 모든 롤아웃과 비교가 불가능해진다.
3. 추론 지연이 얼마나 늘었는가 (켜기 전/후 ms).
4. 어느 (층, 헤드, agg, denoise) 셀을 골랐는지, 그리고 그것이
   `benchmark/ag3s/docs/step-01-attention.json` 의 `best` 와 일치하는지.

## 하지 말 것

- 전체 attention 텐서를 응답에 싣지 말 것 (프레임당 184 MB).
- `return_attn_probs` 를 재구현하지 말 것.
- 로컬 쪽 파일(`src/rby1_bringup/pi05_infer.py`, `benchmark/trajopt/*`, `benchmark/ag3s/*`)을
  고치지 말 것 — 그쪽은 이미 준비돼 있고, 서버는 응답 형식만 맞추면 된다.
