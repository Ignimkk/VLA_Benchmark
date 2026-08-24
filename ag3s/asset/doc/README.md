# AG3S asset

시각 자료를 이미지와 문서로 분리해 관리합니다.

```
asset/
├── image/
│   ├── ag3s_*.png            합성 fixture, 8단계
│   └── rby1_transport/       RB-Y1 씬, 카메라 3대 x 8단계 = 24장
└── doc/                      해설과 측정 결과 (사람이 쓴 것, 재생성되지 않음)
```

`image/`는 **언제든 통째로 재생성**됩니다.

```bash
src/openpi/.venv/bin/python -m benchmark.ag3s.visualization
```

인자 없이 실행하면 `asset/image/`에 8장을 덮어씁니다. 다른 곳에 뽑고 싶으면 `--out <경로>`,
다른 phase로 보고 싶으면 `--phase grasp`를 씁니다. `doc/`는 자동 생성 대상이 아니므로
파이프라인 동작이 바뀌면 손으로 갱신해야 합니다.

렌더링 대상은 `tests/ag3s/fixtures.py`의 결정론적 합성 씬입니다 — 테이블 평면 하나 위에 구
(target), 박스, 실린더가 놓이고, 2링크 mock 로봇이 프레임 안에 들어와 있습니다. 광선 추적으로
depth를 정확히 만들기 때문에 그림에 보이는 것과 GT가 정확히 일치합니다.

- **[ag3s_report.html](ag3s_report.html)** — AG3S 전체 구현 보고서 (원리 · 8단계 · TO 계약 · 결과 · 결함)
  발행본: https://claude.ai/code/artifact/7eff958d-f250-48ea-a864-175ddcb571a6
- [FIGURES.md](FIGURES.md) — 8장 각각이 무엇을 보여주고 무엇을 확인하는 그림인지
- [RBY1_TRANSPORT.md](RBY1_TRANSPORT.md) — RB-Y1 transport 씬 실험 (ZED + D435i, 카메라 3대)
- [rby1_transport_index.json](rby1_transport_index.json) — 위 실험의 기계 판독용 후보 색인

## 파이프라인 해설 · 후속 작업

- [`ag3s_pipeline.html`](ag3s_pipeline.html) — 2D attention + depth가 3D 충돌 제약이 되기까지의
  다섯 단계를 단계별 실측치와 함께 설명한다. 보조 단계(self-filter · support surface ·
  3-카메라 융합 · clearance 정책 · 제약 생성)도 함께.
  발행본: https://claude.ai/code/artifact/a2406356-a646-4cbc-851f-f78a8180cf8d
- [`future_vla.md`](future_vla.md) — RB-Y1으로 fine-tuning된 VLA가 나왔을 때 할 일 체크리스트.
  지금의 attention이 대역품(`gaussian_attention`)이라는 사실에서 출발해, 실물로 갈아끼울 때
  순서대로 확인할 것들을 정리했다.
