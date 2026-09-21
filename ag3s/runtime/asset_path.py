"""자산(URDF · MJCF)이 이 머신 어디에 있는지 찾는다.

## 왜 필요한가

두 종류의 경로가 들어온다.

1. **작업공간 상대경로** — `src/rby1_description/models/rby1a/urdf/model.urdf` 같은 기본값
   (`robot_models.RBY1_URDF`, `experiments.mujoco_source.TRANSPORT_MODEL`).
2. **기록에 박힌 절대경로** — `run_0004/meta.json` 의 `model_xml` 은 롤아웃을 **녹화한 PC** 의
   절대경로다: `/home/mk/dev_ws/vla/pi0_TO_ws/src/rby1_description/...`.

둘 다 `src/` 를 거치는데, 이 컨테이너에서 자산의 실제 자리는 `pi05_TO_hybrid/rby1_description/`
이다. 예전에는 `src -> pi05_TO_hybrid` 와 `/home/mk/... -> /mnt/dev/work` 심볼릭 링크로 맞췄고,
**2026-09-14 에 컨테이너가 재시작되면서 `src` 가 사라져 MuJoCo 재생이 전부 죽었다.**

링크에 기대는 대신 여기서 찾는다. 링크가 있으면 그대로 쓰므로 기존 환경도 그대로 돈다.

## 틀린 자산을 집을 위험

`policy_record.replay_scene` 이 모델을 연 직후 **`nq` 와 관절 이름 순서를 기록과 대조**하고
다르면 예외를 던진다 — "조용히 다른 씬을 재생하는" 실패를 막으라고 있는 검사다. 탐색으로
후보를 넓혀도 그 안전 성질은 그대로다.
"""

from __future__ import annotations

import os
import pathlib

#: 자산 트리(`rby1_description/`)가 놓일 수 있는 자리, 작업 루트 기준. 앞에서부터 찾는다.
#: `src` 를 먼저 두는 것은 링크가 살아 있는 환경에서 예전과 똑같이 동작하게 하기 위해서다.
ASSET_ROOTS = ("src", "pi05_TO_hybrid", ".")

#: 이 디렉터리 이름부터가 자산 트리의 뿌리다. 경로에서 이것을 찾아 꼬리를 떼어 낸다.
ASSET_MARKER = "rby1_description"

#: 명시적 오버라이드. 탐색이 실패하거나 다른 자산을 쓰고 싶을 때.
ENV_OVERRIDE = "AG3S_ASSET_ROOT"


def resolve_asset(path: str | os.PathLike, *, what: str = "asset") -> pathlib.Path:
    """`path` 를 이 머신에서 실제로 존재하는 경로로 바꾼다.

    순서:

    1. 준 경로가 그대로 있으면 그것 (링크가 살아 있으면 여기서 끝난다).
    2. `AG3S_ASSET_ROOT` 가 가리키는 루트 아래에서 찾는다.
    3. 경로에서 `rby1_description/` 부터의 꼬리를 떼어 `ASSET_ROOTS` 에 차례로 붙여 본다.

    찾지 못하면 **어디를 뒤졌는지 적어서** `FileNotFoundError` 를 낸다 — 이 실패는 환경 설정
    문제이고, 어디에 두면 되는지를 말해 주지 않으면 진단이 오래 걸린다.
    """
    p = pathlib.Path(path)
    if p.exists():
        return p if p.is_absolute() else p.resolve()

    parts = p.parts
    tail = (pathlib.Path(*parts[parts.index(ASSET_MARKER):])
            if ASSET_MARKER in parts else None)

    tried: list[str] = [str(p)]
    env_root = os.environ.get(ENV_OVERRIDE)
    if env_root and tail is not None:
        cand = pathlib.Path(env_root) / tail
        tried.append(str(cand))
        if cand.exists():
            return cand.resolve()

    if tail is not None:
        here = pathlib.Path.cwd()
        for root in ASSET_ROOTS:
            cand = here / root / tail
            tried.append(str(cand))
            if cand.exists():
                return cand.resolve()

    raise FileNotFoundError(
        f"{what} not found. Tried:\n  " + "\n  ".join(tried) + "\n"
        f"Run from the workspace root, or set {ENV_OVERRIDE} to the directory that contains "
        f"{ASSET_MARKER}/."
    )


__all__ = ["ASSET_MARKER", "ASSET_ROOTS", "ENV_OVERRIDE", "resolve_asset"]
