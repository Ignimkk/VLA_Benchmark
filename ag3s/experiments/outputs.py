"""검증 산출물의 경로 하나로 정하기 — 다시 돌려도 이전 결과가 지워지지 않도록.

각 단계 스크립트는 그림 폴더와 문서 경로를 따로 받는다. 그대로 두면 재실행이 둘 다 덮어쓰고,
그러면 "depth 를 켜기 전과 후" 같은 비교가 원천적으로 불가능해진다 — 비교하려면 이전 것이
남아 있어야 한다.

`--tag` 하나로 셋을 함께 옮긴다.

| | 기본 | `--tag 004` |
|---|---|---|
| 그림 | `asset/image/<단계>/` | `asset/image_004/<단계>/` |
| 문서 | `docs/step-05-separation.md` | `docs/step-05-separation_004.md` |
| 문서 안 이미지 링크 | `../asset/image/<단계>/` | `../asset/image_004/<단계>/` |

링크는 하드코딩하지 않고 **문서 위치에서 그림 폴더까지의 상대 경로로 계산한다.** 그래야
`--out-figs` 로 아무 데나 지정해도 문서의 이미지가 깨지지 않고, 기본값일 때는 지금까지와
정확히 같은 문자열이 나온다.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re


@dataclasses.dataclass(frozen=True)
class Outputs:
    figures: pathlib.Path
    document: pathlib.Path
    #: 문서 안에서 그림을 가리키는 상대 경로 접두사. `f"![...]({img}/fig1.png)"` 로 쓴다.
    image_prefix: str

    def prepare(self) -> "Outputs":
        self.figures.mkdir(parents=True, exist_ok=True)
        self.document.parent.mkdir(parents=True, exist_ok=True)
        return self


def _tagged(path: pathlib.Path, tag: str, *, is_dir_under_image: bool) -> pathlib.Path:
    if is_dir_under_image:
        # .../asset/image/<단계>  ->  .../asset/image_<tag>/<단계>
        parts = list(path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "image":
                parts[i] = f"image_{tag}"
                return pathlib.Path(*parts)
        return path.parent / f"{path.name}_{tag}"
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def resolve(*, out_figs: str | os.PathLike, out_doc: str | os.PathLike,
            tag: str | None = None) -> Outputs:
    """그림 폴더 · 문서 경로 · 이미지 링크 접두사를 함께 정한다.

    `tag` 가 있으면 둘 다 옮기고, 없으면 준 값을 그대로 쓴다. 접두사는 언제나 계산되므로
    호출자가 경로를 어떻게 주든 문서의 링크가 맞는다.
    """
    figs = pathlib.Path(out_figs)
    doc = pathlib.Path(out_doc)
    if tag:
        clean = re.sub(r"[^0-9A-Za-z._-]", "_", str(tag))
        figs = _tagged(figs, clean, is_dir_under_image=True)
        doc = _tagged(doc, clean, is_dir_under_image=False)
    prefix = os.path.relpath(figs, doc.parent).replace(os.sep, "/")
    return Outputs(figures=figs, document=doc, image_prefix=prefix)


def add_tag_argument(ap) -> None:
    """모든 단계 스크립트가 같은 문구로 갖는 `--tag`."""
    ap.add_argument(
        "--tag", default=None,
        help="산출물을 따로 저장할 꼬리표. 예: --tag 004 이면 그림은 "
             "asset/image_004/<단계>/ 로, 문서는 <이름>_004.md 로 나간다. 재실행이 이전 "
             "결과를 덮어쓰지 않게 하려는 것이고, 문서 안 이미지 링크도 함께 따라간다.")


__all__ = ["Outputs", "add_tag_argument", "resolve"]
