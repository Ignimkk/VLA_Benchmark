"""정책 RNG seed 입구 (T39 S1) — openpi 를 고치지 않고 `Policy._rng` 를 다시 놓는다.

π0.5 의 `openpi.policies.policy.Policy` 는 생성 때 `self._rng = rng or jax.random.key(0)` 를 잡고
(`policy.py:65`), `infer` 마다 `self._rng, sample = jax.random.split(self._rng)` 할 뿐 **다시 놓지
않는다** (`policy.py:75`). `Policy.reset` 은 no-op 이다 (`base_policy.py:10-12`). 그래서 run 의
noise 는 그 run 이 서버 요청 순서의 몇 번째인가가 정한다. O1 은 `(episode, seed)` 가 run 을 정하게
하려고 seed 를 명시한다 (`wire.POLICY_SEED`).

openpi 는 vendored 서브모듈이라 고치지 않는다. 대신 감싼 정책 사슬(`SafePolicy → AttentionPolicy →
Policy`, `SeededPolicy → Policy`)을 `_policy` 로 따라 내려가 **`_rng` 를 가진 객체**를 찾고 그 속성을
바꾼다. 못 찾으면 — openpi 가 그 속성 이름을 바꿨거나 PyTorch 정책이면 — **크게 실패한다.** seed 를
조용히 무시하면 짝지은 비교가 짝이 아닌 채로 기록된다.

| 함수 | 쓰는 곳 |
|---|---|
| `rng_owner(policy)` | `_rng` 를 가진 객체. 없으면 `PolicySeedError` |
| `reseed(policy, seed)` | `owner._rng = jax.random.key(seed)` |
| `SeededPolicy` | `--no-safe` 서버의 얇은 감싸개 — 요청의 맨 키 `policy_seed` 를 **벗겨** 소비한다 |
| `apply_request_seed(policy, scene)` | safe · `--no-perception` 경로 — `ag3s/policy_seed` 를 소비한다 |
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from benchmark.trajopt import wire

__all__ = ["PolicySeedError", "rng_owner", "describe_rng_owner", "reseed",
           "apply_request_seed", "SeededPolicy", "MAX_WRAP_DEPTH", "XLA_DETERMINISTIC_FLAG",
           "describe_xla_determinism"]

log = logging.getLogger(__name__)

#: 감싼 사슬을 따라 내려갈 최대 깊이. 지금은 많아야 셋 (`SafePolicy → AttentionPolicy → Policy`).
#: 순환 참조에 빠지지 않게 하는 상한이다.
MAX_WRAP_DEPTH = 8

#: openpi `Policy` 가 RNG 를 담는 속성 이름 (`policy.py:65`).
RNG_ATTR = "_rng"


class PolicySeedError(RuntimeError):
    """seed 를 적용할 수 없다. 조용히 넘기지 않는다."""


def rng_owner(policy) -> Any:
    """`policy` 사슬에서 `_rng` 를 가진 첫 객체.

    Raises:
        PolicySeedError: 못 찾았거나 그 객체가 PyTorch 정책이다 (`_is_pytorch_model`, `_rng` 를 안 씀).
    """
    seen = []
    obj = policy
    for _ in range(MAX_WRAP_DEPTH):
        if obj is None:
            break
        seen.append(type(obj).__name__)
        if hasattr(obj, RNG_ATTR):
            if getattr(obj, "_is_pytorch_model", False):
                raise PolicySeedError(
                    f"{type(obj).__name__} is a PyTorch policy; it does not sample from "
                    f"`{RNG_ATTR}`, so a policy_seed cannot set its noise")
            return obj
        obj = getattr(obj, "_policy", None)
    raise PolicySeedError(
        f"no `{RNG_ATTR}` attribute along the policy chain {' -> '.join(seen) or '(empty)'}. "
        "openpi `Policy` keeps its JAX key there (`openpi/policies/policy.py:65`); if it moved, "
        "`benchmark/trajopt/policy_seed.py` must follow it. Refusing to ignore the seed — "
        "a silently ignored seed records unpaired runs as paired")


def describe_rng_owner(policy) -> str:
    """시작 로그 한 줄. 못 찾으면 예외 대신 경고 문장 — seed 를 안 보내는 실행은 막지 않는다."""
    try:
        owner = rng_owner(policy)
    except PolicySeedError as exc:
        return f"policy_seed entry UNAVAILABLE ({exc}); requests carrying a seed will fail"
    return (f"policy_seed entry: re-keys {type(owner).__name__}.{RNG_ATTR} "
            "(fresh server = jax.random.key(0)); requests without a seed leave it untouched")


#: 서버 **프로세스 사이** 비트 동일에 필요한 XLA flag (T39 S1 smoke, 2026-10-02).
#:
#: seed 를 고정해도 XLA GPU autotuning 이 프로세스마다 다른 커널을 고르면 같은 관측 · 같은 key 에서
#: 청크가 다르다 — E0 seed 0 첫 청크 3.1e-3, seed 18071 5.0e-3 (t=0 qpos 는 비트 동일). 한 프로세스
#: 안에서는 비트 동일이다. 이 flag 를 주면 서로 다른 두 서버 (E0 · E3b 둘 다) 가 6 청크 비트 동일이었다.
XLA_DETERMINISTIC_FLAG = "--xla_gpu_autotune_level=0"


def describe_xla_determinism(xla_flags: Optional[str]) -> str:
    """시작 로그 한 줄 — 이 서버의 seed 고정 run 이 **어디까지** 재현되는가."""
    flags = (xla_flags or "").split()
    if XLA_DETERMINISTIC_FLAG in flags:
        return (f"XLA autotuning off ({XLA_DETERMINISTIC_FLAG}): a fixed policy_seed reproduces "
                "bit-identically across server processes")
    return (f"XLA autotuning ON (no {XLA_DETERMINISTIC_FLAG} in XLA_FLAGS): a fixed policy_seed "
            "is bit-identical only WITHIN this server process; another process may pick other "
            "GPU kernels and differ by ~1e-3 from the first chunk")


def _jax_key(seed: int):
    import jax

    return jax.random.key(seed)


def reseed(policy, seed: Any, *, key_fn: Optional[Callable[[int], Any]] = None) -> Any:
    """정책 RNG 를 `key_fn(seed)` (기본 `jax.random.key`) 로 다시 놓는다. 그 객체를 돌려준다."""
    seed = wire.check_policy_seed(seed)
    owner = rng_owner(policy)
    setattr(owner, RNG_ATTR, (key_fn or _jax_key)(seed))
    return owner


def apply_request_seed(policy, scene: dict[str, Any], *, seq: Any = None,
                       key_fn: Optional[Callable[[int], Any]] = None) -> Optional[int]:
    """safe 경로 — `scene` (`strip_request` 의 두 번째 값) 에 seed 가 있으면 적용하고 그 값을 돌려준다.

    호출자는 이것을 그 요청의 `policy.infer` **바로 앞**에서 부른다. 없으면 아무것도 안 하고 `None`.
    """
    seed = wire.unpack_policy_seed(scene)
    if seed is None:
        return None
    owner = reseed(policy, seed, key_fn=key_fn)
    log.info("[policy_seed] seq=%s re-keyed %s.%s = jax.random.key(%d)",
             seq, type(owner).__name__, RNG_ATTR, seed)
    return seed


class SeededPolicy:
    """`--no-safe` 서버의 감싸개. 요청의 맨 키 `policy_seed` 를 **벗겨** 소비한다.

    - 키가 없으면 `obs` 를 **그대로**(같은 객체) 넘긴다 — T39 전과 같은 동작.
    - 있으면 `policy.infer` 직전에 RNG 를 다시 놓고, 키를 뺀 사본을 넘긴다 (정책 입력 변환은 모르는
      키를 만나면 깨지거나 — 더 나쁘게 — 조용히 실을 수 있다). 결과에 `policy_seed` 를 회신한다.
    - `metadata` 는 안쪽 것을 그대로 — 접속 때 보내는 바이트가 바뀌지 않는다.
    """

    def __init__(self, policy, *, key_fn: Optional[Callable[[int], Any]] = None):
        self._policy = policy
        self._key_fn = key_fn
        self._requests = 0
        #: 마지막으로 적용한 seed 와 그 요청 번호 (서버 수명 안에서 1 부터). 진단용.
        self.last_seed: Optional[int] = None
        self.last_seed_request: Optional[int] = None

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        self._requests += 1
        if wire.POLICY_SEED not in obs:
            return self._policy.infer(obs, **kwargs)
        obs = dict(obs)
        seed = wire.check_policy_seed(obs.pop(wire.POLICY_SEED))
        owner = reseed(self._policy, seed, key_fn=self._key_fn)
        self.last_seed, self.last_seed_request = seed, self._requests
        log.info("[policy_seed] request #%d re-keyed %s.%s = jax.random.key(%d)",
                 self._requests, type(owner).__name__, RNG_ATTR, seed)
        result = dict(self._policy.infer(obs, **kwargs))
        result[wire.POLICY_SEED] = seed
        return result
