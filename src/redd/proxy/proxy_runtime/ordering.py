"""
Proxy Reordering Utilities

This module centralizes the logic for ordering proxies in the proxy-runtime
pipeline.

The core idea (which we refer to as **ReD-Ordering** / **reording**) is to
sort proxies by their **rejection efficiency**:

    rejection_efficiency = (1 - pass_rate) / cost

Where:
- **pass_rate** is the fraction of documents a proxy is expected to let through
- **cost** is a relative measure of how expensive the proxy is to run

Proxies with **higher rejection efficiency** are run earlier in the cascade
to implement a fail-fast, cost-aware pipeline.

This logic used to live inside `ProxyExecutor` only; it is now factored out
so that other components (or experiments) can reuse the same ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class ProxyLike(Protocol):
    """
    Minimal protocol for a proxy that can be reordered.

    Both `ConformalProxy` and `EmbeddingProxy` satisfy this interface.
    """

    @property
    def cost(self) -> float:  # pragma: no cover - simple property
        ...

    @property
    def pass_rate(self) -> float:  # pragma: no cover - simple property
        ...

    @property
    def rejection_efficiency(self) -> float:  # pragma: no cover - simple property
        ...


G = TypeVar("G", bound=ProxyLike)


@dataclass(frozen=True)
class ProxyOrderingStats:
    """
    Summary statistics about a particular proxy ordering.

    This can be useful for logging and analysis when experimenting with
    different ordering strategies.
    """

    num_proxies: int
    average_cost: float
    average_pass_rate: float


def compute_rejection_efficiency(pass_rate: float, cost: float) -> float:
    """
    Compute rejection efficiency given pass rate and cost.

    This is the scalar objective we use for **reording**.
    """
    if cost <= 0:
        return float("inf")
    return (1.0 - pass_rate) / cost


def reording(proxies: Iterable[G]) -> List[G]:
    """
    Reorder proxies using ReD-Ordering (cost-aware rejection efficiency).

    Proxies are sorted in **descending** order of `rejection_efficiency` so that
    cheap and strict proxies run earlier in the cascade.

    Args:
        proxies: Iterable of proxy-like objects implementing `ProxyLike`

    Returns:
        List of proxies sorted by decreasing rejection efficiency
    """
    proxy_list = list(proxies)

    # Sort by the already-defined `rejection_efficiency` property so we do not
    # duplicate implementation details here.
    proxy_list.sort(key=lambda g: g.rejection_efficiency, reverse=True)
    return proxy_list


def summarize_ordering(proxies: Iterable[ProxyLike]) -> ProxyOrderingStats:
    """
    Compute simple aggregate statistics for a given proxy ordering.

    This is purely for diagnostics / logging and does not influence ordering.
    """
    proxy_list = list(proxies)
    if not proxy_list:
        return ProxyOrderingStats(num_proxies=0, average_cost=0.0, average_pass_rate=0.0)

    total_cost = sum(g.cost for g in proxy_list)
    total_pass_rate = sum(g.pass_rate for g in proxy_list)

    return ProxyOrderingStats(
        num_proxies=len(proxy_list),
        average_cost=total_cost / len(proxy_list),
        average_pass_rate=total_pass_rate / len(proxy_list),
    )
