from __future__ import annotations

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, initial=0, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.initial = initial

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def update(self, n: int = 1) -> None:
            _ = n

        def close(self) -> None:
            return None

    def _tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


tqdm = _tqdm

__all__ = ["tqdm"]
