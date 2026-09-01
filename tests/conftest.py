"""Suppress noisy Starlette deprecation that pytest would otherwise surface.

`fastapi.testclient` re-exports `starlette.testclient.TestClient` which currently
emits  StarletteDeprecationWarning:  Using `httpx` with `starlette.testclient`
is deprecated; install `httpx2` instead.
There is no stable `httpx2` package yet — the warning is unactionable for this
project. Hide it at collection time so the test run stays green without flags.
"""
from __future__ import annotations

import warnings

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Also ignore the exact message in case the category changes in a future release
warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)
