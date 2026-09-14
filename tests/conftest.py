from __future__ import annotations
import warnings
try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings('ignore', category=StarletteDeprecationWarning)
except ImportError:
    warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='Using `httpx` with `starlette.testclient` is deprecated.*')
