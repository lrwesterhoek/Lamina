"""Puts python/ on sys.path so the plot scripts can be run directly from any cwd.
Every script here starts with `import _bootstrap`; after that, `config` and `dtfelib` resolve normally."""

import sys
from pathlib import Path

_python_root = str(Path(__file__).resolve().parents[1])
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)
