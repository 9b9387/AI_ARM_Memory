"""从项目根目录运行入口：将 src 加入路径后委托给 arm_memory 包。"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from arm_memory.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
