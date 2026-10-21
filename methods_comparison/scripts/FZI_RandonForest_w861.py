# -*- coding: utf-8 -*-
"""
Legacy entry point -- delegates to run_fzi_rf_861.py (Well 861 integrated dataset).

Predicting FZI_lab using RandomForestRegressor (Well #861).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_fzi_rf_861 import main

if __name__ == "__main__":
    raise SystemExit(main())
