"""Запустить main_bvs_v2: 6 задач × 2 режима × 5 сидов = 60 эпизодов.

Это просто шорткат к двум yaml-конфигам через run_experiment, чтобы
напомнить про порядок и распечатать итоги. Конфиги:
  - runner/configs/main_bvs_v2_baseline.yaml  (baseline + dummy coach)
  - runner/configs/main_bvs_v2_scaffold.yaml  (scaffold + real coach)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runner.run import run_experiment

CFG_BASE = _REPO_ROOT / "runner" / "configs" / "main_bvs_v2_baseline.yaml"
CFG_SCAF = _REPO_ROOT / "runner" / "configs" / "main_bvs_v2_scaffold.yaml"


def main() -> None:
    for name, cfg_path in [("BASELINE+DUMMY", CFG_BASE), ("SCAFFOLD+REAL", CFG_SCAF)]:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        print(f"\n=== {name} ({cfg_path.relative_to(_REPO_ROOT)}) ===", flush=True)
        t = time.time()
        paths = run_experiment(cfg)
        print(f"  готово: {len(paths)} эпизодов за {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
