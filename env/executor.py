"""Исполнитель кода решения: настоящая песочница + сохранённый stub-режим.

run_solution(code, task, mode) -> (result_text, val_score | None)

Режимы:
  mode="run"    — выполнить код, распарсить VAL_SCORE из stdout, вернуть скор.
  mode="submit" — то же + переменная PREDICT=1, скрипт пишет predictions.csv.
                  Test-скор здесь НЕ считаем (нет доступа к y_test — это работа
                  coach/grader), возвращаем (текст, None).
  mode="stub"   — старое фейковое поведение для офлайн-тестов без sklearn.

Песочница:
  - отдельная временная рабочая директория на каждый вызов;
  - в неё кладём ТОЛЬКО train.csv и test_features.csv задачи (никаких
    hidden_labels) и сам solution.py;
  - subprocess с таймаутом ~120с, cwd = workdir, захват stdout/stderr;
  - пути к данным передаём скрипту через env: TRAIN_PATH, TEST_PATH;
  - любая ошибка/таймаут → (текст ошибки, None), без исключений наружу.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ─────────────────────────── stub ───────────────────────────

_VAL_CURVE: list[float] = [0.75, 0.79, 0.82, 0.84, 0.86]
_SUBMIT_SCORE: float = 0.835
_call_count: int = 0

_TIMEOUT_SEC = 120
_VAL_RE = re.compile(r"VAL_SCORE\s*=\s*([-+]?\d*\.?\d+)")


def reset_executor() -> None:
    """Сбросить состояние stub-режима (для повторных эпизодов в одном процессе)."""
    global _call_count
    _call_count = 0


def _run_stub(mode: str, task) -> tuple[str, Optional[float]]:
    global _call_count
    if mode == "submit":
        return (
            f"submission accepted. test {task.metric}={_SUBMIT_SCORE:.3f}",
            _SUBMIT_SCORE,
        )
    idx = min(_call_count, len(_VAL_CURVE) - 1)
    score = _VAL_CURVE[idx]
    _call_count += 1
    return f"VAL_SCORE={score:.3f}", score


# ─────────────────────────── real sandbox ───────────────────────────

def _parse_val_score(stdout: str) -> Optional[float]:
    matches = _VAL_RE.findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def run_solution(
    code: str,
    task,
    mode: str = "run",
    predictions_out: Optional[str] = None,
) -> tuple[str, Optional[float]]:
    """Исполнить код. В submit-режиме, если задан predictions_out, копируем туда
    predictions.csv (рабочая директория удаляется в finally — путь должен
    пережить очистку, чтобы grader мог посчитать метрику)."""
    if mode == "stub":
        return _run_stub("run", task)

    if not code or not code.strip():
        return "executor: пустой код решения", None

    workdir = Path(tempfile.mkdtemp(prefix="mlgym_sol_"))
    try:
        train_dst = workdir / "train.csv"
        try:
            shutil.copyfile(task.train_path, train_dst)
        except OSError as e:
            return f"executor: не удалось прочитать train ({e})", None

        env = dict(os.environ)
        env["TRAIN_PATH"] = str(train_dst)

        # Фичи теста доступны агенту; y_test — нет.
        test_dst = workdir / "test_features.csv"
        if task.test_features_path and Path(task.test_features_path).exists():
            shutil.copyfile(task.test_features_path, test_dst)
            env["TEST_PATH"] = str(test_dst)

        if mode == "submit":
            env["PREDICT"] = "1"

        (workdir / "solution.py").write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "solution.py"],
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return f"executor: таймаут {_TIMEOUT_SEC}s", None

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            tail = stderr.strip()[-1500:]
            return f"executor: код вышел с ошибкой (rc={proc.returncode})\n{tail}", None

        if mode == "submit":
            preds = workdir / "predictions.csv"
            if not preds.exists():
                return "executor: submit-режим, но predictions.csv не создан", None
            n = sum(1 for _ in preds.open(encoding="utf-8")) - 1
            # Сохраняем predictions.csv за пределы temp-папки для grader'а.
            if predictions_out is not None:
                try:
                    shutil.copyfile(preds, predictions_out)
                except OSError as e:
                    return f"executor: не удалось сохранить predictions.csv ({e})", None
            # Test-скор считает grader (есть доступ к y_test), не executor.
            return f"submission accepted: predictions.csv ({max(n, 0)} строк)", None

        score = _parse_val_score(stdout)
        if score is None:
            tail = stdout.strip()[-1000:]
            return f"executor: VAL_SCORE не найден в stdout\n{tail}", None
        return f"VAL_SCORE={score:.4f}", score
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
