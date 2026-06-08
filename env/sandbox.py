"""Sandbox-абстракция для исполнения кода агента.

Две реализации:
  - PermissiveSandbox: текущее поведение через subprocess.run, full FS access.
    Используется в dev/CI и в challenge-evaluation на «честном» уровне.
  - RestrictiveSandbox: TODO(prod) — намеренный плейсхолдер. Настоящий sandbox
    (bwrap / firejail / контейнер per-run + размонтированный hidden_labels/)
    — отдельная инженерная задача на 1–2 дня (см. README § Privacy).

Переключатель: env-var MLGYM_SANDBOX ∈ {"permissive" (default), "restrictive"}.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class Sandbox(ABC):
    """Запустить произвольный python-код с заданными ENV и cwd."""

    name: str

    @abstractmethod
    def run(self, code: str, *, env: dict, cwd: Path, timeout: int) -> SandboxResult: ...


class PermissiveSandbox(Sandbox):
    """Текущее поведение — subprocess.run. Никаких ограничений FS / network."""

    name = "permissive"

    def run(self, code: str, *, env: dict, cwd: Path, timeout: int) -> SandboxResult:
        (cwd / "solution.py").write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "solution.py"],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxResult(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(returncode=-1, stdout="", stderr="timeout", timed_out=True)


class RestrictiveSandbox(Sandbox):
    """Заглушка. Настоящий sandbox — отдельная задача (bwrap/firejail/контейнер).

    Включается через MLGYM_SANDBOX=restrictive; сейчас сразу raises NotImpl.
    Контракт прописан в README § Privacy; реализация — следующий инженерный
    блок (не входит в PR1)."""

    name = "restrictive"

    def run(self, code: str, *, env: dict, cwd: Path, timeout: int) -> SandboxResult:
        raise NotImplementedError(
            "Restrictive sandbox is intentionally unimplemented in PR1. "
            "Real implementation requires bwrap/firejail/container per run "
            "with hidden_labels/ unmounted. See README § Privacy. "
            "Set MLGYM_SANDBOX=permissive for dev runs."
        )


def get_sandbox() -> Sandbox:
    """Прочитать MLGYM_SANDBOX и вернуть нужную реализацию."""
    name = os.environ.get("MLGYM_SANDBOX", "permissive").lower()
    if name == "permissive":
        return PermissiveSandbox()
    if name == "restrictive":
        return RestrictiveSandbox()
    raise ValueError(f"MLGYM_SANDBOX={name!r}; expected 'permissive' or 'restrictive'")
