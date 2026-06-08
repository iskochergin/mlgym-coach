"""Candidate registry + [CHOOSE:cand_id] парсер.

Жизненный цикл: Train → Validate → Choose → Replay → Final.
- Train+Validate: каждый успешный RUN, давший val_score, регистрирует Candidate
  (predict_code = текущий код агента, считается, что он содержит и train, и
  predict-ветку под PREDICT=1; это требуется в системном промпте).
- Choose: агент явно `[ACTION:submit]` с content `[CHOOSE:cand_3]\\n...`.
- Replay: env применяет Candidate.predict_code (вместо последнего кода) к raw test.
- Fallback: если CHOOSE не вызван — выбирается best_by_validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Candidate:
    candidate_id: str          # авто: cand_1, cand_2, ...
    val_score: Optional[float] # из validation RUN
    predict_code: str          # код, который применяется к raw test rows
    step_idx: int              # на каком шаге зарегистрирован


@dataclass
class CandidateRegistry:
    """Хранит кандидаты эпизода. Лежит в Env state."""

    metric_higher_better: bool
    _items: list[Candidate] = field(default_factory=list)
    _counter: int = 0

    def register(self, predict_code: str, val_score: Optional[float], step_idx: int) -> str:
        self._counter += 1
        cid = f"cand_{self._counter}"
        self._items.append(Candidate(
            candidate_id=cid,
            val_score=val_score,
            predict_code=predict_code,
            step_idx=step_idx,
        ))
        return cid

    def get(self, candidate_id: str) -> Optional[Candidate]:
        for c in self._items:
            if c.candidate_id == candidate_id:
                return c
        return None

    def all(self) -> list[Candidate]:
        return list(self._items)

    def best_by_validation(self) -> Optional[Candidate]:
        scored = [c for c in self._items if c.val_score is not None]
        if not scored:
            return None
        if self.metric_higher_better:
            return max(scored, key=lambda c: c.val_score)
        return min(scored, key=lambda c: c.val_score)


_CHOOSE_RE = re.compile(r"^\s*\[CHOOSE:(?P<cid>[A-Za-z0-9_-]+)\]\s*", re.IGNORECASE)


def parse_choose_marker(content: str) -> Optional[str]:
    """Если content SUBMIT-действия начинается с `[CHOOSE:cand_id]` — вернуть cand_id.
    Иначе None — env применит fallback (best_by_validation)."""
    m = _CHOOSE_RE.match(content or "")
    return m.group("cid") if m else None
