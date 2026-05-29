"""Справочник цен моделей и конвертер бюджета (токены ⇄ доллары).

Цены актуальны на дату создания файла (2026-05-29), указаны per 1M токенов в USD.
Подсмотрены на openai.com/api/pricing. При необходимости обновить — правь
MODEL_PRICING вручную (формат одинаковый для всех моделей).
"""
from __future__ import annotations

# Цена за 1M токенов в USD. Чтобы добавить модель — впиши такой же словарь.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}

# Типичный input/output сплит токенов для ReAct-агента (доли, сумма = 1.0).
DEFAULT_IO_SPLIT = (0.75, 0.25)

# Модель по умолчанию для UI-пересчёта.
DEFAULT_MODEL = "gpt-5-mini"


def _pricing(model: str) -> dict[str, float]:
    """Цены модели; для неизвестной — падать не будем, берём дефолтную."""
    return MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])


def blended_rate(model: str) -> float:
    """Блендированная цена за 1M токенов (USD) по DEFAULT_IO_SPLIT."""
    p = _pricing(model)
    in_share, out_share = DEFAULT_IO_SPLIT
    return in_share * p["input"] + out_share * p["output"]


def tokens_to_dollars(tokens: int, model: str = DEFAULT_MODEL) -> float:
    """Перевести число токенов в доллары по блендированной цене."""
    return (max(0, int(tokens)) / 1_000_000) * blended_rate(model)


def dollars_to_tokens(dollars: float, model: str = DEFAULT_MODEL) -> int:
    """Перевести доллары в целое число токенов по блендированной цене."""
    rate = blended_rate(model)
    if rate <= 0:
        return 0
    return int((max(0.0, float(dollars)) / rate) * 1_000_000)
