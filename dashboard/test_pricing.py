"""Смоук конвертера цен. Запуск: python -m dashboard.test_pricing"""
from __future__ import annotations

from dashboard.pricing import (
    DEFAULT_MODEL,
    blended_rate,
    dollars_to_tokens,
    tokens_to_dollars,
)


def main() -> None:
    model = DEFAULT_MODEL
    print(f"model = {model}")
    print(f"blended_rate = ${blended_rate(model):.4f} / 1M токенов")

    t = dollars_to_tokens(1.0, model)
    print(f"1$ ≈ {t:,} токенов")

    d = tokens_to_dollars(100_000, model)
    print(f"100000 токенов ≈ ${d:.2f}")

    # round-trip-проверка (грубая, из-за округления до целых токенов)
    back = tokens_to_dollars(dollars_to_tokens(1.0, model), model)
    assert abs(back - 1.0) < 0.01, back
    print("round-trip 1$ ok")


if __name__ == "__main__":
    main()
