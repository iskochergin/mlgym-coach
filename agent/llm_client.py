"""Model-neutral, OpenAI-совместимый LLM-клиент + MockLLM для офлайн-прогона.

Реальный режим: говорит с любым OpenAI-совместимым chat/completions эндпоинтом
(base_url + api_key + model из конфига/окружения).

Mock-режим (MLGYM_LLM=mock или нет ключа): возвращает заранее заготовленные
ответы в нашем тегированном формате (см. env/parser.py), чтобы BaselineAgent
крутился end-to-end без реального ключа и без траты токенов.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    timeout: float = 60.0

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


def use_mock() -> bool:
    """Mock включён явно (MLGYM_LLM=mock) или когда нет API-ключа."""
    if os.environ.get("MLGYM_LLM", "").lower() == "mock":
        return True
    return not os.environ.get("OPENAI_API_KEY")


class LLMClient:
    """Тонкая обёртка над chat/completions. Без сторонних SDK — голый urllib."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def complete(self, messages: list[dict], max_tokens: int = 800) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# ─────────────────────────── Mock ───────────────────────────

# Универсальный скрипт решения: определяет тип задачи по числу уникальных
# значений target, обучается на TRAIN_PATH, печатает VAL_SCORE=<float>,
# а в submit-режиме (есть PREDICT=1) пишет predictions.csv по TEST_PATH.
_MOCK_SOLUTION = '''\
import os
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv(os.environ["TRAIN_PATH"])
target = "target" if "target" in train.columns else train.columns[-1]
X = train.drop(columns=[target])
y = train[target]
is_clf = y.nunique() <= 10

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

if is_clf:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, proba)
else:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    score = mean_squared_error(y_val, pred) ** 0.5

print(f"VAL_SCORE={score:.4f}")

if os.environ.get("PREDICT") == "1":
    model.fit(X, y)
    X_test = pd.read_csv(os.environ["TEST_PATH"])
    if is_clf:
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)
    pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
'''

# Заготовленная последовательность ходов ReAct-агента. Последний ответ
# (SUBMIT) повторяется, если модель спросят ещё раз.
_MOCK_SCRIPT: list[str] = [
    "[ACTION:plan]\nБинарная или регрессионная задача: построю RandomForest-бейзлайн, "
    "проверю на отложенной выборке, затем сабмит.\n[/ACTION]",
    f"[ACTION:code]\n{_MOCK_SOLUTION}[/ACTION]",
    "[ACTION:run]\nrun current solution\n[/ACTION]",
    "[ACTION:submit]\nsubmit predictions for the hidden test set\n[/ACTION]",
]


class MockLLM:
    """Stateful-заглушка: на последовательные complete() выдаёт ходы из _MOCK_SCRIPT."""

    def __init__(self, script: Optional[list[str]] = None) -> None:
        self._script = list(script) if script is not None else list(_MOCK_SCRIPT)
        self._idx = 0

    def complete(self, messages: list[dict], max_tokens: int = 800) -> str:
        idx = min(self._idx, len(self._script) - 1)
        self._idx += 1
        return self._script[idx]


def build_client(config: Optional[LLMConfig] = None):
    """Фабрика: MockLLM в офлайн-режиме, иначе настоящий LLMClient."""
    if use_mock():
        return MockLLM()
    return LLMClient(config)
