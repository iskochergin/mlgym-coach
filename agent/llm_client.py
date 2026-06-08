"""LLM-клиент для агента: реальный OpenAI (через openai SDK) + MockLLM-фолбэк.

Конфиг строго через окружение / .env (см. SECRETS.md), секреты в репо не попадают:
  OPENAI_API_KEY   — ключ (плейсхолдер sk-REPLACE-ME трактуется как «ключа нет»)
  OPENAI_BASE_URL  — базовый URL (для прокси / OpenAI-совместимых эндпоинтов)
  OPENAI_MODEL     — модель (дефолт gpt-5-mini)
  MLGYM_LLM        — mock | openai

Выбор клиента (build_client):
  MLGYM_LLM=mock, либо ключа нет, либо ключ == sk-REPLACE-ME → MockLLM
  иначе → реальный OpenAIClient (если openai SDK не установлен — фолбэк на MockLLM).
MockLLM возвращает детерминированные ответы в тегированном формате (env/parser.py),
чтобы BaselineAgent крутился end-to-end без сети и трат токенов.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Подхватываем .env из корня репо (поиск вверх по дереву от CWD).
load_dotenv()

_PLACEHOLDER_KEY = "sk-REPLACE-ME"
_DEFAULT_MODEL = "gpt-5-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"

# Чтобы лог выбора клиента печатался один раз за процесс.
_startup_logged = False


@dataclass
class LLMConfig:
    model: str = _DEFAULT_MODEL
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 60.0

    @staticmethod
    def from_env() -> "LLMConfig":
        mode = os.environ.get("MLGYM_LLM", "").lower()
        if mode == "deepseek":
            base = os.environ.get("DEEPSEEK_BASE_URL") or _DEEPSEEK_DEFAULT_BASE_URL
            return LLMConfig(
                model=os.environ.get("DEEPSEEK_MODEL") or _DEEPSEEK_DEFAULT_MODEL,
                base_url=base.rstrip("/"),
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
        # OpenAI (default)
        base = os.environ.get("OPENAI_BASE_URL")
        return LLMConfig(
            model=os.environ.get("OPENAI_MODEL") or _DEFAULT_MODEL,
            base_url=base.rstrip("/") if base else None,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


def use_mock() -> bool:
    """MockLLM если: MLGYM_LLM=mock, либо ключа нет для выбранного провайдера."""
    mode = os.environ.get("MLGYM_LLM", "").lower()
    if mode == "mock":
        return True
    if mode == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        return (not api_key) or api_key == "sk-REPLACE-ME"
    # По умолчанию считаем openai
    api_key = os.environ.get("OPENAI_API_KEY")
    return (not api_key) or api_key == _PLACEHOLDER_KEY


class OpenAIClient:
    """Реальный клиент через openai SDK. Сохраняет сигнатуру complete()."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        from openai import OpenAI  # ленивый импорт: нет SDK → ImportError ловит build_client

        kwargs: dict = {"api_key": self.config.api_key, "timeout": self.config.timeout}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        # Реальные usage с последнего вызова (None = не доступно). Прокидывается
        # дальше в BaselineAgent.last_tokens и оттуда в Env.step(tokens_used=...).
        self.last_usage_total: Optional[int] = None
        self.last_usage_prompt: Optional[int] = None
        self.last_usage_completion: Optional[int] = None

    def complete(self, messages: list[dict], max_tokens: int = 800) -> str:
        """Резильентный complete: пробуем современные параметры (gpt-5-family —
        max_completion_tokens + reasoning_effort=minimal, чтобы скрытый reasoning
        не съедал ответ), при ошибке откатываемся к более старым."""
        base = {"model": self.config.model, "messages": messages}
        modern = {**base, "max_completion_tokens": max_tokens, "reasoning_effort": "minimal"}
        try:
            resp = self._client.chat.completions.create(**modern)
        except Exception as e:
            msg = str(e).lower()
            if "reasoning_effort" in msg:
                modern.pop("reasoning_effort", None)
                try:
                    resp = self._client.chat.completions.create(**modern)
                except Exception as e2:
                    if "max_completion_tokens" in str(e2).lower() or "unsupported parameter" in str(e2).lower():
                        resp = self._client.chat.completions.create(**base, max_tokens=max_tokens)
                    else:
                        raise
            elif "max_completion_tokens" in msg or "unsupported parameter" in msg:
                resp = self._client.chat.completions.create(**base, max_tokens=max_tokens)
            else:
                raise
        # Сохраняем реальный usage от OpenAI (если есть в ответе).
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                self.last_usage_prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
                self.last_usage_completion = int(getattr(usage, "completion_tokens", 0) or 0)
                self.last_usage_total = int(
                    getattr(usage, "total_tokens", self.last_usage_prompt + self.last_usage_completion)
                )
            except (TypeError, ValueError):
                self.last_usage_total = None
        return resp.choices[0].message.content or ""


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
    pd.DataFrame({"pred": preds}).to_csv("predictions.csv", index=False)
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
        # У моков реального usage нет — потребитель будет фолбэчиться на эвристику.
        self.last_usage_total: Optional[int] = None
        self.last_usage_prompt: Optional[int] = None
        self.last_usage_completion: Optional[int] = None

    def complete(self, messages: list[dict], max_tokens: int = 800) -> str:
        idx = min(self._idx, len(self._script) - 1)
        self._idx += 1
        return self._script[idx]


def _log_once(message: str) -> None:
    global _startup_logged
    if not _startup_logged:
        print(f"[llm_client] {message}", file=sys.stderr)
        _startup_logged = True


def build_client(config: Optional[LLMConfig] = None):
    """Фабрика клиента. Никогда не логирует ключ.

    MockLLM, если включён mock / нет ключа / ключ-плейсхолдер, либо если openai
    SDK не установлен (тогда фолбэк с предупреждением в stderr)."""
    if use_mock():
        _log_once(f"клиент=MockLLM (MLGYM_LLM={os.environ.get('MLGYM_LLM', '')!r}, ключ не задан/плейсхолдер)")
        return MockLLM()

    cfg = config or LLMConfig.from_env()
    mode = os.environ.get("MLGYM_LLM", "").lower() or "openai"
    try:
        client = OpenAIClient(cfg)
    except Exception as e:  # напр. openai SDK не установлен
        _log_once(f"openai SDK недоступен ({e!r}); фолбэк на MockLLM")
        return MockLLM()

    _log_once(f"клиент={mode.upper() if mode != 'openai' else 'OpenAI'}, модель={cfg.model}, base_url={cfg.base_url or _DEFAULT_BASE_URL}")
    return client
