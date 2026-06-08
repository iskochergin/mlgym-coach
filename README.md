# mlgym-coach

> A gym for benchmarking how much a structured pipeline + Socratic hints improve LLM agents on tabular ML tasks — baseline vs scaffold, under a fixed token budget.

Этот файл — **полный контекст проекта**. Его можно целиком отдать любому AI-агенту (Claude и др.), чтобы он понял, что мы строим, как устроена архитектура, какие интерфейсы трогать и в какой зоне он сейчас помогает. Если ты агент — сначала прочитай весь файл, потом раздел **«Инструкция для AI-агента»** в конце.

---

## Идея

Проверяем одну гипотезу: **насколько структурный пайплайн + наводящие подсказки улучшают LLM-агента на классических табличных ML-задачах** по сравнению с голой LLM.

- Даётся LLM и среда (`env`), в которой агент решает табличную ML-задачу.
- Агент действует пошагово: `LLM → action → env → feedback`.
- Внутри среды — **скрытый чек-лист** идеального пайплайна. Агент его не видит; среда по нему выдаёт наводящие подсказки.
- Жёсткий **лимит токенов** на эпизод. Агент видит всё, кроме `y_test`.
- Итог — сравнение двух конфигураций: `baseline` (голая LLM) vs `scaffold` (LLM + пайплайн + подсказки), на фиксированных модели и датасете, на нескольких сидах.

Подсказки **непрямые**: не «убери выбросы в колонке X», а наводящие, чтобы агент додумал сам. Степень прямоты — экспериментальная переменная (см. лестницу L1–L3 ниже).

---

## Как это работает (петля)

```
        ┌─────────── Observation (state, stage, hints, tokens_left) ───────────┐
        │                                                                      │
        ▼                                                                      │
   ┌─────────┐        Action         ┌─────────────────────────────┐          │
   │   LLM   │ ───────────────────▶  │            Env              │ ─────────┘
   │ (agent) │                       │  parser · sandbox · budget  │
   └─────────┘                       │            │                │
                                     │            ▼                │
                                     │   ┌──────────────────┐      │
                                     │   │  Coach (hidden)  │      │
                                     │   │ checklist+hints  │      │
                                     │   └──────────────────┘      │
                                     └─────────────────────────────┘
                                                  │ submit
                                                  ▼
                                           Grader → final test score
```

Правила доступа к данным:
- Агент видит: `train (X+y)`, описание задачи, метрику, фичи теста (`X_test`), подсказки.
- Агент **не видит**: `y_test` и код грейдера. Метрика на тесте считается только при `submit`.
- Ловим утечки: переобучение на валидации, попытки вытащить лейблы из train.

---

## Идеальный пайплайн (v1)

Линейный, 5 стадий, один цикл. Это «трек» агента и одновременно скелет чек-листа Coach (одна группа пунктов на стадию).

```
1 · EDA               →  распределения, пропуски, типы, утечки
2 · Бейзлайн          →  простейшая модель, первый валид-скор
3 · Улучшение (цикл)  →  фичи · модель · тюнинг — пока растёт скор
4 · Сабмит            →  предсказание на тесте, финальный скор
```

На каждом пункте чек-листа Coach держит **детектор** (закрыл ли агент шаг) и **лестницу подсказок**:

| Уровень | Что выдаём (пример: «проверить распределение таргета») |
|---|---|
| L0 | молчим (по умолчанию) |
| L1 — направление | «обрати внимание, как устроена твоя целевая переменная — это влияет на выбор лосса» |
| L2 — вопрос | «ты смотрел на скошенность распределений? уверен, что метрика к ней устойчива?» |
| L3 — прямая | «прологарифмируй таргет» (в норме не доходим) |

Подсказка выдаётся **только для непокрытых пунктов** и эскалирует уровень, если агент N шагов топчется.

Намеренно простая версия: без ветвлений, без отдельной стадии «ансамбль/калибровка» (она пока внутри «Улучшения»), цикл один. Когда базовая версия заработает — стадию 4 можно разбить.

---

## Раскладка репозитория

Папка-на-владельца: границы в git совпадают с зонами ответственности → почти нет конфликтов.

```
mlgym-coach/
  core/        # КОНТРАКТЫ: Task, Action, Observation, Stage, Hint, Step, EpisodeResult
               #   types.py — FROZEN; меняется только через синк
  env/         # [owner: Ваня] gym-петля, парсер действий, песочница, токен-бюджет
               #   gym.py · executor.py (v1 stub) · dummy_coach.py · run_slice.py (демо)
  agent/       # [owner: Ваня] baseline (ReAct) + scaffold (tree-search)
               #   scripted.py — агент-заглушка без LLM, для тестов петли
  coach/       # [owner: Егор] чек-лист, детекторы, подсказки L1–L3, грейдер, метрики
  dashboard/   # [owner: Амели] streamlit-вьюер прогонов поверх EpisodeResult.json
  tasks/       # [owner: Софа] датасеты + спеки задач (train + спрятанный test + метрика)
  runner/      # [owner: Софа] харнесс экспериментов (фиксирует LLM+датасет, N сидов)
  reports/     # [owner: Софа] агрегированные результаты, таблицы baseline vs scaffold

  examples/    # эталонные EpisodeResult-фикстуры (episode_baseline.json, episode_scaffold.json)
               #   + make_examples.py — генератор через core.types. Input для Амели/Софы.
  runs/        # артефакты прогонов (slice_demo.json и т.п.). В .gitignore, не коммитим.
```

Единственная общая зона редактирования — `core/`. Замораживается на старте, меняется только через быстрый синк.

Сейчас в репо лежит **тонкий сквозной слайс**: `env/` + `agent/scripted.py` крутят петлю end-to-end на заглушках (`executor.py` возвращает фейковые скоры, `dummy_coach.py` молчит). Запуск демо: `python env/run_slice.py` — пишет результат в `runs/slice_demo.json`. Формат гарантированно совпадает с контрактом `core/types.py`.

---

## Контракты `core/` (интерфейс, против которого кодят все)

Это draft v1. Все модули импортируют отсюда и **не меняют поля без синка**.

```python
# core/types.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Stage(str, Enum):
    UNDERSTAND = "understand"
    EDA        = "eda"
    BASELINE   = "baseline"
    IMPROVE    = "improve"
    SUBMIT     = "submit"

class ActionType(str, Enum):
    PLAN   = "plan"     # рассуждение / план
    EDA    = "eda"      # исследование данных
    CODE   = "code"     # написать / переписать код решения
    RUN    = "run"      # исполнить текущий код, получить валид-метрику
    SUBMIT = "submit"   # финал: предсказание на тесте

@dataclass
class Task:
    id: str
    description: str            # текст задачи + определение метрики
    metric: str                 # "roc_auc" | "rmse" | ...
    metric_higher_better: bool
    train_path: str             # CSV с X + y
    test_features_path: str     # X_test без лейблов
    # y_test живёт ТОЛЬКО в Grader, агенту недоступен

@dataclass
class Action:
    type: ActionType
    content: str                # текст или код — зависит от type

@dataclass
class Hint:
    stage: Stage
    item_id: str                # на какой пункт чек-листа нацелена
    level: int                  # 1..3 (L1 наводящая → L3 прямая)
    text: str

@dataclass
class Step:
    idx: int
    stage: Stage
    action: Action
    result: str                 # что вернула среда (stdout / ошибка / метрика)
    val_score: Optional[float]
    tokens_used: int
    hints: list[Hint] = field(default_factory=list)

@dataclass
class Observation:
    task: Task
    stage: Stage                # текущая стадия агента
    history: list[Step]
    last_result: Optional[str]
    val_score: Optional[float]
    tokens_left: int
    hints: list[Hint] = field(default_factory=list)   # подсказки этого хода

@dataclass
class EpisodeResult:
    task_id: str
    agent: str                  # "baseline" | "scaffold"
    seed: int
    steps: list[Step]
    final_test_score: Optional[float]
    checklist_coverage: float   # 0..1, заполняет Coach
    total_tokens: int
    config: dict = field(default_factory=dict)
```

Ключевые сигнатуры модулей:

```python
# env/gym.py
class Env:
    def reset(self, task: Task) -> Observation: ...
    def step(self, action: Action) -> Observation: ...   # двигает состояние, дёргает Coach
    def result(self) -> EpisodeResult: ...

# agent/base.py
class Agent:
    def act(self, obs: Observation) -> Action: ...

# coach/coach.py
class Coach:
    def assess(self, obs: Observation) -> tuple[float, list[Hint]]:
        """Возвращает текущее покрытие чек-листа (0..1) и подсказки на этот ход."""

# coach/grader.py
def score_submission(task: Task, predictions) -> float: ...   # считает метрику на y_test
```

Глю-точки: `Observation` (Env→Agent, читает Coach), `Hint` (Coach→Env→Agent), `EpisodeResult` (всё → dashboard + runner).

---

**Ваня — начальника**
Ваня ведёт `agent/` + `env/` и владеет `core/`. Остальные трое — независимые папки.

**Егор — Coach** (`coach/`)
Пишет скрытый чек-лист из 5 стадий (каждая — набор пунктов), детекторы, которые по `Observation` понимают, какие пункты агент закрыл, лестницу подсказок L1–L3 на каждый пункт с политикой выдачи (только непройденные, эскалация если топчется), и метрики покрытия + статистику по подсказкам. Грейдер (`score_submission`) тоже здесь.
*Мокает: `Observation` и пара фейковых траекторий — реальная среда не нужна.*

**Амели — Interface** (`dashboard/`)
Дашборд (Streamlit поверх `EpisodeResult.json`): список прогонов, таймлайн действий агента (стадия → action → результат хода), прогресс по 5 стадиям с выданными подсказками, графики токенов/скора по шагам, сравнение двух прогонов (baseline vs scaffold).
*Мокает: один-два примера `EpisodeResult.json` руками — работает полностью автономно, без чужого кода.*

**Софа — Runner + датасеты** (`tasks/`, `runner/`, `reports/`)
2–3 простых табличных датасета (train + спрятанный test + функция метрики + `Task`-спека) с защитой от утечки `y_test`; раннер, который фиксирует LLM и датасет и гоняет эксперимент на N сидах через yaml-конфиг; агрегатор (среднее ± SE, таблица baseline vs scaffold) с выгрузкой в `reports/`.
*Мокает: фейковый env, возвращающий случайный скор, пока реальный не готов.*

---

## Порядок работ

1. **Нулевой день (Ваня, ~час):** заморозить `core/types.py`. Все импортируют, остальное мокают.
2. **Тонкий сквозной слайс (вместе, пара дней):** 1 датасет, baseline-агент, тупой Coach без подсказок, env крутится, runner выдаёт `EpisodeResult`, dashboard его показывает. Цель — состыковать все модули один раз и выловить косяки интерфейсов.
3. **Параллель:** каждый допиливает свою папку против контрактов.
4. **Эксперимент (вместе):** baseline vs scaffold + ablation по слоям, ≥3 сида, среднее ± SE → `reports/`.

---

## Оценка

- **Outcome:** скор на held-out тесте; доля задач с ≥X% улучшения над фикс. бейзлайном.
- **Process:** покрытие чек-листа (сколько шагов агент закрыл сам), сколько и какого уровня подсказок ушло.
- **Efficiency:** токены / стоимость до заданного скора; скор при фиксированном бюджете.
- **Статистика:** ≥3 сида на конфигурацию (агенты высоковариативны), среднее ± SE.
- **Анти-чит:** прирост от качества модели, а не от утечки/оверфита на валидации.

---

## Стек / конвенции

- Python 3.11+, type hints, `dataclasses`. Без тяжёлых фреймворков в v1.
- Линт/формат: `ruff` + `black`.
- LLM-клиент — model-neutral (OpenAI-совместимый API), модель задаётся в конфиге.
- Песочница: исполнение кода агента в изолированном процессе/контейнере.

---

## Инструкция для AI-агента

Если этот файл отдали тебе, чтобы помочь одному из участников:

1. **Определи зону.** Спроси (или выведи из задачи), в какой папке работаешь: `coach/`, `dashboard/` или `tasks/`+`runner/`. Не лезь в чужие папки.
2. **Кодь против `core/`.** Импортируй типы из `core/types.py`, не меняй их поля. Если поля реально не хватает — не правь молча, а вынеси это как вопрос для синка по контрактам.
3. **Мокай чужое.** Тебе не нужен чужой готовый код: Coach мокает `Observation`, Interface — `EpisodeResult.json`, Runner — фейковый env. Это и есть смысл параллельной работы.
4. **Держи v1 простым.** Пять стадий, один цикл, простые детекторы (правила по коду/истории; LLM-judge — потом). Не добавляй стадии и фичи, которых нет в этом файле, без явной просьбы.
5. **Не нарушай правила данных.** Никогда не давай агенту доступ к `y_test` или коду грейдера. Любой код, читающий тестовые лейблы, — только внутри `coach/grader.py`.
6. **Сверяйся с этим README** как с источником правды по архитектуре, контрактам и границам ответственности.

---

## Запуск через Docker (просто прогнать всё)

Один контейнер: Streamlit-дашборд + раннер + агент.

1. Прописать ключ OpenAI в `.env` в корне репо (см. `SECRETS.md`):
   ```
   OPENAI_API_KEY=sk-...        # твой ключ; sk-REPLACE-ME → включится MockLLM
   OPENAI_MODEL=gpt-5-mini
   MLGYM_LLM=openai             # mock | openai
   ```
   `.env` в гит не коммитится и НЕ попадает в образ — ключ прокидывается в контейнер
   в рантайме через `env_file` в `docker-compose.yml`.

2. Поднять:
   ```bash
   docker compose up --build
   ```

3. Открыть дашборд: **http://localhost:8501**

Данные переживают пересборку образа — `runs/` и `tasks/uploads/` смонтированы как
volume. Песочница исполнения кода в этом варианте НЕ изолирована (агент пишет наш же
baseline-код); строгая изоляция — задача на потом, когда будет исполняться чужой код.

---

## AutoML Gym Challenge — protocols & contracts

### Interaction modes (4)

Каждый эпизод гонится в одном из 4 режимов. Контракт «что видит агент» и «что
делает Env» жёстко зафиксирован.

| mode | agent | coach | stage_policy | что видит агент между шагами |
|---|---|---|---|---|
| `single_shot` | `SingleShotAgent` | DummyCoach | flexible (не применяется) | ничего: ровно один LLM-вызов → CODE + RUN + SUBMIT[CHOOSE:cand_1] |
| `repeated_single_shot` | `RepeatedSingleShotAgent` (N=3) | DummyCoach | flexible (не применяется) | только список scalar `val_score` предыдущих попыток; никакого кода, traceback, hints |
| `fixed` | `BaselineAgent` | real `Coach` | `FixedTransitionsPolicy` (EDA→Baseline→Improve→Submit, без возвратов) | observation + hints; любой «прыжок» назад/вперёд подменяется на nudge |
| `flexible` | `BaselineAgent` | real `Coach` | `FlexibleTransitionsPolicy` | observation + hints; агент сам выбирает следующий шаг |

Backward-compat алиасы (`baseline` → no_coach_flexible, `scaffold` → flexible)
живут в `scripts/run_all_parallel.py` для v2-конфигов.

### Candidate lifecycle: Train → Validate → Choose → Replay → Final

1. **Train + Validate.** Каждый успешный RUN с непустым `VAL_SCORE` АВТОМАТИЧЕСКИ
   регистрирует `Candidate(cand_N, val_score, predict_code, step_idx)` в
   `CandidateRegistry`. `predict_code = последний принятый CODE-блок`; от агента
   ТРЕБУЕТСЯ держать в CODE обе ветки — train+VAL_SCORE и `if PREDICT=1: predict`.
2. **Choose.** Агент перед SUBMIT может явно зафиксировать кандидат через префикс
   в content: `[CHOOSE:cand_3]`. Поддерживается keyword `[CHOOSE:best]`.
3. **Fallback.** Если CHOOSE не вызван или cand_id невалиден — Env берёт
   `registry.best_by_validation()` и пишет в `EpisodeResult.config['failure_modes']`
   соответствующий маркер (`invalid_choose_id:X` если был неверный id).
4. **Replay.** SUBMIT запускает Sandbox с `predict_code` ВЫБРАННОГО кандидата
   (не последнего CODE-блока) и PREDICT=1, получает `predictions.csv`.
5. **Final.** Grader (`coach/grader.py`) считает метрику по `hidden_labels/y_test.csv`.
   `EpisodeResult.final_test_score` — это число, и оно НЕ возвращается обратно агенту.

### Privacy contract

Что физически защищено сейчас:
* Hidden labels (`tasks/hidden_labels/`) **не копируются** в workdir; ничего из
  них не попадает в `TRAIN_PATH` или `TEST_PATH` окружения subprocess.
* После первого вызова grader'а Env **залочен** (`_locked=True`); повторный
  SUBMIT возвращает `None` и добавляет `failure_modes += ["double_final_grade_attempt"]`.
* `final_test_score` пишется ТОЛЬКО в `EpisodeResult` и не показывается агенту
  в обsservation следующего шага.

Что НЕ защищено сейчас (явно):
* Sandbox запускает subprocess с **full FS access** — агентский код в принципе
  может прочитать `tasks/hidden_labels/<id>/y_test.csv` напрямую по пути. PR2
  включает shallow detector через regex на `predict_code`, но это **не
  криптографическая защита**, expert evasion (base64-encoded path, экзотические
  способы открытия файла) пройдут мимо.
* Реальный sandbox (bwrap / firejail / контейнер per-run с unmounted
  `hidden_labels/`) — отдельная инженерная задача, реализация в
  `env/sandbox.py:RestrictiveSandbox` намеренно raises `NotImplementedError`.
  Переключатель: `MLGYM_SANDBOX=permissive` (default) | `restrictive`.

Production-grade challenge evaluation **должна** включать restrictive sandbox.

### How to add a new mode

1. Описать contract (что видит агент, какая stage_policy, какой coach).
2. Если новый агент — создать `agent/<name>.py` с методом `act(obs) -> Action` и
   опциональным `last_tokens`.
3. Если новая stage_policy — добавить в `env/stage_policy.py`,
   расширить `resolve_stage_policy`.
4. Зарегистрировать в `scripts/run_4modes_v3.py::MODES` как
   `(mode_name, coach_kind, policy_name)` и в `_agent_factory`.

### How to add a new task

1. Добавить функцию `prepare_<task>()` в `tasks/prepare_datasets.py` по образцу
   `prepare_titanic_survival`. Использовать `_write_split(task_id, features, target)` —
   она автоматически создаст:
   - `tasks/data/<task_id>/train.csv` (X + target)
   - `tasks/data/<task_id>/test_features.csv` (только X)
   - `tasks/hidden_labels/<task_id>/y_test.csv` (только target)
2. Создать spec `tasks/specs/<task_id>.yaml` с обязательными полями id /
   description / metric / metric_higher_better / *_path.
3. Запустить `python3 -m tasks.prepare_datasets`.
4. Добавить путь к spec в `runner/configs/*.yaml` или в `scripts/run_4modes_v3.py::TASKS`.

### How to reproduce main_4modes_v3 experiment

```bash
# .env содержит OPENAI_API_KEY=sk-... + OPENAI_MODEL + MLGYM_LLM=openai
docker compose up -d --build           # (опционально, если нужен дашборд)
MLGYM_LLM=openai OPENAI_MODEL=gpt-5-mini MLGYM_RUN_TAG=v3_5mini \
  python3 scripts/run_4modes_v3.py --workers 3 --tag v3_5mini
# Параллельно — другие модели:
MLGYM_LLM=openai OPENAI_MODEL=gpt-4o-mini MLGYM_RUN_TAG=v3_4omini \
  python3 scripts/run_4modes_v3.py --workers 3 --tag v3_4omini
MLGYM_LLM=openai OPENAI_MODEL=gpt-5-nano MLGYM_RUN_TAG=v3_5nano \
  python3 scripts/run_4modes_v3.py --workers 3 --tag v3_5nano
# Итог: 3 × 72 = 216 эпизодов в runs/main_4modes_v3_*/. Reproducibility:
# seeds [0,1,2], deterministic kwargs, фиксированный SYSTEM prompt, кэш не используется.
```
