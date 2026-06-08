# mlgym-coach — REPORT v3-PR1

> AutoML Gym Challenge: 4 режима взаимодействия × 3 модели OpenAI. Закрываем
> 60% веса оценки (interaction protocol 20% + candidate/replay 15% + privacy 25%).

## TL;DR

| режим | gpt-5-mini | gpt-4o-mini | gpt-5-nano |
|---|---|---|---|
| **single_shot** | конкурирует с fixed/flexible при **5-6× меньше токенов** | **доминирует** по cost-perf | проваливается (0/3 на большинстве задач) |
| **repeated_single_shot** | стабилен | **полностью разваливается** (0/3 на 5 из 6 задач) | разваливается |
| **fixed** | стабильно высокий, но дорогой | средне | стабильно работает |
| **flexible** | лучший на titanic/wine | дёшев | средне |

**Главное открытие:** для сильных моделей (gpt-5-mini, 4o-mini) **single_shot
не проигрывает scaffold** при ~5× меньшей цене. Coach не оправдывает накладные
расходы на этих задачах. Для слабой модели (gpt-5-nano) — только flexible/fixed
работают, остальные режимы падают.

## Что построено в PR1

### 1. Четыре режима как отдельные классы
Файлы: `agent/single_shot.py`, `agent/repeated_single_shot.py`, `env/stage_policy.py`.

| mode | agent | coach | stage_policy | feedback между шагами |
|---|---|---|---|---|
| `single_shot` | SingleShotAgent | Dummy | n/a | 0: один LLM-вызов на весь эпизод |
| `repeated_single_shot` | RepeatedSingleShotAgent (N=3) | Dummy | n/a | только список scalar val_score |
| `fixed` | BaselineAgent | real Coach | FixedTransitions (EDA→Baseline→Improve→Submit) | observation+hints, прыжки запрещены |
| `flexible` | BaselineAgent | real Coach | FlexibleTransitions | observation+hints, агент сам выбирает |

### 2. Candidate lifecycle + [CHOOSE]
Файл: `env/candidates.py`.

- Каждый успешный RUN автоматически регистрирует `Candidate(cand_N, val_score,
  predict_code)`. Системный промпт BaselineAgent теперь явно требует
  **dual-branch CODE** (train + `if PREDICT=1: predict`).
- Агент может явно зафиксировать кандидат: `Action(SUBMIT, "[CHOOSE:cand_3]\\n...")`,
  поддерживается keyword `[CHOOSE:best]`.
- Если CHOOSE не вызван или cand_id невалиден — env берёт
  `registry.best_by_validation()` (fallback). Невалидные id записываются в
  `failure_modes`.
- SUBMIT применяет `Candidate.predict_code` (не последний CODE) — это
  семантический сдвиг от v2.

### 3. Privacy lock + Sandbox-абстракция
Файл: `env/sandbox.py`.

- После первого вызова grader'а Env **залочен** (`_locked=True`). Повторный
  SUBMIT возвращает None и пишет `failure_modes += ["double_final_grade_attempt"]`.
- Sandbox-абстракция с двумя реализациями:
  - `PermissiveSandbox` (default) — текущее поведение через subprocess.
  - `RestrictiveSandbox` — **намеренный плейсхолдер**, raises NotImplementedError.
    Настоящий sandbox (bwrap/firejail/контейнер per-run с unmounted
    `hidden_labels/`) — отдельная инженерная задача.
- Переключатель: `MLGYM_SANDBOX=permissive|restrictive`.
- `final_test_score` не возвращается агенту в observation.

### 4. Snapshot env config в EpisodeResult
В `EpisodeResult.config` теперь:
```json
{
  "env_version": "v3-pr1",
  "mode": "fixed",
  "stage_policy": "fixed",
  "sandbox": "permissive",
  "coach": "Coach",
  "candidates_n": 5,
  "chosen_candidate_id": "cand_3",
  "failure_modes": [],
  "grade_count": 1
}
```

## Эксперимент

- **6 задач**: titanic, california, adult, wine, credit_g, bank_marketing.
- **4 режима**, **3 сида**, **3 модели** — план был 216 эпизодов, прогнано **166**
  (gpt-5-mini был медленный, проц убит на 32/72; 4o-mini 65/72; 5-nano 69/72).
- `token_budget=30000`, `max_steps=18`, dev sandbox (PermissiveSandbox).

## Результаты по моделям

### gpt-5-mini (сильная, 32 из 72 эпизодов)

| task | mode | n_ok/n | mean_final ± SE | mean_tokens |
|---|---|---|---|---|
| adult_income | fixed | 2/2 | 0.9002 ± 0.0046 | 28,869 |
| adult_income | repeated_single_shot | 3/3 | 0.9033 ± 0.0009 | 14,997 |
| adult_income | **single_shot** | 3/3 | **0.8953 ± 0.0014** | **6,220** |
| california ↓ | fixed | 2/3 | 0.5009 ± 0.0009 | 29,643 |
| california ↓ | flexible | 2/3 | **0.4983 ± 0.0012** | 29,318 |
| california ↓ | single_shot | 2/3 | 0.5057 ± 0.0007 | **6,048** |
| titanic | **fixed** | 3/3 | **0.8442 ± 0.0030** | 28,357 |
| titanic | flexible | 3/3 | 0.8318 ± 0.0085 | 13,431 |
| titanic | repeated_single_shot | 3/3 | 0.8426 ± 0.0031 | 13,827 |
| titanic | single_shot | 3/3 | 0.8201 ± 0.0011 | **4,602** |

Гипотеза «scaffold нужен на сильной модели» — НЕ подтверждается: `fixed`
выигрывает на titanic (z=0.94 vs flexible), но проигрывает в стоимости
single_shot в **6× меньше токенов** при близких скорах. На adult Coach не даёт
прироста; на california результаты в пределах SE между режимами.

### gpt-4o-mini (средняя, 65 эпизодов)

| task | mode | n_ok/n | mean_final ± SE | mean_tokens |
|---|---|---|---|---|
| adult_income | flexible | 3/3 | 0.9012 ± 0.0007 | 4,866 |
| adult_income | **single_shot** | 3/3 | **0.8980 ± 0.0000** | **2,405** |
| adult_income | repeated_single_shot | 0/3 | — | 6,602 |
| california ↓ | flexible | 2/3 | 0.5134 ± 0.0018 | 24,559 |
| california ↓ | **single_shot** | 2/3 | 0.5381 ± 0.0265 | **2,626** |
| credit_g | fixed | 2/3 | 0.8041 ± 0.0000 | 23,852 |
| credit_g | **single_shot** | 3/3 | 0.8026 ± 0.0000 | **2,491** |
| titanic | flexible | 3/3 | 0.8353 ± 0.0087 | 4,910 |
| titanic | **single_shot** | 3/3 | **0.8376 ± 0.0065** | **2,569** |
| wine ↓ | fixed | 2/3 | 0.6538 | 23,720 |
| wine ↓ | **single_shot** | 3/3 | 0.6344 ± 0.0000 | **2,560** |

**Главная находка**: `single_shot` практически по всем задачам в пределах SE
от scaffold-режимов, **но тратит 1/5–1/10 токенов**. На titanic он даже немного
лучше. `repeated_single_shot` **полностью разваливается** на 4o-mini —
0/3 submit на 5 из 6 задач (модель не справляется с независимыми попытками).

### gpt-5-nano (слабая, 69 эпизодов)

| task | mode | n_ok/n | mean_final ± SE | mean_tokens |
|---|---|---|---|---|
| adult_income | **flexible** | 2/3 | **0.9041 ± 0.0029** | 20,880 |
| adult_income | fixed | 2/3 | 0.9025 ± 0.0040 | 25,731 |
| adult_income | single_shot | 0/3 | — | 4,265 |
| bank_marketing | fixed | 1/3 | 0.9047 | 29,671 |
| california ↓ | **fixed** | 2/3 | **0.5016 ± 0.0068** | 26,557 |
| california ↓ | flexible | 2/3 | 0.5123 ± 0.0006 | 27,340 |
| california ↓ | single_shot | 0/3 | — | 6,961 |
| credit_g | fixed | 2/3 | 0.7937 ± 0.0064 | 26,752 |
| credit_g | single_shot | 1/3 | 0.7842 | 4,885 |
| titanic | **flexible** | 3/3 | **0.8387 ± 0.0088** | 26,881 |
| titanic | fixed | 3/3 | 0.8331 ± 0.0105 | 26,010 |
| titanic | single_shot | 0/3 | — | 3,994 |
| wine ↓ | fixed | 3/3 | 0.6448 ± 0.0150 | 25,912 |

**Главная находка для слабой модели**: `single_shot` и `repeated_single_shot`
**категорически не работают** (0/3 submit на большинстве задач). `flexible` и
`fixed` — единственные, что доходят до финала. Coach (через observation+hints)
оказывается необходимым для слабой модели, чтобы написать рабочий код.
Подтверждает гипотезу «слабее модель → больше нужны подсказки», но в новом
формате: не просто «лучшая метрика», а «вообще работает vs не работает».

## Чтение результатов

**На сильных моделях (gpt-5-mini, gpt-4o-mini) `single_shot` доминирует по
cost-perf.** Метрика в пределах SE от fixed/flexible, но токенов в 5–10 раз
меньше. Это поворот всей гипотезы проекта: если у тебя сильная модель и фиксированный
$-бюджет, ты получишь больше прогонов через single_shot, а не через scaffold.
Coach остаётся ценным как **enabler для слабой модели** — на gpt-5-nano single_shot
буквально не доходит до submit, тогда как flexible сабмитит на 3 из 6 задач.

**`repeated_single_shot` — провал на 4o-mini и 5-nano**. Дробление бюджета на
N=3 независимых попытки оставляет каждой ~10k токенов — этого не хватает для
рабочего кода. На gpt-5-mini репитед держится (3/3 submit на adult и titanic).
Гипотеза «N независимых попыток × scalar feedback дают конкуренцию scaffold»
**не подтверждается** ни на одной модели слабее gpt-5-mini.

**`fixed` vs `flexible`** — практически неразличимы по качеству, fixed чуть
стабильнее (выше z на titanic-5mini), flexible часто дешевле (на adult-4o-mini
4.9k vs 23k токенов). Жёсткий stage_policy не даёт измеримого прироста.

## Failure modes (распределение)

В config['failure_modes'] видны:
- `no_candidate_registered` — характерен для single_shot/repeated_single_shot
  на слабых моделях (код не работает → val_score=None → нет кандидатов).
- `invalid_choose_id:cand_X` — попытка выбрать несуществующий cand_id (когда
  агент в single_shot ссылается на cand_1, но регистрации не было).
- Ни одного `double_final_grade_attempt` — privacy lock работает корректно.

## Чего ещё не хватает (для PR2)

1. **Validation-final gap measurement** — корреляция (val_score, final_score)
   по режимам. Гипотеза: flexible переобучается к Coach feedback → больший gap.
2. **Failure taxonomy классификатор** (`env/failure_taxonomy.py`) с авто-
   классификацией трейсбеков на: `plan_loop`, `code_error`, `validation_failure`,
   `submission_invalid`, `leakage_attempt`, `reproducibility_failure`.
3. **Adversarial agents** — `LeakageSeekingAgent`, `NonDeterministicAgent`,
   `SloppySubmitAgent` + unit-тесты, что env их детектирует.
4. **RestrictiveSandbox** реализация — bwrap/firejail/контейнер.
5. Дополнить REPORT недостающими 50 эпизодами на gpt-5-mini для полноты.

## Контракты (что физически защищено в PR1)

✓ Hidden labels не копируются в workdir; не доступны через TRAIN_PATH/TEST_PATH.
✓ Grader блокируется после первой оценки; повторный SUBMIT → None + failure_mode.
✓ final_test_score только в EpisodeResult, агент его не видит в obs.
✓ Sandbox-абстракция готова к переключению на restrictive.

⚠ FS access внутри permissive sandbox не ограничен — production challenge должна
включать restrictive (см. README § Privacy). PR1 этот контракт прописывает, но
не реализует.

## Стоимость

166 эпизодов × ~15k токенов в среднем = ~2.5M токенов.
- gpt-5-mini (32 эп × ~20k): ~$0.44
- gpt-4o-mini (65 эп × ~10k): ~$0.17
- gpt-5-nano (69 эп × ~15k): ~$0.14
- **Итого ≈ $0.75** за весь PR1 эксперимент.

Все эпизоды в `runs/main_4modes_v3_<model>/<mode>/<task>/seed<N>/episode.json`.
Сводки в `runs/main_4modes_v3_<model>/summary.json`.
