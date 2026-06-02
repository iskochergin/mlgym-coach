from core.types import Observation, Hint, Stage


class Coach:
    def __init__(self):
        self.item_attempts: dict[str, int] = {}

        self.CHECKLIST = {
            Stage.EDA: [
                "feature_types", "missing_values", "target_analysis", "duplicates",
                "correlations", "outliers", "categorical_encoding",
                "distribution_shift", "leakage_check"
            ],
            Stage.BASELINE: ["validation_split", "baseline_model", "baseline_score"],
            Stage.IMPROVE: [
                "new_model", "feature_engineering", "hyperparameter_tuning",
                "overfitting_check", "experiment_comparison"
            ],
            Stage.SUBMIT: ["best_model_selected", "retrain_full_data", "test_prediction", "submission"]
        }

        self.KEYWORDS = {
            "feature_types": ["dtype", "dtypes", "select_dtypes"],
            "missing_values": ["isnull", "isna", "fillna", "SimpleImputer"],
            "target_analysis": ["value_counts", "target", "distribution"],
            "duplicates": ["duplicated", "drop_duplicates"],
            "correlations": ["corr", "heatmap"],
            "outliers": ["outlier", "boxplot", "z-score", "quantile"],
            "categorical_encoding": ["OneHotEncoder", "LabelEncoder", "get_dummies", "OrdinalEncoder"],
            "distribution_shift": ["distribution", "shift", "drift"],
            "leakage_check": ["leakage", "data leak"],

            "validation_split": ["train_test_split", "KFold", "StratifiedKFold", "cross_val_score"],
            "baseline_model": ["LogisticRegression", "LinearRegression", "RandomForest", "XGB", "LightGBM", "fit("],
            "baseline_score": ["roc_auc", "accuracy", "f1", "rmse", "mae", "score"],

            "new_model": ["CatBoost", "GradientBoosting", "ensemble"],
            "feature_engineering": ["feature engineering", "new feature", "polynomial", "interaction"],
            "hyperparameter_tuning": ["GridSearchCV", "RandomizedSearchCV", "Optuna"],
            "overfitting_check": ["overfit", "train score", "valid score", "regularization"],
            "experiment_comparison": ["compare", "better", "improvement"],

            "best_model_selected": ["best_model", "final_model"],
            "retrain_full_data": ["retrain", "full data", "all_data"],
            "test_prediction": ["predict", "X_test"],
            "submission": ["submit", "prediction.csv", "predict"]
        }

        self.HINT_TEXTS = {
            "feature_types": {
                1: "Обрати внимание на типы данных в колонках.",
                2: "Ты проверял, правильно ли определились dtypes?",
                3: "Используй df.info() или df.dtypes, чтобы изучить типы признаков."
            },
            "missing_values": {
                1: "Обрати внимание на качество заполненности признаков.",
                2: "Ты проверял количество пропусков?",
                3: "Используй isnull() и обработай пропуски."
            },
            "target_analysis": {
                1: "Изучи целевую переменную.",
                2: "Какой баланс классов или распределение у таргета?",
                3: "Используй value_counts() или построй гистограмму целевой переменной."
            },
            "duplicates": {
                1: "Проверь данные на дубликаты.",
                2: "Есть ли в таблице повторяющиеся строки?",
                3: "Воспользуйся методом duplicated() или drop_duplicates()."
            },
            "correlations": {
                1: "Посмотри на взаимосвязи между признаками.",
                2: "Есть ли сильно коррелирующие признаки?",
                3: "Построй матрицу корреляций с помощью df.corr() и визуализируй её."
            },
            "outliers": {
                1: "Проверь данные на наличие аномалий.",
                2: "Есть ли в данных нетипичные выбросы?",
                3: "Используй boxplot или метод межквартильного размаха (IQR) для поиска выбросов."
            },
            "categorical_encoding": {
                1: "Подумай, как перевести категории в числа.",
                2: "Какие методы кодирования категориальных признаков ты знаешь?",
                3: "Используй OneHotEncoder, LabelEncoder или pd.get_dummies()."
            },
            "distribution_shift": {
                1: "Проверь, не отличаются ли данные в трейне и тесте.",
                2: "Совпадают ли распределения признаков в разных частях выборки?",
                3: "Сравни гистограммы или статистики признаков для трейна и теста."
            },
            "leakage_check": {
                1: "Убедись, что в признаках нет утечек из будущего.",
                2: "Нет ли признаков, которые напрямую зависят от таргета и недоступны при предсказании?",
                3: "Проверь, не попал ли таргет или его производные в признаки (data leakage)."
            },
            "validation_split": {
                1: "Подумай, как ты будешь проверять качество модели.",
                2: "Какой метод валидации (hold-out или K-fold) лучше подходит?",
                3: "Используй train_test_split или KFold из sklearn.model_selection."
            },
            "baseline_model": {
                1: "Начни с построения простого базового решения.",
                2: "Какую простую модель можно обучить первой?",
                3: "Попробуй LogisticRegression или RandomForest с дефолтными параметрами."
            },
            "baseline_score": {
                1: "Оцени качество твоего бейзлайна.",
                2: "Какое значение метрики получилось на валидации?",
                3: "Вычисли нужную метрику (accuracy, roc_auc и т.д.) на валидационной выборке."
            },
            "new_model": {
                1: "Попробуй более сложные алгоритмы.",
                2: "Может быть, градиентный бустинг даст результат лучше?",
                3: "Попробуй использовать XGBoost, LightGBM или CatBoost."
            },
            "feature_engineering": {
                1: "Попробуй создать новые признаки.",
                2: "Какие комбинации существующих колонок могут быть полезны?",
                3: "Создай новые фичи, например, взаимодействия признаков или полиномиальные признаки."
            },
            "hyperparameter_tuning": {
                1: "Попробуй настроить параметры модели.",
                2: "Ты пробовал перебирать параметры по сетке или использовать Optuna?",
                3: "Используй GridSearchCV, RandomizedSearchCV или библиотеку Optuna."
            },
            "overfitting_check": {
                1: "Проверь модель на переобучение.",
                2: "Насколько сильно отличается скор на трейне и на валидации?",
                3: "Сравни метрики на обучающей и валидационной выборках и добавь регуляризацию, если нужно."
            },
            "experiment_comparison": {
                1: "Сравни результаты текущего эксперимента с предыдущими.",
                2: "Стало ли качество лучше после твоих изменений?",
                3: "Зафиксируй улучшение метрики относительно твоего бейзлайна."
            },
            "best_model_selected": {
                1: "Выбери лучшую версию модели.",
                2: "Какая из обученных моделей показала лучший результат на валидации?",
                3: "Определи финальную модель с оптимальными гиперпараметрами."
            },
            "retrain_full_data": {
                1: "Обучи финальную модель на всех доступных данных.",
                2: "Ты объединил трейн и валидацию перед финальным предсказанием?",
                3: "Выполни .fit() на полном наборе данных (train + val)."
            },
            "test_prediction": {
                1: "Сделай предсказание на тестовых данных.",
                2: "Используй X_test для получения финальных ответов.",
                3: "Примени метод .predict() или .predict_proba() к тестовым признакам."
            },
            "submission": {
                1: "Подготовь файл с ответами для отправки.",
                2: "Формат файла соответствует требуемому (например, CSV)?",
                3: "Сохрани предсказания в файл (например, submission.csv) и вызови команду submit."
            }
        }

    def _detect_completed_items(self, obs: Observation) -> set[str]:
        full_text = ""
        for step in obs.history:
            full_text += (step.action.content or "") + " " + (step.result or "") + " "
        full_text = full_text.lower()

        completed = set()
        for item_id, keywords in self.KEYWORDS.items():
            if any(kw.lower() in full_text for kw in keywords):
                completed.add(item_id)
        return completed

    def assess(self, obs: Observation) -> tuple[float, dict[str, float], list[Hint]]:
        completed_items = self._detect_completed_items(obs)

        stage_coverage = {}
        for stage, items in self.CHECKLIST.items():
            completed_in_stage = [it for it in items if it in completed_items]
            stage_coverage[stage.value] = len(completed_in_stage) / len(items) if items else 0.0

        total_items_list = []
        for stage_items in self.CHECKLIST.values():
            total_items_list.extend(stage_items)

        total_count = len(total_items_list)
        completed_count = len(completed_items)
        coverage = completed_count / total_count if total_count > 0 else 0.0

        hints = []
        current_stage_items = self.CHECKLIST.get(obs.stage, [])
        target_item = None
        for item_id in current_stage_items:
            if item_id not in completed_items:
                target_item = item_id
                break

        if target_item:
            self.item_attempts[target_item] = self.item_attempts.get(target_item, 0) + 1
            attempts = self.item_attempts[target_item]

            if attempts <= 3:
                level = 1
            elif attempts <= 6:
                level = 2
            else:
                level = 3

            text = self.HINT_TEXTS[target_item][level]
            hints.append(Hint(
                stage=obs.stage,
                item_id=target_item,
                level=level,
                text=text
            ))

        return coverage, stage_coverage, hints