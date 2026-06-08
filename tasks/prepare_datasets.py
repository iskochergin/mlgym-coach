from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

# macOS Python.framework часто не видит системные CA → fetch_openml ломается
# на CERTIFICATE_VERIFY_FAILED. Подтягиваем bundle из certifi.
try:
    import certifi

    _ctx = ssl.create_default_context(cafile=certifi.where())
    _opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
    urllib.request.install_opener(_opener)
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HIDDEN_LABELS_DIR = ROOT / "hidden_labels"
RANDOM_STATE = 20260527
TEST_SIZE = 0.25


def _clean_columns(columns: list[str]) -> list[str]:
    return [
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        for column in columns
    ]


def _write_split(
    task_id: str,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    stratify: pd.Series | None = None,
) -> None:
    task_data_dir = DATA_DIR / task_id
    task_hidden_dir = HIDDEN_LABELS_DIR / task_id
    task_data_dir.mkdir(parents=True, exist_ok=True)
    task_hidden_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    train = x_train.reset_index(drop=True).copy()
    train["target"] = y_train.reset_index(drop=True)

    train.to_csv(task_data_dir / "train.csv", index=False)
    x_test.reset_index(drop=True).to_csv(task_data_dir / "test_features.csv", index=False)
    y_test.reset_index(drop=True).to_frame("target").to_csv(
        task_hidden_dir / "y_test.csv",
        index=False,
    )


def prepare_breast_cancer() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features.columns = _clean_columns(list(features.columns))
    target = dataset.target.rename("target")

    _write_split(
        "breast_cancer_roc_auc",
        features,
        target,
        stratify=target,
    )


def prepare_diabetes() -> None:
    dataset = load_diabetes(as_frame=True)
    features = dataset.data.copy()
    features.columns = _clean_columns(list(features.columns))
    target = dataset.target.rename("target")

    _write_split("diabetes_rmse", features, target)


# ───────────────────────────── v2 датасеты ─────────────────────────────
# Все из sklearn / fetch_openml — без скачивания внешних файлов вручную.
# Подобраны так, чтобы baseline-качество было в диапазоне 0.6–0.85
# (где есть пространство для роста на scaffold).


def prepare_titanic_survival() -> None:
    """Бинарная классификация по titanic. Baseline accuracy ~0.78–0.82."""
    bundle = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    # Только полезные колонки, без утечек (boat/body/home.dest → исход).
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
    df = df[keep]
    df["target"] = df["survived"].astype(int)
    df = df.drop(columns=["survived"])
    # Категории → строки (агент сам решит, как кодировать); пропуски оставляем.
    for col in ("sex", "embarked"):
        df[col] = df[col].astype("string")
    features = df.drop(columns=["target"])
    features.columns = _clean_columns(list(features.columns))
    target = df["target"].rename("target")
    _write_split("titanic_survival", features, target, stratify=target)


def prepare_california_housing() -> None:
    """Регрессия по California Housing. Baseline RMSE ~0.55–0.65 (target в $100k).
    Сабсэмплим до ~5k строк — иначе агент тратит много времени на fit на каждом RUN."""
    dataset = fetch_california_housing(as_frame=True)
    features = dataset.data.copy()
    features.columns = _clean_columns(list(features.columns))
    target = dataset.target.rename("target")
    if len(features) > 5000:
        sampled = features.sample(n=5000, random_state=RANDOM_STATE)
        features = sampled.reset_index(drop=True)
        target = target.loc[sampled.index].reset_index(drop=True).rename("target")
    _write_split("california_housing", features, target)


def prepare_adult_income() -> None:
    """Бинарная классификация (доход >50K). Baseline ROC AUC ~0.83–0.87.
    Сабсэмплим до ~5k — полный adult слишком большой для интерактивных прогонов."""
    bundle = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    target_raw = df.get("class")
    if target_raw is None:
        target_raw = df["target"] if "target" in df.columns else df.iloc[:, -1]
    target = (target_raw.astype(str).str.strip() == ">50K").astype(int).rename("target")
    # Дедуплицируем имена колонок, чтобы df.drop не упал на дубликатах ("class","class").
    drop_cols = list(dict.fromkeys([c for c in df.columns if c == "class" or c == target_raw.name]))
    features = df.drop(columns=drop_cols)
    features.columns = _clean_columns(list(features.columns))
    if len(features) > 5000:
        # Стратифицированный сабсэмпл: capим n_pos/n_neg доступным количеством, чтобы
        # не упасть на pos.sample(n > len(positives)).
        df2 = features.copy()
        df2["__t"] = target.values
        positives = df2[df2["__t"] == 1]
        negatives = df2[df2["__t"] == 0]
        n_pos = min(int(round(5000 * target.mean())), len(positives))
        n_pos = max(n_pos, 1)
        n_neg = min(5000 - n_pos, len(negatives))
        pos = positives.sample(n=n_pos, random_state=RANDOM_STATE)
        neg = negatives.sample(n=n_neg, random_state=RANDOM_STATE)
        sampled = pd.concat([pos, neg]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        target = sampled["__t"].rename("target")
        features = sampled.drop(columns=["__t"])
    _write_split("adult_income", features, target, stratify=target)


def prepare_credit_g() -> None:
    """Классика German Credit (OpenML id=31). 1000 строк, бинарная классификация
    кредитного риска по 20 признакам (mix категорий и чисел). Baseline RF AUC ~0.78."""
    bundle = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    target_col = "class" if "class" in df.columns else df.columns[-1]
    target = (df[target_col].astype(str).str.strip() == "good").astype(int).rename("target")
    features = df.drop(columns=[target_col])
    features.columns = _clean_columns(list(features.columns))
    _write_split("credit_g", features, target, stratify=target)


def prepare_bank_marketing() -> None:
    """Bank Marketing (OpenML id=1461). Классический Kaggle-starter datatset.
    Бинарная классификация подписки на депозит, ~45k строк → сабсэмпл 5k.
    Сильный класс-дисбаланс (~12% позитивных). Baseline RF AUC ~0.91."""
    bundle = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    target_col = "Class" if "Class" in df.columns else (
        "class" if "class" in df.columns else df.columns[-1])
    target = (df[target_col].astype(str).str.strip() == "2").astype(int).rename("target")
    # Если разметка 1/2 не сработала — пробуем yes/no
    if target.sum() == 0:
        target = (df[target_col].astype(str).str.strip().str.lower() == "yes").astype(int).rename("target")
    features = df.drop(columns=[target_col])
    features.columns = _clean_columns(list(features.columns))
    if len(features) > 5000:
        df2 = features.copy()
        df2["__t"] = target.values
        positives = df2[df2["__t"] == 1]
        negatives = df2[df2["__t"] == 0]
        # Сохраняем оригинальную долю позитивов
        pos_frac = max(0.05, min(0.5, target.mean()))
        n_pos = min(int(round(5000 * pos_frac)), len(positives))
        n_pos = max(n_pos, 50)
        n_neg = min(5000 - n_pos, len(negatives))
        pos = positives.sample(n=n_pos, random_state=RANDOM_STATE)
        neg = negatives.sample(n=n_neg, random_state=RANDOM_STATE)
        sampled = pd.concat([pos, neg]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        target = sampled["__t"].rename("target")
        features = sampled.drop(columns=["__t"])
    _write_split("bank_marketing", features, target, stratify=target)


def prepare_wine_quality() -> None:
    """Регрессия (оценка качества вина 0–10). Baseline RMSE ~0.6–0.7, шумная."""
    bundle = fetch_openml("wine-quality-red", version=1, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    target_col = "class" if "class" in df.columns else df.columns[-1]
    raw_target = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    # Дропаем строки с NaN в target — иначе они утекут в hidden y_test и сломают RMSE.
    mask = raw_target.notna()
    df = df.loc[mask].reset_index(drop=True)
    target = raw_target.loc[mask].reset_index(drop=True).rename("target")
    features = df.drop(columns=[target_col])
    features.columns = _clean_columns(list(features.columns))
    _write_split("wine_quality", features, target)


def main() -> None:
    prepare_breast_cancer()
    prepare_diabetes()
    prepare_titanic_survival()
    prepare_california_housing()
    prepare_adult_income()
    prepare_wine_quality()
    prepare_credit_g()
    prepare_bank_marketing()


if __name__ == "__main__":
    main()
