import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error
)
from core.types import Task

def score_submission(task: Task, predictions_path: str, hidden_labels_path: str) -> float:
    """
    Вычисляет метрику для предсказаний агента.
    
    Args:
        task: Объект Task с описанием задачи и метрики.
        predictions_path: Путь к CSV с предсказаниями агента.
        hidden_labels_path: Путь к CSV с истинными метками.
        
    Returns:
        float: Значение метрики.
        
    Raises:
        ValueError: Если данные некорректны (разная длина, NaN и т.д.)
    """
    # Шаг 1 — загрузка данных
    # Сначала пытаемся прочитать с заголовком
    y_test_df = pd.read_csv(hidden_labels_path)
    preds_df = pd.read_csv(predictions_path)
    
    # Извлекаем y_test
    if "target" in y_test_df.columns:
        y_test = y_test_df["target"]
    else:
        # Если 'target' не найден, предполагаем, что файла без заголовка и перечитываем
        y_test_df = pd.read_csv(hidden_labels_path, header=None)
        y_test = y_test_df.iloc[:, 0]

    # Извлекаем predictions
    if "pred" in preds_df.columns:
        preds = preds_df["pred"]
    elif "target" in preds_df.columns:
        preds = preds_df["target"]
    else:
        # Если ни 'pred', ни 'target' не найдены, перечитываем без заголовка
        preds_df = pd.read_csv(predictions_path, header=None)
        preds = preds_df.iloc[:, 0]
        
    # Проверка на NaN
    if preds.isna().any():
        raise ValueError("Predictions contain NaN values")
        
    # Шаг 2 — выравнивание
    y_test_np = y_test.to_numpy()
    preds_np = preds.to_numpy()
    
    if len(y_test_np) != len(preds_np):
        raise ValueError(f"Length mismatch: y_test ({len(y_test_np)}) != predictions ({len(preds_np)})")
        
    # Шаг 3 & 4 — выбор и вычисление метрики
    metric = task.metric.lower()
    
    if metric == "roc_auc":
        return float(roc_auc_score(y_test_np, preds_np))
    elif metric == "accuracy":
        # Для accuracy обычно нужны дискретные метки. 
        # Если в preds_np вероятности, sklearn может ругнуться или отработать неверно в зависимости от типа y_test.
        # Но по условию мы просто вызываем accuracy_score.
        return float(accuracy_score(y_test_np, preds_np))
    elif metric == "f1":
        return float(f1_score(y_test_np, preds_np))
    elif metric == "rmse":
        mse = mean_squared_error(y_test_np, preds_np)
        return float(np.sqrt(mse))
    elif metric == "mae":
        return float(mean_absolute_error(y_test_np, preds_np))
    else:
        raise ValueError(f"Unsupported metric: {task.metric}")
