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

def _load_single_column(path: str, expected_names: list) -> pd.Series:
    """
    Вспомогательная функция для надежной загрузки одной колонки из CSV.
    1. Пробуем прочитать с заголовком.
    2. Если есть одна из expected_names, используем её.
    3. Если колонок всего одна, используем её (это покрывает 'prediction' и др.).
    4. Если ничего не подошло или данных мало, пробуем без заголовка.
    """
    df = pd.read_csv(path)
    
    # Ищем по приоритетным именам
    for name in expected_names:
        if name in df.columns:
            return df[name]
            
    # Если ровно одна колонка — используем её независимо от имени
    if len(df.columns) == 1:
        # Проверяем, не является ли заголовок данными.
        # Если в файле нет заголовка, pandas считает первую строку данных заголовком.
        # Если при этом данных было больше 1 строки, то первая строка попала в заголовок,
        # и её тип (в df.columns[0]) будет совпадать с типом данных в колонке.
        # Если заголовок — это строка '0', '1' и т.д., а данные тоже числа — скорее всего это данные.
        
        column_name = str(df.columns[0])
        try:
            # Если имя колонки — это число (инт или флоат), то это скорее всего данные
            float(column_name)
            is_numeric_header = True
        except ValueError:
            is_numeric_header = False

        if is_numeric_header or len(df) == 0:
             df_no_header = pd.read_csv(path, header=None)
             return df_no_header.iloc[:, 0]
             
        return df.iloc[:, 0]
        
    # Если не нашли и колонок много/ноль, пробуем без заголовка
    df_no_header = pd.read_csv(path, header=None)
    return df_no_header.iloc[:, 0]

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
    y_test = _load_single_column(hidden_labels_path, ["target"])
    preds = _load_single_column(predictions_path, ["pred", "target"])
        
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
