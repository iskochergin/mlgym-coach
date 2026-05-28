import os
import pandas as pd
import numpy as np
from coach.grader import score_submission
from core.types import Task

def test_grader():
    # Создаем временные CSV файлы
    y_test_path = "temp_y_test.csv"
    preds_path = "temp_preds.csv"
    
    # 1. Тест ROC AUC
    pd.DataFrame({"target": [0, 1, 0, 1]}).to_csv(y_test_path, index=False)
    pd.DataFrame({"pred": [0.1, 0.9, 0.2, 0.8]}).to_csv(preds_path, index=False)
    
    task = Task(id="test", description="", metric="roc_auc", metric_higher_better=True, train_path="", test_features_path="")
    score = score_submission(task, preds_path, y_test_path)
    print(f"ROC AUC: {score}")
    assert score == 1.0
    
    # 2. Тест RMSE
    pd.DataFrame({"target": [10, 20]}).to_csv(y_test_path, index=False)
    pd.DataFrame({"pred": [12, 18]}).to_csv(preds_path, index=False)
    
    task.metric = "rmse"
    score = score_submission(task, preds_path, y_test_path)
    print(f"RMSE: {score}")
    # sqrt((2^2 + 2^2)/2) = sqrt(4) = 2
    assert score == 2.0
    
    # 3. Тест на разную длину
    pd.DataFrame({"target": [1, 2, 3]}).to_csv(y_test_path, index=False)
    pd.DataFrame({"pred": [1, 2]}).to_csv(preds_path, index=False)
    try:
        score_submission(task, preds_path, y_test_path)
        assert False, "Should have raised ValueError for length mismatch"
    except ValueError as e:
        print(f"Caught expected length mismatch: {e}")
        
    # 4. Тест на NaN
    pd.DataFrame({"target": [1, 2]}).to_csv(y_test_path, index=False)
    pd.DataFrame({"pred": [1, np.nan]}).to_csv(preds_path, index=False)
    try:
        score_submission(task, preds_path, y_test_path)
        assert False, "Should have raised ValueError for NaN"
    except ValueError as e:
        print(f"Caught expected NaN error: {e}")

    # 5. Тест разных имен колонок
    pd.DataFrame([1, 0, 1]).to_csv(y_test_path, index=False, header=False) # без хедера, первый столбец
    pd.DataFrame({"target": [1, 0, 1]}).to_csv(preds_path, index=False)
    task.metric = "accuracy"
    score = score_submission(task, preds_path, y_test_path)
    print(f"Accuracy: {score}")
    assert score == 1.0

    # Очистка
    if os.path.exists(y_test_path): os.remove(y_test_path)
    if os.path.exists(preds_path): os.remove(preds_path)
    print("All tests passed!")

if __name__ == "__main__":
    test_grader()
