from __future__ import annotations
import re
from typing import Any, Optional
from core.types import EpisodeResult, Stage, Step, ActionType
from coach.coach import Coach

def get_stage_coverage(steps: list[Step]) -> dict[str, float]:
    """Вычисляет покрытие чек-листа отдельно для каждой стадии."""
    coach = Coach()
    stage_items = coach.CHECKLIST
    
    # Собираем все ключевые слова для каждого пункта
    completed_items = set()
    for step in steps:
        content = (step.action.content or "").lower()
        result = (step.result or "").lower()
        combined = content + " " + result
        
        for stage, items in stage_items.items():
            for item in items:
                keywords = coach.KEYWORDS.get(item, [])
                if any(kw.lower() in combined for kw in keywords):
                    completed_items.add((stage, item))
    
    coverage = {}
    for stage in Stage:
        items = stage_items.get(stage, [])
        if not items:
            coverage[stage.value] = 0.0
            continue
        
        completed = [it for it in items if (stage, it) in completed_items]
        coverage[stage.value] = len(completed) / len(items)
        
    return coverage

def get_best_run_info(steps: list[Step], metric_higher_better: bool = True) -> dict[str, Any]:
    """Находит лучший запуск и пытается извлечь модель и гиперпараметры."""
    best_score = None
    best_step = None
    
    for step in steps:
        if step.val_score is not None:
            if best_score is None:
                best_score = step.val_score
                best_step = step
            else:
                if metric_higher_better:
                    if step.val_score > best_score:
                        best_score = step.val_score
                        best_step = step
                else:
                    if step.val_score < best_score:
                        best_score = step.val_score
                        best_step = step
    
    if best_step is None:
        return {
            "best_score": None,
            "model": "N/A",
            "hyperparameters": "N/A"
        }
    
    model_code = ""
    for i in range(best_step.idx, -1, -1):
        if steps[i].action.type == ActionType.CODE:
            model_code = steps[i].action.content
            break
            
    model_name, params = extract_model_info(model_code)
    
    return {
        "best_score": best_score,
        "model": model_name,
        "hyperparameters": params
    }

def extract_model_info(code: str) -> tuple[str, str]:
    """Извлекает название модели и гиперпараметры из Python кода."""
    if not code:
        return "N/A", "N/A"
    
    # Популярные модели
    models = {
        "XGBoost": ["XGBClassifier", "XGBRegressor", "xgb"],
        "LightGBM": ["LGBMClassifier", "LGBMRegressor", "lgb"],
        "CatBoost": ["CatBoostClassifier", "CatBoostRegressor", "catboost"],
        "RandomForest": ["RandomForestClassifier", "RandomForestRegressor"],
        "LogisticRegression": ["LogisticRegression"],
        "LinearRegression": ["LinearRegression"],
        "GradientBoosting": ["GradientBoostingClassifier", "GradientBoostingRegressor"],
    }
    
    detected_model = "Unknown"
    for name, keywords in models.items():
        if any(kw in code for kw in keywords):
            detected_model = name
            break
            
    params = []
    param_matches = re.findall(r"(\w+)\s*=\s*([^,)\s]+)", code)
    
    exclude = {"self", "train", "test", "X", "y", "True", "False", "None", "Path", "pd", "np"}
    for k, v in param_matches:
        if k not in exclude and len(k) > 2:
            ml_keywords = {"depth", "rate", "estimator", "leaf", "alpha", "lambda", "subsample", "colsample"}
            if any(mk in k.lower() for mk in ml_keywords):
                params.append(f"{k}={v}")
                
    if not params:
        param_block = re.search(r"params\s*=\s*({.*?})", code, re.DOTALL)
        if param_block:
            return detected_model, param_block.group(1).strip()
        return detected_model, "N/A"
        
    return detected_model, "\n".join(params)
