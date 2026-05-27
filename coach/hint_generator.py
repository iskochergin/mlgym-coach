from typing import Dict, Optional, List

class HintGenerator:
    """Генератор подсказок по уровням (L0-L3)"""
    
    HINT_TEMPLATES = {
        "eda": {
            "L1": "Обрати внимание на распределение целевой переменной",
            "L2": "Ты посмотрел на пропущенные значения и выбросы?",
            "L3": "Используй df.describe() и df.isnull().sum()"
        },
        "feature_engineering": {
            "L1": "Подумай о взаимодействии признаков",
            "L2": "Закодировал ли ты категориальные переменные?",
            "L3": "Примени OneHotEncoder для категориальных признаков"
        },
        "baseline_model": {
            "L1": "Начни с простой модели для базовой линии",
            "L2": "Попробуй LogisticRegression или RandomForest",
            "L3": "from sklearn.linear_model import LogisticRegression"
        },
        "model_selection": {
            "L1": "Сравни несколько моделей",
            "L2": "Используй кросс-валидацию для выбора",
            "L3": "Примени cross_val_score с cv=5"
        },
        "tuning": {
            "L1": "Оптимизируй гиперпараметры выбранной модели",
            "L2": "Используй GridSearchCV или RandomizedSearchCV",
            "L3": "param_grid = {'n_estimators': [100, 200]}"
        }
    }
    
    def __init__(self):
        self.hint_levels = {stage: 0 for stage in self.HINT_TEMPLATES}
        self.stuck_counter = {stage: 0 for stage in self.HINT_TEMPLATES}
    
    def generate(self, stage: str, action: Dict, 
                 result: Dict, coverage: Dict) -> Optional[Dict]:
        """Сгенерировать подсказку для текущей стадии"""
        if coverage.get(stage, False):
            return None
        
        # Проверка, застрял ли агент
        if not self._is_progress(action, result):
            self.stuck_counter[stage] += 1
        else:
            self.stuck_counter[stage] = 0
        
        # Эскалация уровня подсказки
        if self.stuck_counter[stage] > 2:
            self.hint_levels[stage] = min(self.hint_levels[stage] + 1, 3)
            self.stuck_counter[stage] = 0
        
        level = self.hint_levels[stage]
        if level == 0:
            return None
        
        hint_text = self.HINT_TEMPLATES.get(stage, {}).get(f"L{level}", "")
        
        return {
            "text": hint_text,
            "level": f"L{level}",
            "stage": stage
        }
    
    def _is_progress(self, action: Dict, result: Dict) -> bool:
        """Проверить, был ли прогресс в действии"""
        return "error" not in result and bool(result)