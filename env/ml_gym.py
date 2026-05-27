from typing import Dict, Any, Tuple
from env.base import BaseEnv
from env.sandbox import CodeSandbox
from env.validators import ActionValidator
from coach.hint_generator import HintGenerator
import json
from pathlib import Path

class MLGymEnv(BaseEnv):
    """ML Gym среда с 5 стадиями и скрытым чек-листом"""
    
    STAGES = ["eda", "feature_engineering", "baseline_model", 
              "model_selection", "tuning"]
    
    def __init__(self, task_config: Dict, token_budget: int = 10000):
        super().__init__()
        self.task_config = task_config
        self.token_budget = token_budget
        self.sandbox = CodeSandbox()
        self.validator = ActionValidator()
        self.hint_generator = HintGenerator()
        
        self.current_stage = 0
        self.tokens_used = 0
        self.steps = []
        self.checklist_coverage = {stage: False for stage in self.STAGES}
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Выполнить шаг агента"""
        stage = self.STAGES[self.current_stage]
        
        # Валидация действия
        is_valid, error_msg = self.validator.validate(action, stage)
        
        # Выполнение кода в песочнице
        if is_valid and "code" in action:
            result = self.sandbox.execute(action["code"])
            score = self._calculate_validation_score(result)
        else:
            result = {"error": error_msg} if not is_valid else {}
            score = 0.0
        
        # Генерация подсказки если нужно
        hint = self.hint_generator.generate(
            stage=stage,
            action=action,
            result=result,
            coverage=self.checklist_coverage
        )
        
        # Обновление состояния
        step_info = {
            "stage": stage,
            "action": action,
            "env_response": result,
            "tokens_used": action.get("tokens", 0),
            "valid_score": score,
            "hint": hint
        }
        self.steps.append(step_info)
        self.tokens_used += action.get("tokens", 0)
        
        # Проверка перехода на следующую стадию
        if self._should_advance_stage(action, result):
            self.checklist_coverage[stage] = True
            self.current_stage = min(self.current_stage + 1, len(self.STAGES) - 1)
        
        done = self.tokens_used >= self.token_budget or stage == "tuning"
        
        return result, score, done, {"stage": stage, "hint": hint}
    
    def save_episode_result(self, episode_id: str, agent_type: str, 
                           final_score: float, output_dir: str = "reports"):
        """Сохранить результат эпизода в JSON"""
        Path(output_dir).mkdir(exist_ok=True)
        
        result = {
            "episode_id": episode_id,
            "task_name": self.task_config["name"],
            "agent_type": agent_type,
            "seed": self.task_config.get("seed", 42),
            "final_score": final_score,
            "total_tokens": self.tokens_used,
            "stages_order": self.STAGES,
            "checklist_coverage": self.checklist_coverage,
            "steps": self.steps
        }
        
        output_path = Path(output_dir) / f"{episode_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        return output_path