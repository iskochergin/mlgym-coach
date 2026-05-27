from typing import Dict, Optional
from env.ml_gym import MLGymEnv
from agent.llm_agent import LLMAgent
from agent.scaffold_agent import ScaffoldAgent
from tasks.base import Task
import uuid

class EpisodeRunner:
    """Запуск эпизода взаимодействия агента со средой"""
    
    def __init__(self, task: Task, agent_type: str = "baseline",
                 config: Optional[Dict] = None):
        self.task = task
        self.agent_type = agent_type
        self.config = config or {}
        
        self.env = MLGymEnv(
            task_config=task.to_dict(),
            token_budget=self.config.get("token_budget", 10000)
        )
        
        if agent_type == "baseline":
            self.agent = LLMAgent(model=self.config.get("model", "gpt-3.5-turbo"))
        elif agent_type == "scaffold":
            self.agent = ScaffoldAgent(
                model=self.config.get("model", "gpt-4"),
                use_tree_search=self.config.get("use_tree_search", True)
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run(self, seed: Optional[int] = None) -> Dict:
        """Запустить эпизод"""
        episode_id = f"ep_{self.agent_type}_{seed or uuid.uuid4().hex[:6]}"
        
        self.env.reset(seed=seed)
        self.agent.reset()
        
        done = False
        while not done:
            # Агент принимает решение
            action = self.agent.decide(
                stage=self.env.STAGES[self.env.current_stage],
                state=self.env.get_state(),
                history=self.env.steps
            )
            
            # Среда выполняет действие
            observation, reward, done, info = self.env.step(action)
            
            # Агент получает обратную связь
            self.agent.observe(observation, reward, info)
        
        # Сохранение результата
        final_score = self.env.evaluate_final()
        self.env.save_episode_result(
            episode_id=episode_id,
            agent_type=self.agent_type,
            final_score=final_score
        )
        
        return {
            "episode_id": episode_id,
            "final_score": final_score,
            "total_tokens": self.env.tokens_used,
            "steps": len(self.env.steps)
        }