import unittest
from core.types import Observation, Task, Step, Action, ActionType, Stage
from coach.coach import Coach


class TestCoachTrajectories(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            id="test_task",
            description="test",
            metric="mae",
            metric_higher_better=False,
            train_path="",
            test_features_path=""
        )
        self.coach = Coach()

    def test_good_trajectory(self):
        """Агент последовательно выполняет пункты, покрытие растет."""
        history = []

        # Шаг 1: Проверка типов (feature_types)
        history.append(Step(0, Stage.EDA, Action(ActionType.CODE, "df.dtypes"), "int64"))
        obs = Observation(self.task, Stage.EDA, history)
        cov1, hints1 = self.coach.assess(obs)
        self.assertGreater(cov1, 0)
        self.assertEqual(hints1[0].item_id, "missing_values")

        # Шаг 2: Проверка пропусков (missing_values)
        history.append(Step(1, Stage.EDA, Action(ActionType.CODE, "df.isna().sum()"), "0"))
        obs = Observation(self.task, Stage.EDA, history)
        cov2, hints2 = self.coach.assess(obs)
        self.assertGreater(cov2, cov1)
        self.assertEqual(hints2[0].item_id, "target_analysis")

    def test_lazy_trajectory(self):
        """Агент игнорирует пункт, подсказки эскалируются (L1 -> L2 -> L3)."""
        history = []
        # Агент сделал что-то не по чеклисту (например, просто вывел голову таблицы)
        history.append(Step(0, Stage.EDA, Action(ActionType.CODE, "df.head()"), "some data"))
        obs = Observation(self.task, Stage.EDA, history)

        # 1-3 попытки: L1
        for i in range(3):
            cov, hints = self.coach.assess(obs)
            self.assertEqual(hints[0].item_id, "feature_types")
            self.assertEqual(hints[0].level, 1)

        # 4-6 попытки: L2
        for i in range(3):
            cov, hints = self.coach.assess(obs)
            self.assertEqual(hints[0].item_id, "feature_types")
            self.assertEqual(hints[0].level, 2)

        # 7+ попытки: L3
        cov, hints = self.coach.assess(obs)
        self.assertEqual(hints[0].item_id, "feature_types")
        self.assertEqual(hints[0].level, 3)


if __name__ == "__main__":
    unittest.main()