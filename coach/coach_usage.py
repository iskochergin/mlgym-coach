from core.types import Observation, Task, Step, Action, ActionType, Stage
from coach.coach import Coach

def example_usage():
    # 1. Инициализация задачи и тренера
    task = Task(
        id="titanic_binary_classification",
        description="Предсказать выживаемость пассажиров Титаника",
        metric="accuracy",
        metric_higher_better=True,
        train_path="train.csv",
        test_features_path="test.csv"
    )
    coach = Coach()

    # 2. Начало работы (пустая история)
    obs = Observation(task=task, stage=Stage.EDA, history=[])
    
    coverage, hints = coach.assess(obs)
    print(f"Начальное покрытие: {coverage}")
    if hints:
        print(f"Первая подсказка: {hints[0].text}")

    # 3. Агент выполнил EDA (проверил типы)
    step1 = Step(
        idx=0,
        stage=Stage.EDA,
        action=Action(type=ActionType.CODE, content="df.info()"),
        result="Column dtypes: int64, object, float64"
    )
    obs.history.append(step1)
    
    coverage, hints = coach.assess(obs)
    print(f"\nПокрытие после первого шага: {coverage:.2f}")
    if hints:
        # Так как 'dtypes' был в результате, пункт 'feature_types' закрыт.
        # Теперь тренер предложит следующий пункт в EDA - 'missing_values'
        print(f"Следующая подсказка: {hints[0].text}")

if __name__ == "__main__":
    example_usage()
