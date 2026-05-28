from core.types import Observation, Task, Step, Action, ActionType, Stage
from coach.coach import Coach


def test_coach():
    # Mock task
    task = Task(id="test", description="desc", metric="auc", metric_higher_better=True, train_path="",
                test_features_path="")

    # 1. Тест определения выполненных пунктов
    history = [
        Step(idx=0, stage=Stage.EDA, action=Action(type=ActionType.CODE, content="df.dtypes"), result="int, float"),
        Step(idx=1, stage=Stage.EDA, action=Action(type=ActionType.CODE, content="df.isnull().sum()"), result="0")
    ]
    obs = Observation(task=task, stage=Stage.EDA, history=history)
    coach = Coach()
    coverage, hints = coach.assess(obs)

    print(f"Coverage after 2 items: {coverage:.4f}")
    assert coverage == 2 / 21
    assert len(hints) == 1
    assert hints[0].item_id == "target_analysis"  # первый незакрытый в EDA после feature_types и missing_values

    # 2. Тест эскалации
    print("\nTesting escalation for 'target_analysis':")
    for i in range(1, 10):
        coverage, hints = coach.assess(obs)
        if hints:
            print(f"Call {i}: Level {hints[0].level}, Text: {hints[0].text[:30]}...")
            if i <= 2:  # так как 1-й вызов уже был выше, это 2, 3 попытки
                assert hints[0].level == 1
            elif i <= 5:  # 4, 5, 6 попытки
                assert hints[0].level == 2
            else:  # 7+ попытки
                assert hints[0].level == 3

    # 3. Тест смены стадии
    history.append(
        Step(idx=2, stage=Stage.EDA, action=Action(type=ActionType.CODE, content="df['target'].value_counts()"),
             result="ok"))
    # Закроем все в EDA (9 штук)
    all_eda_keywords = ["duplicated", "corr", "boxplot", "get_dummies", "distribution", "leakage"]
    for i, kw in enumerate(all_eda_keywords):
        history.append(Step(idx=i + 3, stage=Stage.EDA, action=Action(type=ActionType.CODE, content=kw), result="ok"))

    obs = Observation(task=task, stage=Stage.BASELINE, history=history)
    coverage, hints = coach.assess(obs)
    print(f"\nCoverage after EDA closed: {coverage:.4f}")
    assert coverage == 9 / 21
    assert hints[0].item_id == "validation_split"
    assert hints[0].stage == Stage.BASELINE


if __name__ == "__main__":
    test_coach()
    print("\nAll tests passed!")