from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice, Scorer
from inspect_ai.solver import Solver, multiple_choice

NUM_SAMPLES = 200
ANSWER_KEY_CONVERTION_DICT = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D'
}

@task
def arc_task(solver: Solver | None = None, scorer: Scorer | None = None) -> Task:
    return Task(
        dataset=hf_dataset(
            path="allenai/ai2_arc",
            name="ARC-Challenge",
            split="test",
            sample_fields=record_to_sample,
            limit=NUM_SAMPLES,
            seed=0,
            shuffle=True,
        ),
        solver=solver or multiple_choice(),
        scorer=scorer or choice(),
    )



def record_to_sample(record: dict[str, Any]) -> Sample:
    input=record["question"]
    choices = record["choices"]['text']
    target = record["answerKey"]
    # Some questions have numbers insted of letters as answer key, so we convert them to keep constency with inspect templates
    if target in ANSWER_KEY_CONVERTION_DICT:
        target = ANSWER_KEY_CONVERTION_DICT[target]


    return Sample(
        input=input, choices=choices, target=target
    )