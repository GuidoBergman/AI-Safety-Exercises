from arc import arc_task, record_to_sample
from inspect_ai.model import get_model, GenerateConfig
from inspect_ai.solver import solver, TaskState, Generate, system_message
from inspect_ai import Task, task
from inspect_ai.scorer import scorer, mean, stderr, Target, Score
from inspect_ai.dataset import hf_dataset
import torch


# Prompt from https://github.com/UKGovernmentBEIS/inspect_ai/blob/c181cd9cd8e75eee8913f5dafca53e7daa16ff0c/src/inspect_ai/solver/_multiple_choice.py#L83
SINGLE_ANSWER_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


NUM_TOP_LOGPROBS = 20


@task
def arc_task_with_logprobs(fewshot: int | None = None) -> Task:
    # The fewshot logic is similar to https://inspect.aisi.org.uk/tutorial.html#sec-gsm8k
    if fewshot:
        fewshots=hf_dataset(
            path="allenai/ai2_arc",
            name="ARC-Challenge",
            split="train",
            sample_fields=record_to_sample,
            limit=fewshot,
            seed=0,
            shuffle=True,
        )
        fewshot_prompt = "\n\n".join([sample_to_fewshot(sample) for sample in fewshots]) 
    else:
        fewshot_prompt = None

        

    solver = multiple_choice_with_logprobs(fewshot_prompt = fewshot_prompt)
    return arc_task(solver=solver, scorer=choice_with_logprobs())
    

@solver
def multiple_choice_with_logprobs(prompt: str | None = SINGLE_ANSWER_TEMPLATE, fewshot_prompt: str | None = None):
    model = get_model()
    config = GenerateConfig(logprobs=True, top_logprobs=NUM_TOP_LOGPROBS)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        formatted_choices, letters = format_choices(state.choices)
        formatted_prompt = prompt.format(
            question=state.input,
            choices=formatted_choices,
            letters=",".join(letters)
        )
        formatted_prompt = fewshot_prompt + '\n\n' + formatted_prompt if fewshot_prompt else formatted_prompt
        output = await model.generate(formatted_prompt, config=config)
        state.metadata['logprobs'] = output.choices[0].logprobs
        state.output = output
        state.messages.append(output.message)
        return state
    
    return solve

@scorer(metrics=[mean(), stderr()])
def choice_with_logprobs():

    async def score(state: TaskState, target: Target):
        logprobs = state.metadata['logprobs']
        logprobs_last_token = logprobs.content[-1].top_logprobs
        logprobs_tensor = torch.tensor([entry.logprob for entry in logprobs_last_token])
        probs = torch.softmax(logprobs_tensor, dim=0)
        target = target.text.lower()
        correct_answer_probability = 0
        dic_probabilities = {}
        for i, entry in enumerate(logprobs_last_token):
            # Maybe there is more than token with the correct answer (e.g., 'A' and 'A\n')
            if target in entry.token.lower():
                correct_answer_probability += probs[i].item()

            dic_probabilities[entry.token] = probs[i].item()
        
        return Score(
            value = correct_answer_probability,
            answer=state.output.completion,
            metadata = dic_probabilities
        )

    return score



def sample_to_fewshot(sample):
    formatted_choices, letters = format_choices(sample.choices)
    return (
        f"{sample.input}\nChoices:\n"
        + f"{formatted_choices}\n"
        + f"ANSWER: {sample.target}"
    )


def format_choices(choices: list) -> tuple[str, list]:
     letters = [chr(65+i) for i in range(len(choices))]

     formatted_choices = "\n".join(
                [f"{letter}) {choice if isinstance(choice, str) else choice.value}" for letter, choice in zip(letters, choices)]
     )

     return formatted_choices, letters