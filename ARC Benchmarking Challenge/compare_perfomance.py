from inspect_ai.log import read_eval_log,EvalLog
from inspect_ai.scorer import INCORRECT


DEFAULT_LOG_BASEPATH = 'logs/'

GPT_41_NANO_LOGS_PATH = DEFAULT_LOG_BASEPATH + '2025-07-15T12-32-19-03-00_arc-task_45sJxyW8oiWwU3NszXWgxZ.eval'

GPT_4O_MINI_LOGS_PATH = DEFAULT_LOG_BASEPATH + '2025-07-15T12-30-44-03-00_arc-task_4ELJqEiKCy5qwnJfC97W9x.eval'  

def compare_eval_logs(model1_name: str, model1_log_path: str, model2_name: str, model2_log_path: str):
    model1_logs = read_log(model1_log_path)
    if not model1_logs:
        return
    
    model1_incorrect_questions = get_incorrect_questions(model1_logs)
    print_results(model1_name, model1_logs, model1_incorrect_questions)


    model2_logs = read_log(model2_log_path)
    if not model2_logs:
        return
    
    model2_incorrect_questions = get_incorrect_questions(model2_logs)
    print_results(model2_name, model2_logs, model2_incorrect_questions)

    print(compare_incorrect_questions(model1_name, model1_incorrect_questions, model2_name, model2_incorrect_questions))



def read_log(log_path: str) -> EvalLog | None:
    try:
        log = read_eval_log(log_path)
    except Exception as e:
        print('Error reading log:', e)
        return None

    if log is None:
        print('Log not found')
        return None

    if log.samples is None:
        print('Log has no samples')
        return None
    
    if log.results is None:
        print('Log has no results')
        return None

    return log


def get_incorrect_questions(log: EvalLog) -> list:
    return {s.id: s.input + '\n\t- ' + '\n\t- '.join(s.choices) for s in log.samples if s.scores['choice'].value == INCORRECT}

def print_results(model_name, logs, incorrect_questions):
    print('=' * 50)
    print(f'{model_name} results')
    print('incorrect questions:', f"{len(incorrect_questions)}/{len(logs.samples)}")
    print()

    for metric in logs.results.scores[0].metrics.values():
        print(f"{metric.name}: {metric.value}")

def compare_incorrect_questions(model1_name: str, model1_incorrect_questions: dict, model2_name: str, model2_incorrect_questions: dict):
    model1_only_wrong = {q_id: q_input for q_id, q_input in model1_incorrect_questions.items() if q_id not in model2_incorrect_questions}
    model2_only_wrong = {q_id: q_input for q_id, q_input in model2_incorrect_questions.items() if q_id not in model1_incorrect_questions}
    both_wrong = {q_id: q_input for q_id, q_input in model1_incorrect_questions.items() if q_id in model2_incorrect_questions}

    print('\n' + '=' * 50)
    print('Comparison of Incorrect Questions')
    print(f'Questions only {model1_name} got wrong ({len(model1_only_wrong)}):')
    for q_id, q_input in model1_only_wrong.items():
        print(f'  -{q_input}')

    print(f'\nQuestions only {model2_name} got wrong ({len(model2_only_wrong)}):')
    for q_id, q_input in model2_only_wrong.items():
        print(f'  -{q_input}')

    print(f'\nQuestions both models got wrong ({len(both_wrong)}):')
    for q_id, q_input in both_wrong.items():
        print(f'  -{q_input}')


compare_eval_logs(
    'GPT 4.1 Nano',
    GPT_41_NANO_LOGS_PATH,
    'GPT 4o Mini',
    GPT_4O_MINI_LOGS_PATH
)
