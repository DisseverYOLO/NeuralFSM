import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from baselines import COT, COT_SC, direct, llm_debate, self_refine, spp


def _ensure_openai_env() -> None:
    """
    IMPORTANT: Do NOT hardcode keys in code.
    Set these in your shell before running:
      - OPENAI_API_KEY
      - OPENAI_API_BASE (optional; e.g., https://openrouter.ai/api/v1)
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set it as an environment variable "
            "(do not commit keys to the repo)."
        )


def create_prompt(question, correct_answer, wrong_answers):
    """Create a prompt with randomized option order and return (prompt, correct_label)."""
    all_answers = wrong_answers.copy()
    all_answers.append(correct_answer)
    random.shuffle(all_answers)

    correct_position = all_answers.index(correct_answer)

    prompt = f"{question}\n"
    options = ["(a)", "(b)", "(c)", "(d)"]
    for idx, answer in enumerate(all_answers):
        prompt += f"{options[idx]} {answer}\n"

    prompt += "\nFollow the answer format: <|submit|> <fill in the answer's label>, for example: <|submit|> (a)"
    return prompt, options[correct_position]


def process_row(row, method_name):
    """Process one question and return correctness."""
    question = row["Question"]
    correct_answer = row["Correct Answer"]
    wrong_answers = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]

    prompt, correct_label = create_prompt(question, correct_answer, wrong_answers)

    model_answer = ""
    if method_name == "direct":
        try:
            model_answer = direct(prompt)
        except Exception:
            traceback.print_exc()
    elif method_name == "cot":
        model_answer = COT(prompt)
    elif method_name == "cot_sc":
        model_answer = COT_SC(prompt)
    elif method_name == "llm_debate":
        model_answer = llm_debate(prompt)
    elif method_name == "self_refine":
        model_answer = self_refine(prompt)
    elif method_name == "spp":
        model_answer = spp(prompt)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    is_correct = False
    if "<|submit|>" in model_answer:
        try:
            submitted_answer = model_answer.split("<|submit|>")[1].strip()
            is_correct = correct_label.lower() in submitted_answer.lower()
        except Exception:
            is_correct = False

    return is_correct


def evaluate_model(method_name, max_workers=8):
    df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")
    total = len(df)
    correct = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row, method_name): index for index, row in df.iterrows()}
        with tqdm(total=total, desc=f"Evaluating {method_name}") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        correct += 1
                except Exception as e:
                    index = futures[future]
                    print(f"Error processing question {index} with {method_name}: {e}")
                pbar.update(1)

    accuracy = correct / total if total > 0 else 0
    print(f"{method_name} accuracy: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    _ensure_openai_env()
    random.seed(42)
    max_workers = min(8, os.cpu_count() or 8)

    methods = ["cot_sc", "llm_debate", "self_refine", "spp"]
    results = {}
    for method in methods:
        print(f"\nStarting evaluation of {method}...")
        results[method] = evaluate_model(method, max_workers=max_workers)

    print("\n=== Final Results ===")
    for method, accuracy in results.items():
        print(f"{method}: {accuracy * 100:.2f}%")


