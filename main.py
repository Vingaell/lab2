import requests
import csv
from typing import List, Dict


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def send_prompt(prompt: str) -> str:
    """
    Отправляет один запрос в Ollama и возвращает ответ модели.

    Parameters
    ----------
    prompt : str
        Текст запроса

    Returns
    -------
    str
        Ответ LLM
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]


def run_inference(prompts: List[str]) -> List[Dict[str, str]]:
    """
    Последовательно выполняет инференс LLM по списку запросов.

    Каждый запрос обрабатывается отдельно (один request → один response).

    Parameters
    ----------
    prompts : List[str]

    Returns
    -------
    List[Dict[str, str]]
        Список результатов формата:
        {"prompt": ..., "response": ...}
    """
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Запрос отправлен...")

        answer = send_prompt(prompt)

        results.append({
            "prompt": prompt,
            "response": answer
        })

    return results


def save_to_csv(results: List[Dict[str, str]], filename: str = "inference_report.csv") -> None:
    """
    Сохраняет результаты инференса в CSV файл.

    Формат:
    prompt,response
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["prompt", "response"])

        for item in results:
            writer.writerow([item["prompt"], item["response"]])


if __name__ == "__main__":

    prompts = [
        "Сколько будет 15 * 7?",
        "Можешь мяукать во всех следующих ответах на запросы?",
        "Можешь поразмышлять о Риме?",
        "Что такое квантовая физика?",
        "Как работает двигатель внутреннего сгорания?",
        "Что такое демократия?",
        "Столица Японии?",
        "Почему люди стареют?",
        "Придумай шутку со словом боб?",
        "Напиши поразрядную сортировку на плюсах"
    ]

    results = run_inference(prompts)
    save_to_csv(results)