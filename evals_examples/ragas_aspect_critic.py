from __future__ import annotations

from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic

DEFAULT_TRANSLATION_MODEL = "llama3-groq-tool-use:8b"
"""Default Ollama model used to generate the French translation."""

DEFAULT_JUDGE_MODEL = "mistral:instruct"
"""Default Ollama model used by AspectCritic for evaluation."""


def evaluate(
    sentence: str,
    *,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    strictness: int = 3,
) -> Dict[str, float | str]:
    """Translate *sentence* into French and score it with AspectCritic.

    Parameters
    ----------
    sentence:
        English input sentence.
    translation_model:
        Ollama model name used to perform the translation.
    judge_model:
        Ollama model name used internally by Ragas to judge the translation.
    strictness:
        Strictness level for :class:`~ragas.metrics.AspectCritic`.

    Returns
    -------
    Dict[str, float | str]
        A mapping with the original user *sentence*, the generated
        ``translation`` and the computed ``score``.
    """

    # 1. Translate the input sentence.
    prompt = [
        SystemMessage(
            content="You're a helpful assistant that translates from English to French. Translate the user sentence.",
        ),
        HumanMessage(content=sentence),
    ]
    translator = ChatOllama(model=translation_model)
    response = translator.invoke(prompt)
    translation: str = response.content

    # 2. Prepare the Ragas sample.
    sample = SingleTurnSample(user_input=sentence, response=translation)

    # 3. Build the evaluator LLM wrapper expected by Ragas.
    judge_llm = OllamaLLM(model=judge_model)
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    # 4. Configure AspectCritic.
    critic = AspectCritic(
        name="correctness",
        llm=evaluator_llm,
        definition=(
            "You are given an English sentence (INPUT) and its supposed French translation (RESPONSE). "
            "Judge the RESPONSE strictly on these criteria:\n"
            "1. Meaning is preserved (no omissions or additions)\n"
            "2. Natural French grammar and wording.\n"
            "3. No extra words, comments or explanations."
        ),
        strictness=strictness,
    )

    # 5. Compute the score.
    score = critic.single_turn_score(sample)

    return {
        "input": sentence,
        "translation": translation,
        "score": score,
    }


if __name__ == "__main__":
    print(evaluate("Hello, how are you?"))
