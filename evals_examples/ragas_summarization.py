from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import SummarizationScore


DEFAULT_SUMMARIZATION_MODEL = "llama3-groq-tool-use:8b"
"""Default Ollama model used to summarize the text."""

DEFAULT_JUDGE_MODEL = "mistral:instruct"


def evaluate(
    text: str,
    reference_summary: str,
    *,
    summarization_model: str = DEFAULT_SUMMARIZATION_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, float | str]:
    """Summarizes *text*

    Parameters
    ----------
    text:
        The complete text that we'd like to summarize.
    summarization_model:
        Ollama model name used to perform the summarization.

    Returns
    -------
    dict[str, float | str]
        A mapping with the original ``text``, the generated ``summary``
        and the computed ``score``.
    """

    prompt = [
        SystemMessage(
            content="Summarize the input text in detail. Do not include any headings, introductions, explanations, or extra commentary. Return only the summary text."
        ),
        HumanMessage(content=text),
    ]
    summarizer = ChatOllama(model=summarization_model)
    response = summarizer.invoke(prompt)

    sample = SingleTurnSample(
        response=response.content, reference_contexts=[reference_summary]
    )

    judge_llm = OllamaLLM(model=judge_model)
    critic = SummarizationScore(llm=LangchainLLMWrapper(langchain_llm=judge_llm))

    score = critic.single_turn_score(sample)

    return {"input": text, "summary": response.content, "score": score}


if __name__ == "__main__":
    text = (
        "Linux was already established in 2004, but it was fragmented into proprietary and unsupported "
        "community editions, and free software was not a part of everyday life for most computer users."
        "That’s when Mark Shuttleworth gathered a small team of Debian developers who together founded"
        "Canonical and set out to create an easy-to-use Linux desktop called Ubuntu."
        "The mission for Ubuntu is both social and economic. First, we deliver the world’s free software,"
        "freely, to everybody on the same terms. Whether you are a student in India or a global bank, you"
        "can download and use Ubuntu free of charge. Second, we aim to cut the cost of professional services"
        "- support, management, maintenance, operations - for people who use Ubuntu at scale, through a portfolio"
        "of services provided by Canonical which ultimately fund the improvement of the platform."
    )

    reference_summary = (
        "In 2004, Linux was fragmented and not widely accessible to everyday users. Mark Shuttleworth and a team"
        "of Debian developers founded Canonical to create Ubuntu, a user-friendly Linux desktop. Ubuntu's mission"
        "is both social and economic: to provide free software to everyone equally and to reduce the cost of"
        "large-scale professional services through Canonical’s offerings, which help fund ongoing platform development."
    )
    print(evaluate(text=text, reference_summary=reference_summary))
