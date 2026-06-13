"""
NLI backends for the claim-level benchmark (benchmark.py).

Three backends share one async interface:

    async backend.classify(items) -> {id: verdict}
        items = [{"id", "claim", "premise"}], verdict in {SUPPORT, CONTRADICT, NEI}

- RobertaBackend   : local fine-tuned RoBERTa sequence classifier (no API).
- GPT35NLIBackend  : prompted NLI with gpt-3.5-turbo (resume's "prompted GPT-3.5").
- LLMJudgeBackend  : gpt-3.5-turbo asked to act as a factual-consistency judge.

The two GPT backends require OPENAI_API_KEY; `available()` reports whether a
backend can run so the harness can skip (not fake) it.
"""

import os
import asyncio

from dotenv import load_dotenv

load_dotenv()

SUPPORT, CONTRADICT, NEI = "SUPPORT", "CONTRADICT", "NEI"
GPT35_MODEL = os.environ.get("GPT35_MODEL", "gpt-3.5-turbo")
_CONCURRENCY = int(os.environ.get("BENCHMARK_CONCURRENCY", 8))
_TEMPERATURE = float(os.environ.get("NLI_TEMPERATURE", 0.0))


def parse_label(text: str) -> str:
    """Map a free-text model reply onto SUPPORT / CONTRADICT / NEI by its lead."""
    head = (text or "").strip().upper()
    head = head.lstrip("`*-#:. \n")
    if head.startswith("ENTAIL") or head.startswith("SUPPORT"):
        return SUPPORT
    if head.startswith("CONTRADICT"):
        return CONTRADICT
    if head.startswith("NEUTRAL") or head.startswith("NOT") or head.startswith("NEI"):
        return NEI
    # fall back to a scan if the label is not at the very start
    if "CONTRADICT" in head:
        return CONTRADICT
    if "ENTAIL" in head or "SUPPORT" in head:
        return SUPPORT
    return NEI


# ── Local RoBERTa ────────────────────────────────────────────────────────────────

class RobertaBackend:
    name = "RoBERTa (fine-tuned)"

    def __init__(self, runner):
        # runner: level3.roberta_runner.RobertaNLIRunner (already loaded)
        self._runner = runner

    def available(self) -> bool:
        return self._runner is not None

    async def classify(self, items: list[dict]) -> dict:
        # Local + synchronous; run in a worker thread so the event loop is free.
        def _work():
            out = {}
            for it in items:
                if not it.get("premise"):
                    out[it["id"]] = NEI
                    continue
                out[it["id"]] = self._runner.predict(it["premise"][:2000], it["claim"])
            return out
        return await asyncio.to_thread(_work)


# ── GPT-3.5 backends ─────────────────────────────────────────────────────────────

class _OpenAIBackend:
    """Shared async OpenAI plumbing for the prompted-NLI and judge backends."""

    name = "openai"
    system_prompt = ""
    user_template = ""

    def __init__(self):
        self._key = os.environ.get("OPENAI_API_KEY")
        self._llm = None

    def available(self) -> bool:
        return bool(self._key) and not self._key.startswith("#")

    def _client(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=GPT35_MODEL, api_key=self._key, temperature=_TEMPERATURE,
            )
        return self._llm

    async def _one(self, item, sem):
        from langchain_core.messages import SystemMessage, HumanMessage
        if not item.get("premise"):
            return item["id"], NEI
        prompt = self.user_template.format(
            premise=item["premise"][:1800], claim=item["claim"],
        )
        async with sem:
            try:
                resp = await self._client().ainvoke(
                    [SystemMessage(content=self.system_prompt),
                     HumanMessage(content=prompt)]
                )
                return item["id"], parse_label(resp.content)
            except Exception as e:
                print(f"    [{self.name}] error on id={item['id']}: {e}")
                return item["id"], NEI

    async def classify(self, items: list[dict]) -> dict:
        sem = asyncio.Semaphore(_CONCURRENCY)
        pairs = await asyncio.gather(*(self._one(it, sem) for it in items))
        return dict(pairs)


class GPT35NLIBackend(_OpenAIBackend):
    name = "GPT-3.5 prompted NLI"
    system_prompt = (
        "You are a scientific NLI (Natural Language Inference) system. Given an "
        "abstract (premise) and a claim (hypothesis), decide the relationship. "
        "Respond with exactly one word on the first line: "
        "ENTAILMENT, CONTRADICTION, or NEUTRAL."
    )
    user_template = (
        "Abstract: {premise}\n\nClaim: {claim}\n\n"
        "Label (ENTAILMENT / CONTRADICTION / NEUTRAL):"
    )


class LLMJudgeBackend(_OpenAIBackend):
    name = "GPT-3.5 LLM-judge"
    system_prompt = (
        "You are a meticulous fact-checking judge evaluating whether a scientific "
        "claim is factually consistent with the provided evidence abstract. "
        "Weigh the evidence and return a verdict.\n"
        "Output exactly one word on the first line:\n"
        "SUPPORTED      - the abstract supports the claim\n"
        "CONTRADICTED   - the abstract contradicts the claim\n"
        "NOT_ENOUGH_INFO- the abstract does not settle the claim"
    )
    user_template = (
        "Evidence abstract:\n{premise}\n\nClaim to judge:\n{claim}\n\n"
        "Verdict (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO):"
    )
