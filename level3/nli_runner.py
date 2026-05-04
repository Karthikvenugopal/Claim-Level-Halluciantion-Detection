
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from bm25_retriever import CorpusRetriever
from dotenv import load_dotenv

load_dotenv()

class NLIRunner:
    def __init__(self, claims: list, corpus: dict, retriever: CorpusRetriever):
        self.claims = claims
        self.corpus = corpus
        self.retriever = retriever
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.nli_temp = float(os.environ.get("NLI_TEMPERATURE"))
        self.nli_system_prompt = (
            "You are a scientific NLI (Natural Language Inference) system. "
            "Given an abstract and a claim, output exactly one word on the first line: "
            "ENTAILMENT, CONTRADICTION, or NEUTRAL. Then briefly explain."
        )

        self.nli_user_prompt = """\
            Abstract: {abstract}

            Claim: {claim}

            Label (ENTAILMENT / CONTRADICTION / NEUTRAL):
            """
        self.nli_map = {"ENTAILMENT": "SUPPORT", "CONTRADICTION": "CONTRADICT", "NEUTRAL": "NEI"}
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_key, temperature=self.nli_temp)
        self.results = []

    async def run_nli(self):
        for i, item in enumerate(self.claims):
            # Resolve abstract — oracle first, BM25 fallback
            abstract_text = None
            for cid in item.get("cited_doc_ids", []):
                if cid in self.corpus:
                    abstract_text = " ".join(self.corpus[cid]["abstract"])
                    break
            if abstract_text is None:
                top = self.retriever.retrieve(item["claim"], k=1)
                if top:
                    abstract_text = " ".join(top[0]["abstract"])

            if not abstract_text:
                verdict, raw = "NEI", "NO_ABSTRACT"
            else:
                prompt = self.nli_user_prompt.format(abstract=abstract_text[:1500], claim=item["claim"])
                try:
                    resp = await self.llm.ainvoke([SystemMessage(content=self.nli_system_prompt),
                                            HumanMessage(content=prompt)])
                    raw  = resp.content.strip()
                    head = raw.upper()[:20]
                    if head.startswith("ENTAILMENT"):
                        verdict = "SUPPORT"
                    elif head.startswith("CONTRADICTION"):
                        verdict = "CONTRADICT"
                    else:
                        verdict = "NEI"
                except Exception as e:
                    print(f"  [{i+1}] NLI error: {e}")
                    verdict, raw = "NEI", f"ERROR: {e}"

            print(f"  [{i+1:2d}/{len(self.claims)}] nli={verdict}  ({item['claim'][:55]})")
            self.results.append({"id": item["id"], "nli_verdict": verdict, "nli_raw": raw})

        return self.results


