from collections import Counter
import os
from langchain_openai import ChatOpenAI
from uqlm import LongTextUQ
from bm25_retriever import CorpusRetriever
from dotenv import load_dotenv

load_dotenv()

class UQLMRunner:
    def __init__(self, claims: list, corpus: dict, retriever: CorpusRetriever):
        self.claims = claims
        self.corpus = corpus
        self.retriever = retriever
        self.label_template_prompt = """\
            Based ONLY on the following scientific abstract, evaluate the claim.

            Abstract: {abstract}

            Claim: {claim}

            Begin your response with exactly one label — SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO — then explain your reasoning.\
            """

        self.label_template_prompt_no_abstract = """\
            Evaluate the following scientific claim based on general evidence.

            Claim: {claim}

            Begin with exactly one label: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO. Then explain.\
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.7)
        self.luq = LongTextUQ(llm=self.llm, scorers=["entailment"], response_refinement=False)
        self.prompts = []
        self.relevance_gate_threshold = float(os.environ.get("RELEVANCE_GATE_THRESHOLD"))
        self.uqlm_num_responses = int(os.environ.get("UQLM_NUM_RESPONSES"))

    def parse_label(self, text: str) -> str:
        t = text.strip().upper()
        head = t[:int(os.environ.get("LABEL_SCAN_CHARS"))]

        if head.startswith("NOT_ENOUGH_INFO") or head.startswith("NOT ENOUGH"):
            return "NEI"
        if head.startswith("SUPPORTED"):
            return "SUPPORT"
        if head.startswith("CONTRADICTED"):
            return "CONTRADICT"

        if "NOT_ENOUGH_INFO" in head or "NOT ENOUGH INFO" in head:
            return "NEI"
        if "CONTRADICTED" in head:
            return "CONTRADICT"
        if "SUPPORTED" in head:
            return "SUPPORT"

        return "NEI"
    
    def create_prompts(self):
        for c in self.claims:
                # First try oracle cited_doc_ids, then fall back to open retrieval
                abstract_text = None
                for cid in c.get("cited_doc_ids", []):
                    if cid in self.corpus:
                        abstract_text = " ".join(self.corpus[cid]["abstract"])
                        break

                if abstract_text is None:
                    top_docs = self.retriever.retrieve(c["claim"], k=1)
                    if top_docs and top_docs[0]["bm25_score"] > self.relevance_gate_threshold:
                        abstract_text = " ".join(top_docs[0]["abstract"])

                if abstract_text:
                    prompt = self.label_template_prompt.format(abstract=abstract_text, claim=c["claim"])
                else:
                    prompt = self.label_template_prompt_no_abstract.format(claim=c["claim"])

                self.prompts.append(prompt)

    async def run_uqlm(self):
        self.create_prompts()
        raw = await self.luq.generate_and_score(prompts=self.prompts, num_responses=self.uqlm_num_responses)
        df  = raw.to_df()

        results = []
        for i, (item, (_, row)) in enumerate(zip(self.claims, df.iterrows())):
            # Collect all sampled responses for label parsing
            all_responses = []
            primary = row.get("response", "")
            if primary and isinstance(primary, str):
                all_responses.append(primary)
            sampled = row.get("sampled_responses", [])
            if isinstance(sampled, list):
                all_responses.extend(str(r) for r in sampled if r)

            # Majority vote
            parsed_labels = [self.parse_label(r) for r in all_responses]
            if parsed_labels:
                count      = Counter(parsed_labels)
                verdict, n = count.most_common(1)[0]
                confidence = round(n / len(parsed_labels), 3)
            else:
                verdict, confidence = "NEI", 0.0
                count = Counter()

            # NLI entailment as consistency measure
            try:
                cd = row["claims_data"] or []
            except (KeyError, TypeError):
                cd = []

            if cd:
                scores = [c.get("entailment", c.get("score", 0.5)) for c in cd]
                avg_entailment = round(sum(scores) / len(scores), 4)
            else:
                avg_entailment = 0.5

            print(
                f"  [{i+1:2d}/{len(self.claims)}] vote={verdict}({confidence:.0%}) "
                f"entail={avg_entailment:.3f}  ({item['claim'][:45]})"
            )
            results.append({
                "id":                  item["id"],
                "uqlm_verdict":        verdict,
                "uqlm_confidence":     confidence,
                "uqlm_avg_entailment": avg_entailment,
                "uqlm_label_counts":   dict(count),
                "uqlm_claims_data":    cd,
            })

        return results




    

