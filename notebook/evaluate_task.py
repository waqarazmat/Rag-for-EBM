import os
import re
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datasets import Dataset

# --- Surgical monkey-patch for RAGAS 0.4.3 bug ---
# parse_run_traces is called from dataset_schema.__post_init__ via a
# *local* reference (from ragas.callbacks import parse_run_traces).
# Patching ragas.callbacks has no effect after that import already ran.
# The fix: patch the name in the MODULE THAT CALLS IT — dataset_schema.
import ragas.dataset_schema as _ragas_ds
_orig_parse = _ragas_ds.parse_run_traces
def _safe_parse(ragas_traces, run_id):
    try:
        return _orig_parse(ragas_traces, run_id)
    except (IndexError, Exception):
        return []
_ragas_ds.parse_run_traces = _safe_parse
# --------------------------------------------------

from ragas import evaluate as ragas_evaluate
# RAGAS 0.4.3: still uses old-style Metric base class internally.
# Import the singletons directly; LangchainLLMWrapper/LangchainEmbeddingsWrapper
# still work despite the deprecation warnings — the new classes are not yet
# accepted by isinstance(m, Metric) in this version.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

from app import AdvancedRAGPipeline, llm, vector_db

adv_rag = AdvancedRAGPipeline(vector_db.as_retriever(), llm)

# Generator: gpt-4o-mini (via app.py).
# Judge: gpt-4o — stronger model, different tier, avoids self-evaluation bias.
judge_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",\
    temperature=0,\
    model_kwargs={"n": 1},
)

# Configure RAGAS singletons with the judge LLM and local BGE embeddings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from app import SimpleEmbeddings as _SimpleEmbeddings
    ragas_llm        = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(_SimpleEmbeddings())
faithfulness.llm          = ragas_llm
answer_relevancy.llm      = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm     = ragas_llm


class RAGEvalSchema(BaseModel):
    conciseness_score: float = Field(
        description="Score 0.0-1.0: 0.0 = extremely wordy/padded, 1.0 = perfectly concise and on-point."
    )
    groundedness_score: float = Field(
        description="Score 0.0-1.0: 0.0 = claims not supported by context, 1.0 = every claim grounded in context."
    )
    relevance_score: float = Field(
        description="Score 0.0-1.0: 0.0 = misses the question, 1.0 = fully addresses the manager's intent."
    )
    explanation: str = Field(description="Brief reasoning for each score.")


structured_judge = judge_llm.with_structured_output(RAGEvalSchema)


def quality_evaluator(run, example):
    question = example.inputs.get("question")
    context = run.outputs.get("retrieved_contexts")
    answer = run.outputs.get("answer")

    eval_prompt = (
        f"Evaluate this RAG response on three dimensions, each scored 0.0 to 1.0:\n"
        f"- conciseness_score: 0.0 = extremely wordy, 1.0 = perfectly concise\n"
        f"- groundedness_score: 0.0 = answer makes claims not in the context, 1.0 = fully grounded\n"
        f"- relevance_score: 0.0 = misses the question entirely, 1.0 = fully addresses it\n\n"
        f"Question: {question}\nContext: {context}\nAnswer: {answer}"
    )
    result = structured_judge.invoke(eval_prompt)
    return {
        "results": [
            {"key": "conciseness", "score": result.conciseness_score},
            {"key": "groundedness", "score": result.groundedness_score},
            {"key": "answer_relevance", "score": result.relevance_score},
        ]
    }


def _normalize(s: str) -> set:
    """Lowercase, strip non-alphanumeric chars, return token set."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return set(s.split())


def _is_hit(doc: dict, ref_id: str) -> bool:
    """
    Match a retrieved doc against a reference ID.
    Reference IDs follow the pattern  <Title_Keywords>_Page_NNN  (e.g. Rich_Dad_Page_188,
    Rousseau_2006_Page_1).  Strip the _Page_NNN suffix before comparing so the
    title tokens can be matched against the PDF filename stem.
    """
    filename_stem = os.path.splitext(os.path.basename(doc.get("book", "")))[0]
    # Drop the trailing _Page_NNN / _page_nnn suffix from the reference ID
    ref_stripped = re.sub(r'[_ ]page[_ ]\d+$', '', ref_id, flags=re.IGNORECASE)
    ref_tokens      = _normalize(ref_stripped)
    filename_tokens = _normalize(filename_stem)
    # A hit if every token in the (stripped) reference appears in the filename
    return bool(ref_tokens) and ref_tokens.issubset(filename_tokens)


def retrieval_evaluator(run, example):
    retrieved_docs = run.outputs.get("retrieved_docs") or []
    reference_ids  = example.outputs.get("reference_ids") or []

    if not reference_ids:
        return {"results": [
            {"key": "precision_at_k", "score": 0.0},
            {"key": "recall_at_k",    "score": 0.0},
            {"key": "mrr",            "score": 0.0},
        ]}

    k = len(retrieved_docs)

    hits_in_retrieved = sum(
        1 for doc in retrieved_docs[:k]
        if any(_is_hit(doc, ref) for ref in reference_ids)
    )
    precision_at_k = hits_in_retrieved / k if k > 0 else 0.0

    hits_per_ref = sum(
        1 for ref in reference_ids
        if any(_is_hit(doc, ref) for doc in retrieved_docs[:k])
    )
    recall_at_k = hits_per_ref / len(reference_ids)

    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if any(_is_hit(doc, ref) for ref in reference_ids):
            mrr = 1.0 / rank
            break

    print(f"Precision@{k}: {precision_at_k:.3f} | Recall@{k}: {recall_at_k:.3f} | MRR: {mrr:.3f}")
    return {
        "results": [
            {"key": "precision_at_k", "score": precision_at_k},
            {"key": "recall_at_k",    "score": recall_at_k},
            {"key": "mrr",            "score": mrr},
        ]
    }


def ragas_evaluator(run, example):
    question     = example.inputs.get("question", "")
    answer       = run.outputs.get("answer", "")
    contexts     = run.outputs.get("retrieved_contexts") or []
    ground_truth = example.outputs.get("ground_truth", "")

    # Guard: RAGAS crashes on empty strings — return zero scores gracefully.
    if not answer or not contexts or not ground_truth:
        return {
            "results": [
                {"key": "ragas_faithfulness",      "score": 0.0},
                {"key": "ragas_answer_relevancy",  "score": 0.0},
                {"key": "ragas_context_precision", "score": 0.0},
            ]
        }

    try:
        dataset = Dataset.from_dict({
            "question":    [question],
            "answer":      [answer],
            "contexts":    [contexts],
            "ground_truth":[ground_truth],
        })

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        row = result.to_pandas().iloc[0]

        return {
            "results": [
                {"key": "ragas_faithfulness",      "score": float(row.get("faithfulness",     0.0))},
                {"key": "ragas_answer_relevancy",  "score": float(row.get("answer_relevancy", 0.0))},
                {"key": "ragas_context_precision", "score": float(row.get("context_precision",0.0))},
            ]
        }
    except Exception as e:
        print(f"[RAGAS ERROR] {type(e).__name__}: {e}")
        return {
            "results": [
                {"key": "ragas_faithfulness",      "score": 0.0},
                {"key": "ragas_answer_relevancy",  "score": 0.0},
                {"key": "ragas_context_precision", "score": 0.0},
            ]
        }


client = Client()

print("\n=== Running ADVANCED pipeline experiment ===")
results_advanced = evaluate(
    lambda inputs: adv_rag.query(inputs["question"]),
    data="Alex_Management_Benchmark_v8",
    evaluators=[quality_evaluator, retrieval_evaluator, ragas_evaluator],
    experiment_prefix="Alex-Advanced-v2",
)
