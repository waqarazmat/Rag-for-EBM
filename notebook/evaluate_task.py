import os
from langsmith import Client, evaluate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
# importing from app.py to use the same RAG pipeline and vector DB setup for evaluation
from app import AdvancedRAGPipeline, llm, vector_db 

# 1. Setup Alex & Judge
adv_rag = AdvancedRAGPipeline(vector_db.as_retriever(), llm)
judge_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 2. Schema for LLM Judge (Conciseness 0-1)
class RAGEvalSchema(BaseModel):
    conciseness_score: int = Field(description="Score 0-1: 0 is wordy, 1 is perfectly sharp.")
    groundedness_score: float = Field(description="Score 0-1: Is the answer strictly from context?")
    relevance_score: float = Field(description="Score 0-1: Does it answer the manager's intent?")
    explanation: str = Field(description="Reasoning for scores.")

structured_judge = judge_llm.with_structured_output(RAGEvalSchema)

# 3. LLM Quality Evaluator
def quality_evaluator(run, example):
    question = example.inputs.get("question")
    context = run.outputs.get("retrieved_contexts")
    answer = run.outputs.get("answer")

    eval_prompt = f"Evaluate Alex's response for: Question: {question}, Context: {context}, Answer: {answer}. Conciseness scale 1-5, others 0-1."
    result = structured_judge.invoke(eval_prompt)
    return {
        "results": [
            {"key": "conciseness", "score": result.conciseness_score},
            {"key": "groundedness", "score": result.groundedness_score},
            {"key": "answer_relevance", "score": result.relevance_score}
        ]
    }

def retrieval_evaluator(run, example):
    retrieved_docs = run.outputs.get("retrieved_docs",) or []
    reference_ids = example.outputs.get("reference_ids",) or []
    
    hits = 0
    for ref in reference_ids:
        # show reference keywords in filename for loose matching, ignore short words
        ref_keywords = [w.lower() for w in ref.split("_") if len(w) > 2]
        
        for doc in retrieved_docs:
            filename = os.path.basename(doc['book']).lower()
            # if any keyword from reference is in the retrieved doc filename, count as hit (loose check)
            if any(kw in filename for kw in ref_keywords):
                hits += 1
                break

    precision = hits / len(retrieved_docs) if retrieved_docs else 0
    recall = hits / len(reference_ids) if reference_ids else 0
    
    print(f"--- LOOSE CHECK --- Hits: {hits} | Ref: {reference_ids}")
    return {
        "results": [
            {"key": "retrieval_precision", "score": precision},
            {"key": "retrieval_recall", "score": recall}
        ]
    }

client = Client()
results = evaluate(
    lambda inputs: adv_rag.query(inputs["question"]),
    data="Alex_Management_Benchmark_v6", 
    evaluators=[quality_evaluator, retrieval_evaluator], # both quality and retrieval evaluation functions are there to be run for each example
    experiment_prefix="Alex-Full-Metrics-Final"
)