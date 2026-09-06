"""
eval.py
-------
Standalone evaluation harness for the RAG pipeline. Run against whatever
is currently indexed in your ChromaDB (./chroma_db by default) using a
labeled question set you write for your own test PDFs.

Reports:
  - Retrieval hit-rate: did a chunk from the expected source document
    make it into the top-k results after reranking?
  - Answer correctness: LLM-as-judge comparison against a reference
    answer you supply (skipped for questions with no reference_answer).
  - Latency per stage: retrieval, rerank, generation, total — averaged
    across the eval set.

Usage:
    python eval.py --eval-set eval_set.json

    # Point at different models/settings if you're not using the defaults:
    python eval.py --eval-set eval_set.json --llm-model llama3.2 \
        --embed-model nomic-embed-text --retrieve-k 20 --rerank-k 5

Writes a full results JSON (eval_results.json by default) alongside a
printed summary table.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import ollama

from vector_store import VectorStore, DEFAULT_PERSIST_DIR
from reranker import Reranker
from rag_pipeline import RAGPipeline


def load_eval_set(path: str) -> List[Dict]:
    with open(path) as f:
        items = json.load(f)
    for i, item in enumerate(items):
        if "question" not in item:
            raise ValueError(f"eval set item {i} is missing a 'question' field")
    return items


def judge_answer(question: str, reference_answer: str, generated_answer: str, llm_model: str) -> bool:
    """
    LLM-as-judge: grades whether the generated answer conveys the same key
    facts as a reference answer you wrote by hand. This is a coarse signal
    (single CORRECT/INCORRECT verdict, no partial credit) — good enough to
    track whether pipeline changes help or hurt, not a substitute for
    spot-checking the actual generated_answer text in the results file.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are grading a RAG system's answer against a reference "
                "answer. Reply with exactly one word: CORRECT if the "
                "generated answer conveys the same key facts as the "
                "reference answer, or INCORRECT if it misses key facts, "
                "contradicts the reference, or is a refusal/non-answer when "
                "the reference has a real answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION: {question}\n\n"
                f"REFERENCE ANSWER: {reference_answer}\n\n"
                f"GENERATED ANSWER: {generated_answer}"
            ),
        },
    ]
    response = ollama.chat(model=llm_model, messages=messages)
    verdict = response["message"]["content"].strip().upper()
    return "INCORRECT" not in verdict and "CORRECT" in verdict


def run_eval(
    eval_set_path: str,
    embed_model: str,
    llm_model: str,
    retrieve_k: int,
    rerank_k: int,
    persist_dir: str,
) -> List[Dict]:
    store = VectorStore(embed_model=embed_model, persist_dir=persist_dir)
    if len(store) == 0:
        raise RuntimeError(
            f"No documents indexed in '{persist_dir}'. Process your test PDFs "
            "through the Streamlit app first, then run this eval against the "
            "same chroma_db."
        )

    reranker = Reranker()
    pipeline = RAGPipeline(
        vector_store=store,
        reranker=reranker,
        llm_model=llm_model,
        retrieve_k=retrieve_k,
        rerank_k=rerank_k,
    )

    items = load_eval_set(eval_set_path)
    results = []

    for item in items:
        result = pipeline.answer_with_metrics(
            item["question"], source_filter=item.get("source_filter")
        )
        retrieved_sources = {c["metadata"].get("source") for c in result["sources"]}
        expected_source = item.get("expected_source")
        hit = expected_source in retrieved_sources if expected_source else None

        correct = None
        if item.get("reference_answer"):
            correct = judge_answer(
                item["question"], item["reference_answer"], result["answer"], llm_model
            )

        row = {
            "id": item.get("id"),
            "question": item["question"],
            "expected_source": expected_source,
            "retrieved_sources": sorted(s for s in retrieved_sources if s),
            "retrieval_hit": hit,
            "answer_correct": correct,
            "generated_answer": result["answer"],
            "timing": result["timing"],
        }
        results.append(row)

        hit_str = "n/a" if hit is None else ("HIT" if hit else "MISS")
        correct_str = "n/a" if correct is None else ("CORRECT" if correct else "INCORRECT")
        print(
            f"[{item.get('id', '?')}] retrieval={hit_str:5s}  "
            f"answer={correct_str:9s}  total={result['timing']['total']:.2f}s"
        )

    return results


def summarize(results: List[Dict]) -> Dict:
    n = len(results)

    scored_hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    hit_rate = sum(scored_hits) / len(scored_hits) if scored_hits else None

    graded = [r for r in results if r["answer_correct"] is not None]
    correctness = sum(r["answer_correct"] for r in graded) / len(graded) if graded else None

    def avg(stage):
        return sum(r["timing"][stage] for r in results) / n if n else 0.0

    summary = {
        "num_questions": n,
        "retrieval_hit_rate": hit_rate,
        "retrieval_hit_rate_n": len(scored_hits),
        "answer_correctness": correctness,
        "answer_correctness_n": len(graded),
        "avg_retrieval_sec": round(avg("retrieval"), 3),
        "avg_rerank_sec": round(avg("rerank"), 3),
        "avg_generation_sec": round(avg("generation"), 3),
        "avg_total_sec": round(avg("total"), 3),
    }

    print("\n=== Evaluation Summary ===")
    print(f"Questions evaluated:   {n}")
    if hit_rate is not None:
        print(f"Retrieval hit-rate:    {hit_rate:.1%}  ({len(scored_hits)} questions with an expected_source)")
    else:
        print("Retrieval hit-rate:    N/A (no expected_source fields in eval set)")
    if correctness is not None:
        print(f"Answer correctness:    {correctness:.1%}  ({len(graded)} questions with a reference_answer)")
    else:
        print("Answer correctness:    N/A (no reference_answer fields in eval set)")
    print(f"Avg retrieval time:    {summary['avg_retrieval_sec']:.2f}s")
    print(f"Avg rerank time:       {summary['avg_rerank_sec']:.2f}s")
    print(f"Avg generation time:   {summary['avg_generation_sec']:.2f}s")
    print(f"Avg total latency:     {summary['avg_total_sec']:.2f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a labeled question set.")
    parser.add_argument("--eval-set", default="eval_set.json", help="Path to the labeled question set JSON.")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="llama3.2")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    if not Path(args.eval_set).exists():
        raise SystemExit(
            f"Eval set not found: {args.eval_set}\n"
            "Copy eval_set.example.json to eval_set.json and fill it in with "
            "questions about your own test PDFs first."
        )

    results = run_eval(
        args.eval_set, args.embed_model, args.llm_model, args.retrieve_k, args.rerank_k, args.persist_dir
    )
    summary = summarize(results)

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()