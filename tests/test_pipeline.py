import sys
import os
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import TextCleaner, DocumentChunker, DocumentLoader, DataPipeline
from utils.evaluation import RetrievalEvaluator, ResponseEvaluator


def test_text_cleaner():
    cleaner = TextCleaner()

    raw = "  This   has   extra    spaces   and\n\n\n\nnewlines  "
    cleaned = cleaner.clean(raw)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned

    financial = "$ 1,234,567 revenue in Q4 2024 for FY 24"
    normalized = cleaner.normalize_financial(financial)
    assert "$1" in normalized or "1234567" in normalized

    print("[PASS] TextCleaner")


def test_document_chunker():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence is a bit longer to push over the limit."
    chunks = chunker.chunk_text(text, {"source": "test"})

    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(c["source"] == "test" for c in chunks)

    for chunk in chunks:
        assert chunk["char_count"] == len(chunk["text"])

    print(f"[PASS] DocumentChunker - produced {len(chunks)} chunks")


def test_data_pipeline():
    pipeline = DataPipeline()
    text = (
        "Apple Inc reported revenue of $383 billion in fiscal year 2024. "
        "The iPhone segment generated $200 billion in revenue. "
        "Services revenue reached a record $96 billion. "
        "Mac revenue was $29 billion driven by M3 chip adoption. "
        "Operating margin expanded to 32 percent year over year."
    )
    df = pipeline.process_single_text(text, "test_doc")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "text" in df.columns
    assert "text_length" in df.columns
    assert "word_count" in df.columns

    print(f"[PASS] DataPipeline - {len(df)} chunks, columns: {list(df.columns)}")


def test_retrieval_evaluator():
    evaluator = RetrievalEvaluator()

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc6"]

    p_at_3 = evaluator.precision_at_k(retrieved, relevant, k=3)
    assert 0 <= p_at_3 <= 1
    assert abs(p_at_3 - 2/3) < 0.01

    r_at_5 = evaluator.recall_at_k(retrieved, relevant, k=5)
    assert 0 <= r_at_5 <= 1
    assert abs(r_at_5 - 2/3) < 0.01

    f1_at_3 = evaluator.f1_at_k(retrieved, relevant, k=3)
    assert 0 <= f1_at_3 <= 1

    mrr_score = evaluator.mrr(retrieved, relevant)
    assert mrr_score == 1.0

    map_score = evaluator.average_precision(retrieved, relevant)
    assert 0 <= map_score <= 1

    print(f"[PASS] RetrievalEvaluator - P@3={p_at_3:.3f}, R@5={r_at_5:.3f}, MRR={mrr_score:.3f}")


def test_response_evaluator():
    evaluator = ResponseEvaluator()

    response = "Apple reported revenue of $383 billion with strong iPhone sales and growing services."
    query = "What was Apple's revenue?"
    context = [
        "Apple Inc reported total revenue of $383.3 billion for fiscal year 2024.",
        "iPhone revenue was $200.6 billion with services reaching $96.2 billion.",
    ]

    relevance = evaluator.context_relevance(response, context)
    assert 0 <= relevance <= 1

    faithfulness = evaluator.faithfulness_score(response, context)
    assert 0 <= faithfulness <= 1

    completeness = evaluator.response_completeness(response, query)
    assert 0 <= completeness <= 1

    metrics = evaluator.evaluate_response(response, query, context)
    assert "quality_score" in metrics
    assert 0 <= metrics["quality_score"] <= 1

    similarity = evaluator.semantic_similarity(response, context[0])
    assert 0 <= similarity <= 1

    print(f"[PASS] ResponseEvaluator - quality={metrics['quality_score']:.3f}, faithfulness={faithfulness:.3f}")


def test_response_evaluator_edge_cases():
    evaluator = ResponseEvaluator()

    assert evaluator.context_relevance("test", []) == 0.0
    assert evaluator.faithfulness_score("test", []) == 0.0
    assert evaluator.response_completeness("answer", "") == 1.0

    metrics = evaluator.evaluate_response("response", "query", ["context"], reference="reference answer")
    assert "semantic_similarity" in metrics

    print("[PASS] ResponseEvaluator edge cases")


def run_all_tests():
    print("=" * 60)
    print("Running Financial RAG Assistant Tests")
    print("=" * 60)
    start = time.time()

    tests = [
        test_text_cleaner,
        test_document_chunker,
        test_data_pipeline,
        test_retrieval_evaluator,
        test_response_evaluator,
        test_response_evaluator_edge_cases,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.2f}s)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
