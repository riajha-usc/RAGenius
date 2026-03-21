import sys
import time
import json

from rag_pipeline import RAGPipeline
from config import SAMPLE_DOCS_DIR


def main():
    print("=" * 70)
    print("  LLM-Powered Financial RAG Assistant - End-to-End Demo")
    print("=" * 70)

    print("\n[1/5] Initializing RAG Pipeline...")
    start = time.time()
    pipeline = RAGPipeline(lazy_load=False)
    print(f"  Models loaded in {time.time() - start:.1f}s")

    status = pipeline.get_status()
    print(f"  Embedding model: {status['embedding_model']['model_name']}")
    print(f"  Embedding dimension: {status['embedding_model']['dimension']}")
    print(f"  Generation model: {status['generation_model']['model_name']}")
    print(f"  Generation params: {status['generation_model']['total_parameters_millions']}M parameters")

    print(f"\n[2/5] Indexing documents from {SAMPLE_DOCS_DIR}...")
    index_result = pipeline.index_documents(str(SAMPLE_DOCS_DIR))
    print(f"  Documents processed: {index_result['documents_processed']}")
    print(f"  Total chunks: {index_result['total_chunks']}")
    print(f"  Index type: {index_result['index_stats']['index_type']}")
    print(f"  Indexing time: {index_result['processing_time_seconds']}s")

    print("\n[3/5] Running Financial Queries...")
    queries = [
        {
            "query": "What was Apple's total revenue and services growth in fiscal year 2024?",
            "template": "default",
        },
        {
            "query": "Compare Tesla and Ford's operating margins and profitability",
            "template": "chain_of_thought",
        },
        {
            "query": "What are NVIDIA's key risk factors related to China?",
            "template": "concise",
        },
        {
            "query": "How is Microsoft's Azure cloud business performing and what role does AI play?",
            "template": "chain_of_thought",
        },
        {
            "query": "What are Amazon's main competitive and regulatory risks?",
            "template": "default",
        },
    ]

    all_results = []
    for i, q in enumerate(queries):
        print(f"\n  Query {i+1}: {q['query']}")
        print(f"  Template: {q['template']}")

        result = pipeline.query(
            question=q["query"],
            template=q["template"],
            top_k=5,
            evaluate=True,
        )
        all_results.append(result)

        print(f"  Answer: {result['answer'][:150]}...")
        print(f"  Sources: {len(result['sources'])} chunks retrieved")
        print(f"  Top source score: {result['sources'][0]['score']:.4f}")
        print(f"  Timing: retrieval={result['timing']['retrieval_seconds']}s, "
              f"generation={result['timing']['generation_seconds']}s, "
              f"total={result['timing']['total_seconds']}s")

        if "evaluation" in result:
            eval_data = result["evaluation"]
            resp_metrics = eval_data.get("response", {})
            print(f"  Quality Score: {resp_metrics.get('quality_score', 0):.3f}")
            print(f"  Faithfulness: {resp_metrics.get('faithfulness', 0):.3f}")
            print(f"  Context Relevance: {resp_metrics.get('context_relevance', 0):.3f}")
            print(f"  Completeness: {resp_metrics.get('completeness', 0):.3f}")

    print("\n[4/5] Comparing Prompt Templates...")
    comparison = pipeline.compare_templates(
        "What was Apple's revenue growth and key financial metrics?",
        top_k=5,
    )
    print(f"\n  Template Comparison Results:")
    for template, data in comparison.items():
        metrics = data["metrics"]
        print(f"  {template:20s} | quality={metrics['quality_score']:.3f} "
              f"| faithfulness={metrics['faithfulness']:.3f} "
              f"| latency={data['latency']:.3f}s "
              f"| tokens={data['tokens']}")

    print("\n[5/5] Running Prompt Optimization (Grid Search)...")
    eval_queries = [
        {"query": "What was Apple's revenue in 2024?", "reference": "Apple reported $383.3 billion in revenue."},
        {"query": "What is NVIDIA's gross margin?", "reference": "NVIDIA achieved 73.8% gross margin."},
        {"query": "How did Tesla's energy segment perform?", "reference": "Energy segment revenue grew 67% with 30.5% margins."},
    ]

    optimization = pipeline.optimize_prompts(
        eval_queries=eval_queries,
        param_grid={
            "template": ["default", "chain_of_thought", "concise"],
            "temperature": [0.1, 0.3, 0.5],
            "top_k_retrieval": [3, 5],
        },
    )

    best = optimization["best_params"]
    print(f"  Best template: {best.get('template', 'N/A')}")
    print(f"  Best temperature: {best.get('temperature', 'N/A')}")
    print(f"  Best top_k: {best.get('top_k_retrieval', 'N/A')}")
    print(f"  Best quality score: {best.get('avg_quality_score', 0):.4f}")
    print(f"  Score std: {best.get('std_quality_score', 0):.4f}")

    print(f"\n  Top 5 configurations:")
    for i, r in enumerate(optimization["all_results"][:5]):
        print(f"    {i+1}. template={r['template']:20s} temp={r['temperature']:.1f} "
              f"top_k={r['top_k_retrieval']} score={r['avg_quality_score']:.4f}")

    print("\n[+] Saving FAISS index...")
    pipeline.save_index()
    print("  Index saved successfully")

    print("\n" + "=" * 70)
    print("  Pipeline Status Summary")
    print("=" * 70)
    final_status = pipeline.get_status()
    print(json.dumps(final_status, indent=2, default=str))

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
