import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


class RetrievalEvaluator:

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        if k:
            retrieved = retrieved[:k]
        if not retrieved:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved if doc in relevant_set)
        return hits / len(retrieved)

    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        if k:
            retrieved = retrieved[:k]
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved if doc in relevant_set)
        return hits / len(relevant_set)

    def f1_at_k(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        hits = 0
        sum_precisions = 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                hits += 1
                sum_precisions += hits / (i + 1)
        if not relevant_set:
            return 0.0
        return sum_precisions / len(relevant_set)

    def ndcg_at_k(self, scores: List[float], relevant_scores: List[float], k: int = None) -> float:
        if k:
            scores = scores[:k]
            relevant_scores = relevant_scores[:k]
        try:
            return float(ndcg_score(
                np.array([relevant_scores]),
                np.array([scores]),
            ))
        except Exception:
            return 0.0

    def evaluate_retrieval(
        self,
        retrieved_docs: List[Dict],
        relevant_doc_ids: List[str],
        k_values: List[int] = None,
    ) -> Dict:
        if k_values is None:
            k_values = [1, 3, 5, 10]

        retrieved_ids = [
            doc.get("metadata", {}).get("source", str(doc.get("doc_index", "")))
            for doc in retrieved_docs
        ]

        results = {}
        for k in k_values:
            results[f"precision@{k}"] = self.precision_at_k(retrieved_ids, relevant_doc_ids, k)
            results[f"recall@{k}"] = self.recall_at_k(retrieved_ids, relevant_doc_ids, k)
            results[f"f1@{k}"] = self.f1_at_k(retrieved_ids, relevant_doc_ids, k)

        results["mrr"] = self.mrr(retrieved_ids, relevant_doc_ids)
        results["map"] = self.average_precision(retrieved_ids, relevant_doc_ids)

        return results


class ResponseEvaluator:

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

    def semantic_similarity(self, response: str, reference: str) -> float:
        try:
            tfidf_matrix = self.tfidf.fit_transform([response, reference])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    def context_relevance(self, response: str, context_chunks: List[str]) -> float:
        if not context_chunks:
            return 0.0
        try:
            all_texts = [response] + context_chunks
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            return float(np.mean(similarities))
        except Exception:
            return 0.0

    def faithfulness_score(self, response: str, context_chunks: List[str]) -> float:
        if not context_chunks:
            return 0.0
        response_words = set(response.lower().split())
        context_words = set()
        for chunk in context_chunks:
            context_words.update(chunk.lower().split())

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when", "that",
            "this", "these", "those", "it", "its", "they", "them", "their",
        }

        response_content = response_words - stopwords
        if not response_content:
            return 0.0
        grounded = response_content.intersection(context_words)
        return len(grounded) / len(response_content)

    def response_completeness(self, response: str, query: str) -> float:
        query_words = set(query.lower().split()) - {
            "what", "how", "why", "when", "where", "who", "which",
            "is", "are", "was", "were", "the", "a", "an", "of", "in",
            "for", "to", "and", "or", "do", "does", "did", "can",
        }
        if not query_words:
            return 1.0
        response_lower = response.lower()
        covered = sum(1 for w in query_words if w in response_lower)
        return covered / len(query_words)

    def evaluate_response(
        self,
        response: str,
        query: str,
        context_chunks: List[str],
        reference: Optional[str] = None,
    ) -> Dict:
        metrics = {
            "context_relevance": self.context_relevance(response, context_chunks),
            "faithfulness": self.faithfulness_score(response, context_chunks),
            "completeness": self.response_completeness(response, query),
            "response_length": len(response),
            "word_count": len(response.split()),
        }

        if reference:
            metrics["semantic_similarity"] = self.semantic_similarity(response, reference)

        quality_weights = {
            "context_relevance": 0.3,
            "faithfulness": 0.4,
            "completeness": 0.3,
        }
        metrics["quality_score"] = sum(
            metrics[k] * v for k, v in quality_weights.items()
        )

        return metrics


class PromptOptimizer:

    def __init__(self, generator, embedding_engine, vector_store):
        self.generator = generator
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.response_evaluator = ResponseEvaluator()

    def grid_search(
        self,
        eval_queries: List[Dict],
        param_grid: Dict = None,
    ) -> Dict:
        if param_grid is None:
            param_grid = {
                "template": ["default", "chain_of_thought", "concise"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
                "top_k_retrieval": [3, 5, 7, 10],
            }

        grid = list(ParameterGrid(param_grid))
        results = []

        for params in grid:
            param_scores = []
            for eval_item in eval_queries:
                query = eval_item["query"]
                reference = eval_item.get("reference", "")
                relevant_docs = eval_item.get("relevant_docs", [])

                query_embedding = self.embedding_engine.encode_query(query)
                retrieved = self.vector_store.search(
                    query_embedding,
                    top_k=params.get("top_k_retrieval", 5),
                )

                context_chunks = retrieved
                gen_result = self.generator.generate_with_context(
                    query=query,
                    context_chunks=context_chunks,
                    template=params.get("template", "default"),
                    temperature=params.get("temperature", 0.3),
                )

                context_texts = [c["text"] for c in context_chunks]
                eval_metrics = self.response_evaluator.evaluate_response(
                    response=gen_result["generated_text"],
                    query=query,
                    context_chunks=context_texts,
                    reference=reference if reference else None,
                )
                param_scores.append(eval_metrics["quality_score"])

            avg_score = np.mean(param_scores) if param_scores else 0.0
            std_score = np.std(param_scores) if param_scores else 0.0
            results.append({
                **params,
                "avg_quality_score": avg_score,
                "std_quality_score": std_score,
                "n_queries": len(eval_queries),
            })

        results_df = pd.DataFrame(results).sort_values("avg_quality_score", ascending=False)
        best = results_df.iloc[0].to_dict() if len(results_df) > 0 else {}

        return {
            "best_params": best,
            "all_results": results_df.to_dict("records"),
            "param_grid": param_grid,
        }

    def evaluate_prompt_strategy(
        self,
        query: str,
        context_chunks: List[Dict],
        templates: List[str] = None,
    ) -> Dict:
        if templates is None:
            templates = list(PROMPT_TEMPLATES.keys())

        from config import PROMPT_TEMPLATES
        template_results = {}

        for template in templates:
            if template not in PROMPT_TEMPLATES:
                continue

            start_time = time.time()
            gen_result = self.generator.generate_with_context(
                query=query,
                context_chunks=context_chunks,
                template=template,
            )
            latency = time.time() - start_time

            context_texts = [c["text"] for c in context_chunks]
            eval_metrics = self.response_evaluator.evaluate_response(
                response=gen_result["generated_text"],
                query=query,
                context_chunks=context_texts,
            )

            template_results[template] = {
                "response": gen_result["generated_text"],
                "metrics": eval_metrics,
                "latency": latency,
                "tokens": gen_result["total_tokens"],
            }

        return template_results


class PipelineEvaluator:

    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.response_evaluator = ResponseEvaluator()

    def evaluate_full_pipeline(
        self,
        query: str,
        retrieved_docs: List[Dict],
        generated_response: str,
        relevant_doc_ids: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
        latency: Optional[float] = None,
    ) -> Dict:
        retrieval_metrics = {}
        if relevant_doc_ids:
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                retrieved_docs, relevant_doc_ids
            )

        context_texts = [doc["text"] for doc in retrieved_docs]
        response_metrics = self.response_evaluator.evaluate_response(
            response=generated_response,
            query=query,
            context_chunks=context_texts,
            reference=reference_answer,
        )

        combined = {
            "retrieval": retrieval_metrics,
            "response": response_metrics,
            "overall_quality": response_metrics.get("quality_score", 0.0),
        }

        if latency is not None:
            combined["latency_seconds"] = latency

        return combined
