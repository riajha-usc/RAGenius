import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from utils.preprocessing import DataPipeline, TextCleaner
from utils.embeddings import EmbeddingEngine
from utils.vector_store import FAISSVectorStore
from utils.generator import LLMGenerator
from utils.evaluation import PipelineEvaluator, PromptOptimizer, ResponseEvaluator
from config import TOP_K_RETRIEVAL, EMBEDDING_DIMENSION


class RAGPipeline:

    def __init__(self, lazy_load: bool = False):
        self.data_pipeline = DataPipeline()
        self.cleaner = TextCleaner()
        self.evaluator = PipelineEvaluator()
        self.response_evaluator = ResponseEvaluator()
        self.documents_df = None
        self.is_indexed = False

        if not lazy_load:
            self._initialize_models()
        else:
            self.embedding_engine = None
            self.generator = None
            self.vector_store = None

    def _initialize_models(self):
        self.embedding_engine = EmbeddingEngine()
        self.generator = LLMGenerator()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_engine.dimension
        )

    def ensure_models(self):
        if self.embedding_engine is None:
            self._initialize_models()

    def index_documents(self, directory: str = None) -> Dict:
        self.ensure_models()
        start_time = time.time()

        self.documents_df = self.data_pipeline.run(directory)
        if self.documents_df.empty:
            return {"status": "error", "message": "No documents found"}

        texts = self.documents_df["text"].tolist()
        embeddings = self.embedding_engine.encode_documents(texts)

        metadata_list = []
        for _, row in self.documents_df.iterrows():
            metadata_list.append({
                "source": row.get("source", "unknown"),
                "chunk_index": row.get("chunk_index", 0),
                "doc_id": row.get("doc_id", 0),
                "chunk_id": row.get("chunk_id", 0),
            })

        self.vector_store.clear()
        self.vector_store.add_documents(embeddings, texts, metadata_list)
        self.is_indexed = True

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "documents_processed": self.documents_df["source"].nunique(),
            "total_chunks": len(texts),
            "embedding_dimension": self.embedding_engine.dimension,
            "index_stats": self.vector_store.get_stats(),
            "processing_time_seconds": round(elapsed, 2),
        }

    def index_text(self, text: str, source: str = "direct_input") -> Dict:
        self.ensure_models()
        start_time = time.time()

        df = self.data_pipeline.process_single_text(text, source)
        if df.empty:
            return {"status": "error", "message": "No text to index"}

        if self.documents_df is not None:
            self.documents_df = pd.concat([self.documents_df, df], ignore_index=True)
        else:
            self.documents_df = df

        texts = df["text"].tolist()
        embeddings = self.embedding_engine.encode_documents(texts)

        metadata_list = [{"source": source, "chunk_index": i} for i in range(len(texts))]
        self.vector_store.add_documents(embeddings, texts, metadata_list)
        self.is_indexed = True

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "chunks_added": len(texts),
            "source": source,
            "processing_time_seconds": round(elapsed, 2),
        }

    def query(
        self,
        question: str,
        top_k: int = TOP_K_RETRIEVAL,
        template: str = "default",
        temperature: float = None,
        evaluate: bool = True,
        relevant_doc_ids: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> Dict:
        self.ensure_models()
        if not self.is_indexed:
            return {"status": "error", "message": "No documents indexed. Index documents first."}

        start_time = time.time()

        cleaned_query = self.cleaner.clean(question)
        cleaned_query = self.cleaner.normalize_financial(cleaned_query)

        query_embedding = self.embedding_engine.encode_query(cleaned_query)
        retrieval_time = time.time()

        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)
        retrieval_elapsed = time.time() - retrieval_time

        gen_kwargs = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        generation_time = time.time()
        gen_result = self.generator.generate_with_context(
            query=cleaned_query,
            context_chunks=retrieved_docs,
            template=template,
            **gen_kwargs,
        )
        generation_elapsed = time.time() - generation_time

        total_elapsed = time.time() - start_time

        result = {
            "status": "success",
            "query": question,
            "answer": gen_result["generated_text"],
            "sources": [
                {
                    "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "score": doc["score"],
                    "metadata": doc["metadata"],
                    "rank": doc["rank"],
                }
                for doc in retrieved_docs
            ],
            "generation_info": {
                "model": gen_result["model"],
                "template": template,
                "input_tokens": gen_result["input_tokens"],
                "output_tokens": gen_result["output_tokens"],
                "total_tokens": gen_result["total_tokens"],
                "parameters": gen_result["parameters"],
            },
            "timing": {
                "retrieval_seconds": round(retrieval_elapsed, 4),
                "generation_seconds": round(generation_elapsed, 4),
                "total_seconds": round(total_elapsed, 4),
            },
        }

        if evaluate:
            context_texts = [doc["text"] for doc in retrieved_docs]
            eval_metrics = self.evaluator.evaluate_full_pipeline(
                query=cleaned_query,
                retrieved_docs=retrieved_docs,
                generated_response=gen_result["generated_text"],
                relevant_doc_ids=relevant_doc_ids,
                reference_answer=reference_answer,
                latency=total_elapsed,
            )
            result["evaluation"] = eval_metrics

        return result

    def get_embedding(self, text: str) -> Dict:
        self.ensure_models()
        embedding = self.embedding_engine.encode(text)
        return {
            "text": text,
            "embedding": embedding.tolist(),
            "dimension": embedding.shape[1],
            "model": self.embedding_engine.model_name,
        }

    def optimize_prompts(self, eval_queries: List[Dict], param_grid: Dict = None) -> Dict:
        self.ensure_models()
        optimizer = PromptOptimizer(
            self.generator, self.embedding_engine, self.vector_store
        )
        return optimizer.grid_search(eval_queries, param_grid)

    def compare_templates(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict:
        self.ensure_models()
        if not self.is_indexed:
            return {"status": "error", "message": "No documents indexed."}

        query_embedding = self.embedding_engine.encode_query(query)
        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)

        optimizer = PromptOptimizer(
            self.generator, self.embedding_engine, self.vector_store
        )
        return optimizer.evaluate_prompt_strategy(query, retrieved_docs)

    def save_index(self, path: str = None):
        if self.vector_store:
            self.vector_store.save(path)

    def load_index(self, path: str = None):
        self.ensure_models()
        self.vector_store.load(path)
        self.is_indexed = True

    def get_status(self) -> Dict:
        status = {
            "models_loaded": self.embedding_engine is not None,
            "is_indexed": self.is_indexed,
        }
        if self.embedding_engine:
            status["embedding_model"] = self.embedding_engine.get_info()
        if self.generator:
            status["generation_model"] = self.generator.get_info()
        if self.vector_store:
            status["vector_store"] = self.vector_store.get_stats()
        if self.documents_df is not None:
            status["documents"] = {
                "total_chunks": len(self.documents_df),
                "unique_sources": int(self.documents_df["source"].nunique()),
                "sources": self.documents_df["source"].unique().tolist(),
            }
        return status
