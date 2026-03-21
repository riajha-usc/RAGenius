import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import INDEX_DIR, FAISS_INDEX_TYPE, FAISS_NPROBE, TOP_K_RETRIEVAL


class FAISSVectorStore:

    def __init__(self, dimension: int, index_type: str = FAISS_INDEX_TYPE):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(dimension, index_type)
        self.documents = []
        self.metadata = []
        self.doc_count = 0

    def _create_index(self, dimension: int, index_type: str):
        if index_type == "FlatIP":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "FlatL2":
            index = faiss.IndexFlatL2(dimension)
        elif index_type.startswith("IVF"):
            parts = index_type.split(",")
            nlist = int(parts[0].replace("IVF", ""))
            quantizer = faiss.IndexFlatIP(dimension)
            if len(parts) > 1 and parts[1].startswith("PQ"):
                m = int(parts[1].replace("PQ", ""))
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            else:
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = FAISS_NPROBE
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(dimension)
        return index

    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[Dict]] = None):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        faiss.normalize_L2(embeddings)

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)

        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

        self.doc_count += len(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, self.doc_count)
        if top_k == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            result = {
                "rank": i + 1,
                "score": float(score),
                "text": self.documents[idx],
                "doc_index": int(idx),
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
            }
            results.append(result)

        return results

    def save(self, path: str = None):
        save_dir = Path(path) if path else INDEX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "store_data.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "doc_count": self.doc_count,
                "dimension": self.dimension,
                "index_type": self.index_type,
            }, f)

    def load(self, path: str = None):
        load_dir = Path(path) if path else INDEX_DIR
        self.index = faiss.read_index(str(load_dir / "index.faiss"))
        with open(load_dir / "store_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.doc_count = data["doc_count"]
        self.dimension = data["dimension"]
        self.index_type = data["index_type"]

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, "is_trained", True),
            "document_count": self.doc_count,
        }

    def clear(self):
        self.index = self._create_index(self.dimension, self.index_type)
        self.documents = []
        self.metadata = []
        self.doc_count = 0
