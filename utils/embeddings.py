import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingEngine:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        test_embedding = self.model.encode(["test"])
        return test_embedding.shape[1]

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(query, normalize=True)

    def encode_documents(self, documents: List[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(documents, batch_size=batch_size, normalize=True)

    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores = np.dot(query_embedding, doc_embeddings.T).flatten()
        return scores

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
        }
