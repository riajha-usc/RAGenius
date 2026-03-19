# LLM-Powered Financial RAG Assistant

End-to-end Retrieval-Augmented Generation pipeline for semantic financial queries.

## Architecture

```
Query → Embedding (HuggingFace) → FAISS Vector Search → Re-ranking → LLM Generation → Evaluation
```

## Tech Stack

- **Python** - Core language
- **HuggingFace Transformers** - LLM generation (flan-t5-base)
- **Sentence-Transformers** - Query/document embeddings (all-MiniLM-L6-v2)
- **FAISS** - Approximate nearest neighbor vector search
- **Pandas / NumPy** - Data preprocessing and analysis
- **Flask** - API orchestration layer
- **Scikit-learn** - Prompt optimization, TF-IDF evaluation, grid search

## Project Structure

```
financial-assistant/
├── app.py                  # Flask API server
├── rag_pipeline.py         # Core RAG pipeline orchestrator
├── config.py               # Configuration and prompt templates
├── demo.py                 # End-to-end demo script
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── utils/
│   ├── preprocessing.py    # Document loading, chunking, cleaning (Pandas/NumPy)
│   ├── embeddings.py       # HuggingFace sentence-transformers encoding
│   ├── vector_store.py     # FAISS index management and retrieval
│   ├── generator.py        # LLM text generation with prompt templates
│   └── evaluation.py       # Metrics, evaluation, prompt optimization (Scikit-learn)
├── data/
│   ├── sample_docs/        # Sample financial documents
│   └── faiss_index/        # Persisted FAISS index
└── tests/
    └── test_pipeline.py    # Unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the Flask API

```bash
python app.py
```

### API Endpoints

| Method | Endpoint                | Description                          |
|--------|-------------------------|--------------------------------------|
| POST   | /api/query              | Submit a semantic financial query    |
| POST   | /api/index/directory    | Index documents from a directory     |
| POST   | /api/index/text         | Index raw text directly              |
| POST   | /api/embed              | Generate embeddings for text         |
| GET    | /api/documents          | List indexed documents               |
| GET    | /api/metrics            | Retrieve pipeline metrics            |
| GET    | /api/status             | Pipeline status and model info       |
| GET    | /api/health             | Health check                         |
| POST   | /api/optimize           | Run prompt optimization grid search  |
| POST   | /api/compare-templates  | Compare prompt template strategies   |
| POST   | /api/index/save         | Persist FAISS index to disk          |
| POST   | /api/index/load         | Load persisted FAISS index           |

### Example API Calls

**Index documents:**
```bash
curl -X POST http://localhost:5000/api/index/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/sample_docs"}'
```

**Query the pipeline:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple revenue in 2024?",
    "top_k": 5,
    "template": "chain_of_thought",
    "evaluate": true
  }'
```

**Run prompt optimization:**
```bash
curl -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "eval_queries": [
      {"query": "What was Apple revenue?", "reference": "Apple reported $383.3B."},
      {"query": "NVIDIA gross margin?", "reference": "73.8% gross margin."}
    ],
    "param_grid": {
      "template": ["default", "chain_of_thought", "concise"],
      "temperature": [0.1, 0.3, 0.5],
      "top_k_retrieval": [3, 5, 7]
    }
  }'
```

### Run the Demo

```bash
python demo.py
```

### Run Tests

```bash
python -m tests.test_pipeline
```

## Evaluation Metrics

### Retrieval Metrics
- Precision@K, Recall@K, F1@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- NDCG@K

### Response Quality Metrics
- Context Relevance (TF-IDF cosine similarity)
- Faithfulness (lexical grounding score)
- Completeness (query term coverage)
- Composite Quality Score (weighted combination)
- Semantic Similarity (against reference answers)

### Prompt Optimization
- Scikit-learn ParameterGrid for systematic search over templates, temperature, and top-k
- Cross-query quality scoring with mean/std aggregation
- Per-template latency and token usage comparison

## Configuration

All parameters are configurable via `.env` or `config.py`:

- `EMBEDDING_MODEL` - HuggingFace model for embeddings
- `GENERATION_MODEL` - HuggingFace model for generation
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - Document chunking parameters
- `TOP_K_RETRIEVAL` - Number of chunks to retrieve
- `TEMPERATURE` / `TOP_P` / `TOP_K_GENERATION` - Generation parameters
- `FAISS_INDEX_TYPE` - FAISS index type (FlatIP, FlatL2, IVF, HNSW)