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

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your configuration:

```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GENERATION_MODEL=google/flan-t5-base
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
TEMPERATURE=0.3
FAISS_INDEX_TYPE=FlatIP
FLASK_PORT=5000
```

## Usage

### Run the demo

```bash
python demo.py
```

This indexes sample financial documents, runs queries, compares prompt templates, and runs a grid search optimization.

### Start the API server

```bash
python app.py
```

### Run tests

```bash
python -m tests.test_pipeline
```

## API Endpoints

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

### Example requests

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

## Evaluation Metrics

**Retrieval** - Precision@K, Recall@K, F1@K, MRR, MAP, NDCG@K

**Response quality** - Context relevance (TF-IDF cosine similarity), faithfulness (lexical grounding), completeness (query coverage), composite quality score

**Prompt optimization** - Scikit-learn ParameterGrid search over templates, temperature, and top-k with cross-query scoring