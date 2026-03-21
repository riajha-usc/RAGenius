import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import RAGPipeline
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG


app = Flask(__name__)
CORS(app)

pipeline = RAGPipeline(lazy_load=True)


@app.before_request
def log_request():
    request._start_time = time.time()


@app.after_request
def log_response(response):
    if hasattr(request, "_start_time"):
        elapsed = time.time() - request._start_time
        response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    return response


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(pipeline.get_status())


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    result = pipeline.query(
        question=data["query"],
        top_k=data.get("top_k", 5),
        template=data.get("template", "default"),
        temperature=data.get("temperature"),
        evaluate=data.get("evaluate", True),
        relevant_doc_ids=data.get("relevant_doc_ids"),
        reference_answer=data.get("reference_answer"),
    )
    return jsonify(result)


@app.route("/api/index/directory", methods=["POST"])
def index_directory():
    data = request.get_json() or {}
    directory = data.get("directory")
    result = pipeline.index_documents(directory)
    return jsonify(result)


@app.route("/api/index/text", methods=["POST"])
def index_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = pipeline.index_text(
        text=data["text"],
        source=data.get("source", "direct_input"),
    )
    return jsonify(result)


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = pipeline.get_embedding(data["text"])
    return jsonify(result)


@app.route("/api/documents", methods=["GET"])
def list_documents():
    if pipeline.documents_df is None or pipeline.documents_df.empty:
        return jsonify({"documents": [], "total_chunks": 0})

    df = pipeline.documents_df
    docs = []
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        docs.append({
            "source": source,
            "total_chunks": len(source_df),
            "avg_chunk_length": float(source_df["text_length"].mean()),
            "total_words": int(source_df["word_count"].sum()),
        })

    return jsonify({
        "documents": docs,
        "total_chunks": len(df),
        "total_documents": len(docs),
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    status = pipeline.get_status()
    return jsonify({
        "pipeline_status": status,
        "index_stats": status.get("vector_store", {}),
    })


@app.route("/api/optimize", methods=["POST"])
def optimize_prompts():
    data = request.get_json()
    if not data or "eval_queries" not in data:
        return jsonify({"error": "Missing 'eval_queries' field"}), 400

    result = pipeline.optimize_prompts(
        eval_queries=data["eval_queries"],
        param_grid=data.get("param_grid"),
    )
    return jsonify(result)


@app.route("/api/compare-templates", methods=["POST"])
def compare_templates():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    result = pipeline.compare_templates(
        query=data["query"],
        top_k=data.get("top_k", 5),
    )
    return jsonify(result)


@app.route("/api/index/save", methods=["POST"])
def save_index():
    data = request.get_json() or {}
    try:
        pipeline.save_index(data.get("path"))
        return jsonify({"status": "success", "message": "Index saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/index/load", methods=["POST"])
def load_index():
    data = request.get_json() or {}
    try:
        pipeline.load_index(data.get("path"))
        return jsonify({"status": "success", "message": "Index loaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
