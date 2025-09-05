import os
import pandas as pd
import pyterrier as pt
from flask import Flask, request, jsonify
import requests
import warnings
import re 

# configurring the part and setup
print("--- Configuring Application ---")

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "var", "pyterrier_index")
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "documents.csv")

INITIAL_RETRIEVAL_DEPTH = 4
FINAL_RERANK_DEPTH = 2

OPENROUTER_API_KEY = ""
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
RERANKER_MODEL = "openai/gpt-4o-mini"
GENERATION_MODEL = "openai/gpt-4o-mini"

# Initializing pyterrier and loading the index
print("\n--- Initialising PyTerrier and Loading Index ---")
warnings.filterwarnings("ignore", category=DeprecationWarning)

if not pt.java.started():
    print("Starting PyTerrier...")
    pt.java.set_memory_limit(4096)
    pt.java.init()
    print("✅ PyTerrier started successfully.")

if not os.path.exists(INDEX_DIR):
    print(f"Index not found at {INDEX_DIR}. Please run final_setup.py first!")
    exit()

index = pt.IndexFactory.of(INDEX_DIR)
print("✅ Index loaded successfully.")

documents_df = pd.read_csv(CSV_PATH)
docs_for_lookup = documents_df.set_index('docno')
retriever = pt.BatchRetrieve(index, wmodel="BM25", num_results=INITIAL_RETRIEVAL_DEPTH)
print("✅ Retriever is ready.")

# RAG Pipeline
def clean_query(query: str) -> str:
    """Removes special characters from a query to make it safe for Terrier."""
    return re.sub(r'[^\w\s]', '', query)

def call_llm(prompt, model, temperature=0.0, max_tokens=150):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=45)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: LLM API call failed. {e}"

def rerank_with_llm(documents, query):
    print(f"Reranking {len(documents)} documents with LLM...")
    for doc in documents:
        prompt = f"On a scale of 0 to 1, how relevant is this document to the query?\n\nQuery: {query}\nDocument: {doc.get('text', '')}\n\nProvide only the numeric score."
        try:
            response = call_llm(prompt, model=RERANKER_MODEL, temperature=0.0, max_tokens=10)
            score = float(response.strip())
        except (ValueError, TypeError):
            score = 0.0
        doc['llm_score'] = score
    return sorted(documents, key=lambda x: x.get('llm_score', 0.0), reverse=True)

def generate_final_answer(reranked_docs, query):
    print(f"Generating final answer using top {len(reranked_docs)} documents.")
    context = "\n\n---\n\n".join([doc['text'] for doc in reranked_docs])
    prompt = f"Using the following documents, answer the question clearly.\n\nDocuments:\n{context}\n\nQuestion: {query}"
    return call_llm(prompt, model=GENERATION_MODEL, temperature=0.3, max_tokens=500)

def search_pyterrier(query: str):
    results_df = retriever.search(query)
    merged_results = pd.merge(results_df, docs_for_lookup, left_on='docno', right_index=True)
    return merged_results.to_dict(orient='records')

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    if not request.json or 'query' not in request.json:
        return jsonify({"error": "Bad Request"}), 400

    original_query = request.json['query']
    print(f"\n\n--- New Query Received ---")
    print(f"Original Query: '{original_query}'")

    cleaned_query = clean_query(original_query)
    print(f"Cleaned Query for Terrier: '{cleaned_query}'")

    initial_docs = search_pyterrier(cleaned_query)
    if not initial_docs:
        return jsonify({"answer": "I couldn't find any documents."})

    reranked_docs = rerank_with_llm(initial_docs, original_query)

    top_docs = reranked_docs[:FINAL_RERANK_DEPTH]
    final_answer = generate_final_answer(top_docs, original_query)
    
    response_data = {
        "question": original_query,
        "answer": final_answer,
        "reranked_documents": [
            {"docno": doc['docno'], "text": doc.get('text', ''), "llm_score": doc.get('llm_score')}
            for doc in reranked_docs
        ]
    }
    
    print("--- Pipeline Complete ---")
    return jsonify(response_data)

if __name__ == '__main__':
    print("\nStarting Flask API server...")
    print(f"Send POST requests to http://127.0.0.1:5000/query")
    print("Example: curl -X POST -H \"Content-Type: application/json\" -d '{\"query\": \"What is BM25?\"}' http://127.0.0.1:5000/query")

    app.run(host='0.0.0.0', port=5000)
