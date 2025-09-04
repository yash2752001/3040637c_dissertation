import os
import re
from flask import Flask, request, jsonify
import requests
from elasticsearch import Elasticsearch

print("--- Configuring Elasticsearch COVID Application ---")

ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS = "../http_ca.crt"
INDEX_NAME = "beir-trec-covid"

INITIAL_RETRIEVAL_DEPTH = 15
FINAL_RERANK_DEPTH = 10

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
RERANKER_MODEL = "openai/gpt-4o-mini"
GENERATION_MODEL = "openai/gpt-4o-mini"

print("\n--- Connecting to Elasticsearch ---")
try:
    es = Elasticsearch(
        [ES_HOST],
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=True,
        ca_certs=ES_CA_CERTS
    )
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch.")
    print(" Connected to Elasticsearch.")
except Exception as e:
    print(f" Failed to connect to Elasticsearch: {e}")
    exit()

def call_llm(prompt, model, temperature=0.0, max_tokens=500):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: LLM API call failed. {e}"

def rerank_with_llm(documents, query):
    print(f"Reranking {len(documents)} documents with LLM...")
    for doc in documents:
        prompt = f"On a scale of 0 to 1, how relevant is this document to the query?\n\nQuery: {query}\nDocument: {doc.get('text', '')[:1000]}\n\nProvide only the numeric score."
        try:
            response = call_llm(prompt, model=RERANKER_MODEL, temperature=0.0, max_tokens=10)
            score = float(response.strip())
        except (ValueError, TypeError):
            score = 0.0
        doc['llm_score'] = score
    return sorted(documents, key=lambda x: x.get('llm_score', 0.0), reverse=True)

def generate_final_answer(reranked_docs, query):
    print(f" Generating final answer using top {len(reranked_docs)} documents.")
    context = "\n\n---\n\n".join([doc['text'] for doc in reranked_docs])
    prompt = f"Using the following documents, answer the question clearly.\n\nDocuments:\n{context[:12000]}\n\nQuestion: {query}"
    return call_llm(prompt, model=GENERATION_MODEL, temperature=0.3, max_tokens=800)

def search_elasticsearch(query: str):
    """Performs a search using Elasticsearch."""
    search_body = {
        "size": INITIAL_RETRIEVAL_DEPTH,
        "query": {
            "match": {
                "text": query
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body)
    
    results = []
    for hit in response['hits']['hits']:
        results.append({
            "docno": hit['_id'],
            "es_score": hit['_score'],
            "text": hit['_source'].get('title', '') + ' ' + hit['_source'].get('text', '')
        })
    return results

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    if not request.json or 'query' not in request.json:
        return jsonify({"error": "Bad Request"}), 400

    original_query = request.json['query']
    
    initial_docs = search_elasticsearch(original_query)
    if not initial_docs:
        return jsonify({"answer": "I couldn't find any documents."})

    reranked_docs = rerank_with_llm(initial_docs, original_query)
    top_docs = reranked_docs[:FINAL_RERANK_DEPTH]
    final_answer = generate_final_answer(top_docs, original_query)
    
    response_data = {
        "question": original_query, 
        "answer": final_answer, 
        "reranked_documents": [
            {
                "docno": doc['docno'], 
                "llm_score": doc.get('llm_score'),
                "es_score": doc.get('es_score')
            } 
            for doc in reranked_docs
        ]
    }
    return jsonify(response_data)

if __name__ == '__main__':
    print("\n🌍 Starting Elasticsearch COVID Flask API server...")
    app.run(host='0.0.0.0', port=5002)