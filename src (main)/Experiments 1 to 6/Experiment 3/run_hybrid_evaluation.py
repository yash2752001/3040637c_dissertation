# run_final_hybrid_benchmark.py
# A single, powerful script for Task 3 that performs a rigorous, professional evaluation.
# It compares the Hybrid workflow against the best individual workflow (PyTerrier)
# using multiple datasets and statistical significance testing.

import os
import re
import pandas as pd
from tqdm import tqdm
import warnings
import textwrap
import json

# --- Import All Necessary Libraries ---
import pyterrier as pt
from elasticsearch import Elasticsearch
from beir.datasets.data_loader import GenericDataLoader
from scipy.stats import ttest_rel # For statistical significance
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Master Configuration ---
print("--- Configuring Final Hybrid Benchmark for Task 3 ---")

# We will run the evaluation on these three standard BEIR datasets
DATASET_NAMES = ["trec-covid", "nfcorpus", "scifact"]
NUM_QUERIES_TO_SAMPLE = 25 # Use a sample of 25 queries from each dataset

# Ollama local server address
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"

# Elasticsearch Connection Details
ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS = "../http_ca.crt"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# RAG Settings
INITIAL_RETRIEVAL_DEPTH = 10 # Retrieve 10 from each
FINAL_RERANK_DEPTH = 10

# --- 2. Initialize Systems ---
print("\n--- Initializing PyTerrier and Elasticsearch ---")
if not pt.java.started(): pt.java.init()
try:
    es_client = Elasticsearch([ES_HOST], basic_auth=(ES_USER, ES_PASSWORD), verify_certs=True, ca_certs=ES_CA_CERTS)
    if not es_client.ping(): raise ConnectionError("Could not connect.")
    print("✅ PyTerrier & Elasticsearch are ready.")
except Exception as e:
    print(f"❌ Could not connect to Elasticsearch: {e}. Please ensure the service is running.")
    exit()

# --- 3. Define Core "Evo Agent" RAG Functions ---
def call_ollama_api(prompt: str, is_json=False) -> str:
    try:
        payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
        if is_json: payload["format"] = "json"
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        response_json = json.loads(response.text)
        return response_json.get('response', '')
    except Exception as e:
        return f"Error from local API: {e}"

def rerank_with_llm(documents, query):
    for doc in documents:
        prompt = f"Score the relevance of the document for the query on a scale of 0.0 to 1.0. Query: {query}. Document: {doc.get('text', '')[:2000]}. Respond with only the numeric score."
        try:
            response_text = call_ollama_api(prompt)
            score_match = re.search(r'(\d\.\d+)', response_text)
            score = float(score_match.group(1)) if score_match else 0.0
        except: score = 0.0
        doc['llm_score'] = score
    return sorted(documents, key=lambda x: x.get('llm_score', 0.0), reverse=True)

def generate_final_answer(reranked_docs, query):
    context = "\n\n".join([doc['text'] for doc in reranked_docs])
    prompt = f"Using these documents, answer the question clearly.\n\nDocuments:\n{context[:7000]}\n\nQuestion: {query}"
    return call_ollama_api(prompt)

def evaluate_with_ai_judge(question, answer, contexts):
    prompt = f"""You are an impartial AI evaluator. Evaluate the answer based on the question and context.
Provide scores from 0.0 to 1.0 for 'faithfulness' and 'answer_relevancy'.
Respond ONLY with a single JSON object in the format: {{"faithfulness": <score>, "answer_relevancy": <score>}}
CONTEXT: {contexts}
QUESTION: {question}
ANSWER: {answer}
JSON Response:"""
    try:
        evaluation_str = call_ollama_api(prompt, is_json=True)
        return json.loads(evaluation_str)
    except: return {"faithfulness": 0, "answer_relevancy": 0}

# --- 4. Main Evaluation Loop ---
final_summary_results = []

for ds_name in DATASET_NAMES:
    print(f"\n\n=======================================================")
    print(f"         RUNNING HYBRID BENCHMARK FOR: {ds_name}         ")
    print("=======================================================")

    # --- Data Loading and Indexing ---
    dataset_path = os.path.join(os.getcwd(), f"../Stats/{ds_name}")
    corpus, queries, _ = GenericDataLoader(data_folder=dataset_path).load(split="test")
    pt_index_dir = os.path.join(SCRIPT_DIR, f"../Stats/var/{ds_name}_pt_index")
    pt_index = pt.IndexFactory.of(pt_index_dir)
    es_index_name = f"beir-{ds_name.replace('-', '_')}"
    
    docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]}).set_index('docno')
    
    query_items = list(queries.items())[:NUM_QUERIES_TO_SAMPLE]
    print(f"✅ Loaded data for {ds_name} and sampled {len(query_items)} queries.")
    
    pt_retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=INITIAL_RETRIEVAL_DEPTH)
    def search_elasticsearch(query_text):
        search_body = {"size": INITIAL_RETRIEVAL_DEPTH, "query": {"match": {"text": query_text}}}
        response = es_client.search(index=es_index_name, body=search_body)
        return [{"docno": h['_id'], "text": h['_source'].get('title', '') + ' ' + h['_source'].get('text', '')} for h in response['hits']['hits']]
    def search_pyterrier(query_text):
        cleaned_query = re.sub(r'[^\w\s]', '', query_text)
        results_df = pt_retriever.search(cleaned_query)
        return pd.merge(results_df, docs_df, left_on='docno', right_index=True).to_dict(orient='records')

    # --- Run Pipelines and Judge ---
    per_query_scores = []
    for qid, query in tqdm(query_items, desc=f"Processing {ds_name}"):
        # PyTerrier Workflow (Baseline)
        pt_initial = search_pyterrier(query)
        pt_eval = {"faithfulness": 0, "answer_relevancy": 0}
        if pt_initial:
            pt_reranked = rerank_with_llm(pt_initial, query)
            pt_top = pt_reranked[:FINAL_RERANK_DEPTH]
            pt_answer = generate_final_answer(pt_top, query)
            pt_eval = evaluate_with_ai_judge(query, pt_answer, [d['text'] for d in pt_top])

        # Hybrid Workflow
        es_initial = search_elasticsearch(query)
        combined_docs_dict = {doc['docno']: doc for doc in pt_initial + es_initial}
        combined_docs = list(combined_docs_dict.values())
        
        hybrid_eval = {"faithfulness": 0, "answer_relevancy": 0}
        if combined_docs:
            hybrid_reranked = rerank_with_llm(combined_docs, query)
            hybrid_top = hybrid_reranked[:FINAL_RERANK_DEPTH]
            hybrid_answer = generate_final_answer(hybrid_top, query)
            hybrid_eval = evaluate_with_ai_judge(query, hybrid_answer, [d['text'] for d in hybrid_top])

        per_query_scores.append({
            'pt_faithfulness': pt_eval.get('faithfulness', 0),
            'pt_relevancy': pt_eval.get('answer_relevancy', 0),
            'hybrid_faithfulness': hybrid_eval.get('faithfulness', 0),
            'hybrid_relevancy': hybrid_eval.get('answer_relevancy', 0),
        })
    
    # --- Calculate Averages and Statistical Significance ---
    scores_df = pd.DataFrame(per_query_scores)
    avg_scores = scores_df.mean()
    
    _, p_faithfulness = ttest_rel(scores_df['pt_faithfulness'], scores_df['hybrid_faithfulness'])
    _, p_relevancy = ttest_rel(scores_df['pt_relevancy'], scores_df['hybrid_relevancy'])
    
    # --- Display Results ---
    print(f"\n--- Results for {ds_name} ---")
    summary_data = {
        "Metric": ["Faithfulness", "Relevancy"],
        "PyTerrier Workflow (Baseline)": [avg_scores['pt_faithfulness'], avg_scores['pt_relevancy']],
        "Hybrid Workflow": [avg_scores['hybrid_faithfulness'], avg_scores['hybrid_relevancy']],
        "p-value": [p_faithfulness, p_relevancy]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))

print("\n\n=======================================================")
print("              ALL BENCHMARKS COMPLETE              ")
print("=======================================================")
