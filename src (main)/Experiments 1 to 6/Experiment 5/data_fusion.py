# run_final_fusion_benchmark.py
# The definitive script for Task 5. It is slow but rigorous and meets all requirements.
# It uses the local Llama3 model for all AI tasks.

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
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from scipy.stats import ttest_rel # For statistical significance
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Master Configuration ---
print("--- Configuring Final Benchmark for Task 5: Data Fusion ---")

# We will run the evaluation on these three standard BEIR datasets
DATASET_NAMES = ["trec-covid", "nfcorpus", "scifact"]
NUM_QUERIES_TO_SAMPLE = 25 # Use a sample of 25 queries from each dataset

# Ollama local server address
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Elasticsearch Connection Details
ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS = "../http_ca.crt"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# RAG Settings
INITIAL_RETRIEVAL_DEPTH = 10 # Retrieve 10 from each engine
FINAL_RERANK_DEPTH = 10

# --- 2. The EvoAgent Class ---
class EvoAgent:
    """An agent that uses a local LLM for all intelligent tasks."""
    def __init__(self, model_name):
        self.model = model_name
        print(f"✅ EvoAgent initialized with local model: {self.model}")

    def _call_ollama(self, prompt: str, is_json=False) -> str:
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            if is_json: payload["format"] = "json"
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            return json.loads(response.text).get('response', '')
        except Exception as e:
            return f"Error from local API: {e}"
    
    def rerank(self, documents, query):
        """Reranks documents using the powerful but slow Llama3 model."""
        for doc in documents:
            prompt = f"On a scale of 0.0 to 1.0, how relevant is this document to the query? Query: {query}. Document: {doc.get('text', '')[:2000]}. Respond with only the numeric score."
            try:
                response_text = self._call_ollama(prompt)
                score_match = re.search(r'(\d\.\d+)', response_text)
                score = float(score_match.group(1)) if score_match else 0.0
            except: score = 0.0
            doc['llm_score'] = score
        return sorted(documents, key=lambda x: x.get('llm_score', 0.0), reverse=True)

    def generate_answer(self, context_docs, query):
        context = "\n\n".join([doc['text'] for doc in context_docs])
        prompt = f"Using these documents, answer the question clearly.\n\nDocuments:\n{context[:7000]}\n\nQuestion: {query}"
        return self._call_ollama(prompt)

    def judge_answer(self, question, answer, contexts):
        prompt = f"""You are an impartial AI evaluator. Evaluate the answer based on the question and context.
Provide scores from 0.0 to 1.0 for 'faithfulness' and 'answer_relevancy'.
Respond ONLY with a single JSON object in the format: {{"faithfulness": <score>, "answer_relevancy": <score>}}
CONTEXT: {contexts}
QUESTION: {question}
ANSWER: {answer}
JSON Response:"""
        try:
            evaluation_str = self._call_ollama(prompt, is_json=True)
            return json.loads(evaluation_str)
        except: return {"faithfulness": 0, "answer_relevancy": 0}

# --- 3. Initialize Systems ---
print("\n--- Initializing PyTerrier, Elasticsearch, and EvoAgent ---")
if not pt.java.started(): pt.java.init()
agent = EvoAgent(model_name=OLLAMA_MODEL)
try:
    es_client = Elasticsearch([ES_HOST], basic_auth=(ES_USER, ES_PASSWORD), verify_certs=True, ca_certs=ES_CA_CERTS)
    if not es_client.ping(): raise ConnectionError("Could not connect.")
    print("✅ Elasticsearch is ready.")
except Exception as e:
    print(f"❌ Could not connect to Elasticsearch: {e}. Please ensure the service is running.")
    exit()

# --- 4. Data Fusion Functions ---
def fuse_interleave(pt_docs, es_docs):
    fused, i, j = [], 0, 0
    while i < len(pt_docs) or j < len(es_docs):
        if i < len(pt_docs): fused.append(pt_docs[i]); i += 1
        if j < len(es_docs): fused.append(es_docs[j]); j += 1
    return list({doc['docno']: doc for doc in fused}.values())

def fuse_combsum(pt_docs, es_docs):
    pt_scores = {doc['docno']: doc['score'] for doc in pt_docs}
    es_scores = {doc['docno']: doc['es_score'] for doc in es_docs}
    min_pt, max_pt = (min(pt_scores.values()), max(pt_scores.values())) if pt_scores else (0,1)
    min_es, max_es = (min(es_scores.values()), max(es_scores.values())) if es_scores else (0,1)
    
    all_docs = {doc['docno']: doc for doc in pt_docs + es_docs}
    for docno, doc in all_docs.items():
        norm_pt = (pt_scores.get(docno, 0) - min_pt) / (max_pt - min_pt) if max_pt > min_pt else 0
        norm_es = (es_scores.get(docno, 0) - min_es) / (max_es - min_es) if max_es > min_es else 0
        doc['fused_score'] = norm_pt + norm_es
    return sorted(all_docs.values(), key=lambda x: x['fused_score'], reverse=True)

def fuse_rrf(pt_docs, es_docs, k=60):
    rrf_scores = {}
    all_docs = {doc['docno']: doc for doc in pt_docs + es_docs}
    for i, doc in enumerate(pt_docs):
        rrf_scores[doc['docno']] = rrf_scores.get(doc['docno'], 0) + 1 / (k + i + 1)
    for i, doc in enumerate(es_docs):
        rrf_scores[doc['docno']] = rrf_scores.get(doc['docno'], 0) + 1 / (k + i + 1)
    
    for docno, doc in all_docs.items():
        doc['fused_score'] = rrf_scores.get(docno, 0)
    return sorted(all_docs.values(), key=lambda x: x['fused_score'], reverse=True)

# --- 5. Main Evaluation Loop ---
for ds_name in DATASET_NAMES:
    print(f"\n\n=======================================================")
    print(f"         RUNNING FUSION BENCHMARK FOR: {ds_name}         ")
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
        return [{"docno": h['_id'], "es_score": h['_score'], "text": h['_source'].get('title', '') + ' ' + h['_source'].get('text', '')} for h in response['hits']['hits']]
    def search_pyterrier(query_text):
        cleaned_query = re.sub(r'[^\w\s]', '', query_text)
        results_df = pt_retriever.search(cleaned_query)
        return pd.merge(results_df, docs_df, left_on='docno', right_index=True).to_dict(orient='records')

    # --- Run Pipelines and Judge ---
    per_query_scores = []
    for qid, query in tqdm(query_items, desc=f"Processing {ds_name}"):
        pt_docs = search_pyterrier(query)
        es_docs = search_elasticsearch(query)
        
        eval_scores = {}
        fusion_methods = {
            "Interleave": fuse_interleave(pt_docs, es_docs),
            "CombSUM": fuse_combsum(pt_docs, es_docs),
            "RRF": fuse_rrf(pt_docs, es_docs)
        }
        for name, fused_docs in fusion_methods.items():
            reranked_docs = agent.rerank(fused_docs, query)
            top_docs = reranked_docs[:FINAL_RERANK_DEPTH]
            answer = agent.generate_answer(top_docs, query)
            evaluation = agent.judge_answer(query, answer, [d['text'] for d in top_docs])
            eval_scores[f'{name}_faithfulness'] = evaluation.get('faithfulness', 0)
            eval_scores[f'{name}_relevancy'] = evaluation.get('answer_relevancy', 0)
        per_query_scores.append(eval_scores)
    
    # --- Calculate Averages and Statistical Significance ---
    scores_df = pd.DataFrame(per_query_scores)
    avg_scores = scores_df.mean()
    
    _, p_faith_combsum = ttest_rel(scores_df['Interleave_faithfulness'], scores_df['CombSUM_faithfulness'])
    _, p_rel_combsum = ttest_rel(scores_df['Interleave_relevancy'], scores_df['CombSUM_relevancy'])
    _, p_faith_rrf = ttest_rel(scores_df['Interleave_faithfulness'], scores_df['RRF_faithfulness'])
    _, p_rel_rrf = ttest_rel(scores_df['Interleave_relevancy'], scores_df['RRF_relevancy'])
    
    # --- Display Results ---
    print(f"\n--- Results for {ds_name} ---")
    summary_data = {
        "Metric": ["Faithfulness", "Relevancy"],
        "Interleave (Baseline)": [avg_scores['Interleave_faithfulness'], avg_scores['Interleave_relevancy']],
        "CombSUM (Score-based)": [avg_scores['CombSUM_faithfulness'], avg_scores['CombSUM_relevancy']],
        "RRF (Rank-based)": [avg_scores['RRF_faithfulness'], avg_scores['RRF_relevancy']]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))
    print("\n--- Statistical Significance (vs. Interleave Baseline) ---")
    print(f"CombSUM Faithfulness p-value: {p_faith_combsum:.4f}")
    print(f"CombSUM Relevancy p-value:    {p_rel_combsum:.4f}")
    print(f"RRF Faithfulness p-value:     {p_faith_rrf:.4f}")
    print(f"RRF Relevancy p-value:        {p_rel_rrf:.4f}")

print("\n\n=======================================================")
print("              ALL BENCHMARKS COMPLETE              ")
print("=======================================================")
