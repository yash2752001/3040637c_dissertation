import os
import re
import pandas as pd
from tqdm import tqdm
import warnings
import textwrap
import time
import json

import pyterrier as pt
from elasticsearch import Elasticsearch
from beir.datasets.data_loader import GenericDataLoader
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("--- Configuring Automated Workflow Evaluation ---")

TOGETHER_API_KEY = "tgp_v1_l8l7Jc7zA00bZY9CwryARqj_LqO38sp0q8rHB_tM04k"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
LLM_MODEL = "meta-llama/Llama-3-8b-chat-hf"

ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS = "../http_ca.crt"
ES_INDEX_NAME = "beir-trec-covid"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PT_INDEX_DIR = os.path.join(SCRIPT_DIR, "../Stats/var/trec_covid_index")
DATASET_PATH = os.path.join(SCRIPT_DIR, "../Stats/trec-covid")

INITIAL_RETRIEVAL_DEPTH = 15
FINAL_RERANK_DEPTH = 10
QUERIES = [
    "what is the incubation period of covid?",
    "social impact of covid-19 pandemic",
    "long term effects of coronavirus"
]

print("\n--- Initializing PyTerrier and Elasticsearch ---")
if not pt.java.started(): pt.java.init()
pt_index = pt.IndexFactory.of(PT_INDEX_DIR)
pt_retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=INITIAL_RETRIEVAL_DEPTH)
print(" PyTerrier is ready.")
try:
    es_client = Elasticsearch([ES_HOST], basic_auth=(ES_USER, ES_PASSWORD), verify_certs=True, ca_certs=ES_CA_CERTS)
    if not es_client.ping(): raise ConnectionError("Could not connect.")
    print(" Elasticsearch is ready.")
except Exception as e:
    print(f" Could not connect to Elasticsearch: {e}. Please ensure the service is running.")
    exit()

print("\n--- Loading Document Corpus ---")
corpus, _, _ = GenericDataLoader(data_folder=DATASET_PATH).load(split="test")
docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]}).set_index('docno')
print(f" Loaded {len(docs_df)} documents.")

def call_together_api(prompt: str, max_tokens: int, is_json=False) -> str:
    """A unified function to call the Together AI API."""
    try:
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        if is_json:
            payload["response_format"] = {"type": "json_object"}
            
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  - API Error: {e}")
        return f"Error from API: {e}"

def rerank_with_llm(documents, query):
    for doc in documents:
        prompt = f"Score the relevance of the document for the query on a scale of 0.0 to 1.0. Query: {query}. Document: {doc.get('text', '')[:2000]}. Respond with only the numeric score."
        try:
            response_text = call_together_api(prompt, 10)
            score_match = re.search(r'(\d\.\d+)', response_text)
            score = float(score_match.group(1)) if score_match else 0.0
        except: score = 0.0
        doc['llm_score'] = score
        time.sleep(2) 
    return sorted(documents, key=lambda x: x.get('llm_score', 0.0), reverse=True)

def generate_final_answer(reranked_docs, query):
    context = "\n\n".join([doc['text'] for doc in reranked_docs])
    prompt = f"Using these documents, answer the question clearly.\n\nDocuments:\n{context[:7000]}\n\nQuestion: {query}"
    return call_together_api(prompt, 1024)

def evaluate_with_ai_judge(result):
    """Uses an AI judge to automatically score the faithfulness and relevance."""
    if not result or 'answer' not in result or 'contexts' not in result:
        return {"faithfulness": 0, "answer_relevancy": 0}

    prompt = f"""You are an impartial AI evaluator. Your task is to evaluate a generated answer based on a given question and context.
Provide a score from 0.0 to 1.0 for 'faithfulness' and 'answer_relevancy'.
- Faithfulness: Does the answer strictly stick to the facts presented in the context?
- Answer Relevancy: Is the answer directly relevant to the question?
Respond ONLY with a single JSON object in the format: {{"faithfulness": <score>, "answer_relevancy": <score>}}

CONTEXT: {result['contexts']}
QUESTION: {result['question']}
ANSWER: {result['answer']}
JSON Response:"""
    try:
        evaluation_str = call_together_api(prompt, 100, is_json=True)
        return json.loads(evaluation_str)
    except Exception as e:
        print(f"  - AI Judge Error: {e}")
        return {"faithfulness": 0, "answer_relevancy": 0}

def run_full_pipeline(retriever_name, retriever_func, query):
    print(f"\n--- Running Full RAG Pipeline for: {retriever_name} ---")
    initial_docs = retriever_func(query)
    if not initial_docs: return None
    reranked_docs = rerank_with_llm(initial_docs, query)
    top_docs = reranked_docs[:FINAL_RERANK_DEPTH]
    final_answer = generate_final_answer(top_docs, query)
    return {'question': query, 'answer': final_answer, 'contexts': [doc.get('text', '') for doc in top_docs]}

def search_pyterrier(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    results_df = pt_retriever.search(cleaned_query)
    return pd.merge(results_df, docs_df, left_on='docno', right_index=True).to_dict(orient='records')

def search_elasticsearch(query):
    search_body = {"size": INITIAL_RETRIEVAL_DEPTH, "query": {"match": {"text": query}}}
    response = es_client.search(index=ES_INDEX_NAME, body=search_body)
    return [{"docno": h['_id'], "text": h['_source'].get('title', '') + ' ' + h['_source'].get('text', '')} for h in response['hits']['hits']]

evaluation_results = []
for query in tqdm(QUERIES, desc="Evaluating Queries"):
    pyterrier_result = run_full_pipeline("PyTerrier", search_pyterrier, query)
    elastic_result = run_full_pipeline("Elasticsearch", search_elasticsearch, query)
    
    print("  - Judging PyTerrier response...")
    pyterrier_eval = evaluate_with_ai_judge(pyterrier_result)
    time.sleep(2)
    
    print("  - Judging Elasticsearch response...")
    elastic_eval = evaluate_with_ai_judge(elastic_result)
    time.sleep(2)

    evaluation_results.append({
        "Query": query,
        "PyTerrier_Faithfulness": pyterrier_eval.get('faithfulness', 0),
        "PyTerrier_Relevancy": pyterrier_eval.get('answer_relevancy', 0),
        "Elastic_Faithfulness": elastic_eval.get('faithfulness', 0),
        "Elastic_Relevancy": elastic_eval.get('answer_relevancy', 0),
    })

print("\n\n==============================================")
print("      FINAL AUTOMATED WORKFLOW EVALUATION     ")
print("==============================================")
summary_df = pd.DataFrame(evaluation_results)
print(summary_df.round(3))
print("\n--- Average Scores ---")
print(summary_df.drop(columns=['Query']).mean())
print("\n==============================================")
