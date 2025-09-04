# run_agent_evaluation.py
# The final, fast evaluation script, re-engineered to use an explicit EvoAgent class.

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
import requests
from sentence_transformers.cross_encoder import CrossEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Master Configuration ---
print("--- Configuring FAST Local Workflow Evaluation with EvoAgent ---")

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
ES_HOST, ES_USER, ES_PASSWORD = "https://localhost:9200", "elastic", "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS, ES_INDEX_NAME = "../http_ca.crt", "beir-trec-covid"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PT_INDEX_DIR = os.path.join(SCRIPT_DIR, "../Stats/var/trec_covid_index")
DATASET_PATH = os.path.join(SCRIPT_DIR, "../Stats/trec-covid")
INITIAL_RETRIEVAL_DEPTH, FINAL_RERANK_DEPTH = 15, 10
QUERIES = [
    "what is the incubation period of covid?",
    "social impact of covid-19 pandemic",
    "long term effects of coronavirus"
]

# --- 2. The EvoAgent Class ---
class EvoAgent:
    """An agent that uses different models for different tasks."""
    def __init__(self, reranker_model_name, generation_model_name):
        print(f"Initializing EvoAgent with Reranker: {reranker_model_name} and Generator: {generation_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)
        self.generator_model = generation_model_name
        self.judge_model = generation_model_name # Use the same powerful model for judging
        print("✅ EvoAgent Initialized.")

    def _call_ollama(self, prompt, is_json=False):
        try:
            payload = {"model": self.generator_model, "prompt": prompt, "stream": False}
            if is_json: payload["format"] = "json"
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            return json.loads(response.text).get('response', '')
        except Exception as e:
            return f"Error from Ollama API: {e}"

    def rerank(self, documents, query):
        """Reranks documents using a fast cross-encoder."""
        model_input = [[query, doc.get('text', '')] for doc in documents]
        scores = self.reranker.predict(model_input)
        for i, doc in enumerate(documents):
            doc['rerank_score'] = scores[i]
        return sorted(documents, key=lambda x: x.get('rerank_score', 0.0), reverse=True)

    def generate(self, reranked_docs, query):
        """Generates an answer using a powerful LLM."""
        context = "\n\n".join([doc['text'] for doc in reranked_docs])
        prompt = f"Using these documents, answer the question clearly.\n\nDocuments:\n{context[:7000]}\n\nQuestion: {query}"
        return self._call_ollama(prompt)
    
    def judge(self, question, answer, contexts):
        """Acts as an AI judge to score a response."""
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

# --- 3. Initialize Systems & Load Data ---
print("\n--- Initializing PyTerrier, Elasticsearch, and EvoAgent ---")
if not pt.java.started(): pt.java.init()
pt_index = pt.IndexFactory.of(PT_INDEX_DIR)
pt_retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=INITIAL_RETRIEVAL_DEPTH)
print("✅ PyTerrier is ready.")
try:
    es_client = Elasticsearch([ES_HOST], basic_auth=(ES_USER, ES_PASSWORD), verify_certs=True, ca_certs=ES_CA_CERTS)
    if not es_client.ping(): raise ConnectionError("Could not connect.")
    print("✅ Elasticsearch is ready.")
except Exception as e:
    print(f"❌ Could not connect to Elasticsearch: {e}. Please ensure the service is running.")
    exit()

# Create our agent instance
agent = EvoAgent(
    reranker_model_name='cross-encoder/ms-marco-TinyBERT-L-2',
    generation_model_name='llama3'
)

corpus, _, _ = GenericDataLoader(data_folder=DATASET_PATH).load(split="test")
docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]}).set_index('docno')
print(f"✅ Loaded {len(docs_df)} documents.")

# --- 4. Define Search Functions & Main Pipeline ---
def search_pyterrier(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    results_df = pt_retriever.search(cleaned_query)
    return pd.merge(results_df, docs_df, left_on='docno', right_index=True).to_dict(orient='records')

def search_elasticsearch(query):
    search_body = {"size": INITIAL_RETRIEVAL_DEPTH, "query": {"match": {"text": query}}}
    response = es_client.search(index=ES_INDEX_NAME, body=search_body)
    return [{"docno": h['_id'], "text": h['_source'].get('title', '') + ' ' + h['_source'].get('text', '')} for h in response['hits']['hits']]

def run_full_pipeline(retriever_name, retriever_func, query):
    print(f"\n--- Running Full Pipeline for: {retriever_name} ---")
    initial_docs = retriever_func(query)
    if not initial_docs: return None
    # Use the agent to rerank and generate
    reranked_docs = agent.rerank(initial_docs, query)
    top_docs = reranked_docs[:FINAL_RERANK_DEPTH]
    final_answer = agent.generate(top_docs, query)
    return {'question': query, 'answer': final_answer, 'contexts': [doc.get('text', '') for doc in top_docs]}

# --- 5. Main Evaluation Loop ---
evaluation_results = []
for query in tqdm(QUERIES, desc="Evaluating Queries"):
    pyterrier_result = run_full_pipeline("PyTerrier", search_pyterrier, query)
    elastic_result = run_full_pipeline("Elasticsearch", search_elasticsearch, query)
    
    # Use the agent to judge the results
    print("  - Agent Judging PyTerrier response...")
    pyterrier_eval = agent.judge(pyterrier_result['question'], pyterrier_result['answer'], pyterrier_result['contexts'])
    
    print("  - Agent Judging Elasticsearch response...")
    elastic_eval = agent.judge(elastic_result['question'], elastic_result['answer'], elastic_result['contexts'])

    evaluation_results.append({
        "Query": query,
        "PyTerrier_Faithfulness": pyterrier_eval.get('faithfulness', 0),
        "PyTerrier_Relevancy": pyterrier_eval.get('answer_relevancy', 0),
        "Elastic_Faithfulness": elastic_eval.get('faithfulness', 0),
        "Elastic_Relevancy": elastic_eval.get('answer_relevancy', 0),
    })

# --- 6. Display Final Results ---
print("\n\n==============================================")
print("      FINAL AUTOMATED WORKFLOW EVALUATION (EVOAGENT)     ")
print("==============================================")
summary_df = pd.DataFrame(evaluation_results)
print(summary_df.round(3))
print("\n--- Average Scores ---")
print(summary_df.drop(columns=['Query']).mean())
print("\n==============================================")
