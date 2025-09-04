# run_adaptive_benchmark.py
# A single, powerful script for Task 6 that performs a rigorous, professional evaluation
# of an Adaptive RAG workflow with Query Expansion.

import os
import re
import pandas as pd
from tqdm import tqdm
import warnings
import textwrap
import json

# --- Import All Necessary Libraries ---
import pyterrier as pt
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from scipy.stats import ttest_rel # For statistical significance
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Master Configuration ---
print("--- Configuring Final Benchmark for Task 6: Adaptive RAG ---")

# We will run the evaluation on these three standard BEIR datasets
DATASET_NAMES = ["trec-covid", "nfcorpus", "scifact"]
NUM_QUERIES_TO_SAMPLE = 25 # Use a sample of 25 queries from each dataset

# Ollama local server address
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# This is the threshold for our adaptive logic. If the top document's score is below this,
# the agent will adapt and try a better search method.
ADAPTIVE_SCORE_THRESHOLD = 10.0 

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# RAG Settings
INITIAL_RETRIEVAL_DEPTH = 10
FINAL_RERANK_DEPTH = 5 # Use a smaller k for the final answer to improve faithfulness

# --- 2. The EvoAgent Class ---
class EvoAgent:
    """An agent that uses a local LLM for intelligent tasks."""
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
print("\n--- Initializing PyTerrier and EvoAgent ---")
if not pt.java.started():
    # --- THIS IS THE FIX ---
    # The problematic 'terrier-rewrite' package download is not needed and has been removed.
    pt.java.init()
agent = EvoAgent(model_name=OLLAMA_MODEL)

# --- 4. Main Evaluation Loop ---
for ds_name in DATASET_NAMES:
    print(f"\n\n=======================================================")
    print(f"         RUNNING ADAPTIVE BENCHMARK FOR: {ds_name}         ")
    print("=======================================================")

    # --- Data Loading and Indexing ---
    dataset_path = os.path.join(os.getcwd(), f"../Stats/{ds_name}")
    corpus, queries, _ = GenericDataLoader(data_folder=dataset_path).load(split="test")
    pt_index_dir = os.path.join(SCRIPT_DIR, f"../Stats/var/{ds_name}_pt_index")
    pt_index = pt.IndexFactory.of(pt_index_dir)
    docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]}).set_index('docno')
    query_items = list(queries.items())[:NUM_QUERIES_TO_SAMPLE]
    print(f"✅ Loaded data for {ds_name} and sampled {len(query_items)} queries.")
    
    # --- Define Retrieval Models ---
    standard_retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=INITIAL_RETRIEVAL_DEPTH)
    # This is our advanced query expansion model
    advanced_retriever = standard_retriever >> pt.rewrite.RM3(pt_index) >> standard_retriever

    # --- Run Pipelines and Judge ---
    per_query_scores = []
    for qid, query in tqdm(query_items, desc=f"Processing {ds_name}"):
        cleaned_query = re.sub(r'[^\w\s]', '', query)
        
        # --- Standard Workflow ---
        std_docs_df = standard_retriever.search(cleaned_query)
        std_merged = pd.merge(std_docs_df, docs_df, left_on='docno', right_index=True)
        std_top_docs = std_merged.head(FINAL_RERANK_DEPTH).to_dict(orient='records')
        std_answer = agent.generate_answer(std_top_docs, query)
        std_eval = agent.judge_answer(query, std_answer, [d['text'] for d in std_top_docs])

        # --- Adaptive Workflow ---
        # Agent first checks the quality of the standard search
        top_score = std_docs_df.iloc[0]['score'] if not std_docs_df.empty else 0
        
        if top_score < ADAPTIVE_SCORE_THRESHOLD:
            # If score is low, ADAPT: use the better (but slower) query expansion model
            adaptive_docs_df = advanced_retriever.search(cleaned_query)
        else:
            # If score is good, no need to adapt, just use the standard results
            adaptive_docs_df = std_docs_df
            
        adaptive_merged = pd.merge(adaptive_docs_df, docs_df, left_on='docno', right_index=True)
        adaptive_top_docs = adaptive_merged.head(FINAL_RERANK_DEPTH).to_dict(orient='records')
        adaptive_answer = agent.generate_answer(adaptive_top_docs, query)
        adaptive_eval = agent.judge_answer(query, adaptive_answer, [d['text'] for d in adaptive_top_docs])

        per_query_scores.append({
            'std_faithfulness': std_eval.get('faithfulness', 0),
            'std_relevancy': std_eval.get('answer_relevancy', 0),
            'adaptive_faithfulness': adaptive_eval.get('faithfulness', 0),
            'adaptive_relevancy': adaptive_eval.get('answer_relevancy', 0),
        })
    
    # --- Calculate Averages and Statistical Significance ---
    scores_df = pd.DataFrame(per_query_scores)
    avg_scores = scores_df.mean()
    
    _, p_faithfulness = ttest_rel(scores_df['std_faithfulness'], scores_df['adaptive_faithfulness'])
    _, p_relevancy = ttest_rel(scores_df['std_relevancy'], scores_df['adaptive_relevancy'])
    
    # --- Display Results ---
    print(f"\n--- Results for {ds_name} ---")
    summary_data = {
        "Metric": ["Faithfulness", "Relevancy"],
        "Standard Workflow (Baseline)": [avg_scores['std_faithfulness'], avg_scores['std_relevancy']],
        "Adaptive Workflow": [avg_scores['adaptive_faithfulness'], avg_scores['adaptive_relevancy']],
        "p-value": [p_faithfulness, p_relevancy]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))

print("\n\n=======================================================")
print("              ALL BENCHMARKS COMPLETE              ")
print("=======================================================")
