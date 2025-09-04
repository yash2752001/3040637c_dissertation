# run_final_task4_benchmark.py
# A single, powerful script for Task 4 that performs a rigorous, professional evaluation.
# This version includes robust query cleaning and is compatible with your PyTerrier version.

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
print("--- Configuring Final Benchmark for Task 4 ---")

# We will run the evaluation on these three standard BEIR datasets
DATASET_NAMES = ["trec-covid", "nfcorpus", "scifact"]
NUM_QUERIES_TO_SAMPLE = 25 # Use a sample of 25 queries for AI-based tests

# Ollama local server address
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

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
    pt.java.init()
agent = EvoAgent(model_name=OLLAMA_MODEL)

# --- 4. Main Evaluation Loop ---
for ds_name in DATASET_NAMES:
    print(f"\n\n=======================================================")
    print(f"         RUNNING BENCHMARK FOR: {ds_name}         ")
    print("=======================================================")

    # --- Data Loading and Indexing ---
    dataset_path = os.path.join(os.getcwd(), ds_name)
    if not os.path.exists(dataset_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
        print(f"Downloading {ds_name}...")
        util.download_and_unzip(url, os.getcwd())
    
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
    
    index_dir = os.path.join(SCRIPT_DIR, "var", f"{ds_name}_pt_index")
    if not os.path.exists(index_dir):
        print(f"PyTerrier index not found for {ds_name}. Creating index... (This may take time)")
        docs_df_for_indexing = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]})
        indexer = pt.DFIndexer(index_dir, overwrite=True)
        indexer.index(docs_df_for_indexing["text"], docs_df_for_indexing["docno"])
    pt_index = pt.IndexFactory.of(index_dir)
    
    docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]}).set_index('docno')
    
    # --- THIS IS THE FIX ---
    # We pre-process ALL queries at the start to remove special characters.
    topics_df = pd.DataFrame.from_dict(queries, orient='index', columns=['original_query']).reset_index().rename(columns={'index': 'qid'})
    topics_df['query'] = topics_df['original_query'].apply(lambda q: re.sub(r'[^\w\s]', '', q))
    qrels_df = pd.DataFrame.from_records([(qid, doc_id, int(score)) for qid, docs in qrels.items() for doc_id, score in docs.items()], columns=['qid', 'docno', 'label'])
    
    # --- EXPERIMENT 1: SEARCH TUNING (TF-IDF vs. BM25 vs. BM25 + RM3) ---
    print("\n--- Experiment 1: Comparing Search Models ---")
    tfidf = pt.BatchRetrieve(pt_index, wmodel="Tf")
    bm25 = pt.BatchRetrieve(pt_index, wmodel="BM25")
    bm25_rm3 = bm25 >> pt.rewrite.RM3(pt_index) >> bm25
    
    # Run the experiment using the cleaned queries
    all_results = [tfidf(topics_df), bm25(topics_df), bm25_rm3(topics_df)]
    
    # Calculate the main evaluation table
    results_df = pt.Experiment(
        all_results,
        topics_df,
        qrels_df,
        eval_metrics=["ndcg_cut_10", "map"],
        names=["TF-IDF", "BM25", "BM25 + RM3"]
    )
    
    # Manually calculate p-values
    baseline_scores_df = pt.Experiment([all_results[1]], topics_df, qrels_df, eval_metrics=["ndcg_cut_10"], perquery=True)
    tfidf_scores_df = pt.Experiment([all_results[0]], topics_df, qrels_df, eval_metrics=["ndcg_cut_10"], perquery=True)
    _, p_value_tfidf = ttest_rel(tfidf_scores_df['value'], baseline_scores_df['value'])
    rm3_scores_df = pt.Experiment([all_results[2]], topics_df, qrels_df, eval_metrics=["ndcg_cut_10"], perquery=True)
    _, p_value_rm3 = ttest_rel(rm3_scores_df['value'], baseline_scores_df['value'])

    print(f"\n--- Results for {ds_name} ---")
    print(results_df)
    print(f"\n- p-value (TF-IDF vs BM25): {p_value_tfidf:.4f}")
    print(f"- p-value (BM25+RM3 vs BM25): {p_value_rm3:.4f}")

    # --- EXPERIMENTS 2 & 3 Setup ---
    sampled_topics = topics_df.head(NUM_QUERIES_TO_SAMPLE)
    
    # --- EXPERIMENT 2: QUERY HANDLING (Keywords vs. Full Questions) ---
    print("\n--- Experiment 2: Comparing Query Types ---")
    per_query_scores_qtype = []
    retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=5)
    for _, row in tqdm(sampled_topics.iterrows(), total=len(sampled_topics), desc="Processing Query Types"):
        full_question = row['original_query']
        keyword_query = row['query'] # Use the already cleaned query as the "keyword" version

        docs_full = retriever.search(keyword_query) # Search with cleaned query
        merged_full = pd.merge(docs_full, docs_df, left_on='docno', right_index=True)
        answer_full = agent.generate_answer(merged_full.to_dict(orient='records'), full_question) # Generate with original query
        eval_full = agent.judge_answer(full_question, answer_full, [d['text'] for d in merged_full.to_dict(orient='records')])
        
        docs_keyword = retriever.search(keyword_query)
        merged_keyword = pd.merge(docs_keyword, docs_df, left_on='docno', right_index=True)
        answer_keyword = agent.generate_answer(merged_keyword.to_dict(orient='records'), full_question)
        eval_keyword = agent.judge_answer(full_question, answer_keyword, [d['text'] for d in merged_keyword.to_dict(orient='records')])
        
        per_query_scores_qtype.append({'full_relevancy': eval_full.get('answer_relevancy', 0), 'keyword_relevancy': eval_keyword.get('answer_relevancy', 0)})
        
    qtype_df = pd.DataFrame(per_query_scores_qtype)
    _, p_relevancy = ttest_rel(qtype_df['full_relevancy'], qtype_df['keyword_relevancy'])
    print(f"\n- Average Relevancy (Full Question): {qtype_df['full_relevancy'].mean():.4f}")
    print(f"- Average Relevancy (Keywords):    {qtype_df['keyword_relevancy'].mean():.4f}")
    print(f"- p-value: {p_relevancy:.4f}")

    # --- EXPERIMENT 3: CALCULATION TUNING (k=5 vs. k=15) ---
    print("\n--- Experiment 3: Comparing Retrieval Depth (k) ---")
    per_query_scores_k = []
    retriever_k5 = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=5)
    retriever_k15 = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=15)
    for _, row in tqdm(sampled_topics.iterrows(), total=len(sampled_topics), desc="Processing Retrieval Depth"):
        query = row['original_query']
        cleaned_query = row['query']
        
        docs_k5 = retriever_k5.search(cleaned_query)
        merged_k5 = pd.merge(docs_k5, docs_df, left_on='docno', right_index=True)
        answer_k5 = agent.generate_answer(merged_k5.to_dict(orient='records'), query)
        eval_k5 = agent.judge_answer(query, answer_k5, [d['text'] for d in merged_k5.to_dict(orient='records')])
        
        docs_k15 = retriever_k15.search(cleaned_query)
        merged_k15 = pd.merge(docs_k15, docs_df, left_on='docno', right_index=True)
        answer_k15 = agent.generate_answer(merged_k15.to_dict(orient='records'), query)
        eval_k15 = agent.judge_answer(query, answer_k15, [d['text'] for d in merged_k15.to_dict(orient='records')])
        
        per_query_scores_k.append({'faithfulness_k5': eval_k5.get('faithfulness', 0), 'faithfulness_k15': eval_k15.get('faithfulness', 0)})

    k_df = pd.DataFrame(per_query_scores_k)
    _, p_faithfulness = ttest_rel(k_df['faithfulness_k5'], k_df['faithfulness_k15'])
    print(f"\n- Average Faithfulness (k=5):  {k_df['faithfulness_k5'].mean():.4f}")
    print(f"- Average Faithfulness (k=15): {k_df['faithfulness_k15'].mean():.4f}")
    print(f"- p-value: {p_faithfulness:.4f}")

print("\n\n=======================================================")
print("              ALL BENCHMARKS COMPLETE              ")
print("=======================================================")
