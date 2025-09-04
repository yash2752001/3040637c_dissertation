# run_final_task1_benchmark.py
# A single, powerful script for Task 1 that performs a rigorous, professional evaluation.
# It uses multiple datasets and statistical significance testing, with ZERO API calls.

import os
import re
import pandas as pd
from tqdm import tqdm
import warnings

# --- Import All Necessary Libraries ---
import pyterrier as pt
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from scipy.stats import ttest_rel # For statistical significance

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. Master Configuration ---
print("--- Configuring Final Benchmark for Task 1 ---")

# We will run the evaluation on these three standard BEIR datasets
DATASET_NAMES = ["trec-covid", "nfcorpus", "scifact"]

# Elasticsearch Connection Details
ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "Ne3uf10oYs8sDFXrMrvo"
ES_CA_CERTS = "../http_ca.crt" # Path relative to the 'Stats' folder

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

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

# --- 3. Main Evaluation Loop ---
# The script will loop through each dataset and run the full benchmark
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
    
    # --- PyTerrier Indexing ---
    pt_index_dir = os.path.join(SCRIPT_DIR, "var", f"{ds_name}_pt_index")
    if not os.path.exists(pt_index_dir):
        print(f"PyTerrier index not found for {ds_name}. Creating index...")
        docs_df = pd.DataFrame({'docno': list(corpus.keys()), 'text': [d.get('title', '') + ' ' + d.get('text', '') for d in corpus.values()]})
        indexer = pt.DFIndexer(pt_index_dir, overwrite=True)
        indexer.index(docs_df["text"], docs_df["docno"])
    pt_index = pt.IndexFactory.of(pt_index_dir)

    # --- Elasticsearch Indexing ---
    es_index_name = f"beir-{ds_name.replace('-', '_')}" # Ensure valid index name
    if not es_client.indices.exists(index=es_index_name):
        print(f"Elasticsearch index not found for {ds_name}. Creating index...")
        es_client.indices.create(index=es_index_name)
        actions = [{"_index": es_index_name, "_id": doc_id, "_source": doc} for doc_id, doc in corpus.items()]
        bulk(es_client, actions)

    # --- 4. Define Retrieval Models ---
    pt_retriever = pt.BatchRetrieve(pt_index, wmodel="BM25", num_results=100)

    def search_elasticsearch(query_text):
        search_body = {"size": 100, "query": {"match": {"text": query_text}}}
        response = es_client.search(index=es_index_name, body=search_body)
        return {hit['_id']: hit['_score'] for hit in response['hits']['hits']}

    # --- 5. Run Queries ---
    print(f"\n--- Running {len(queries)} queries for {ds_name} ---")
    pyterrier_results = {}
    elastic_results = {}
    for qid, query in tqdm(queries.items(), desc=f"Processing {ds_name}"):
        pt_res_df = pt_retriever.search(re.sub(r'[^\w\s]', '', query))
        pyterrier_results[qid] = dict(zip(pt_res_df['docno'], pt_res_df['score']))
        elastic_results[qid] = search_elasticsearch(query)

    # --- 6. Evaluate and Calculate Statistical Significance ---
    evaluator = EvaluateRetrieval()
    k_values = [10] # We will focus on the top 10 results
    
    # Get overall scores
    scores_pt = evaluator.evaluate(qrels, pyterrier_results, k_values)
    scores_es = evaluator.evaluate(qrels, elastic_results, k_values)
    
    # Manually calculate nDCG@10 for each query to perform the t-test
    per_query_ndcg_pt = {}
    per_query_ndcg_es = {}
    for qid in queries.keys():
        # Ensure the query exists in the results before evaluating
        if qid in pyterrier_results and qid in elastic_results and qid in qrels:
            per_query_ndcg_pt[qid] = evaluator.evaluate({qid: qrels[qid]}, {qid: pyterrier_results[qid]}, k_values)[0]['NDCG@10']
            per_query_ndcg_es[qid] = evaluator.evaluate({qid: qrels[qid]}, {qid: elastic_results[qid]}, k_values)[0]['NDCG@10']
    
    # Perform paired t-test on the lists of scores
    t_stat, p_value = ttest_rel(list(per_query_ndcg_pt.values()), list(per_query_ndcg_es.values()))

    # --- 7. Display Results ---
    print(f"\n--- Results for {ds_name} ---")
    summary_data = {
        "Metric": ["nDCG@10", "Recall@10", "Precision@10"],
        "PyTerrier": [scores_pt[0]['NDCG@10'], scores_pt[2]['Recall@10'], scores_pt[3]['P@10']],
        "Elasticsearch": [scores_es[0]['NDCG@10'], scores_es[2]['Recall@10'], scores_es[3]['P@10']]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))
    print("\n--- Statistical Significance (for nDCG@10) ---")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = "Elasticsearch" if summary_df.loc[0, "Elasticsearch"] > summary_df.loc[0, "PyTerrier"] else "PyTerrier"
        print(f"Conclusion: The difference is statistically significant. {winner} is the clear winner.")
    else:
        print("Conclusion: The difference is NOT statistically significant.")

print("\n\n=======================================================")
print("              ALL BENCHMARKS COMPLETE              ")
print("=======================================================")
