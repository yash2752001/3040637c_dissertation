import os
import pandas as pd
import pyterrier as pt
import shutil
import warnings

print("--- Step 1: Creating a local dataset ---")

csv_data = """docno,text
doc_1,"PyTerrier is a Python framework for running experiments in information retrieval."
doc_2,"Information retrieval is the science of searching for information in documents."
doc_3,"BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query."
doc_4,"The vaswani collection is a standard test collection for information retrieval models."
doc_5,"A search engine's main components are the crawler, the indexer, and the query processor."
doc_6,"Vector search uses dense vector representations of documents and queries to find semantically similar results."
doc_7,"Language models like GPT can be used for zero-shot reranking by evaluating document relevance without specific training."
doc_8,"The precision and recall are two fundamental metrics for evaluating the quality of a search system."
doc_9,"Term frequency (TF) and inverse document frequency (IDF) are the building blocks of many scoring models, including BM25."
doc_10,"Retrieval-Augmented Generation, or RAG, combines a retriever with a generator model to produce informed answers."
"""

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_FOLDER, "documents.csv")

os.makedirs(DATA_FOLDER, exist_ok=True)

try:
    with open(CSV_PATH, "w", encoding='utf-8') as f:
        f.write(csv_data)
    print(f" Successfully created 'documents.csv' with 10 documents.")
except Exception as e:
    print(f" Failed to create the CSV file. Error: {e}")
    exit()

print("\n--- Step 2: Building the search index ---")

INDEX_DIR = os.path.join(SCRIPT_DIR, "var", "pyterrier_index")

if os.path.exists(INDEX_DIR):
    print(f"Deleting old index at: {INDEX_DIR}")
    shutil.rmtree(INDEX_DIR)
print(f"Ensuring index directory exists at: {INDEX_DIR}")
os.makedirs(INDEX_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=DeprecationWarning)

if not pt.java.started():
    print(" Starting PyTerrier...")
    pt.java.set_memory_limit(4096)
    pt.java.init()
    print(" PyTerrier started successfully.")
else:
    print(" PyTerrier is already running.")
    
try:
    documents_df = pd.read_csv(CSV_PATH)
    print(" Successfully loaded 'documents.csv' into a DataFrame.")
except Exception as e:
    print(f" Failed to load the CSV file into pandas. Error: {e}")
    exit()

try:
    print("⏳ Building index from the DataFrame...")
    indexer = pt.DFIndexer(INDEX_DIR, overwrite=True)
    index_ref = indexer.index(documents_df["text"], documents_df["docno"])
    print(f"Index created successfully at: {INDEX_DIR}")
except Exception as e:
    print(f"An error occurred during indexing: {e}")
    exit()

print("\n\n SUCCESS! ")
print("The local data is created and the index is built with the new, larger dataset.")