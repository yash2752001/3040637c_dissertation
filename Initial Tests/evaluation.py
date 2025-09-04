import time
from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://localhost:9200"],
    verify_certs=True,
    ca_certs="http_ca.crt", 
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo") 
)

index_name = "beir_trec_covid"

def search_elasticsearch(query, top_k=10):
    start_time = time.time()
    response = es.search(
        index=index_name,
        query={"match": {"text": query}},
        size=top_k
    )
    elapsed = time.time() - start_time
    hits = response["hits"]["hits"]
    return hits, elapsed

def main():
    user_query = input("Enter your search query or keyword: ").strip()
    if not user_query:
        print("No query entered, exiting.")
        return

    hits, elapsed = search_elasticsearch(user_query)
    num_results = len(hits)

    print(f"\nSearch completed in {elapsed:.3f} seconds.")
    print(f"Found {num_results} document(s) matching your query.\n")

    if num_results == 0:
        print("No documents found.")
        return

    for i, hit in enumerate(hits, 1):
        title = hit["_source"].get("title", "No Title")
        score = hit["_score"]
        print(f"{i}. Title: {title}")
        print(f"   Elasticsearch Score: {score:.4f}\n")

if __name__ == "__main__":
    main()
