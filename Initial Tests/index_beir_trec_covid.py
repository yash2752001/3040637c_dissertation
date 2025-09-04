from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="http_ca.crt",  
    verify_certs=True,
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo")
)

with open("datasets/trec-covid/corpus.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

actions = []
for doc in docs:
    actions.append({
        "_index": "beir_trec_covid",
        "_id": doc["_id"],
        "_source": {
            "title": doc["title"],
            "text": doc["text"]
        }
    })

helpers.bulk(es, actions)

print("Indexing completed.")
