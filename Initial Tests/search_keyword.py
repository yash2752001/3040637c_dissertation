from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://localhost:9200"],
    verify_certs=True,
    ca_certs="http_ca.crt",
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo")
)

index_name = "beir_trec_covid"

query_text = "covid symptoms"

response = es.search(
    index=index_name,
    query={
        "match": {
            "text": query_text
        }
    },
    size=5
)

print("Total hits:", response['hits']['total']['value'])
print(f"Top documents for query: '{query_text}'")

for hit in response['hits']['hits']:
    print(f"Score: {hit['_score']}")
    print(f"Title: {hit['_source']['title']}")
    print(f"Text snippet: {hit['_source']['text'][:200]}...") 
    print("--------------------")
