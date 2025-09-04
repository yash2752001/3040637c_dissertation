from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://localhost:9200"],
    verify_certs=True,
    ca_certs="http_ca.crt",  
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo")
)

index_name = "beir_trec_covid"

response = es.search(index=index_name, query={"match_all": {}}, size=5)  # get 5 docs only

print("Total hits:", response['hits']['total']['value'])
print("Sample documents:")

for hit in response['hits']['hits']:
    print(f"Score: {hit['_score']}")
    print(f"Source: {hit['_source']}")
    print("--------------------")
