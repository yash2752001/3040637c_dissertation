from elasticsearch import Elasticsearch

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo"),
    verify_certs=False
)

doc = {
    "title": "Hello from Elasticsearch",
    "content": "This is a simple demo document."
}

es.index(index="demo-index", id=1, document=doc, refresh=True)

res = es.search(index="demo-index", query={"match_all": {}})
print("ES search results:", res)
