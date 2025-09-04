from elasticsearch import Elasticsearch
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo"),
    verify_certs=True,
    ca_certs="http_ca.crt",
)

# indexing a test document
doc = {
    "title": "AI Article",
    "content": "Artificial intelligence is transforming industries..."
}
es.index(index="ai-articles", id=1, document=doc)
es.indices.refresh(index="ai-articles")

# searching for the same document
search_resp = es.search(
    index="ai-articles",
    query={"match_all": {}}
)
hits = search_resp["hits"]["hits"]

if not hits:
    print("No documents found.")
    exit()

# extracting text to summarize
doc_content = hits[0]["_source"]["content"]

config = OpenRouterConfig(
    model="openai/gpt-4o-mini",
    openrouter_key="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", 
    temperature=0.3,
    max_tokens=500
)
llm = OpenRouterLLM(config=config)

prompt = f"Summarize this text:\n\n{doc_content}"
summary = llm.generate(prompt=prompt)

print("\n=== LLM Generated Summary ===")
print(summary)

# saving the summary back to Elasticsearch
summary_doc = {
    "original_doc_id": hits[0]["_id"],
    "summary": summary.content
}
es.index(index="ai-article-summaries", document=summary_doc)
print("Saved summary to index 'ai-article-summaries'.")
