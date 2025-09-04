from elasticsearch import Elasticsearch
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo"),
    verify_certs=True,
    ca_certs="http_ca.crt",
)

user_query = input("Enter your question: ")

search_resp = es.search(
    index="ai-articles",
    query={
        "multi_match": {
            "query": user_query,
            "fields": ["title", "content"]
        }
    },
    size=3
)
hits = search_resp["hits"]["hits"]

if not hits:
    print("No documents found for your query.")
    exit()

# combining retrieved content
retrieved_texts = "\n\n".join(
    f"- {hit['_source']['content']}" for hit in hits
)

config = OpenRouterConfig(
    model="openai/gpt-4o-mini",
    openrouter_key="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",
    temperature=0.3,
    max_tokens=500
)
llm = OpenRouterLLM(config=config)

prompt = f"""Using the following retrieved documents, answer the question below.

Documents:
{retrieved_texts}

Question: {user_query}

Answer in a clear and concise paragraph:
"""

response = llm.generate(prompt=prompt)

print("\n=== Generated Answer ===")
print(response.content)

answer_doc = {
    "question": user_query,
    "answer": response.content,
    "retrieved_doc_ids": [hit["_id"] for hit in hits]
}
es.index(index="ai-article-answers", document=answer_doc)

print("\nAnswer saved to index 'ai-article-answers'.")
