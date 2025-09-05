from elasticsearch import Elasticsearch
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

es = Elasticsearch(
    ["https://localhost:9200"],
    verify_certs=True,
    ca_certs="http_ca.crt",
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo")
)

index_name = "beir_trec_covid"

# getting user query
user_query = input("Enter your question: ")

# searching ES for top 5 docs matching 'text' field
search_resp = es.search(
    index=index_name,
    query={"match": {"text": user_query}},
    size=5
)

hits = search_resp["hits"]["hits"]

if not hits:
    print("No documents found for your query.")
    exit()

print(f"Found {len(hits)} documents. Now reranking...")

# LLM configs for both RAG and reranking
llm_config_rag = OpenRouterConfig(
    model="openai/gpt-4o-mini",
    openrouter_key="",
    temperature=0.3,
    max_tokens=500
)
llm_rag = OpenRouterLLM(config=llm_config_rag)

llm_config_rank = OpenRouterConfig(
    model="openai/gpt-4o-mini",
    openrouter_key="",
    temperature=0.0,  
    max_tokens=100
)
llm_rank = OpenRouterLLM(config=llm_config_rank)

# reranking documents with LLM scoring
def score_doc(doc_text, query):
    prompt = f"On a scale of 0 to 1, how relevant is this document to the query?\n\nQuery: {query}\nDocument: {doc_text}\nRelevance score:"
    response = llm_rank.generate(prompt=prompt)
    try:
        score = float(response.content.strip())
    except:
        score = 0.0
    return score

for hit in hits:
    doc_text = hit["_source"]["text"]
    hit["llm_score"] = score_doc(doc_text, user_query)

hits = sorted(hits, key=lambda x: x["llm_score"], reverse=True)

print("\nReranked scores:")
for i, hit in enumerate(hits):
    print(f"{i+1}. Score: {hit['llm_score']:.3f} Title: {hit['_source']['title']}")

# combining top 3 reranked docs for RAG input
top_docs_text = "\n\n".join(f"- {hit['_source']['text']}" for hit in hits[:3])

rag_prompt = f"""Using the following documents, answer the question clearly and concisely.

Documents:
{top_docs_text}

Question: {user_query}
"""

# final answer
rag_response = llm_rag.generate(prompt=rag_prompt)

print("\n=== Final RAG Answer ===")
print(rag_response.content)

# saving Q&A and scores back to Elasticsearch
answer_doc = {
    "question": user_query,
    "answer": rag_response.content,
    "retrieved_docs": [
        {
            "doc_id": hit["_id"],
            "title": hit["_source"]["title"],
            "score": hit["llm_score"]
        }
        for hit in hits[:3]
    ]
}

es.index(index="ai-article-answers", document=answer_doc)
print("\nAnswer saved to 'ai-article-answers' index.")

