import os
import pyterrier as pt
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

def search_and_rag(query, index_path):
    if not pt.started():
        pt.java.init()

    indexref = pt.IndexFactory.of(index_path)
    retriever = pt.terrier.Retriever(indexref, wmodel="BM25")

    results = retriever.search(query).head(5)

    if results.empty:
        print("No documents found.")
        return

    print(f"Top {len(results)} documents:")
    for i, row in results.iterrows():
        print(f"{i+1}. DocNo: {row['docno']}, Score: {row['score']}")

    docs_text = "\n\n".join([f"- {row['docno']}: {row['docno']}" for _, row in results.iterrows()])

    # Setup EvoAgentX LLM for RAG
    llm_config_rag = OpenRouterConfig(
        model="openai/gpt-4o-mini",
        openrouter_key="",
        temperature=0.3,
        max_tokens=500
    )
    llm_rag = OpenRouterLLM(config=llm_config_rag)

    rag_prompt = f"""Using the following documents, answer the question clearly and concisely.

Documents:
{docs_text}

Question: {query}
"""

    rag_response = llm_rag.generate(prompt=rag_prompt)

    print("\n=== Final RAG Answer ===")
    print(rag_response.content)

def main():
    index_path = os.path.abspath(os.path.join(os.getcwd(), "var", "vaswani_index"))
    print(f"Using index at: {index_path}")
    query = input("Enter your search query: ").strip()
    search_and_rag(query, index_path)

if __name__ == "__main__":
    main()

