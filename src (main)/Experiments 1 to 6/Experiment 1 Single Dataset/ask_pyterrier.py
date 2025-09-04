import requests
import json

SERVER_URL = "http://127.0.0.1:5002/query" 

print("RAG Query Client")
print(f"Talking to server at: {SERVER_URL}")
print("Enter your question below, or type 'exit' to quit.")
print("="*50)

while True:
    try:
        user_query = input("Your Question: ")
        if user_query.lower() == 'exit':
            print("Exiting client.")
            break

        headers = {"Content-Type": "application/json"}
        payload = {"query": user_query}

        print("\n...Sending query to server...")
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()

        result = response.json()
        
        print("\n--- Generated Answer ---")
        print(result.get('answer', 'No answer was generated.'))
        
        print("\n--- Top 5 Reranked Documents (for evaluation) ---")
        reranked_docs = result.get('reranked_documents', [])
        for i, doc in enumerate(reranked_docs[:5]):
            doc_id = doc.get('docno')
            llm_score = doc.get('llm_score', 0)
            
            initial_score = doc.get('bm25_score') or doc.get('es_score') or 0.0
            score_type = "BM25" if 'bm25_score' in doc else "ES"

            print(f"{i+1}. Doc ID: {doc_id} | Initial {score_type} Score: {initial_score:.2f} | LLM Rerank Score: {llm_score:.2f}")

        print("\n" + "="*50)

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to the server. Please make sure the server is running. Details: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")