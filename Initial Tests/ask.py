import requests
import json

SERVER_URL = "http://127.0.0.1:5000/query"

print("RAG Query Client")
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
            score = doc.get('llm_score', 0)
            preview = doc.get('text_preview', 'No preview available.')
            print(f"{i+1}. Doc ID: {doc_id} | Score: {score:.2f}")
            print(f"   Preview: {preview}\n")

        print("="*50)

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to the server. Please make sure data_app.py is running. Details: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")