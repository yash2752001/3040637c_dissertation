import os
from elasticsearch import Elasticsearch
from evoagentx.models import OpenRouterConfig, OpenRouterLLM
from evoagentx.agents import Agent

os.environ["OPENROUTER_API_KEY"] = "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"

es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=("elastic", "Ne3uf10oYs8sDFXrMrvo"),
    verify_certs=True,
    ca_certs="http_ca.crt"
)

openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise Exception("OPENROUTER_API_KEY environment variable is not set!")

llm_config = OpenRouterConfig(
    model="openai/gpt-4o-mini",
    openrouter_key=openrouter_key,
    temperature=0.7,
    max_tokens=500,
)

agent1 = Agent(
    name="Agent 1",
    description="Argues in favor of the topic.",
    llm=OpenRouterLLM(config=llm_config)
)

agent2 = Agent(
    name="Agent 2",
    description="Argues against the topic.",
    llm=OpenRouterLLM(config=llm_config)
)


def search_elasticsearch(query, size=3):
    response = es.search(
        index="beir_trec_covid",
        query={"match": {"text": query}},
        size=size
    )
    return response["hits"]["hits"]

def run_debate(question, rounds=3):
    history = []
    current_question = question

    for i in range(rounds):
        docs = search_elasticsearch(current_question)
        docs_text = "\n\n".join([doc["_source"]["text"] for doc in docs])
        prompt1 = f"Using the following documents, argue your point about:\n\n{docs_text}\n\nQuestion: {current_question}\nAnswer:"
        response1 = agent1.llm.generate(prompt=prompt1)
        print(f"Agent 1: {response1.content}\n")
        history.append(("Agent 1", response1.content))

        current_question = response1.content
        docs = search_elasticsearch(current_question)
        docs_text = "\n\n".join([doc["_source"]["text"] for doc in docs])
        prompt2 = f"Using the following documents, rebut the previous statement:\n\n{docs_text}\n\nStatement: {current_question}\nResponse:"
        response2 = agent2.llm.generate(prompt=prompt2)
        print(f"Agent 2: {response2.content}\n")
        history.append(("Agent 2", response2.content))

        current_question = response2.content

    return history

if __name__ == "__main__":
    user_question = input("Enter the debate topic/question: ")
    debate_history = run_debate(user_question)
