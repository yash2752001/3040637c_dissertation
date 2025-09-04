# Agentic AI for Retrieval-Augmented Generation: A Comparative Study

This project implements and evaluates a fully local and reproducible Retrieval-Augmented Generation (RAG) pipeline. It uses the EvoAgentX framework to orchestrate workflows, with PyTerrier for information retrieval and a locally-run Ollama LLM (Llama 3) for answer generation and evaluation.

The repository is organized into two main parts:
* `/Initial Tests`: Contains the scripts and code from the early, exploratory phases of the project, including initial setups with Elasticsearch, Pyterrier and the debate agent system.
* `/src (main)`: Contains the final, unified evaluation script used to run the experiments described in Evaluation section of the dissertation.

## Requirements
* Python 3.9+
* pip
* Ollama

## Installation
1.  **Clone this repository:**
    ```bash
    git clone [Insert Your Repo URL Here]
    cd [your-repo-name]
    ```

2.  **Install Python dependencies:**
    This will install all necessary libraries, including PyTerrier and EvoAgentX.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Ollama:**
    This project requires Ollama to run the Llama 3 model locally.
    * First, [install Ollama](https://ollama.com/) for your operating system.
    * Then, pull the Llama 3 model from the terminal:
        ```bash
        ollama pull llama3
        ```
