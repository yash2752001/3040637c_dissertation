This file contains the source code accompanying the MSc dissertation:

The code implements retrieval-augmented generation (RAG) pipelines, using PyTerrier for retrieval and a local Ollama LLM (Llama 3) for answer generation and evaluation.

The experiments replicate the evaluation tasks described in Chapters 3,4 of the dissertation.



Requirements

A) Install Python 3.9+ and pip.

Then install dependencies:
pip install -r requirements.txt


Additional Setup

1\. Ollama (Local LLM)

This project requires Ollama to run Llama 3 locally.
Install Ollama and pull the model: ollama pull llama3
By default, the scripts expect Ollama’s API server at:
http://localhost:11434/api/generate



2\. EvoAgentX
git clone https://github.com/EvoAgentX/EvoAgentX.git



3\. BEIR Datasets
The scripts will automatically download and unpack BEIR datasets (trec-covid, nfcorpus, scifact) if not already present.

