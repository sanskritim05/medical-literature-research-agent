# Medical Literature Research Agent

Medical Literature Research Agent is a Python app that answers clinical questions using medical literature. The user enters a question in natural language, the app searches PubMed, fetches abstracts, summarizes the findings, and returns a final answer with citations.

## Features

- Search PubMed using the free NCBI Entrez API
- Optional ClinicalTrials.gov search for ongoing studies
- Filter by year range and study type
- Compare two treatments or approaches
- Follow-up questions in the same session
- Confidence score for the evidence
- PDF export
- Simple web interface

## Tech Stack

- Python
- FastAPI
- LangGraph
- Groq or Ollama
- PubMed Entrez API
- ClinicalTrials.gov API
- ChromaDB

## Clone

```bash
git clone https://github.com/sanskritim05/medical-literature-research-agent.git
cd medical-literature-research-agent
```

## Setup

1. Create an environment file:

```bash
cp .env.example .env
```

2. Add your model settings to `.env`.

For Groq:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

For Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
uvicorn main:app --reload
```

5. Open:

```text
http://127.0.0.1:8000
```

## Example Questions

- In adults with acute low back pain, do NSAIDs improve pain and function compared with acetaminophen?
- For type 2 diabetes, do GLP-1 receptor agonists reduce cardiovascular events compared with standard care?
- In children with acute otitis media, when is watchful waiting appropriate compared with immediate antibiotics?
- Compare intratympanic steroids versus oral steroids for idiopathic sudden sensorineural hearing loss.

## Project Structure

```text
medical-literature-research-agent/
├── main.py
├── agent.py
├── pubmed_tool.py
├── frontend/
│   └── index.html
├── .env.example
├── requirements.txt
└── README.md
```
