<a id="readme-top"></a>

<div align="center">
  <h1>Medical Literature Research Agent</h1>
  <p>
    A Python web app that answers clinical questions by searching PubMed, summarizing abstracts, and returning a final evidence-based answer with citations.
  </p>
</div>

## Table of Contents

- [About](#about)
- [Features](#features)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Questions](#example-questions)
- [Project Structure](#project-structure)

## About

Medical Literature Research Agent helps users explore medical evidence through a simple web interface. The user enters a clinical question in natural language, the app searches PubMed, retrieves abstracts, summarizes the findings, and produces a final answer with inline citations and linked references.

It also supports optional filters, treatment comparison, session-based follow-up questions, PDF export, and ongoing trial lookup from ClinicalTrials.gov.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

- Search PubMed using the free NCBI Entrez API
- Summarize abstracts and synthesize a final answer with citations
- Filter by year range and study type
- Compare two treatments or approaches
- Continue with follow-up questions in the same session
- Show a confidence score for the evidence
- Export results as PDF
- Include ongoing studies from ClinicalTrials.gov

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

- Python
- FastAPI
- LangGraph
- Groq or Ollama
- PubMed Entrez API
- ClinicalTrials.gov API
- ChromaDB

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Clone the repository

```bash
git clone https://github.com/sanskritim05/medical-literature-research-agent.git
cd medical-literature-research-agent
```

### Create the environment file

```bash
cp .env.example .env
```

### Configure the model

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

If using Ollama, start it first:

```bash
ollama pull llama3.1:8b
ollama serve
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

1. Enter a clinical question.
2. Optionally choose a study type or date range.
3. Run the search.
4. Review the final answer, citations, and references.
5. Optionally compare two treatments, simplify the answer, or export the results as a PDF.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Example Questions

- In adults with acute low back pain, do NSAIDs improve pain and function compared with acetaminophen?
- For type 2 diabetes, do GLP-1 receptor agonists reduce cardiovascular events compared with standard care?
- In children with acute otitis media, when is watchful waiting appropriate compared with immediate antibiotics?
- Compare intratympanic steroids versus oral steroids for idiopathic sudden sensorineural hearing loss.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```text
medical-literature-research-agent/
├── main.py
├── agent.py
├── pubmed_tool.py
├── frontend/
│   ├── index.html
│   └── styles.css
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
