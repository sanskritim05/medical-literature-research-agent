# Medical Literature Research Agent

An evidence-focused research workspace built with LangGraph, PubMed, ClinicalTrials.gov, and either Groq or Ollama for synthesis. Users can ask a clinical question in natural language, filter by date and study type, run follow-up questions in the same session, compare two treatments, inspect highlighted evidence sentences, export results as PDF, and simplify the answer for a patient-friendly reading level.

## What It Showcases

- LangGraph orchestration for retrieval and synthesis
- PubMed search through the free NCBI Entrez API using `requests` and `xml.etree.ElementTree`
- Optional ClinicalTrials.gov retrieval for ongoing trial context
- ChromaDB caching for previously fetched abstracts
- Confidence scoring to flag weak or conflicting evidence
- Multi-turn session memory for follow-up questions
- Live research activity panel in the frontend
- PDF report export for saving literature summaries
- Optional LangSmith tracing support for observability and screenshots
- Optional Ollama support for fully local inference

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

## Setup

1. Clone the repository and move into the project directory.

```bash
git clone <your-repo-url>
cd medical-literature-research-agent
```

2. Create `.env` from the template.

```bash
cp .env.example .env
```

3. Pick your LLM provider.

Groq:

```bash
# in .env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Ollama:

```bash
# in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

If you use Ollama, start the model locally first:

```bash
ollama pull llama3.1:8b
ollama serve
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

5. Run the app.

```bash
uvicorn main:app --reload
```

6. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## LangSmith Tracing

LangSmith is optional. To enable tracing on the free tier, set these in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=medical-literature-research-agent
LANGSMITH_API_KEY=your_langsmith_key
```

With tracing enabled, LangGraph and LangChain runs can be inspected in LangSmith for debugging, demos, and README screenshots.

## Features

- Standard synthesis mode for one clinical question
- Compare mode for treatment-versus-treatment literature reviews
- Year range and study type filtering
- ClinicalTrials.gov search for ongoing studies
- ChromaDB abstract caching and reuse
- Confidence score with rationale
- History sidebar and follow-up memory
- Evidence highlight sentences extracted from abstracts
- Simplify-answer button for patient-friendly explanations
- PDF export of the answer, confidence note, citations, and trial context

## Example Questions

- In adults with acute low back pain, do NSAIDs improve pain and function compared with acetaminophen?
- For type 2 diabetes, do GLP-1 receptor agonists reduce cardiovascular events compared with standard care?
- In children with acute otitis media, when is watchful waiting appropriate compared with immediate antibiotics?
- Compare intratympanic steroids versus oral steroids for idiopathic sudden sensorineural hearing loss.
- Over the last 2 years, what randomized controlled trials compare semaglutide and tirzepatide for weight loss?

## How It Works

1. The user submits a question and optional filters.
2. The LangGraph workflow retrieves published evidence from PubMed.
3. The app checks ChromaDB for cached abstracts and reuses them when possible.
4. The app optionally pulls ongoing studies from ClinicalTrials.gov.
5. The LLM synthesizes the evidence, summarizes each abstract, and produces a final cited answer.
6. The frontend displays the live activity feed, confidence score, references, evidence highlights, and export options.

## Screenshots

Add screenshots here after running the app:

- `frontend-home.png`
- `compare-mode-results.png`
- `langsmith-trace.png`
- `pdf-report-preview.png`

## Notes

- PubMed and ClinicalTrials.gov are both free data sources.
- Ollama support enables fully local inference and zero API cost after setup.
- Groq can be used when you want hosted inference and faster responses.
