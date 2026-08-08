<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Medical Literature Research Agent</h3>

  <p align="center">
    A Python web app that answers clinical questions by searching PubMed, summarizing abstracts, and returning evidence-based answers with citations.
  </p>
</div>

Demo [https://youtu.be/He-Nh3Mv7L0]
<!-- ABOUT THE PROJECT -->
## About The Project

Medical Literature Research Agent helps users explore medical evidence through a simple web interface. Enter a clinical question in natural language, and the app searches PubMed, retrieves abstracts, summarizes the findings, and produces a final answer with inline citations and linked references.

It also supports optional filters, treatment comparison, session-based follow-up questions, PDF export, and ongoing trial lookup from ClinicalTrials.gov.


### Built With

* [![Python][Python.org]][Python-url]
* [![FastAPI][FastAPI.tiangolo.com]][FastAPI-url]
* [![LangGraph][LangGraph]][LangGraph-url]
* [![Groq][Groq.com]][Groq-url]
* [![Vercel][Vercel.com]][Vercel-url]
* [![React][React.js]][React-url]


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.10 or later
* A [Groq API key](https://console.groq.com) (**required**)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/sanskritim05/medical-literature-research-agent.git
   ```
2. Create the environment file
   ```sh
   cp .env.example .env
   ```
3. Add your credentials to `.env`
   ```sh
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.1-8b-instant
   ```
4. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
5. Install frontend dependencies and build (or run Vite in a second terminal)
   ```sh
   cd web && npm install && npm run build && cd ..
   ```
6. Start the app
   ```sh
   uvicorn main:app --reload
   ```
7. Open in your browser
   ```text
   http://127.0.0.1:8000
   ```

For UI hot reload during development, run `uvicorn main:app --reload` and `cd web && npm run dev` (Vite proxies `/api` to the backend).


## Deploy on Vercel

This app is configured for Vercel’s FastAPI runtime (`main.py` + `vercel.json`).

1. Push the repo to GitHub and import it in [Vercel](https://vercel.com/new).
2. In **Project Settings → Environment Variables**, add:

   | Name | Required | Notes |
   |------|----------|--------|
   | `GROQ_API_KEY` | **Yes** | From [console.groq.com](https://console.groq.com) |
   | `LLM_PROVIDER` | Recommended | Set to `groq` |
   | `GROQ_MODEL` | Optional | Default `llama-3.1-8b-instant` |
   | `NCBI_API_KEY` | Optional | Free NCBI key helps avoid PubMed rate limits on shared IPs |
   | `NCBI_EMAIL` | Optional | Contact email for NCBI E-utilities etiquette |
   | `LANGSMITH_API_KEY` | Optional | Only if you enable LangSmith tracing |

3. Leave **Output Directory** blank in Vercel project settings (do not set it to `public`).
4. Deploy. Research requests can take a while; `vercel.json` sets `maxDuration` to **300 seconds**.

5. Or deploy from the CLI:
   ```sh
   npx vercel
   ```
   Then set the same env vars in the Vercel dashboard (or with `npx vercel env add`).

**Notes**
- Ollama is for local use only; do not set `LLM_PROVIDER=ollama` on Vercel.
- PubMed + ClinicalTrials.gov need no paid keys.
- Session memory is in-process (ephemeral on serverless). Browser history still works via `localStorage`.


<!-- USAGE -->
## Usage

1. Enter a clinical question in natural language.
2. Optionally select a study type or date range.
3. Run the search.
4. Review the final answer, inline citations, and linked references.
5. Optionally compare two treatments, simplify the answer, or export results as a PDF.


<!-- EXAMPLE QUESTIONS -->
## Example Questions

* In adults with acute low back pain, do NSAIDs improve pain and function compared with acetaminophen?
* For type 2 diabetes, do GLP-1 receptor agonists reduce cardiovascular events compared with standard care?
* In children with acute otitis media, when is watchful waiting appropriate compared with immediate antibiotics?
* Compare intratympanic steroids versus oral steroids for idiopathic sudden sensorineural hearing loss.


<!-- PROJECT STRUCTURE -->
## Project Structure

```text
medical-literature-research-agent/
├── main.py
├── agent.py
├── pubmed_tool.py
├── web/                 # Evidentia React UI (Vite)
│   ├── src/
│   └── package.json
├── frontend_dist/       # Built frontend (npm run build)
├── vercel.json
├── runtime.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/sanskritim05/medical-literature-research-agent.svg?style=for-the-badge
[contributors-url]: https://github.com/sanskritim05/medical-literature-research-agent/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sanskritim05/medical-literature-research-agent.svg?style=for-the-badge
[forks-url]: https://github.com/sanskritim05/medical-literature-research-agent/network/members
[stars-shield]: https://img.shields.io/github/stars/sanskritim05/medical-literature-research-agent.svg?style=for-the-badge
[stars-url]: https://github.com/sanskritim05/medical-literature-research-agent/stargazers
[issues-shield]: https://img.shields.io/github/issues/sanskritim05/medical-literature-research-agent.svg?style=for-the-badge
[issues-url]: https://github.com/sanskritim05/medical-literature-research-agent/issues
[license-shield]: https://img.shields.io/github/license/sanskritim05/medical-literature-research-agent.svg?style=for-the-badge
[license-url]: https://github.com/sanskritim05/medical-literature-research-agent/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/your_username
[product-screenshot]: images/screenshot.png
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org
[FastAPI.tiangolo.com]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com
[LangGraph]: https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white
[LangGraph-url]: https://github.com/langchain-ai/langgraph
[Groq.com]: https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white
[Groq-url]: https://groq.com
[Vercel.com]: https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white
[Vercel-url]: https://vercel.com
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://react.dev
