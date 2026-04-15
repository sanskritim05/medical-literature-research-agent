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
* [![ChromaDB][ChromaDB]][ChromaDB-url]


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.8 or later
* A [Groq API key](https://console.groq.com)

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
5. Start the app
   ```sh
   uvicorn main:app --reload
   ```
6. Open in your browser
   ```text
   http://127.0.0.1:8000
   ```


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
├── frontend/
│   ├── index.html
│   └── styles.css
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
[ChromaDB]: https://img.shields.io/badge/ChromaDB-E85D4A?style=for-the-badge&logoColor=white
[ChromaDB-url]: https://www.trychroma.com
