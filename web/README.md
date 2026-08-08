# Evidentia web UI

React + Vite frontend for the Medical Literature Research Agent.

## Local development

Terminal 1 — API:
```sh
uvicorn main:app --reload
```

Terminal 2 — UI:
```sh
cd web
npm install
npm run dev
```

Open the Vite URL (proxies `/api` to `http://127.0.0.1:8000`).

## Production build

```sh
cd web && npm install && npm run build
```

Outputs static files to `../public` for FastAPI / Vercel.
