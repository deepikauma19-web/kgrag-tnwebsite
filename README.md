# Tamil Nadu Schemes RAG

A Docker-ready Streamlit chatbot that answers from official Tamil Nadu scheme pages. The backend crawls the source pages, stores structured records in Neo4j, and builds a shared FAISS index. The frontend searches that index and asks OpenAI to answer only from the retrieved records.

## Project layout

```text
frontend/              Streamlit chat application
backend/               ingestion API and crawler
docker-compose.yml     starts Neo4j, backend, and frontend
.env.example           required configuration template
```

## Deploy with Docker

1. Install Docker Engine and Docker Compose on the VPS.
2. Copy `.env.example` to `.env`, then set a real `OPENAI_API_KEY` and a strong `NEO4J_PASSWORD`.
3. Start the stack:

```bash
docker compose up --build -d
```

4. Open `http://YOUR_VPS_IP:8501`. Use **Refresh Neo4j and FAISS** in the sidebar to create the initial knowledge base. A full crawl can take several minutes.

The only public application port is 8501. Neo4j and the ingestion API stay inside Docker's private network. For production, place a reverse proxy with TLS in front of port 8501.

## Useful commands

```bash
docker compose logs -f
docker compose down
docker compose up --build -d
```

To test a small crawl first, set `MAX_SCHEMES=5` in `.env`, restart the stack, and use Refresh. Set it back to `0` for all listed schemes.

## Configuration

- `NEO4J_URI` defaults to the bundled Neo4j container: `bolt://neo4j:7687`.
- `SCHEME_START_URL`, `MAX_SCHEMES`, and `REQUEST_DELAY_SECONDS` control crawling.
- `OPENAI_CHAT_MODEL` defaults to `gpt-4o-mini`.
- Data is stored in Docker named volumes, so it persists across container rebuilds. It is intentionally excluded from Git.

## GitHub

Do not commit `.env` or generated data. The included `.gitignore` protects both. Create a repository, then run:

```bash
git init
git add .
git commit -m "Initial Docker deployment"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

