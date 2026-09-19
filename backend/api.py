"""HTTP service used by the Streamlit frontend to refresh the knowledge base."""
from threading import Lock
from fastapi import FastAPI, HTTPException
from tn_schemes_ingest import main as run_ingestion

app = FastAPI(title="Tamil Nadu Schemes Ingestion API")
ingestion_lock = Lock()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest():
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion is already in progress.")
    try:
        run_ingestion()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        ingestion_lock.release()
    return {"status": "completed"}

