import sys
import uuid
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import graph, fresh_turn
from langgraph.types import Command

app = FastAPI(title="Research Agent - Stage 3 (Corrective RAG)")

class AskRequests(BaseModel):
    question: str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    decision: str

def _format(result: dict, thread_id: str):
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return {"status": "approval_needed", "prompt": intr.value["prompt"], "thread_id": thread_id}
    return {"status": "done", "answer": result["answer"], "thread_id": thread_id}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequests):
    thread_id = req.thread_id or str(uuid.uuid4())      # honor the client's session, mint only if absent
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(fresh_turn(req.question), config=config)
    return _format(result, thread_id)

