import sys
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import graph, fresh_turn
from langgraph.types import Command

def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=client_ip)
app = FastAPI(title="Research Agent - Stage 3 (Corrective RAG)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("5/minute;50/day")
def ask(request: Request, req: AskRequests):
    thread_id = req.thread_id or str(uuid.uuid4())      # honor the client's session, mint only if absent
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(fresh_turn(req.question), config=config)
    return _format(result, thread_id)

@app.post("/resume")
@limiter.limit("5/minute;50/day")
def resume(request: Request, req: ResumeRequest):
    config= {"configurable": {"thread_id": req.thread_id}}
    if not graph.get_state(config).next:
        raise HTTPException(status_code=409, detail="No pending approval on this thread.")
    result = graph.invoke(Command(resume=req.decision), config=config)
    return _format(result, req.thread_id)