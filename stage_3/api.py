import sys
import uuid
import json as _json
import hashlib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional
import anthropic
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import auth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import graph, fresh_turn
from langgraph.types import Command

STATIC_DIR = Path(__file__).resolve().parent / "static"
auth.init()
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_key(key: str = Security(_key_header)):
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key (X-API-Key header).")
    ident = auth.lookup(key)
    if not ident:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return ident

def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

def rate_key(request: Request):
    k = request.headers.get("x-api-key")
    if k:
        return "key:" + hashlib.sha256(k.encode()).hexdigest()[:16]
    return client_ip(request)


limiter = Limiter(key_func=rate_key)
app = FastAPI(title="Research Agent - Stage 3 (Corrective RAG)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class AskRequests(BaseModel):
    question: str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    decision: str

class MintRequest(BaseModel):
    name: str

def _format(result: dict, thread_id: str):
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return {"status": "approval_needed", "prompt": intr.value["prompt"], "thread_id": thread_id}
    return {"status": "done", "answer": result["answer"], "thread_id": thread_id}

def _authorize_thread(thread_id: str, key_id: str):
    owner = auth.thread_owner(thread_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Unknown thread.")
    if owner != key_id:
        raise HTTPException(status_code=403, detail="Not your thread.")

def _sse(obj):
    return f"data: {_json.dumps(obj)}\n\n"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/ask")
@limiter.limit("5/minute;50/day")
def ask(request: Request, req: AskRequests, ident=Depends(require_key)):
    key_id, _ = ident
    if req.thread_id:
        _authorize_thread(req.thread_id, key_id)
        thread_id = req.thread_id
    else:
        thread_id = str(uuid.uuid4())
        auth.claim_thread(thread_id, key_id)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(fresh_turn(req.question), config=config)
    return _format(result, thread_id)

@app.post("/resume")
@limiter.limit("5/minute;50/day")
def resume(request: Request, req: ResumeRequest, ident=Depends(require_key)):
    _authorize_thread(req.thread_id, ident[0])   # ownership FIRST: strangers learn nothing
    config= {"configurable": {"thread_id": req.thread_id}}
    if not graph.get_state(config).next:
        raise HTTPException(status_code=409, detail="No pending approval on this thread.")
    result = graph.invoke(Command(resume=req.decision), config=config)
    return _format(result, req.thread_id)

@app.exception_handler(anthropic.APIError)
def llm_unavailable(request: Request, exc: anthropic.APIError):
    return JSONResponse(status_code=503,
                        content={"status": "error", "detail": "Model provider unavailable - please retry."})

@app.post("/ask/stream")
@limiter.limit("5/minute;50/day")
def ask_stream(request: Request, req: AskRequests, ident=Depends(require_key)):
    key_id, _ = ident
    if req.thread_id:
        _authorize_thread(req.thread_id, key_id)
        thread_id = req.thread_id
    else:
        thread_id = str(uuid.uuid4())
        auth.claim_thread(thread_id, key_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    def events():
        yield _sse({"event": "start", "thread_id": thread_id})
        final = {}
        for update in graph.stream(fresh_turn(req.question), config=config,
                                   stream_mode="updates"):
            for node, out in update.items():
                if node == "__interrupt__":
                    yield _sse({"event": "approval_needed",
                                "prompt": out[0].value["prompt"], "thread_id": thread_id})
                    return
                final = out or final
                yield _sse({"event": "node", "node": node})
        yield _sse({"event": "answer", "answer": final.get("answer", ""),
                    "thread_id": thread_id})
    
    return StreamingResponse(events(), media_type="text/event-stream")

@app.post("/admin/keys")
def mint_key(request: Request, req: MintRequest, ident=Depends(require_key)):
    if not ident[1]:
        raise HTTPException(status_code=403, detail="Admin only.")
    return {"name": req.name, "key": auth.mint(req.name)}