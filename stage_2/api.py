from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from pathlib import Path

from agent import answer

def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

STATIC_DIR = Path(__file__).resolve().parent / "static"
limiter = Limiter(key_func=client_ip)
app = FastAPI(title="Research Agent (Stage 2)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/ask")
@limiter.limit("5/minute;50/day")
def ask(request: Request, req: AskRequest):
    result = answer(req.question)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "iterations": result["iterations"],
        "timing": result["timing"],
        "usage": result["usage"],
    }