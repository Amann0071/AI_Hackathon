import uuid
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.agent import scheme_agent, AgentState

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Government Scheme-Matching Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Simple in-memory session store for hackathon/local runs
SESSIONS: Dict[str, AgentState] = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


def _new_state() -> AgentState:
    greeting = (
        "Namaste! I can help you discover eligible Indian government schemes and show "
        "near-miss options with clear reasons. You can chat in English or Hindi—what "
        "is your approximate age and what do you currently do (e.g., student, entrepreneur, "
        "small business, or looking for work)?"
    )
    return {
        "messages": [{"role": "assistant", "content": greeting}],
        "profile": {},
        "language": "English",
        "results": {},
        "reply": greeting,
        "is_complete": False,
    }


@app.get("/")
def serve_home():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.post("/reset")
def reset_session(payload: Dict[str, Any] | None = None):
    sid = (payload or {}).get("session_id") or str(uuid.uuid4())
    SESSIONS[sid] = _new_state()
    return {
        "session_id": sid,
        "reply": SESSIONS[sid]["reply"],
        "profile": SESSIONS[sid]["profile"],
        "results": SESSIONS[sid]["results"],
        "is_complete": False,
    }


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    sid = req.session_id or str(uuid.uuid4())
    if sid not in SESSIONS:
        SESSIONS[sid] = _new_state()

    state = SESSIONS[sid]
    updated_messages: List[Dict[str, str]] = list(state.get("messages") or [])
    updated_messages.append({"role": "user", "content": req.message.strip()})

    graph_input: AgentState = {
        "messages": updated_messages,
        "profile": dict(state.get("profile") or {}),
        "language": state.get("language") or "English",
        "results": dict(state.get("results") or {}),
        "reply": "",
        "is_complete": False,
    }

    try:
        output = scheme_agent.invoke(graph_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    updated_messages.append({"role": "assistant", "content": output.get("reply", "")})
    output["messages"] = updated_messages
    SESSIONS[sid] = output

    return {
        "session_id": sid,
        "reply": output.get("reply", ""),
        "profile": output.get("profile", {}),
        "language": output.get("language", "English"),
        "results": output.get("results", {}),
        "is_complete": output.get("is_complete", False),
    }