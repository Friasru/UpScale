from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from upscale.config import CORS_ORIGINS
from upscale.orchestrator import Orchestrator
from upscale.schemas import ChatRequest, ChatResponse

app = FastAPI(title="UpScale API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

orchestrator = Orchestrator()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    return await orchestrator.respond(request)
