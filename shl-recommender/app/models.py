from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str = Field(
        default="",
        description="Short code(s) for the SHL test type, e.g. 'K' (Knowledge & Skills), "
        "'P' (Personality & Behavior), 'S' (Simulations), 'A' (Ability & Aptitude), etc.",
    )


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
