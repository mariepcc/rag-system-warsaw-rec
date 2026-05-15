from schemas.places import PlaceResponse
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if v is None or not str(v).strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 10_000:
            raise ValueError("Message too long — max 10,000 characters")
        return v


class ChatResponse(BaseModel):
    answer: str
    type: Optional[Literal["rag", "followup", "hybrid"]] = None
    session_id: str
    places: list[PlaceResponse] = []
    enough_context: bool


class MessageFilter(BaseModel):
    field: str
    op: str
    value: float | str | int


class ClassificationResult(BaseModel):
    message_type: Literal["rag", "followup", "hybrid"] = Field(
        description="Typ wiadomości: nowe zapytanie, followup, lub hybrid"
    )
    reasoning: str = Field(description="Krótkie uzasadnienie klasyfikacji")
    reformulated_query: Optional[str] = Field(
        default=None,
        description="Przepisane zapytanie uwzględniające kontekst historii (dla rag i hybrid)",
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    message_type: Optional[str] = None
    recommended_places: Optional[Any] = None
    content: str


class SynthesizedResponse(BaseModel):
    answer: str = Field(description="Odpowiedź dla użytkownika")
    type: Optional[Literal["rag", "followup", "hybrid"]] = Field(
        description="Typ wiadomości: nowe zapytanie, followup, lub hybrid"
    )
    enough_context: bool = Field(description="Czy asystent miał wystarczający kontekst")
    recommended_places: List[PlaceResponse] = Field(default_factory=list)
    recommended_place_names: List[str] = Field(
        default_factory=list,
        description="Nazwy poleconych miejsc (do zapisu w historii)",
    )
    _context: Any = None


class ChatSessionSearchResponse(BaseModel):
    id: str
    created_at: datetime
    first_message: Optional[str] = None
