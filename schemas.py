from pydantic import BaseModel


class TeachResponse(BaseModel):
    message: str
    patterns_stored: int


class AnalyzeResponse(BaseModel):
    similarity: float
    matched_explanation: str
    confidence: str  # high, medium, low


class AnalyzeEmptyResponse(BaseModel):
    message: str
    patterns_stored: int


class HealthResponse(BaseModel):
    status: str
    patterns_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
