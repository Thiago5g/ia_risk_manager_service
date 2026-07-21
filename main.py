from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from schemas import (
    TeachResponse,
    AnalyzeResponse,
    AnalyzeEmptyResponse,
    HealthResponse,
    ErrorResponse,
)
from inference import process_and_store_image, compare_image, get_patterns_count

app = FastAPI(
    title="Trade Chart Pattern Recognition",
    description="CLIP-based trade chart pattern recognition using embedding similarity",
    version="1.0.0",
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_image(file: UploadFile) -> None:
    """Validate uploaded file is an image within size limits."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Returns service status and pattern count."""
    return HealthResponse(
        status="ok",
        patterns_count=get_patterns_count(),
    )


@app.post(
    "/teach",
    response_model=TeachResponse,
    responses={400: {"model": ErrorResponse}},
)
async def teach(file: UploadFile = File(...), explanation: str = Form(...)):
    """
    Teach a new chart pattern.

    Upload a trade chart image with an expert annotation. The system generates
    a CLIP embedding and stores it alongside the explanation for future matching.
    """
    validate_image(file)

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    try:
        process_and_store_image(contents, explanation)
        return TeachResponse(
            message="Pattern stored successfully",
            patterns_stored=get_patterns_count(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.post(
    "/analyze",
    response_model=AnalyzeResponse | AnalyzeEmptyResponse,
    responses={400: {"model": ErrorResponse}},
)
async def analyze(file: UploadFile = File(...)):
    """
    Analyze a trade chart against stored patterns.

    Upload a chart image. The system generates a CLIP embedding, computes
    cosine similarity against all stored patterns, and returns the best match
    with its expert annotation.
    """
    validate_image(file)

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    try:
        result = compare_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")

    # No patterns stored yet
    if "message" in result:
        return AnalyzeEmptyResponse(
            message=result["message"],
            patterns_stored=0,
        )

    # Determine confidence level
    similarity = float(result["similarity"])
    if similarity >= 0.85:
        confidence = "high"
    elif similarity >= 0.70:
        confidence = "medium"
    else:
        confidence = "low"

    return AnalyzeResponse(
        similarity=similarity,
        matched_explanation=result["matched_explanation"],
        confidence=confidence,
    )
