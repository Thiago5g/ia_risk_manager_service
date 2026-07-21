from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch
import json
import os
import numpy as np

# Load CLIP model (downloads on first run, cached afterward)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

DB_PATH = os.environ.get("MEMORY_PATH", "./memory.json")


def extract_embedding(image_bytes: bytes) -> np.ndarray:
    """Generate a CLIP embedding vector from raw image bytes."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.squeeze().numpy()


def get_patterns_count() -> int:
    """Return the number of stored patterns."""
    if not os.path.exists(DB_PATH):
        return 0
    try:
        with open(DB_PATH, "r") as f:
            memory = json.load(f)
        return len(memory)
    except (json.JSONDecodeError, IOError):
        return 0


def process_and_store_image(image_bytes: bytes, explanation: str) -> list[float]:
    """
    Generate CLIP embedding for an image and store it with the explanation.

    Returns the embedding vector as a list of floats.
    """
    embedding = extract_embedding(image_bytes).tolist()

    memory: list[dict] = []
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                memory = json.load(f)
        except (json.JSONDecodeError, IOError):
            memory = []

    memory.append({"embedding": embedding, "explanation": explanation})

    with open(DB_PATH, "w") as f:
        json.dump(memory, f)

    return embedding


def compare_image(image_bytes: bytes) -> dict:
    """
    Compare an image against all stored patterns using cosine similarity.

    Returns the best match with similarity score and explanation,
    or a message if no patterns are stored.
    """
    if not os.path.exists(DB_PATH):
        return {"message": "No patterns stored yet. Use /teach to add patterns first."}

    try:
        with open(DB_PATH, "r") as f:
            memory = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"message": "No patterns stored yet. Use /teach to add patterns first."}

    if len(memory) == 0:
        return {"message": "No patterns stored yet. Use /teach to add patterns first."}

    embedding = extract_embedding(image_bytes)

    similarities: list[tuple[float, str]] = []
    for item in memory:
        stored_vec = np.array(item["embedding"])
        sim = cosine_similarity(embedding, stored_vec)
        similarities.append((sim, item["explanation"]))

    similarities.sort(reverse=True, key=lambda x: x[0])

    return {
        "similarity": similarities[0][0],
        "matched_explanation": similarities[0][1],
    }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
