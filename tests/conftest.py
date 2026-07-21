import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
from io import BytesIO
from PIL import Image


@pytest.fixture
def temp_memory_path(tmp_path):
    """Provide a temporary memory.json path for tests."""
    path = str(tmp_path / "memory.json")
    with patch.dict(os.environ, {"MEMORY_PATH": path}):
        # Also patch the module-level DB_PATH
        import inference
        original_path = inference.DB_PATH
        inference.DB_PATH = path
        yield path
        inference.DB_PATH = original_path


@pytest.fixture
def sample_image_bytes():
    """Generate a simple test image as bytes."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_clip():
    """Mock CLIP model to avoid downloading weights in tests."""
    fake_embedding = np.random.randn(512).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)

    with patch("inference.extract_embedding") as mock:
        mock.return_value = fake_embedding
        yield mock


@pytest.fixture
def mock_clip_deterministic():
    """Mock CLIP with deterministic embeddings based on call count."""
    call_count = [0]
    embeddings = [
        np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32),
        np.array([0.9, 0.1, 0.0] + [0.0] * 509, dtype=np.float32),  # Similar to first
        np.array([0.0, 0.0, 1.0] + [0.0] * 509, dtype=np.float32),  # Different
    ]

    def side_effect(image_bytes):
        idx = call_count[0] % len(embeddings)
        call_count[0] += 1
        return embeddings[idx]

    with patch("inference.extract_embedding") as mock:
        mock.side_effect = side_effect
        yield mock
