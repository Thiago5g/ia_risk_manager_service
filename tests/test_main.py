import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from main import app


client = TestClient(app)


def make_image_file(color="red", format="PNG"):
    """Create an in-memory image file for upload."""
    img = Image.new("RGB", (100, 100), color=color)
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    content_type = f"image/{'png' if format == 'PNG' else 'jpeg'}"
    return ("chart.png", buffer, content_type)


class TestHealth:
    def test_health_returns_ok(self, temp_memory_path):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["patterns_count"] == 0

    def test_health_counts_patterns(self, temp_memory_path, mock_clip, sample_image_bytes):
        # Teach a pattern first
        client.post(
            "/teach",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
            data={"explanation": "Test pattern"},
        )
        response = client.get("/health")
        assert response.json()["patterns_count"] == 1


class TestTeach:
    def test_teach_stores_pattern(self, temp_memory_path, mock_clip, sample_image_bytes):
        response = client.post(
            "/teach",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
            data={"explanation": "Bullish engulfing at support"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Pattern stored successfully"
        assert data["patterns_stored"] == 1

    def test_teach_multiple_patterns(self, temp_memory_path, mock_clip, sample_image_bytes):
        for i in range(3):
            client.post(
                "/teach",
                files={"file": ("chart.png", sample_image_bytes, "image/png")},
                data={"explanation": f"Pattern {i}"},
            )
        response = client.get("/health")
        assert response.json()["patterns_count"] == 3

    def test_teach_rejects_non_image(self, temp_memory_path):
        response = client.post(
            "/teach",
            files={"file": ("data.csv", BytesIO(b"a,b,c"), "text/csv")},
            data={"explanation": "Should fail"},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_teach_requires_explanation(self, temp_memory_path, sample_image_bytes):
        response = client.post(
            "/teach",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 422  # Missing required field


class TestAnalyze:
    def test_analyze_with_no_patterns(self, temp_memory_path, mock_clip, sample_image_bytes):
        response = client.post(
            "/analyze",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "No patterns stored" in data["message"]

    def test_analyze_returns_best_match(self, temp_memory_path, mock_clip_deterministic, sample_image_bytes):
        # Teach a pattern (uses embedding[0])
        client.post(
            "/teach",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
            data={"explanation": "Bullish pattern"},
        )

        # Analyze (uses embedding[1] which is similar to embedding[0])
        response = client.post(
            "/analyze",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        assert "matched_explanation" in data
        assert data["matched_explanation"] == "Bullish pattern"
        assert 0 <= data["similarity"] <= 1

    def test_analyze_returns_confidence_level(self, temp_memory_path, mock_clip_deterministic, sample_image_bytes):
        # Teach
        client.post(
            "/teach",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
            data={"explanation": "Test"},
        )
        # Analyze
        response = client.post(
            "/analyze",
            files={"file": ("chart.png", sample_image_bytes, "image/png")},
        )
        data = response.json()
        assert data["confidence"] in ("high", "medium", "low")

    def test_analyze_rejects_non_image(self, temp_memory_path):
        response = client.post(
            "/analyze",
            files={"file": ("data.txt", BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400
