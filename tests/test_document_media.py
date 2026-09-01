"""serving the actual bytes a citation points at, not just a filesystem path"""

from pathlib import Path

from fastapi.testclient import TestClient


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, "application/pdf")})


class TestImageFile:
    def test_serves_the_extracted_images_bytes(self, client: TestClient, manual_pdf_bytes: bytes) -> None:
        document_id = _upload(client, manual_pdf_bytes).json()["id"]
        image_id = client.get(f"/documents/{document_id}/images").json()[0]["id"]

        response = client.get(f"/documents/{document_id}/images/{image_id}/file")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_unknown_image_id_is_404(self, client: TestClient, manual_pdf_bytes: bytes) -> None:
        document_id = _upload(client, manual_pdf_bytes).json()["id"]

        assert client.get(f"/documents/{document_id}/images/does-not-exist/file").status_code == 404

    def test_image_from_a_different_document_is_404(
        self, client: TestClient, manual_pdf_bytes: bytes
    ) -> None:
        """an image id alone is not enough, it must belong to the document in the url"""
        first = _upload(client, manual_pdf_bytes, "first.pdf").json()["id"]
        second = _upload(client, manual_pdf_bytes, "second.pdf").json()["id"]
        image_of_first = client.get(f"/documents/{first}/images").json()[0]["id"]

        assert client.get(f"/documents/{second}/images/{image_of_first}/file").status_code == 404


class TestPagePreview:
    def test_renders_a_page_as_png(self, client: TestClient, manual_pdf_bytes: bytes) -> None:
        document_id = _upload(client, manual_pdf_bytes).json()["id"]

        response = client.get(f"/documents/{document_id}/pages/1/preview")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_repeat_requests_use_the_cache(
        self, client: TestClient, manual_pdf_bytes: bytes, settings
    ) -> None:
        document_id = _upload(client, manual_pdf_bytes).json()["id"]

        client.get(f"/documents/{document_id}/pages/1/preview")
        cached_files = list((Path(settings.preview_dir) / document_id).glob("*.png"))

        assert len(cached_files) == 1
        first_mtime = cached_files[0].stat().st_mtime

        client.get(f"/documents/{document_id}/pages/1/preview")

        assert cached_files[0].stat().st_mtime == first_mtime

    def test_page_out_of_range_is_404(self, client: TestClient, manual_pdf_bytes: bytes) -> None:
        document_id = _upload(client, manual_pdf_bytes).json()["id"]

        assert client.get(f"/documents/{document_id}/pages/999/preview").status_code == 404

    def test_unknown_document_is_404(self, client: TestClient) -> None:
        assert client.get("/documents/does-not-exist/pages/1/preview").status_code == 404
