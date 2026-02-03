"""Tests for document server caching functionality."""

import shutil
import time
from pathlib import Path

import pytest

from automas.mcp.servers.document.docx_reader import extract_images_from_docx
from automas.mcp.servers.document.pdf_reader import extract_images_from_pdf
from automas.mcp.servers.document.pptx_reader import extract_images_from_pptx
from automas.mcp.servers.document.utils import get_file_hash
from automas.mcp.servers.document.xlsx_reader import extract_images_from_xlsx


class TestDocumentCaching:
    """Test caching functionality for document image extraction."""

    @pytest.fixture
    def pdf_file(self):
        return "/home/glhf/Documents/repos/automas-research/examples/data/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf"

    @pytest.fixture
    def docx_file(self):
        return "/home/glhf/Documents/repos/automas-research/examples/data/cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb.docx"

    @pytest.fixture
    def pptx_file(self):
        return "/home/glhf/Documents/repos/automas-research/examples/data/a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx"

    @pytest.fixture
    def xlsx_file(self):
        return "/home/glhf/Documents/repos/automas-research/examples/data/7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f.xlsx"

    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        yield
        cache_dirs = [
            Path.home() / ".automas" / "pdf_cache",
            Path.home() / ".automas" / "docx_cache",
            Path.home() / ".automas" / "pptx_cache",
            Path.home() / ".automas" / "xlsx_cache",
            Path.home() / ".automas" / "pdf_images",
            Path.home() / ".automas" / "docx_images",
            Path.home() / ".automas" / "pptx_images",
            Path.home() / ".automas" / "xlsx_images",
        ]
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

    def test_pdf_caching(self, pdf_file):
        if not Path(pdf_file).exists():
            pytest.skip(f"Test file {pdf_file} not found")

        images_first = extract_images_from_pdf(pdf_file)
        images_cached = extract_images_from_pdf(pdf_file)

        assert len(images_first) == len(images_cached)

        for img1, img2 in zip(images_first, images_cached):
            assert img1.page == img2.page
            assert img1.name == img2.name
            assert Path(img2.path).exists()

    def test_docx_caching(self, docx_file):
        if not Path(docx_file).exists():
            pytest.skip(f"Test file {docx_file} not found")

        images_first = extract_images_from_docx(docx_file)
        images_cached = extract_images_from_docx(docx_file)

        assert len(images_first) == len(images_cached)

        for img1, img2 in zip(images_first, images_cached):
            assert img1.filename == img2.filename
            assert img1.content_type == img2.content_type
            assert Path(img2.path).exists()

    def test_pptx_caching(self, pptx_file):
        if not Path(pptx_file).exists():
            pytest.skip(f"Test file {pptx_file} not found")

        images_first = extract_images_from_pptx(pptx_file)
        images_cached = extract_images_from_pptx(pptx_file)

        assert len(images_first) == len(images_cached)

        for img1, img2 in zip(images_first, images_cached):
            assert img1.slide_number == img2.slide_number
            assert img1.shape_name == img2.shape_name
            assert Path(img2.path).exists()

    def test_xlsx_caching(self, xlsx_file):
        if not Path(xlsx_file).exists():
            pytest.skip(f"Test file {xlsx_file} not found")

        images_first = extract_images_from_xlsx(xlsx_file)
        images_cached = extract_images_from_xlsx(xlsx_file)

        assert len(images_first) == len(images_cached)

        for img1, img2 in zip(images_first, images_cached):
            assert img1.sheet_name == img2.sheet_name
            assert img1.cell == img2.cell
            assert Path(img2.path).exists()

    def test_cache_invalidation_on_missing_files(self, pdf_file):
        if not Path(pdf_file).exists():
            pytest.skip(f"Test file {pdf_file} not found")

        images_first = extract_images_from_pdf(pdf_file)

        if images_first:
            Path(images_first[0].path).unlink()

            images_second = extract_images_from_pdf(pdf_file)

            assert len(images_second) > 0
            assert Path(images_second[0].path).exists()

    def test_multiple_extractions_same_images(self, pdf_file):
        if not Path(pdf_file).exists():
            pytest.skip(f"Test file {pdf_file} not found")

        images_1 = extract_images_from_pdf(pdf_file)
        images_2 = extract_images_from_pdf(pdf_file)
        images_3 = extract_images_from_pdf(pdf_file)

        assert len(images_1) == len(images_2) == len(images_3)

        for img in images_2:
            assert Path(img.path).exists()

        for img in images_3:
            assert Path(img.path).exists()

    def test_cache_performance(self, pdf_file):
        if not Path(pdf_file).exists():
            pytest.skip(f"Test file {pdf_file} not found")

        start = time.time()
        images_first = extract_images_from_pdf(pdf_file)
        first_duration = time.time() - start

        start = time.time()
        images_cached = extract_images_from_pdf(pdf_file)
        cached_duration = time.time() - start

        assert len(images_first) == len(images_cached)
        assert cached_duration <= first_duration * 1.5, "Cached should not be significantly slower"
        print(
            f"\nFirst extraction: {first_duration:.4f}s, Cached: {cached_duration:.4f}s, Speedup: {first_duration / cached_duration:.2f}x"
        )

    def test_cache_file_exists(self, pdf_file):
        if not Path(pdf_file).exists():
            pytest.skip(f"Test file {pdf_file} not found")

        cache_dir = Path.home() / ".automas" / "pdf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        file_hash = get_file_hash(pdf_file)
        cache_file = cache_dir / f"{file_hash}.json"

        if cache_file.exists():
            cache_file.unlink()

        assert not cache_file.exists()

        extract_images_from_pdf(pdf_file)

        assert cache_file.exists()

        import json

        with open(cache_file) as f:
            cache_data = json.load(f)

        assert isinstance(cache_data, list)
        if cache_data:
            assert "path" in cache_data[0]
            assert Path(cache_data[0]["path"]).exists()
