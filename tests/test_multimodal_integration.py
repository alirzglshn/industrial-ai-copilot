"""image embedding with the real clip model

deselected by default, downloads weights on first run:
    pytest -m integration

covers clip's plumbing, deterministic and worth asserting; not its ranking
quality on schematics, which is the known weak point page context exists for
"""

from pathlib import Path

import pytest
from PIL import Image as PILImage

from copilot.retrieval.image_embedder import CLIP_MAX_TOKENS, ClipImageEmbedder

pytestmark = pytest.mark.integration

MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def clip() -> ClipImageEmbedder:
    # module-scoped, since loading clip dominates the runtime of these tests
    return ClipImageEmbedder(MODEL)


@pytest.fixture
def photo(tmp_path: Path):
    def make(name: str, color: tuple[int, int, int]) -> Path:
        path = tmp_path / f"{name}.png"
        PILImage.new("RGB", (224, 224), color=color).save(path)
        return path

    return make


def test_projection_width_is_what_the_collection_is_sized_to(clip: ClipImageEmbedder) -> None:
    assert clip.dimension == 512


def test_images_and_text_land_in_the_same_space(clip: ClipImageEmbedder, photo) -> None:
    """text to image search only works because both towers share one space"""
    image_vectors = clip.embed_images([str(photo("red", (200, 30, 30)))])
    text_vectors = clip.embed_texts(["a photograph"])

    assert len(image_vectors[0]) == len(text_vectors[0]) == clip.dimension


def test_vectors_are_normalized(clip: ClipImageEmbedder, photo) -> None:
    """qdrant's cosine distance and the caption-averaging both assume this"""
    image_vector = clip.embed_images([str(photo("blue", (30, 30, 200)))])[0]
    text_vector = clip.embed_query("a diagram")

    assert sum(v * v for v in image_vector) == pytest.approx(1.0, rel=1e-4)
    assert sum(v * v for v in text_vector) == pytest.approx(1.0, rel=1e-4)


def test_a_colour_query_prefers_the_matching_image(clip: ClipImageEmbedder, photo) -> None:
    """a deliberately easy, in-distribution discrimination, checking wiring not schematic skill"""
    red, blue = photo("red", (220, 20, 20)), photo("blue", (20, 20, 220))
    vectors = clip.embed_images([str(red), str(blue)])
    query = clip.embed_query("a solid red image")

    red_score = sum(a * b for a, b in zip(query, vectors[0]))
    blue_score = sum(a * b for a, b in zip(query, vectors[1]))
    assert red_score > blue_score


def test_unreadable_file_yields_none_without_failing_the_batch(
    clip: ClipImageEmbedder, photo, tmp_path: Path
) -> None:
    good = photo("good", (100, 140, 90))
    missing = tmp_path / "not_here.png"

    vectors = clip.embed_images([str(good), str(missing)])

    assert vectors[0] is not None
    assert vectors[1] is None


def test_overlong_caption_is_truncated_rather_than_raising(clip: ClipImageEmbedder) -> None:
    """clip's text encoder is capped at 77 tokens, captions can exceed it"""
    long_caption = " ".join(["impeller"] * (CLIP_MAX_TOKENS * 3))

    vectors = clip.embed_texts([long_caption])

    assert len(vectors[0]) == clip.dimension


def test_embedding_nothing_is_a_no_op(clip: ClipImageEmbedder) -> None:
    assert clip.embed_images([]) == []
    assert clip.embed_texts([]) == []
