"""regression test: lru_cache'd factories must cache on a plain hashable model
name, not an unhashable settings instance

model constructors are monkeypatched so this needs no real weights
"""

from copilot.core.config import Settings
from copilot.generation import generator as generator_module
from copilot.retrieval import captioner as captioner_module
from copilot.retrieval import embedder as embedder_module
from copilot.retrieval import image_embedder as image_embedder_module


def test_settings_itself_is_genuinely_unhashable() -> None:
    """confirming the premise: without the fix, this is exactly what would raise"""
    try:
        hash(Settings())
        raised = False
    except TypeError:
        raised = True
    assert raised, "Settings became hashable; the rest of this file's premise no longer holds"


def test_get_text_embedder_accepts_a_plain_string_twice(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        embedder_module, "SentenceTransformerEmbedder", lambda name: calls.append(name) or name
    )
    embedder_module.get_text_embedder.cache_clear()

    first = embedder_module.get_text_embedder("BAAI/bge-small-en-v1.5")
    second = embedder_module.get_text_embedder("BAAI/bge-small-en-v1.5")

    assert first is second
    assert calls == ["BAAI/bge-small-en-v1.5"]  # constructed once, not twice


def test_get_image_embedder_accepts_a_plain_string_twice(monkeypatch) -> None:
    monkeypatch.setattr(image_embedder_module, "ClipImageEmbedder", lambda name: name)
    image_embedder_module.get_image_embedder.cache_clear()

    first = image_embedder_module.get_image_embedder("openai/clip-vit-base-patch32")
    second = image_embedder_module.get_image_embedder("openai/clip-vit-base-patch32")

    assert first is second


def test_get_image_captioner_accepts_plain_arguments_twice(monkeypatch) -> None:
    monkeypatch.setattr(
        captioner_module,
        "TransformersImageCaptioner",
        lambda model_name, max_new_tokens, prompt: (model_name, max_new_tokens, prompt),
    )
    captioner_module.get_image_captioner.cache_clear()

    first = captioner_module.get_image_captioner("Salesforce/blip-image-captioning-base")
    second = captioner_module.get_image_captioner("Salesforce/blip-image-captioning-base")

    assert first is second


def test_get_answer_generator_takes_no_settings_argument(monkeypatch) -> None:
    """the real regression case, closed by removing the settings parameter entirely"""
    monkeypatch.setattr(
        generator_module, "LocalLlmAnswerGenerator", lambda model_name, **kw: model_name
    )
    generator_module.get_answer_generator.cache_clear()

    first = generator_module.get_answer_generator()
    second = generator_module.get_answer_generator()

    assert first is second


def test_retrieval_deps_never_passes_a_settings_instance_into_a_cached_call() -> None:
    """static guard against the bug's exact shape re-appearing in retrieval.deps"""
    import ast
    import inspect

    from copilot.retrieval import deps as deps_module

    source = inspect.getsource(deps_module)
    tree = ast.parse(source)

    cached_calls = {"get_text_embedder", "get_image_embedder", "get_image_captioner"}
    offending: list[str] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in cached_calls:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            # the bug's exact shape: passing the bare settings name itself
            if isinstance(arg, ast.Name) and arg.id == "settings":
                offending.append(node.func.id)

    assert offending == [], f"passes a Settings instance directly into: {offending}"
